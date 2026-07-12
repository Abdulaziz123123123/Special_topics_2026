import copy
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import ExperimentConfig
from utils.featLoader import LoadData
from model import FOP
from utils.trainer import Trainer
from utils.evaluator import Evaluator
from utils.earlystop import EarlyStopping


class ClassBalancedMissingModalityDataset(Dataset):
    """
    Keep every original training sample, but replace one modality with zeros
    for a fixed, class-balanced subset.

    Example:
        10 classes x 10 samples = 100 samples
        ratio = 0.20
        -> select 2 samples from every class
        -> zero the selected face embeddings
        -> dataset length stays 100 (not 80 and not 120)
    """

    def __init__(
        self,
        base_dataset,
        ratio,
        modality="face",
        seed=1,
    ):
        if modality not in {"face", "audio"}:
            raise ValueError("modality must be 'face' or 'audio'")

        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio must be between 0.0 and 1.0")

        self.base_dataset = base_dataset
        self.ratio = float(ratio)
        self.modality = modality
        self.seed = int(seed)

        labels = np.asarray(base_dataset.labels).reshape(-1)

        if len(labels) != len(base_dataset):
            raise ValueError(
                "The number of labels does not match the dataset length."
            )

        self.missing_mask = np.zeros(len(base_dataset), dtype=bool)
        self.class_stats = {}

        rng = np.random.default_rng(self.seed)

        for class_label in np.unique(labels):
            class_indices = np.flatnonzero(labels == class_label)
            class_size = len(class_indices)

            # Round to the nearest whole sample, with .5 rounded upward.
            # For 10 samples and ratio 0.20, this gives exactly 2.
            num_missing = int(np.floor(class_size * self.ratio + 0.5))
            num_missing = min(class_size, max(0, num_missing))

            if num_missing > 0:
                selected = rng.choice(
                    class_indices,
                    size=num_missing,
                    replace=False,
                )
                self.missing_mask[selected] = True

            self.class_stats[str(class_label)] = {
                "total": int(class_size),
                "zeroed": int(num_missing),
                "complete": int(class_size - num_missing),
            }

    def __len__(self):
        # No sample is removed.
        return len(self.base_dataset)

    @staticmethod
    def _zero_copy(value):
        """
        Return a zero-valued copy without changing the original stored feature.
        """
        if torch.is_tensor(value):
            return torch.zeros_like(value)

        array = np.asarray(value)
        return np.zeros_like(array)

    def __getitem__(self, index):
        # The existing PolySim loader returns: audio, face, label
        audio, face, label = self.base_dataset[index]

        if self.missing_mask[index]:
            if self.modality == "face":
                face = self._zero_copy(face)
            else:
                audio = self._zero_copy(audio)

        # The selected sample is returned normally with the same label.
        return audio, face, label

    def summary(self):
        total = len(self.base_dataset)
        zeroed = int(self.missing_mask.sum())

        return {
            "strategy": "fixed_class_balanced",
            "modality": self.modality,
            "requested_ratio": self.ratio,
            "dataset_size_before": total,
            "dataset_size_after": len(self),
            "zeroed_samples": zeroed,
            "complete_samples": total - zeroed,
            "actual_ratio": (zeroed / total) if total else 0.0,
            "per_class": self.class_stats,
        }


def save_checkpoint(model, optimizer, config, epoch, metric_value, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "metric": metric_value,
        "early_stop_metric": config.early_stop_metric,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "config": vars(config),
    }

    torch.save(checkpoint, save_path)


def save_results(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


def setup_logger(config):
    logger = logging.getLogger("Experiment")
    logger.setLevel(config.log_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(levelname)s][%(name)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def make_loader(
    csv_path,
    config,
    shuffle=False,
    apply_training_missingness=False,
    logger=None,
):
    base_dataset = LoadData(
        csv_path=csv_path,
        config=config,
        audio_encoder="ecappa_feats_path",
        modality="audiovisual",
    )

    loader_dataset = base_dataset
    missing_summary = None

    if apply_training_missingness and config.apply_class_balanced_missing:
        loader_dataset = ClassBalancedMissingModalityDataset(
            base_dataset=base_dataset,
            ratio=config.missing_ratio,
            modality=config.train_missing_modality,
            seed=config.missing_seed,
        )
        missing_summary = loader_dataset.summary()

        if logger is not None:
            logger.info(
                "Class-balanced training masking | "
                "Before=%d | After=%d | Zeroed=%d | Actual ratio=%.4f",
                missing_summary["dataset_size_before"],
                missing_summary["dataset_size_after"],
                missing_summary["zeroed_samples"],
                missing_summary["actual_ratio"],
            )

            for class_label, class_info in missing_summary["per_class"].items():
                logger.debug(
                    "Class %s | Total=%d | Zeroed=%d | Complete=%d",
                    class_label,
                    class_info["total"],
                    class_info["zeroed"],
                    class_info["complete"],
                )

    loader = DataLoader(
        loader_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
        drop_last=False,
    )

    return base_dataset, loader, missing_summary


def choose_monitor_value(config, p3, p4, p5, p6):
    """Select the metric used for checkpointing and early stopping."""
    metric_name = config.early_stop_metric.lower()

    metric_map = {
        "seen": p3,
        "p3": p3,
        "p4": p4,
        "unseen": p5,
        "p5": p5,
        "p6": p6,
    }

    if metric_name not in metric_map:
        raise ValueError(
            "early_stop_metric must be one of: "
            "seen, unseen, p3, p4, p5, p6"
        )

    return metric_map[metric_name]


def main():
    # --------------------------------------------------
    # Config and reproducibility
    # --------------------------------------------------
    config = ExperimentConfig()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if config.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    logger = setup_logger(config)

    logger.info("=== Experiment started ===")
    logger.info(
        "Seed=%d | Device=%s | Fusion=%s | Version=%s | "
        "Train_Lang=%s | #Classes=%d | UnSeen_Lang=%s | "
        "Missing=%s | Ratio=%.2f | Strategy=class-balanced fixed subset",
        config.seed,
        config.device,
        config.fusion,
        config.version,
        config.seen_lang,
        config.resolved_num_classes,
        config.unseen_lang,
        config.train_missing_modality,
        config.missing_ratio,
    )

    # --------------------------------------------------
    # CSV paths
    # --------------------------------------------------
    train_csv = (
        f"./feature_tracker/"
        f"{config.version}_train_{config.seen_lang}.csv"
    )
    test_csv = (
        f"./feature_tracker/"
        f"{config.version}_test_{config.seen_lang}.csv"
    )
    unseen_csv = (
        f"./feature_tracker/"
        f"{config.version}_test_{config.unseen_lang}.csv"
    )

    logger.info("Train CSV: %s", train_csv)
    logger.info("Test  CSV: %s", test_csv)
    logger.info("Unseen CSV: %s", unseen_csv)

    # --------------------------------------------------
    # Data loaders
    # --------------------------------------------------
    # Apply the professor's requested procedure ONLY to training data:
    # select a fixed percentage within each class, zero that modality,
    # and keep all samples in the dataset.
    _, train_loader, train_missing_summary = make_loader(
        train_csv,
        config,
        shuffle=True,
        apply_training_missingness=True,
        logger=logger,
    )

    # Evaluation datasets remain unchanged here.
    test_dataset, _, _ = make_loader(
        test_csv,
        config,
        shuffle=False,
        apply_training_missingness=False,
        logger=logger,
    )

    unseen_test_dataset, _, _ = make_loader(
        unseen_csv,
        config,
        shuffle=False,
        apply_training_missingness=False,
        logger=logger,
    )

    # --------------------------------------------------
    # Infer feature dimensions
    # --------------------------------------------------
    audio, face, _ = next(iter(train_loader))

    logger.info(
        "Feature dimensions | Audio=%d | Face=%d",
        audio.shape[1],
        face.shape[1],
    )

    model = FOP(
        config=config,
        face_dim=face.shape[1],
        voice_dim=audio.shape[1],
    ).to(config.device)

    logger.info(
        "Model initialized | Params=%.2fM",
        sum(parameter.numel() for parameter in model.parameters()) / 1e6,
    )

    # IMPORTANT:
    # The selected samples have already been zeroed by the dataset wrapper.
    # Disable any old random mini-batch dropout inside Trainer to avoid
    # applying missingness twice.
    trainer_config = copy.copy(config)
    trainer_config.missing_ratio = 0.0

    trainer = Trainer(model, trainer_config)
    evaluator = Evaluator(model, config)

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for alpha in config.alpha_list:
        logger.info("=== Training with alpha=%.3f ===", alpha)

        best_metric = -float("inf")
        best_epoch = -1

        best_scores = {
            "p3": {"accuracy": -float("inf"), "epoch": -1},
            "p4": {"accuracy": -float("inf"), "epoch": -1},
            "p5": {"accuracy": -float("inf"), "epoch": -1},
            "p6": {"accuracy": -float("inf"), "epoch": -1},
        }

        history = []

        ratio_tag = f"{int(round(config.missing_ratio * 100)):03d}pct"

        checkpoint_path = (
            f"./checkpoints/{config.version}_{config.seen_lang}_"
            f"class_balanced_{config.train_missing_modality}_dropout_"
            f"{ratio_tag}_alpha{alpha}_best.pt"
        )

        results_path = (
            f"./results/{config.version}_{config.seen_lang}_"
            f"class_balanced_{config.train_missing_modality}_dropout_"
            f"{ratio_tag}_alpha{alpha}_p3_p6.json"
        )

        early_stopper = EarlyStopping(
            patience=config.early_stop_patience,
            min_delta=config.early_stop_min_delta,
        )

        for epoch in range(config.max_epochs):
            loss = trainer.train_epoch(train_loader, alpha)

            # P3: seen language, both modalities available
            p3 = evaluator.accuracy(
                test_dataset,
                missing_face=False,
            )

            # P4: seen language, face completely missing at inference
            p4 = evaluator.accuracy(
                test_dataset,
                missing_face=True,
            )

            # P5: unseen language, both modalities available
            p5 = evaluator.accuracy(
                unseen_test_dataset,
                missing_face=False,
            )

            # P6: unseen language, face completely missing at inference
            p6 = evaluator.accuracy(
                unseen_test_dataset,
                missing_face=True,
            )

            scores = {
                "p3": p3,
                "p4": p4,
                "p5": p5,
                "p6": p6,
            }

            for name, value in scores.items():
                if value > best_scores[name]["accuracy"]:
                    best_scores[name] = {
                        "accuracy": value,
                        "epoch": epoch,
                    }

            monitor_value = choose_monitor_value(
                config,
                p3,
                p4,
                p5,
                p6,
            )

            if monitor_value > best_metric:
                best_metric = monitor_value
                best_epoch = epoch

                save_checkpoint(
                    model=model,
                    optimizer=trainer.opt,
                    config=config,
                    epoch=epoch,
                    metric_value=monitor_value,
                    save_path=checkpoint_path,
                )

            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss),
                    "p3": p3,
                    "p4": p4,
                    "p5": p5,
                    "p6": p6,
                    "monitor_metric": config.early_stop_metric,
                    "monitor_value": monitor_value,
                }
            )

            logger.info(
                "[alpha=%.3f] Epoch %03d | Loss %.4f | "
                "P3 %.2f | P4 %.2f | P5 %.2f | P6 %.2f",
                alpha,
                epoch,
                loss,
                p3,
                p4,
                p5,
                p6,
            )

            if (
                config.early_stop
                and early_stopper.step(monitor_value)
            ):
                logger.info(
                    "Early stopping triggered at epoch %d "
                    "(best %s accuracy = %.2f)",
                    epoch,
                    config.early_stop_metric,
                    early_stopper.best_score,
                )
                break

        results = {
            "version": config.version,
            "train_language": config.seen_lang,
            "unseen_language": config.unseen_lang,
            "train_missing_modality": config.train_missing_modality,
            "train_missing_ratio_requested": config.missing_ratio,
            "train_missing_strategy": "fixed_class_balanced",
            "train_missing_summary": train_missing_summary,
            "alpha": alpha,
            "early_stop_metric": config.early_stop_metric,
            "best_monitored_metric": best_metric,
            "best_monitored_epoch": best_epoch,
            "best_scores_observed": best_scores,
            "checkpoint_path": checkpoint_path,
            "history": history,
        }

        save_results(results, results_path)

        logger.info("=== Best scores observed for this run ===")
        logger.info(
            "P3 %.2f (epoch %d) | P4 %.2f (epoch %d) | "
            "P5 %.2f (epoch %d) | P6 %.2f (epoch %d)",
            best_scores["p3"]["accuracy"],
            best_scores["p3"]["epoch"],
            best_scores["p4"]["accuracy"],
            best_scores["p4"]["epoch"],
            best_scores["p5"]["accuracy"],
            best_scores["p5"]["epoch"],
            best_scores["p6"]["accuracy"],
            best_scores["p6"]["epoch"],
        )
        logger.info("Checkpoint saved to: %s", checkpoint_path)
        logger.info("Detailed results saved to: %s", results_path)

    logger.info("=== Experiment finished ===")


if __name__ == "__main__":
    main()
