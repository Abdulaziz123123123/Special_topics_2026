from dataclasses import dataclass
from typing import Tuple
import logging
import os


@dataclass
class ExperimentConfig:
    home_dir: str = os.path.dirname(os.path.abspath(__file__))

    seed: int = 1
    device: str = "cpu"
    lr: float = 1e-2
    batch_size: int = 16
    max_epochs: int = 300
    num_workers: int = 3
    alpha_list: Tuple[float, ...] = (0.0,)
    embedding_dim: int = 512
    fusion: str = "linear"  # linear | gated

    version: str = "v3"
    seen_lang: str = "German"  # English | Hindi | German

    # Professor's requested experiment:
    # Select this ratio separately within every class.
    # The selected modality is replaced with zeros.
    # No sample is deleted.
    train_missing_modality: str = "face"
    missing_ratio: float = 0.80
    missing_seed: int = 1
    apply_class_balanced_missing: bool = True

    debug: bool = False

    early_stop: bool = True
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.2
    early_stop_metric: str = "seen"  # seen | unseen | p3 | p4 | p5 | p6

    @property
    def log_level(self):
        return logging.DEBUG if self.debug else logging.INFO

    @property
    def resolved_num_classes(self):
        if self.version == "v1":
            return 70
        if self.version == "v2":
            return 84
        if self.version == "v3":
            return 36

        raise ValueError(f"Unknown version '{self.version}'")

    @property
    def unseen_lang(self):
        if self.version == "v1" and self.seen_lang == "Urdu":
            return "English"
        if self.version == "v2" and self.seen_lang == "Hindi":
            return "English"
        if self.version == "v3" and self.seen_lang == "German":
            return "English"

        raise ValueError(
            f"Invalid version '{self.version}' "
            f"or seen_lang '{self.seen_lang}'."
        )
