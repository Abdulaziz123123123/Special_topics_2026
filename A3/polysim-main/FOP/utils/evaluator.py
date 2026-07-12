import torch


class Evaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self._cached = {}

    def _get_tensors(self, dataset):
        """
        Cache tensors to avoid repeated torch.from_numpy calls.
        """
        key = id(dataset)

        if key not in self._cached:
            self._cached[key] = (
                torch.from_numpy(dataset.face_feats).float(),
                torch.from_numpy(dataset.audio_feats).float(),
                torch.from_numpy(dataset.labels).long(),
            )

        return self._cached[key]

    @staticmethod
    def _apply_missing_modality(
        face,
        audio,
        missing_face=False,
        missing_audio=False,
    ):
        """
        missing_face=True:
            Replace every face embedding with zeros (P4/P6).

        missing_audio=True:
            Replace every audio embedding with zeros.
        """
        if missing_face:
            face = torch.zeros_like(face)

        if missing_audio:
            audio = torch.zeros_like(audio)

        return face, audio

    def accuracy_from_tensors(
        self,
        face,
        audio,
        labels,
        missing_face=False,
        missing_audio=False,
    ):
        self.model.eval()

        face = face.to(
            self.config.device,
            non_blocking=True,
        )
        audio = audio.to(
            self.config.device,
            non_blocking=True,
        )
        labels = labels.to(
            self.config.device,
            non_blocking=True,
        )

        face, audio = self._apply_missing_modality(
            face,
            audio,
            missing_face=missing_face,
            missing_audio=missing_audio,
        )

        with torch.no_grad():
            _, logits, _, _ = self.model(face, audio)
            predictions = logits.argmax(dim=1)
            correct = (predictions == labels).sum().item()

        return 100.0 * correct / labels.size(0)

    def accuracy(
        self,
        dataset,
        missing_face=False,
        missing_audio=False,
    ):
        self.model.eval()

        face, audio, labels = self._get_tensors(dataset)

        return self.accuracy_from_tensors(
            face=face,
            audio=audio,
            labels=labels,
            missing_face=missing_face,
            missing_audio=missing_audio,
        )
