from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import joblib
import numpy as np
from sklearn.svm import LinearSVC

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


CLASSES = ("tank", "vehicle", "empty")
IMAGE_SIZE = 64


if nn is not None:
    class CellCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 24, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(48, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 96),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(96, len(CLASSES)),
            )

        def forward(self, image: "torch.Tensor") -> "torch.Tensor":
            return self.classifier(self.features(image))
else:
    CellCNN = None


class VisionPipeline:
    """Clasifica únicamente píxeles RGB; usa CNN y reserva HOG/SVM como fallback."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        fallback_path: str | Path | None = None,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.fallback_path = Path(fallback_path) if fallback_path else None
        self.classes = list(CLASSES)
        self.model = None
        self.backend = "unavailable"
        self.device = "cpu"
        self._load_models()

    def _load_models(self) -> None:
        if self.model_path and self.model_path.is_file() and torch is not None and CellCNN is not None:
            try:
                payload = torch.load(self.model_path, map_location="cpu", weights_only=True)
                if payload.get("format") == "drone-wars-cnn-v1":
                    model = CellCNN()
                    model.load_state_dict(payload["state_dict"])
                    model.eval()
                    self.model = model
                    self.classes = list(payload["classes"])
                    self.backend = "CNN (PyTorch, CPU)"
                    return
            except (
                OSError,
                RuntimeError,
                ValueError,
                KeyError,
                TypeError,
                AttributeError,
                pickle.UnpicklingError,
            ):
                self.model = None
        if self.fallback_path and self.fallback_path.is_file():
            try:
                fallback = joblib.load(self.fallback_path)
                if isinstance(fallback, dict) and fallback.get("format") == "drone-wars-hog-v1":
                    self.model = fallback["model"]
                    self.classes = list(fallback["classes"])
                    self.backend = "HOG + SVM (fallback)"
            except (OSError, ValueError, EOFError, TypeError, AttributeError):
                self.model = None

    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("classify espera una imagen numpy.ndarray.")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("La imagen debe ser HxW, HxWx3 RGB o HxWx4 BGRA.")
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(image.astype(np.uint8))

    @staticmethod
    def _feature_vector(image: np.ndarray) -> np.ndarray:
        processed = VisionPipeline.preprocess(image)
        gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        hog = cv2.HOGDescriptor(
            (IMAGE_SIZE, IMAGE_SIZE),
            (16, 16),
            (8, 8),
            (8, 8),
            9,
        )
        return hog.compute(gray).reshape(-1).astype(np.float32)

    def fit_cnn(self, images: np.ndarray, labels: np.ndarray, epochs: int = 18) -> None:
        if torch is None or CellCNN is None:
            raise RuntimeError("PyTorch no está instalado; no se puede entrenar la CNN.")
        torch.manual_seed(2026)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.model = CellCNN().cpu()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.0005)
        loss_fn = nn.CrossEntropyLoss()
        encoded = np.asarray([CLASSES.index(label) for label in labels], dtype=np.int64)
        tensor_images = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).float().div_(255.0)
        tensor_labels = torch.from_numpy(encoded)
        generator = torch.Generator().manual_seed(2026)
        for _ in range(epochs):
            self.model.train()
            order = torch.randperm(len(tensor_labels), generator=generator)
            for batch in order.split(64):
                logits = self.model(tensor_images[batch])
                loss = loss_fn(logits, tensor_labels[batch])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        self.model.eval()
        self.classes = list(CLASSES)
        self.backend = "CNN (PyTorch, CPU)"

    def fit_fallback(self, images: np.ndarray, labels: np.ndarray) -> None:
        features = np.vstack([self._feature_vector(image) for image in images])
        self.model = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
        self.model.fit(features, labels)
        self.classes = list(CLASSES)
        self.backend = "HOG + SVM (fallback)"
        if self.fallback_path is not None:
            self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"format": "drone-wars-hog-v1", "model": self.model, "classes": self.classes},
                self.fallback_path,
            )

    def save(self, path: str | Path | None = None) -> None:
        destination = Path(path) if path else self.model_path
        if destination is None or self.model is None or torch is None or CellCNN is None:
            raise RuntimeError("No hay una CNN PyTorch cargada para guardar.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "drone-wars-cnn-v1",
                "classes": self.classes,
                "state_dict": self.model.state_dict(),
            },
            destination,
        )
        self.model_path = destination

    def classify(self, image: np.ndarray) -> Tuple[str, float]:
        processed = self.preprocess(image)
        if self.model is None:
            raise RuntimeError("No hay modelo visual cargado; entrena o instala uno antes de jugar.")
        if self.backend.startswith("CNN"):
            assert torch is not None
            tensor = torch.from_numpy(processed.transpose(2, 0, 1).copy()).float().unsqueeze(0).div_(255.0)
            with torch.inference_mode():
                probabilities = torch.softmax(self.model(tensor), dim=1)[0]
            confidence, index = probabilities.max(dim=0)
            return self.classes[int(index)], float(confidence)
        features = self._feature_vector(processed).reshape(1, -1)
        prediction = str(self.model.predict(features)[0])
        scores = np.asarray(self.model.decision_function(features), dtype=np.float64).reshape(-1)
        shifted = scores - scores.max()
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        confidence = float(probabilities.max())
        return prediction, confidence

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        return self.classify(image)

    def classify_batch(self, images: Iterable[np.ndarray]) -> List[Tuple[str, float]]:
        return [self.classify(image) for image in images]

    def detect_from_frame(self, image: np.ndarray) -> List[str]:
        prediction, _ = self.predict(image)
        return [prediction]


def create_synthetic_sprite(kind: str, size: int = 32) -> np.ndarray:
    """Crea un ejemplo de compatibilidad para pruebas; no se usa como clasificador."""
    canvas = np.full((size, size, 3), (135, 139, 128), dtype=np.uint8)
    if kind == "tank":
        cv2.rectangle(canvas, (5, 11), (size - 6, size - 8), (82, 119, 68), -1)
        cv2.rectangle(canvas, (size // 2 - 3, 5), (size // 2 + 3, size // 2), (49, 58, 45), -1)
        cv2.line(canvas, (size // 2, size // 2), (size // 2 + 9, size // 2 - 8), (38, 44, 37), 3)
    elif kind == "vehicle":
        cv2.rectangle(canvas, (7, 12), (size - 7, size - 9), (50, 117, 153), -1)
        cv2.rectangle(canvas, (11, 7), (size - 11, 14), (61, 75, 79), -1)
        cv2.circle(canvas, (11, size - 6), 3, (31, 35, 38), -1)
        cv2.circle(canvas, (size - 11, size - 6), 3, (31, 35, 38), -1)
    else:
        for offset in range(0, size, 8):
            cv2.line(canvas, (0, offset), (size - 1, offset), (111, 119, 117), 1)
    return canvas
