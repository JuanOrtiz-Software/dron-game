from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pygame
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from art import make_sprite, render_cell_image
from assets_manager import prepare_user_tanks
from vision import CLASSES, IMAGE_SIZE, VisionPipeline

BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "assets"
USER_IMAGE_DIR = ASSET_DIR / "user_tanks"
SAMPLES_PER_CLASS = 800
VALIDATION_PER_CLASS = 240


def _sources() -> Dict[str, List[Tuple[str, np.ndarray]]]:
    pygame.init()
    street = make_sprite("ground_street.png", (IMAGE_SIZE, IMAGE_SIZE))
    sources: Dict[str, List[Tuple[str, np.ndarray]]] = {name: [] for name in CLASSES}
    for variant in range(1, 5):
        sprite = make_sprite(f"tank_{variant:02}_normal.png")
        rendered = render_cell_image(street, sprite, 0, IMAGE_SIZE)
        sources["tank"].append((f"procedural:tank_{variant:02}", rendered))
    for kind in ("car", "pickup", "truck", "helicopter"):
        for variant in range(1, 4):
            sprite = make_sprite(f"{kind}_{variant:02}_normal.png")
            rendered = render_cell_image(street, sprite, 0, IMAGE_SIZE)
            sources["vehicle"].append((f"procedural:{kind}_{variant:02}", rendered))
    for index in range(8):
        base = make_sprite("ground_street.png" if index % 2 == 0 else "ground_grass.png", (IMAGE_SIZE, IMAGE_SIZE))
        arr = pygame.surfarray.array3d(base).transpose(1, 0, 2).copy()
        rng = np.random.default_rng(4000 + index)
        arr = np.clip(arr.astype(np.float32) * rng.uniform(0.88, 1.12) + rng.uniform(-12, 12), 0, 255).astype(np.uint8)
        sources["empty"].append((f"procedural:ground_{index:02}", arr))

    for source_path, processed_path, _ in prepare_user_tanks(ASSET_DIR):
        try:
            sprite = pygame.image.load(str(processed_path))
        except (pygame.error, OSError) as exc:
            raise RuntimeError(f"No se pudo cargar la imagen procesada {source_path.name}: {exc}") from exc
        # User photos always contribute to training; validation stays on held-out source sprites.
        rendered = render_cell_image(street, sprite, 0, IMAGE_SIZE)
        sources["tank"].append((f"user:{processed_path.stem}", rendered))
    downloaded = ASSET_DIR / "downloads" / "tank_04_normal.png"
    if downloaded.is_file():
        try:
            sprite = pygame.image.load(str(downloaded))
        except (pygame.error, OSError) as exc:
            raise RuntimeError(f"No se pudo cargar el sprite descargado {downloaded.name}: {exc}") from exc
        sources["tank"].append(("downloaded:commons_tank", render_cell_image(street, sprite, 0, IMAGE_SIZE)))
    return sources


def _augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = image.shape[:2]
    angle = float(rng.uniform(0, 360))
    scale = float(rng.uniform(0.72, 1.22))
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    if rng.random() < 0.5:
        image = cv2.flip(image, 1)
    background = rng.integers(55, 190, size=(height, width, 3), dtype=np.uint8)
    transformed = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT if rng.random() < 0.5 else cv2.BORDER_CONSTANT,
        borderValue=tuple(int(value) for value in background[0, 0]),
    )
    random_background = rng.integers(48, 190, size=(height, width, 3), dtype=np.uint8)
    random_background = cv2.GaussianBlur(random_background, (9, 9), 2)
    if rng.random() < 0.35:
        transformed = cv2.addWeighted(transformed, 0.88, random_background, 0.12, 0)
    alpha = float(rng.uniform(0.72, 1.28))
    beta = float(rng.uniform(-22, 22))
    transformed = cv2.convertScaleAbs(transformed, alpha=alpha, beta=beta)
    if rng.random() < 0.35:
        overlay = np.zeros_like(transformed)
        x0, y0 = (int(rng.integers(width)), int(rng.integers(height)))
        x1, y1 = (int(rng.integers(width)), int(rng.integers(height)))
        cv2.line(overlay, (x0, y0), (x1, y1), (32, 31, 29), int(rng.integers(18, 75)), cv2.LINE_AA)
        transformed = cv2.addWeighted(transformed, 1, overlay, 0.22, 0)
    if rng.random() < 0.25:
        transformed = cv2.GaussianBlur(transformed, (3, 3), float(rng.uniform(0.2, 1.2)))
    if rng.random() < 0.15:
        encode_quality = int(rng.integers(35, 91))
        ok, encoded = cv2.imencode(".jpg", transformed, [cv2.IMWRITE_JPEG_QUALITY, encode_quality])
        if ok:
            transformed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    noise = rng.normal(0, rng.uniform(0, 16), transformed.shape)
    return np.clip(transformed.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _split_sources(
    sources: Dict[str, List[Tuple[str, np.ndarray]]],
) -> Tuple[Dict[str, list[np.ndarray]], Dict[str, list[np.ndarray]], Dict[str, list[str]]]:
    rng = random.Random(2026)
    training: Dict[str, list[np.ndarray]] = defaultdict(list)
    validation: Dict[str, list[np.ndarray]] = defaultdict(list)
    split_ids: Dict[str, list[str]] = {}
    for label in CLASSES:
        procedural = [item for item in sources[label] if item[0].startswith("procedural:")]
        additional_training = [
            item for item in sources[label]
            if item[0].startswith(("user:", "downloaded:"))
        ]
        rng.shuffle(procedural)
        held_count = max(1, round(len(procedural) * 0.25))
        held_ids = {source_id for source_id, _ in procedural[:held_count]}
        train_items = [(source_id, image) for source_id, image in procedural if source_id not in held_ids]
        train_items.extend(additional_training)
        validation_items = [(source_id, image) for source_id, image in procedural if source_id in held_ids]
        if not train_items or not validation_items:
            raise RuntimeError(f"No hay fuentes suficientes para separar entrenamiento y validación: {label}.")
        split_ids[label] = [
            f"train:{source_id}" for source_id, _ in train_items
        ] + [f"validation:{source_id}" for source_id, _ in validation_items]
        for _ in range(SAMPLES_PER_CLASS):
            source_id, image = rng.choice(train_items)
            training[label].append(_augment(image, np.random.default_rng(rng.randrange(2**32))))
        for _ in range(VALIDATION_PER_CLASS):
            source_id, image = rng.choice(validation_items)
            validation[label].append(_augment(image, np.random.default_rng(rng.randrange(2**32))))
    return training, validation, split_ids


def _flatten(dataset: Dict[str, list[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    images = np.concatenate([np.stack(dataset[label]) for label in CLASSES])
    labels = np.concatenate(
        [np.full(len(dataset[label]), label, dtype=object) for label in CLASSES]
    )
    return images, labels


def train_pipeline(output_path: Path, epochs: int = 80) -> VisionPipeline:
    sources = _sources()
    training, validation, source_split = _split_sources(sources)
    train_images, train_labels = _flatten(training)
    validation_images, validation_labels = _flatten(validation)
    classifier = VisionPipeline(output_path, output_path.parent / "vision_sklearn.joblib")
    classifier.fit_cnn(train_images, train_labels, epochs=epochs)
    predictions = [classifier.classify(image) for image in validation_images]
    y_pred = np.asarray([prediction for prediction, _ in predictions])
    accuracy = float(accuracy_score(validation_labels, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        validation_labels, y_pred, labels=list(CLASSES), zero_division=0
    )
    metrics = {
        "accuracy": accuracy,
        "labels": list(CLASSES),
        "confusion_matrix": confusion_matrix(validation_labels, y_pred, labels=list(CLASSES)).tolist(),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(CLASSES)
        },
        "training_samples": {label: len(training[label]) for label in CLASSES},
        "validation_samples": {label: len(validation[label]) for label in CLASSES},
        "source_split": source_split,
        "evaluation": "augmentaciones generadas desde sprites fuente excluidos del entrenamiento",
        "pipeline": classifier.backend,
    }
    classifier.save(output_path)
    metrics_path = output_path.parent / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    model_info = {
        "training_date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pipeline": classifier.backend,
        "status": "ok",
        "device": "cpu",
        "source_sprite_count": {label: len(sources[label]) for label in CLASSES},
        "source_manifest": {
            label: sorted(source_id for source_id, _ in sources[label])
            for label in CLASSES
        },
        "user_images_used": sum(source_id.startswith("user:") for source_id, _ in sources["tank"]),
        "training_samples_per_class": SAMPLES_PER_CLASS,
        "validation_samples_per_class": VALIDATION_PER_CLASS,
        "epochs": epochs,
        "metrics_file": metrics_path.name,
        "fallback_file": "vision_sklearn.joblib",
        "accuracy": accuracy,
    }
    (output_path.parent / "model_info.json").write_text(
        json.dumps(model_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fallback = VisionPipeline(None, output_path.parent / "vision_sklearn.joblib")
    fallback.fit_fallback(train_images, train_labels)
    print(f"Entrenamiento completado: {classifier.backend}; dispositivo CPU.")
    print(f"Muestras entrenamiento/validación por clase: {SAMPLES_PER_CLASS}/{VALIDATION_PER_CLASS}.")
    print(f"Exactitud en sprites fuente reservados: {accuracy:.4f}")
    print(f"Matriz de confusión: {metrics['confusion_matrix']}")
    print(f"Imágenes propias incluidas en entrenamiento: {model_info['user_images_used']}")
    return classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena y evalúa la CNN visual del juego.")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "models" / "vision.pt")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs debe ser positivo.")
    train_pipeline(args.output, args.epochs)


if __name__ == "__main__":
    main()
