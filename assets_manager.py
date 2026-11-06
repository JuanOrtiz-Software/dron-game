from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

LOGGER = logging.getLogger("drone_wars.assets")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
PROCESSED_SIZE = (96, 96)


def image_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def process_tank_image(source: Path, normal_path: Path, destroyed_path: Path) -> None:
    try:
        with Image.open(source) as opened:
            rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"No se pudo decodificar la imagen {source}: {exc}") from exc

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    if min(height, width) < 8:
        raise ValueError(f"La imagen es demasiado pequeña para procesar: {source}")
    mask = np.zeros((height, width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    inset_x = max(1, width // 25)
    inset_y = max(1, height // 25)
    try:
        cv2.grabCut(
            bgr,
            mask,
            (inset_x, inset_y, max(1, width - 2 * inset_x), max(1, height - 2 * inset_y)),
            background_model,
            foreground_model,
            5,
            cv2.GC_INIT_WITH_RECT,
        )
        alpha = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
        if cv2.countNonZero(alpha) < height * width * 0.01:
            raise cv2.error("background-removal", "grabCut", "foreground mask is empty", "", 0)
    except cv2.error as exc:
        LOGGER.warning("No se pudo quitar el fondo de %s (%s); se conservará la foto completa.", source.name, exc)
        alpha = np.full((height, width), 255, dtype=np.uint8)

    points = cv2.findNonZero(alpha)
    if points is None:
        raise ValueError(f"No se detectó ningún primer plano en {source}.")
    x, y, crop_width, crop_height = cv2.boundingRect(points)
    foreground = cv2.cvtColor(bgr[y : y + crop_height, x : x + crop_width], cv2.COLOR_BGR2RGBA)
    foreground[:, :, 3] = alpha[y : y + crop_height, x : x + crop_width]
    image = Image.fromarray(foreground)
    image.thumbnail((82, 82), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", PROCESSED_SIZE, (0, 0, 0, 0))
    canvas.alpha_composite(
        image,
        ((PROCESSED_SIZE[0] - image.width) // 2, (PROCESSED_SIZE[1] - image.height) // 2),
    )
    normal_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(normal_path, format="PNG")

    destroyed = np.asarray(canvas).copy()
    rgba = destroyed[:, :, :3].astype(np.float32)
    rgba *= 0.48
    destroyed[:, :, :3] = np.clip(rgba, 0, 255).astype(np.uint8)
    cv2.circle(destroyed, (67, 25), 9, (222, 72, 54, 225), -1, cv2.LINE_AA)
    cv2.circle(destroyed, (67, 25), 5, (255, 139, 58, 240), -1, cv2.LINE_AA)
    Image.fromarray(destroyed).save(destroyed_path, format="PNG")


def prepare_user_tanks(asset_dir: Path) -> List[Tuple[Path, Path, Path]]:
    source_dir = asset_dir / "user_tanks"
    processed_dir = source_dir / "processed"
    processed: List[Tuple[Path, Path, Path]] = []
    for index, source in enumerate(image_files(source_dir), start=1):
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:8]
        normal = processed_dir / f"user_tank_{index:02}_{digest}_normal.png"
        destroyed = processed_dir / f"user_tank_{index:02}_{digest}_destroyed.png"
        if not normal.is_file() or not destroyed.is_file():
            process_tank_image(source, normal, destroyed)
        processed.append((source, normal, destroyed))
    return processed


def write_asset_credits(
    asset_dir: Path,
    downloaded_metadata: Iterable[dict[str, str]],
    procedural_names: Iterable[str],
) -> None:
    users = image_files(asset_dir / "user_tanks")
    downloaded = list(downloaded_metadata)
    catalog = list(procedural_names)
    lines = [
        "# Manifiesto de activos",
        "",
        f"Recuento por origen: usuario = {len(users)}, descargado = {len(downloaded)}, procedural = {len(catalog)}.",
        "",
        "## Imágenes del usuario",
    ]
    lines.extend(f"- `user_tanks/{path.name}` — imagen aportada por el usuario." for path in users)
    if not users:
        lines.append("- Ninguna. `assets/user_tanks/` está vacío.")
    lines.extend(["", "## Imágenes descargadas"])
    for record in downloaded:
        lines.append(
            f"- `{record['filename']}` — {record['title']} — licencia: {record['license']} — "
            f"autor: {record['artist']} — fuente: {record['source']}."
        )
    if not downloaded:
        lines.append("- Ninguna descarga verificada.")
    lines.extend(["", "## Activos procedurales"])
    lines.extend(f"- `{name}` — generado localmente por el juego." for name in catalog)
    (asset_dir / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    credits = [
        "# Créditos de activos",
        "",
        f"Recuento en esta ejecución: usuario = {len(users)}, descargado = {len(downloaded)}, procedural = {len(catalog)}.",
        "",
    ]
    credits.extend(
        f"- {record['filename']} — {record['title']}; {record['license']}; {record['artist']}; {record['source']}"
        for record in downloaded
    )
    credits.extend(f"- `user_tanks/{path.name}` — imagen aportada por el usuario." for path in users)
    credits.append("- Todos los elementos restantes de la interfaz y del mundo se generan con pygame.")
    (asset_dir / "CREDITS.md").write_text("\n".join(credits) + "\n", encoding="utf-8")
