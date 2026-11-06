from __future__ import annotations

import html
import json
import logging
import re
import ssl
import time
from pathlib import Path
from typing import Dict, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi
from PIL import Image, UnidentifiedImageError

from assets_manager import process_tank_image

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "DroneWarsEducationalGame/1.0 (asset attribution and licensing; contact: project-maintainer)"
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_LICENSES = {
    "cc0",
    "cc by 2.0",
    "cc by 3.0",
    "cc by 4.0",
    "public domain",
    "pd",
}
LOGGER = logging.getLogger("drone_wars.assets")


def _read_https(url: str, timeout: int = 12, retries: int = 3) -> bytes:
    if not url.startswith("https://"):
        raise ValueError(f"Se rechazó URL no HTTPS: {url}")
    context = ssl.create_default_context(cafile=certifi.where())
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout, context=context) as response:
                payload = response.read(MAX_DOWNLOAD_BYTES + 1)
            if len(payload) > MAX_DOWNLOAD_BYTES:
                raise ValueError("El recurso supera el límite de 5 MB.")
            if len(payload) < 200:
                raise ValueError(f"El recurso es demasiado pequeño ({len(payload)} bytes).")
            return payload
        except (HTTPError, URLError, TimeoutError, OSError, ssl.SSLError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
    if last_error is None:
        raise RuntimeError("Falló la descarga sin producir un error de red.")
    raise RuntimeError(f"No se pudo descargar {url}: {last_error}") from last_error


def _clean_metadata(value: dict[str, object]) -> str:
    raw = str(value.get("value", ""))
    return html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()


def _license_key(value: str) -> str:
    lowered = value.lower()
    if "sharealike" in lowered or re.search(r"\bby[- ]?(?:sa|nc|nd)\b", lowered):
        return "not permitted"
    normalized = re.sub(r"\s+", " ", lowered.replace("-", " ")).strip()
    if normalized.startswith("cc0"):
        return "cc0"
    if normalized.startswith("cc by"):
        version_match = re.search(r"([234])(?:\.0)?", normalized)
        return f"cc by {version_match.group(1)}.0" if version_match else normalized
    if "public domain" in normalized or normalized.startswith("pd"):
        return "public domain"
    return normalized


def _commons_candidates() -> Iterable[dict[str, object]]:
    parameters = {
        "action": "query",
        "generator": "search",
        "gsrsearch": 'filetype:bitmap "toy tank"',
        "gsrnamespace": "6",
        "gsrlimit": "30",
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime|size",
        "iiurlwidth": "512",
        "format": "json",
        "formatversion": "2",
    }
    url = f"{COMMONS_API}?{urlencode(parameters)}"
    payload = _read_https(url)
    try:
        response = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Wikimedia Commons devolvió JSON inválido: {exc}") from exc
    pages = response.get("query", {}).get("pages", [])
    for page in pages:
        title = str(page.get("title", "")).lower()
        if "tank" not in title or "toy" not in title:
            continue
        imageinfo = page.get("imageinfo", [])
        if imageinfo:
            yield {
                "title": page.get("title", ""),
                "info": imageinfo[0],
            }


def ensure_asset_pack(asset_dir: Path) -> list[str]:
    """Obtiene imágenes de tanque mediante la API oficial de Commons y valida su licencia."""
    asset_dir.mkdir(parents=True, exist_ok=True)
    log_path = asset_dir.parent.parent / "logs" / "asset_downloads.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    logger = logging.getLogger("drone_wars.assets")
    logger.setLevel(logging.INFO)
    if not any(isinstance(existing, logging.FileHandler) and existing.baseFilename == handler.baseFilename for existing in logger.handlers):
        logger.addHandler(handler)
    else:
        handler.close()

    metadata_path = asset_dir / "metadata.json"
    records: list[dict[str, str]] = []
    if metadata_path.is_file():
        try:
            records = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.exception("No se pudo leer %s: %s", metadata_path, exc)
            raise RuntimeError(f"Metadatos de descarga inválidos: {metadata_path}") from exc
    if records and all(
        _license_key(item.get("license", "")) in ALLOWED_LICENSES
        and (asset_dir / item["normal_file"]).is_file()
        and (asset_dir / item["destroyed_file"]).is_file()
        for item in records
    ):
        return [item["filename"] for item in records]
    if records:
        logger.warning("Se eliminarán recursos en caché con licencia no permitida o archivos incompletos.")
        for item in records:
            for key in ("normal_file", "destroyed_file", "source_file"):
                name = item.get(key)
                if name:
                    target = (asset_dir / name).resolve()
                    if target.parent == asset_dir.resolve() and target.is_file():
                        target.unlink()
        metadata_path.unlink(missing_ok=True)
    stale_original = asset_dir / "commons_tank.png"
    if stale_original.is_file():
        stale_original.unlink()

    try:
        candidates = list(_commons_candidates())
    except (OSError, RuntimeError, ValueError, ssl.SSLError) as exc:
        logger.exception("No se pudo consultar la API de Wikimedia Commons: %s", exc)
        return []

    for candidate in candidates:
        info = candidate["info"]
        extmetadata = info.get("extmetadata", {})
        license_name = _clean_metadata(extmetadata.get("LicenseShortName", {}))
        if _license_key(license_name) not in ALLOWED_LICENSES:
            logger.info("Descarga omitida por licencia no permitida: %s (%s)", candidate["title"], license_name)
            continue
        source_url = str(info.get("thumburl") or info.get("url") or "")
        try:
            payload = _read_https(source_url)
            from io import BytesIO

            with Image.open(BytesIO(payload)) as image:
                image.verify()
            suffix = ".png"
            original = asset_dir / f"commons_tank{suffix}"
            original.write_bytes(payload)
            normal = asset_dir / "tank_04_normal.png"
            destroyed = asset_dir / "tank_04_destroyed.png"
            process_tank_image(original, normal, destroyed)
            record = {
                "filename": normal.name,
                "title": str(candidate["title"]),
                "license": license_name,
                "artist": _clean_metadata(extmetadata.get("Artist", {})) or "Autoría indicada en Wikimedia Commons",
                "source": str(info.get("descriptionurl") or candidate["title"]),
                "normal_file": normal.name,
                "destroyed_file": destroyed.name,
                "source_file": original.name,
            }
            records = [record]
            metadata_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            logger.info("Activo descargado y procesado: %s; licencia %s; fuente %s", normal, license_name, record["source"])
            return [normal.name]
        except (OSError, RuntimeError, ValueError, UnidentifiedImageError, ssl.SSLError) as exc:
            logger.exception("Se rechazó el recurso %s: %s", candidate["title"], exc)
    logger.warning("La API no proporcionó imágenes con licencias permitidas; se conserva el catálogo procedural.")
    return []


def load_download_metadata(asset_dir: Path) -> list[dict[str, str]]:
    path = asset_dir / "metadata.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.exception("No se pudo leer el manifiesto de descargas %s: %s", path, exc)
        raise RuntimeError(f"Manifiesto de descargas inválido: {path}") from exc
    if not isinstance(data, list):
        raise RuntimeError(f"El manifiesto de descargas debe ser una lista: {path}")
    approved: list[dict[str, str]] = []
    for item in data:
        title = str(item.get("title", "")).lower() if isinstance(item, dict) else ""
        if (
            not isinstance(item, dict)
            or _license_key(str(item.get("license", ""))) not in ALLOWED_LICENSES
            or "toy" not in title
            or "tank" not in title
        ):
            LOGGER.warning("Se ignoró un activo cacheado con una licencia no permitida.")
            continue
        normal_name = str(item.get("normal_file", ""))
        destroyed_name = str(item.get("destroyed_file", ""))
        normal = (asset_dir / normal_name).resolve()
        destroyed = (asset_dir / destroyed_name).resolve()
        if (
            normal.parent != asset_dir.resolve()
            or destroyed.parent != asset_dir.resolve()
            or not normal.is_file()
            or not destroyed.is_file()
        ):
            LOGGER.warning("Se ignoró un activo cacheado incompleto: %s.", normal_name)
            continue
        approved.append({str(key): str(value) for key, value in item.items()})
    return approved
