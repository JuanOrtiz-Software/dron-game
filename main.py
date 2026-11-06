from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import queue
import random
import re
import ssl
import subprocess
import sys
import threading
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.request import urlopen

try:
    import numpy as np
    import pygame

    from art import make_sprite, render_cell_image, sprite_catalog, sprite_size
    from assets_manager import prepare_user_tanks, write_asset_credits
    from config import (
        AMMO_MAX,
        BATTERY_MAX,
        CONFIDENCE_THRESHOLD,
        GRID_SIZE,
        MAX_SWEEPS,
        MODE_3_VISION_RADIUS,
        SHOT_TICKS,
        TICK_SECONDS,
        VISION_RADIUS,
    )
    from download_assets import ensure_asset_pack, load_download_metadata
    from drone import Drone
    from entities import Entity
    from reference.reference_algorithms import arbol_desde_grid, bfs as oracle_bfs, dfs as oracle_dfs
    from search import LiveSearch
    from stats import drone_statistics, save_match_result
    from vision import VisionPipeline, create_synthetic_sprite
    from world import World
except (ImportError, OSError) as exc:
    print(f"No se pudo cargar un paquete requerido: {exc}")
    print(f"Instálalo con: {sys.executable} -m pip install -r requirements.txt")
    if sys.stdin.isatty():
        try:
            input("Presiona ENTER para salir...")
        except EOFError:
            print("No hay entrada de teclado; se cierra el diagnóstico.")
    raise SystemExit(1) from exc

BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "assets"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"
SCREENSHOT_DIR = BASE_DIR / "docs" / "screenshots"
MODEL_FILE = MODEL_DIR / "vision.pt"
FALLBACK_MODEL_FILE = MODEL_DIR / "vision_sklearn.joblib"
MODEL_INFO_FILE = MODEL_DIR / "model_info.json"
METRICS_FILE = MODEL_DIR / "metrics.json"
MANIFEST_FILE = ASSET_DIR / "MANIFEST.md"
USER_TANK_DIR = ASSET_DIR / "user_tanks"
CELL_SIZE = 28
SCREEN_SIZE = (1180, 800)
BOARD_ORIGIN = (310, 134)
PANEL_SIZE = (294, 202)
Position = Tuple[int, int]


def _algorithm_integrity_check() -> str:
    cases = (
        ([[0, 0, 0], [0, 1, 0], [0, 0, 0]], (0, 0), (2, 2)),
        ([[0, 1, 0], [1, 1, 0], [0, 1, 0]], (0, 0), (2, 2)),
        ([[0, 0], [0, 0]], (0, 0), (0, 0)),
    )
    comparisons = 0
    for grid, start, goal in cases:
        adjacency = arbol_desde_grid(grid)
        if start not in adjacency or goal not in adjacency:
            raise AssertionError("El caso de integridad contiene una celda bloqueada.")
        for algorithm, oracle in (("bfs", oracle_bfs), ("dfs", oracle_dfs)):
            expected_nodes, expected_found = oracle(start, goal, adjacency)
            search = LiveSearch(start, goal, adjacency, algorithm)
            while not search.finished:
                search.step()
            if search.history != expected_nodes or search.result is not expected_found:
                raise AssertionError(f"{algorithm.upper()} difiere del oráculo.")
            comparisons += 1
    return f"{comparisons} secuencias comparadas con el oráculo"


def _run_search(
    start: Position,
    goal: Optional[Position],
    adjacency: Dict[Position, List[Position]],
    algorithm: str,
) -> Tuple[List[Position], bool]:
    search = LiveSearch(start, goal, adjacency, algorithm)
    while not search.finished:
        search.step()
    return search.history, search.result


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    action: str
    icon: str = "icon_drone.png"
    disabled: bool = False


class GameApp:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.running = True
        self.screen: Optional[pygame.Surface] = None
        self.clock = pygame.time.Clock()
        self.font: Optional[pygame.font.Font] = None
        self.small_font: Optional[pygame.font.Font] = None
        self.large_font: Optional[pygame.font.Font] = None
        self.sprites: Dict[str, pygame.Surface] = {}
        self.loading_queue: queue.Queue[Tuple[str, Any]] = queue.Queue()
        self.cancel_preparation = threading.Event()
        self.loading_thread: Optional[threading.Thread] = None
        self.loading_progress = 0.0
        self.loading_step = "Comprobando activos y modelo visual"
        self.loading_messages: List[str] = []
        self.loading_started = time.monotonic()
        self.loading_done = False
        self.force_retrain = False
        self.state = "loading"
        self.world: Optional[World] = None
        self.red_drone: Optional[Drone] = None
        self.blue_drone: Optional[Drone] = None
        self.vision: Optional[VisionPipeline] = None
        self.model_info: Dict[str, Any] = {}
        self.model_metrics: Dict[str, Any] = {}
        self.error_text = ""
        self.error_traceback = ""
        self.error_type = ""
        self.error_message = ""
        self.error_frame = ""
        self.log_warning = ""
        self.buttons: List[Button] = []
        self.mode_index = 0
        self.mode_names = (
            "BUSCAR Y DESTRUIR",
            "¿QUIÉN GANA?",
            "MISIÓN CON RECURSOS LIMITADOS",
        )
        self.mode2_first = "red"
        self.mode2_order: List[str] = []
        self.mode2_turn_index = 0
        self.mode2_stats: Dict[str, Dict[str, int | float | bool]] = {}
        self.world_snapshot: Optional[Dict[str, Any]] = None
        self.match_seed = 0
        self.winner = ""
        self.results_path: Optional[Path] = None
        self.results_dir = RESULTS_DIR
        self.conflict_priority = 0
        self.tick_count = 0
        self.match_ticks = 0
        self.result_text = ""
        self.effects: List[Tuple[Position, int]] = []
        self.demo_effects = False
        self.fonts_ready = False

    def _make_dirs(self) -> None:
        for path in (ASSET_DIR, USER_TANK_DIR, MODEL_DIR, LOG_DIR, RESULTS_DIR, SCREENSHOT_DIR):
            path.mkdir(parents=True, exist_ok=True)

    def _init_display(self) -> None:
        os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "center")
        pygame.init()
        pygame.font.init()
        desktop = pygame.display.get_desktop_sizes()
        desktop_width, desktop_height = desktop[0] if desktop else SCREEN_SIZE
        if pygame.display.get_driver() == "dummy":
            width, height = SCREEN_SIZE
        else:
            width = min(SCREEN_SIZE[0], max(900, desktop_width - 48))
            height = min(SCREEN_SIZE[1], max(650, desktop_height - 32))
        logical_size = (width, height)
        try:
            self.screen = pygame.display.set_mode(logical_size, pygame.SCALED | pygame.RESIZABLE)
        except pygame.error:
            self.screen = pygame.display.set_mode(logical_size, pygame.RESIZABLE)
        pygame.display.set_caption("DRONE WARS · CIUDAD DE JUGUETE")
        pygame.event.pump()
        self.font = self._font(20)
        self.small_font = self._font(15)
        self.large_font = self._font(31, bold=True)
        self.fonts_ready = True
        print(f"Ventana pygame abierta: {self.screen.get_size()} · SDL {pygame.get_sdl_version()} · {pygame.display.get_driver()}")

    @staticmethod
    def _font(size: int, bold: bool = False) -> pygame.font.Font:
        for family in ("DejaVu Sans", "Helvetica", "Liberation Sans", "sans"):
            path = pygame.font.match_font(family, bold=bold)
            if path:
                return pygame.font.Font(path, size)
        return pygame.font.Font(None, size)

    def _ensure_sprites(self) -> None:
        catalog = sprite_catalog()
        user_tanks = prepare_user_tanks(ASSET_DIR)
        for name in catalog:
            path = ASSET_DIR / name
            try:
                if not path.exists():
                    pygame.image.save(make_sprite(name, sprite_size(name)), str(path))
                self.sprites[name] = pygame.image.load(str(path)).convert_alpha()
            except (pygame.error, OSError) as exc:
                print(f"AVISO: activo {name} no disponible ({exc}); se genera un reemplazo.")
                replacement = make_sprite(name, sprite_size(name))
                self.sprites[name] = replacement
                pygame.image.save(replacement, str(path))
        downloaded_dir = ASSET_DIR / "downloads"
        downloaded_normal = downloaded_dir / "tank_04_normal.png"
        downloaded_destroyed = downloaded_dir / "tank_04_destroyed.png"
        if downloaded_normal.is_file() and downloaded_destroyed.is_file():
            self.sprites["tank_04_normal.png"] = pygame.image.load(str(downloaded_normal)).convert_alpha()
            self.sprites["tank_04_destroyed.png"] = pygame.image.load(str(downloaded_destroyed)).convert_alpha()
        used_user_names = set()
        for variant, (_, normal, destroyed) in enumerate(user_tanks[:4], start=1):
            normal_name = f"tank_{variant:02}_normal.png"
            destroyed_name = f"tank_{variant:02}_destroyed.png"
            self.sprites[normal_name] = pygame.image.load(str(normal)).convert_alpha()
            self.sprites[destroyed_name] = pygame.image.load(str(destroyed)).convert_alpha()
            used_user_names.update((normal_name, destroyed_name))
        procedural_names = [
            name for name in catalog if name not in used_user_names
            and not (name.startswith("tank_04_") and downloaded_normal.is_file())
        ]
        write_asset_credits(ASSET_DIR, load_download_metadata(downloaded_dir), procedural_names)

    def _ensure_error_sprites(self) -> None:
        for name in sprite_catalog():
            if name not in self.sprites:
                self.sprites[name] = make_sprite(name, sprite_size(name))

    def _sprite(self, name: str, size: Optional[Tuple[int, int]] = None) -> pygame.Surface:
        if name not in self.sprites:
            if name not in sprite_catalog():
                raise KeyError(f"El sprite {name!r} no pertenece al catálogo.")
            print(f"AVISO: falta el sprite {name}; se genera un reemplazo procedural.")
            self.sprites[name] = make_sprite(name, sprite_size(name))
        image = self.sprites[name]
        return pygame.transform.smoothscale(image, size) if size and image.get_size() != size else image

    def _text(
        self,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = (238, 241, 232),
        font: Optional[pygame.font.Font] = None,
        center: bool = False,
    ) -> pygame.Rect:
        assert self.screen is not None
        font = font or self.font
        assert font is not None
        rendered = font.render(text, True, color)
        rect = rendered.get_rect(center=position) if center else rendered.get_rect(topleft=position)
        self.screen.blit(rendered, rect)
        return rect

    def _button(self, rect: pygame.Rect, label: str, action: str, icon: str = "icon_drone.png") -> Button:
        button = Button(rect, label, action, icon)
        self.buttons.append(button)
        self.screen.blit(self._sprite("button.png", rect.size), rect)
        icon_size = min(40, rect.height - 16)
        self.screen.blit(self._sprite(icon, (icon_size, icon_size)), (rect.x + 12, rect.centery - icon_size // 2))
        text_left = rect.x + icon_size + 24
        text_right = rect.right - 12
        available_width = text_right - text_left
        label_font = self.font
        if label_font.size(label)[0] > available_width:
            font_size = 18
            label_font = self._font(font_size)
            while label_font.size(label)[0] > available_width and font_size > 12:
                font_size -= 1
                label_font = self._font(font_size)
        self._text(label, ((text_left + text_right) // 2, rect.centery), font=label_font, center=True)
        return button

    def _panel(self, rect: pygame.Rect) -> None:
        self.screen.blit(self._sprite("hud_panel.png", rect.size), rect)

    def _log(self, message: str) -> None:
        self.loading_messages.append(message)
        self.loading_messages = self.loading_messages[-6:]
        print(message)

    def prepare_worker(self) -> None:
        self.loading_started = time.monotonic()

        def worker() -> None:
            try:
                self.loading_queue.put(("progress", (0.08, "Verificando y procesando activos visuales", "Consultando fuentes oficiales y preparando el catálogo...")))
                downloaded = list(ensure_asset_pack(ASSET_DIR / "downloads"))
                download_records = load_download_metadata(ASSET_DIR / "downloads")
                if downloaded:
                    self.loading_queue.put(("log", f"Activos externos descargados y procesados: {len(downloaded)}."))
                elif not download_records:
                    self.loading_queue.put(("log", "No hay imágenes descargadas permitidas; se conserva el catálogo procedural."))
                write_asset_credits(
                    ASSET_DIR,
                    download_records,
                    [
                        name for name in sprite_catalog()
                        if not (
                            name.startswith("tank_04_")
                            and any(record["normal_file"] == "tank_04_normal.png" for record in download_records)
                        )
                    ],
                )
                user_images = prepare_user_tanks(ASSET_DIR)
                user_sources = [f"user:{normal.stem}" for _, normal, _ in user_images]
                self.loading_queue.put(("assets_ready", (download_records, user_images)))
                current_source_manifest = {
                    "tank": sorted(
                        [f"procedural:tank_{variant:02}" for variant in range(1, 5)]
                        + user_sources
                        + (["downloaded:commons_tank"] if download_records else [])
                    ),
                    "vehicle": sorted(
                        f"procedural:{kind}_{variant:02}"
                        for kind in ("car", "pickup", "truck", "helicopter")
                        for variant in range(1, 4)
                    ),
                    "empty": sorted(f"procedural:ground_{index:02}" for index in range(8)),
                }
                if not self.force_retrain and MODEL_INFO_FILE.is_file():
                    existing_info = json.loads(MODEL_INFO_FILE.read_text(encoding="utf-8"))
                    if not isinstance(existing_info, dict) or existing_info.get("source_manifest") != current_source_manifest:
                        self.force_retrain = True
                        self.loading_queue.put(("log", "Cambió el catálogo de imágenes; se reentrena el modelo para incluir las fuentes nuevas."))
                self.loading_queue.put(("progress", (0.2, "Validando modelo y preparando el clasificador", "Revisando el modelo visual...")))
                model: Optional[VisionPipeline] = None
                if not self.force_retrain and MODEL_FILE.is_file() and METRICS_FILE.is_file():
                    try:
                        model = VisionPipeline(MODEL_FILE, FALLBACK_MODEL_FILE)
                        if model.backend == "unavailable":
                            raise ValueError("No hay un backend visual cargado.")
                        self.loading_queue.put(("log", "Modelo local encontrado; no se reentrena."))
                    except (OSError, ValueError, EOFError, TypeError, AttributeError, RuntimeError) as exc:
                        self.loading_queue.put(("log", f"El modelo local no se pudo validar: {exc}"))
                        model = None
                if model is None and not self.cancel_preparation.is_set():
                    self.loading_queue.put(("progress", (0.55, "Entrenando el modelo visual", "El modelo no existe o está dañado; entrenando el clasificador...")))
                    command = [
                        sys.executable,
                        str(BASE_DIR / "train_vision.py"),
                        "--output",
                        str(MODEL_FILE),
                    ]
                    process = subprocess.Popen(command, cwd=BASE_DIR)
                    deadline = time.monotonic() + 600
                    while process.poll() is None:
                        if self.cancel_preparation.is_set():
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            fallback = VisionPipeline(None, FALLBACK_MODEL_FILE)
                            self.loading_queue.put(("log", f"Preparación cancelada; backend disponible: {fallback.backend}."))
                            self.loading_queue.put(("model", (fallback if fallback.backend != "unavailable" else None, {"pipeline": fallback.backend, "status": "fallback"}, {})))
                            self.loading_queue.put(("done", None))
                            return
                        if time.monotonic() >= deadline:
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            raise subprocess.TimeoutExpired(command, 600)
                        time.sleep(0.2)
                    if process.returncode:
                        raise subprocess.CalledProcessError(process.returncode, command)
                    model = VisionPipeline(MODEL_FILE, FALLBACK_MODEL_FILE)
                    if model.backend == "unavailable":
                        raise RuntimeError("El entrenamiento terminó sin producir un modelo cargable.")
                    self.force_retrain = False
                    self.loading_queue.put(("log", "Modelo entrenado y validado."))
                if model is not None:
                    info = json.loads(MODEL_INFO_FILE.read_text(encoding="utf-8")) if MODEL_INFO_FILE.exists() else {}
                    metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8")) if METRICS_FILE.exists() else {}
                    info["pipeline"] = model.backend
                    self.loading_queue.put(("model", (model, info, metrics)))
                else:
                    fallback = VisionPipeline(None, FALLBACK_MODEL_FILE)
                    self.loading_queue.put(("model", (fallback if fallback.backend != "unavailable" else None, {"pipeline": fallback.backend, "status": "fallback"}, {})))
                self.loading_queue.put(("progress", (1.0, "Preparación lista", "Usando imágenes propias/procedurales y el modelo disponible.")))
                self.loading_queue.put(("done", None))
            except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError, RuntimeError) as exc:
                fallback = VisionPipeline(None, FALLBACK_MODEL_FILE)
                self.loading_queue.put(("log", f"Preparación no disponible: {exc}; backend alterno: {fallback.backend}."))
                self.loading_queue.put(("model", (fallback if fallback.backend != "unavailable" else None, {"pipeline": fallback.backend, "status": "fallback", "reason": str(exc)}, {})))
                self.loading_queue.put(("done", None))

        self.loading_thread = threading.Thread(target=worker, name="preparacion-modelo", daemon=True)
        self.loading_thread.start()

    def _read_loading_messages(self) -> None:
        while True:
            try:
                kind, payload = self.loading_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                self.loading_progress, self.loading_step, message = payload
                self._log(message)
            elif kind == "log":
                self._log(str(payload))
            elif kind == "model":
                self.vision, self.model_info, self.model_metrics = payload
                self.model_info.setdefault("pipeline", "No disponible" if self.vision is None else self.vision.backend)
                self.model_info.setdefault("status", "fallback" if self.vision is None else "ok")
            elif kind == "assets_ready":
                download_records, user_images = payload
                for variant, (_, normal, destroyed) in enumerate(user_images[:4], start=1):
                    self.sprites[f"tank_{variant:02}_normal.png"] = pygame.image.load(str(normal)).convert_alpha()
                    self.sprites[f"tank_{variant:02}_destroyed.png"] = pygame.image.load(str(destroyed)).convert_alpha()
                for record in download_records:
                    if record["normal_file"] == "tank_04_normal.png":
                        directory = ASSET_DIR / "downloads"
                        self.sprites["tank_04_normal.png"] = pygame.image.load(
                            str(directory / record["normal_file"])
                        ).convert_alpha()
                        self.sprites["tank_04_destroyed.png"] = pygame.image.load(
                            str(directory / record["destroyed_file"])
                        ).convert_alpha()
            elif kind == "done":
                self.loading_done = True
                self.state = "model_summary"

    def _create_drone(self, side: str) -> Drone:
        assert self.world is not None
        is_red = side == "red"
        drone = Drone(
            "Dron 1 · Rojo" if is_red else "Dron 2 · Azul",
            (227, 74, 65) if is_red else (76, 143, 240),
            self.world.red_start if is_red else self.world.blue_start,
            "bfs" if is_red else "dfs",
            self.world.goal,
            self.world.adjacency,
            self.vision,
            MODE_3_VISION_RADIUS if self.mode_index == 2 else VISION_RADIUS,
            self.match_seed,
        )
        if is_red:
            self.red_drone = drone
        else:
            self.blue_drone = drone
        return drone

    def start_match(self, mode_index: int, seed: Optional[int] = None) -> None:
        if self.vision is None:
            self.error_text = "No hay un modelo de visión disponible. Reentrena el modelo o revisa el log."
            self.state = "error"
            return
        self.mode_index = mode_index
        self.match_seed = seed if seed is not None else 173 + mode_index * 31
        self.world = World(self.match_seed, include_stations=mode_index == 2)
        self.world_snapshot = self.world.snapshot()
        self.red_drone = self._create_drone("red")
        self.blue_drone = self._create_drone("blue")
        self.mode2_order = [self.mode2_first, "blue" if self.mode2_first == "red" else "red"] if mode_index == 1 else []
        self.mode2_turn_index = 0
        self.mode2_stats = {}
        self.results_path = None
        self.winner = ""
        self.conflict_priority = 0
        self.match_ticks = 0
        self.tick_count = 0
        self.result_text = ""
        self.effects.clear()
        self.state = "match"
        for drone in self._active_drones():
            drone.observe_world(self.world, self.sprites, CELL_SIZE)

    def _active_drones(self) -> List[Drone]:
        if self.mode_index == 1 and self.mode2_order:
            side = self.mode2_order[self.mode2_turn_index]
            drone = self.red_drone if side == "red" else self.blue_drone
            return [drone] if drone is not None else []
        return [drone for drone in (self.red_drone, self.blue_drone) if drone is not None]

    def _drone_records(self) -> Dict[str, Dict[str, int | float | bool]]:
        if self.mode_index == 1:
            return dict(self.mode2_stats)
        records: Dict[str, Dict[str, int | float | bool]] = {}
        if self.red_drone:
            records["dron_1_bfs"] = drone_statistics(self.red_drone)
        if self.blue_drone:
            records["dron_2_dfs"] = drone_statistics(self.blue_drone)
        return records

    def _save_results(self) -> None:
        records = self._drone_records()
        self.results_path = save_match_result(
            self.results_dir,
            self.mode_names[self.mode_index],
            self.match_seed,
            {
                "grid_size": GRID_SIZE,
                "vision_radius": MODE_3_VISION_RADIUS if self.mode_index == 2 else VISION_RADIUS,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "shot_ticks": SHOT_TICKS,
                "battery_max": BATTERY_MAX,
                "ammo_max": AMMO_MAX,
            },
            records,
            self.winner,
        )

    def _finish_mode2_turn(self) -> None:
        side = self.mode2_order[self.mode2_turn_index]
        drone = self.red_drone if side == "red" else self.blue_drone
        assert drone is not None and self.world is not None
        self.mode2_stats["dron_1_bfs" if side == "red" else "dron_2_dfs"] = drone_statistics(drone)
        self.mode2_turn_index += 1
        if self.mode2_turn_index < len(self.mode2_order):
            self.world.restore(self.world_snapshot or {})
            next_side = self.mode2_order[self.mode2_turn_index]
            next_drone = self._create_drone(next_side)
            self.tick_count = 0
            next_drone.observe_world(self.world, self.sprites, CELL_SIZE)
            return
        self._choose_mode2_winner()
        self.state = "results"
        self._save_results()

    def _choose_mode2_winner(self) -> None:
        red = self.mode2_stats["dron_1_bfs"]
        blue = self.mode2_stats["dron_2_dfs"]
        def rank(stats: Dict[str, int | float | bool]) -> tuple[float, ...]:
            return (
                float(stats["score"]),
                float(stats["tanks_destroyed"]),
                -float(stats["vehicles_destroyed_wrongly"]),
                -float(stats["false_negatives"]),
                float(stats["recognition_accuracy_pct"]),
                -float(stats["elapsed_time_s"]),
                -float(stats["real_steps"]),
                -float(stats["nodes_expanded"]),
            )
        if rank(red) == rank(blue):
            self.winner = "Empate"
            self.result_text = "Empate: BFS y DFS obtuvieron el mismo resultado."
        elif rank(red) > rank(blue):
            self.winner = "Dron 1 · BFS"
            self.result_text = "Gana BFS: mejor resultado global en score, aciertos y errores."
        else:
            self.winner = "Dron 2 · DFS"
            self.result_text = "Gana DFS: mejor resultado global en score, aciertos y errores."

    def _finish_match(self) -> None:
        assert self.red_drone is not None and self.blue_drone is not None
        if self.mode_index == 2:
            ranked = [
                drone
                for drone in (self.red_drone, self.blue_drone)
                if drone.reached_goal and not drone.crashed
            ]
            if ranked:
                ranked.sort(key=lambda drone: (-drone.score, drone.elapsed_time_s))
                if len(ranked) == 1:
                    self.winner = ranked[0].name
                    other = self.blue_drone if ranked[0] is self.red_drone else self.red_drone
                    self.result_text = (
                        f"Gana {self.winner}: llegó a la meta con vida; "
                        f"{other.name} no llegó con vida."
                    )
                elif (
                    ranked[0].score == ranked[1].score
                    and ranked[0].elapsed_time_s == ranked[1].elapsed_time_s
                ):
                    self.winner = "Empate"
                    self.result_text = "Empate entre los drones que llegaron a la meta con vida."
                else:
                    self.winner = ranked[0].name
                    self.result_text = (
                        f"Gana {self.winner}: score más alto entre quienes llegaron con vida; "
                        "el tiempo desempata."
                    )
            else:
                ranked = [self.red_drone, self.blue_drone]
                ranked.sort(key=lambda drone: -drone.score)
                if ranked[0].score == ranked[1].score:
                    self.winner = "Empate"
                    self.result_text = "Ambos se estrellaron con el mismo score."
                else:
                    self.winner = ranked[0].name
                    self.result_text = f"Ambos se estrellaron; gana {self.winner} por score."
        else:
            ranked = sorted(
                (self.red_drone, self.blue_drone),
                key=lambda drone: (-drone.score, drone.elapsed_time_s, drone.real_steps),
            )
            if (
                ranked[0].score == ranked[1].score
                and ranked[0].elapsed_time_s == ranked[1].elapsed_time_s
                and ranked[0].real_steps == ranked[1].real_steps
            ):
                self.winner = "Empate"
            else:
                self.winner = ranked[0].name
            self.result_text = (
                "Empate según score, tiempo y pasos."
                if self.winner == "Empate"
                else f"Gana {self.winner} según score, tiempo y pasos."
            )
        self.state = "results"
        self._save_results()

    def advance_match(self) -> None:
        if not self.world or not self.red_drone or not self.blue_drone:
            return
        self.match_ticks += 1
        self.tick_count += 1
        resource_mode = self.mode_index == 2
        active = self._active_drones()
        pending_at_start = {
            drone.name: drone.pending_shot_id
            for drone in active
            if drone.pending_shot_id is not None
        }
        for drone in active:
            was_crashed = drone.crashed
            drone.advance(self.world, resource_mode)
            if drone.resource_phase is None:
                drone.observe_world(self.world, self.sprites, CELL_SIZE)
            if drone.crashed and not was_crashed:
                self.effects.append((drone.current, self.tick_count))

        by_id = {entity.entity_id: entity for entity in self.world.entities}
        for drone in active:
            if drone.name not in pending_at_start:
                continue
            destroyed_id = drone.advance_shot()
            if destroyed_id is None:
                continue
            target = by_id[destroyed_id]
            if not target.destroy(drone.name, self.tick_count):
                continue
            if target.kind == "tank":
                drone.tanks_destroyed += 1
            else:
                drone.vehicles_destroyed_wrongly += 1
            self.effects.append((target.position, self.tick_count))
            for observer in (self.red_drone, self.blue_drone):
                if observer:
                    observer.forget_entity(target)

        candidates = {
            drone.name: drone.target_candidates(self.world)
            for drone in active
            if drone.active and drone.pending_shot_id is None and not drone.goal_phase
        }
        common_targets = set.intersection(
            *(
                {entity.entity_id for entity in entities[:1]}
                for entities in candidates.values()
                if entities
            )
        ) if sum(bool(entities) for entities in candidates.values()) > 1 else set()
        priority_order = active
        if common_targets:
            priority_order = active[self.conflict_priority :] + active[: self.conflict_priority]
            self.conflict_priority = 1 - self.conflict_priority
        reserved = {
            drone.pending_shot_id
            for drone in active
            if drone.pending_shot_id is not None
        }
        for drone in priority_order:
            for target in candidates.get(drone.name, []):
                if target.entity_id in reserved or not target.alive:
                    continue
                if drone.begin_shot(target, resource_mode):
                    reserved.add(target.entity_id)
                    if resource_mode and drone.battery <= 0:
                        drone.pending_shot_id = None
                        drone.shot_ticks_remaining = 0
                        drone.crashed = True
                        drone.finished = True
                        drone.stopped_at_ticks = drone.elapsed_ticks
                        self.effects.append((drone.current, self.tick_count))
                    break

        if not self.world.tanks_alive():
            self.world.activate_goal()
        elif resource_mode and any(drone.active and drone.ammo == 0 for drone in active):
            self.world.activate_goal()
        elif self.mode_index == 1 and active and active[0].done_searching:
            self.world.activate_goal()
        elif self.mode_index != 1 and active and all(
            drone.done_searching or drone.finished for drone in active
        ):
            self.world.activate_goal()

        if self.mode_index == 1:
            if active and (active[0].finished or active[0].crashed):
                self._finish_mode2_turn()
            return
        if all(drone.finished or drone.crashed for drone in (self.red_drone, self.blue_drone)):
            self._finish_match()
        elif self.match_ticks >= 30000:
            raise RuntimeError("La partida excedió 30 000 ticks sin completar el objetivo.")

    def _draw_loading(self) -> None:
        self.screen.fill((23, 39, 49))
        self.screen.blit(self._sprite("background_city.png", self.screen.get_size()), (0, 0))
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((14, 25, 34, 210))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(self._sprite("logo.png", (700, 138)), (240, 68))
        self._text("DRONE WARS", (660, 137), font=self.large_font, center=True)
        self._text("PREPARANDO JUEGO", (self.screen.get_width() // 2, 270), font=self.large_font, center=True)
        self._text(self.loading_step, (self.screen.get_width() // 2, 317), center=True)
        bar = pygame.Rect(160, 360, self.screen.get_width() - 320, 30)
        self.screen.blit(self._sprite("hud_panel.png", bar.size), bar)
        fill_width = int((bar.width - 12) * self.loading_progress)
        if fill_width:
            fill_rect = pygame.Rect(bar.x + 6, bar.y + 6, fill_width, 18)
            self.screen.blit(self._sprite("bar_progress.png", fill_rect.size), fill_rect)
        elapsed = time.monotonic() - self.loading_started
        remaining = max(1, int(elapsed * (1 - self.loading_progress) / max(self.loading_progress, 0.03)))
        estimate = f"{remaining} s" if elapsed >= 2 else "calculando"
        self._text(f"Progreso {self.loading_progress * 100:.0f}% · estimación restante: {estimate}", (self.screen.get_width() // 2, 410), center=True)
        for index, message in enumerate(self.loading_messages[-4:]):
            self._text(message[:90], (190, 458 + index * 29), (221, 231, 220), self.small_font)
        self.buttons = []
        self._button(pygame.Rect(447, 660, 286, 58), "CANCELAR", "cancel", "icon_cancel.png")
        pygame.display.flip()

    def _draw_model_summary(self) -> None:
        self.screen.fill((23, 39, 49))
        self.screen.blit(self._sprite("background_city.png", self.screen.get_size()), (0, 0))
        self._panel(pygame.Rect(210, 80, 760, 600))
        self.screen.blit(self._sprite("medal.png", (98, 98)), (250, 114))
        self._text("RESUMEN DEL MODELO", (590, 150), font=self.large_font, center=True)
        metrics = self.model_metrics
        accuracy = metrics.get("accuracy", self.model_info.get("accuracy", "n/d"))
        if isinstance(accuracy, (float, int)):
            accuracy = f"{accuracy:.1%}"
        lines = [
            f"Backend activo: {self.model_info.get('pipeline', 'No disponible')}",
            f"Estado: {self.model_info.get('status', 'listo')}",
            f"Precisión de validación: {accuracy}",
            "Clases: tanque · vehículo · calle",
        ]
        for index, line in enumerate(lines):
            self._text(line, (330, 260 + index * 44))
        matrix = metrics.get("confusion_matrix")
        if matrix:
            self._text("Matriz de confusión", (590, 465), font=self.font, center=True)
            labels = ("tanque", "vehículo", "vacío")
            for row, values in enumerate(matrix[:3]):
                text = " · ".join(f"{labels[col]}: {values[col]}" for col in range(min(3, len(values))))
                self._text(text, (360, 500 + row * 28), color=(195, 216, 206), font=self.small_font)
        self.buttons = []
        self._button(pygame.Rect(440, 590, 300, 58), "CONTINUAR", "menu", "icon_score.png")
        pygame.display.flip()

    def _draw_menu(self) -> None:
        self.screen.blit(self._sprite("background_city.png", self.screen.get_size()), (0, 0))
        shade = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        shade.fill((17, 26, 34, 130))
        self.screen.blit(shade, (0, 0))
        self.screen.blit(self._sprite("logo.png", (710, 140)), (235, 70))
        self._text("DRONE WARS · CIUDAD DE JUGUETE", (645, 145), font=self.large_font, center=True)
        self._text("Explora, observa y compara cada ruta", (590, 231), color=(229, 222, 189), center=True)
        self.buttons = []
        self._button(pygame.Rect(440, 320, 300, 62), "JUGAR", "play", "icon_drone.png")
        self._button(pygame.Rect(440, 400, 300, 62), "INFORMACIÓN DEL MODELO", "info", "icon_nodes.png")
        self._button(pygame.Rect(440, 480, 300, 62), "REENTRENAR MODELO", "retrain", "icon_nodes.png")
        self._button(pygame.Rect(440, 560, 300, 62), "SALIR", "quit", "icon_error.png")
        self._text("ESC · salir", (590, 650), color=(221, 221, 204), font=self.small_font, center=True)
        pygame.display.flip()

    def _draw_modes(self) -> None:
        self.screen.blit(self._sprite("background_city.png", self.screen.get_size()), (0, 0))
        self._panel(pygame.Rect(260, 55, 660, 690))
        self._text("ELIGE EL MODO DE VUELO", (590, 96), font=self.large_font, center=True)
        self.buttons = []
        for index, label in enumerate(self.mode_names):
            y = 175 + index * 137
            rect = pygame.Rect(350, y, 480, 96)
            self._button(rect, label, f"mode_{index}", f"mode_{index + 1:02}.png")
        self._button(pygame.Rect(440, 620, 300, 58), "VOLVER", "menu", "icon_error.png")
        pygame.display.flip()

    def _draw_drone_select(self) -> None:
        self.screen.blit(self._sprite("background_city.png", self.screen.get_size()), (0, 0))
        self._panel(pygame.Rect(180, 68, 820, 650))
        self._text("MODO 2 · SELECCIÓN DE DRONES", (590, 112), font=self.large_font, center=True)
        self.screen.blit(self._sprite("drone_blue_0.png", (110, 110)), (315, 170))
        self.screen.blit(self._sprite("drone_red_0.png", (110, 110)), (745, 170))
        self._text("DRON 2 · AZUL", (370, 302), center=True)
        self._text("DRON 1 · ROJO", (800, 302), center=True)
        self._text(f"Algoritmo: DFS · {'PRIMERO' if self.mode2_first == 'blue' else 'SEGUNDO'}", (370, 352), center=True)
        self._text(f"Algoritmo: BFS · {'PRIMERO' if self.mode2_first == 'red' else 'SEGUNDO'}", (800, 352), center=True)
        self.buttons = []
        self._button(pygame.Rect(250, 397, 240, 54), "JUGAR DFS PRIMERO", "first_blue", "icon_drone.png")
        self._button(pygame.Rect(680, 397, 240, 54), "JUGAR BFS PRIMERO", "first_red", "icon_drone.png")
        self._button(pygame.Rect(365, 515, 450, 62), "INICIAR COMPARACIÓN", "start_mode_2", "icon_score.png")
        self._button(pygame.Rect(440, 620, 300, 54), "VOLVER", "modes", "icon_error.png")
        pygame.display.flip()

    def _draw_error(self) -> None:
        self.screen.fill((42, 26, 34))
        self.screen.blit(self._sprite("background_city.png", self.screen.get_size()), (0, 0))
        shade = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        shade.fill((38, 17, 24, 215))
        self.screen.blit(shade, (0, 0))
        self._panel(pygame.Rect(160, 100, 860, 570))
        self.screen.blit(self._sprite("icon_error.png", (66, 66)), (220, 137))
        self._text("Ocurrió un error", (620, 167), font=self.large_font, center=True)
        if self.error_type:
            self._text(f"Excepción: {self.error_type}", (230, 240), color=(255, 202, 190))
            self._text(f"Mensaje: {self.error_message[:90]}", (230, 278), color=(255, 202, 190))
            self._text(f"Origen: {self.error_frame}", (230, 316), color=(218, 228, 221))
        else:
            self._text(self.error_text[:100], (230, 260), color=(255, 202, 190))
        log_message = self.log_warning or "Detalle técnico: logs/error.log"
        self._text(log_message[:100], (230, 354), color=(218, 228, 221))
        self.buttons = []
        self._button(pygame.Rect(207, 440, 245, 60), "REINTENTAR", "retry", "icon_drone.png")
        self._button(pygame.Rect(468, 440, 280, 60), "VOLVER AL MENÚ", "menu", "icon_score.png")
        self._button(pygame.Rect(764, 440, 208, 60), "SALIR", "quit", "icon_error.png")
        pygame.display.flip()

    def _draw_drone_hud(self, drone: Drone, left: bool) -> None:
        x = 12 if left else self.screen.get_width() - PANEL_SIZE[0] - 12
        rect = pygame.Rect(x, 12, *PANEL_SIZE)
        self._panel(rect)
        color = drone.color
        title = "DRON 2 · AZUL" if left else "DRON 1 · ROJO"
        self._text(title, (rect.x + 18, rect.y + 11), color=color, font=self.font)
        stats = drone_statistics(drone)
        lines = (
            f"Score {stats['score']} · TP {stats['tanks_destroyed']} · FP {stats['vehicles_destroyed_wrongly']}",
            f"FN {stats['false_negatives']} · Recon. {stats['recognition_accuracy_pct']:.0f}% · Tiro {stats['shot_precision_pct']:.0f}%",
            f"Celdas {stats['cells_explored']} · Nodos {stats['nodes_expanded']} · Pasos {stats['real_steps']}",
            f"Tiempo {stats['elapsed_time_s']:.1f} s · {stats['elapsed_ticks']} ticks",
        )
        for index, line in enumerate(lines):
            self._text(line, (rect.x + 16, rect.y + 43 + index * 23), font=self.small_font)
        if self.mode_index == 2:
            self._draw_bar(rect.x + 16, rect.y + 137, "icon_battery.png", drone.battery, (75, 186, 128), BATTERY_MAX)
            self._draw_bar(rect.x + 158, rect.y + 137, "icon_ammo.png", drone.ammo, (228, 160, 73), AMMO_MAX)
            self._text(
                f"Recargas {drone.recharges} · Choque {'SÍ' if drone.crashed else 'NO'}",
                (rect.x + 16, rect.y + 170),
                font=self.small_font,
            )

    def _draw_bar(
        self,
        x: int,
        y: int,
        icon: str,
        value: float,
        color: Tuple[int, int, int],
        maximum: float = 100.0,
    ) -> None:
        self.screen.blit(self._sprite(icon, (21, 21)), (x, y))
        rect = pygame.Rect(x + 27, y + 6, 90, 10)
        self.screen.blit(self._sprite("bar_track.png", rect.size), rect)
        if value > 0:
            fraction = min(1.0, value / maximum)
            sprite_name = "bar_battery.png" if icon == "icon_battery.png" else "bar_ammo.png"
            fill_rect = pygame.Rect(rect.x, rect.y, int(rect.width * fraction), rect.height)
            if fill_rect.width:
                self.screen.blit(self._sprite(sprite_name, fill_rect.size), fill_rect)
        self._text(f"{value:g}", (rect.right + 4, y + 2), font=self.small_font)

    def _draw_match(self) -> None:
        assert self.world and self.red_drone and self.blue_drone
        self.screen.fill((28, 40, 46))
        map_rect = pygame.Rect(*BOARD_ORIGIN, GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE)
        frame_rect = map_rect.inflate(10, 10)
        self.screen.blit(self._sprite("map_frame.png", frame_rect.size), frame_rect)
        active = self._active_drones()
        visible = set().union(*(drone.visible for drone in active))
        explored = set().union(*(drone.explored for drone in active))
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                position = (x, y)
                rect = pygame.Rect(map_rect.x + x * CELL_SIZE, map_rect.y + y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                self.screen.blit(self._sprite("ground_street.png", rect.size), rect)
                if self.world.grid[y][x]:
                    index = ((x * 7 + y * 13 + self.mode_index) % 6) + 1
                    self.screen.blit(self._sprite(f"building_{index:02}.png", rect.size), rect)
                if position not in visible and position not in explored:
                    self.screen.blit(self._sprite("fog.png", rect.size), rect)
                elif position not in visible:
                    shade = self._sprite("fog.png", rect.size).copy()
                    shade.set_alpha(85)
                    self.screen.blit(shade, rect)
        for entity in self.world.entities:
            if entity.kind == "charging_station" and self.mode_index != 2:
                continue
            if (
                entity.kind != "charging_station"
                and entity.position not in visible
                and entity.position not in explored
            ):
                continue
            if entity.kind == "charging_station":
                sprite = self._sprite(f"charging_{self.tick_count // 8 % 2}.png", (CELL_SIZE, CELL_SIZE))
            else:
                sprite = self._sprite(entity.sprite_name, (CELL_SIZE, CELL_SIZE))
                sprite = pygame.transform.rotate(sprite, -entity.orientation)
            px, py = entity.position
            self.screen.blit(
                sprite,
                sprite.get_rect(
                    center=(
                        map_rect.x + px * CELL_SIZE + CELL_SIZE // 2,
                        map_rect.y + py * CELL_SIZE + CELL_SIZE // 2,
                    )
                ),
            )
            if not entity.alive and entity.destroyed_tick is not None:
                smoke_age = self.tick_count - entity.destroyed_tick
                if 0 <= smoke_age < 48:
                    smoke_frame = (smoke_age // 8) % 2
                    smoke = self._sprite(f"smoke_{smoke_frame:02}.png", (CELL_SIZE, CELL_SIZE))
                    self.screen.blit(smoke, (map_rect.x + px * CELL_SIZE, map_rect.y + py * CELL_SIZE))
        if self.world.goal_exists and self.world.goal is not None:
            gx, gy = self.world.goal
            pulse_size = CELL_SIZE if self.tick_count % 20 < 10 else CELL_SIZE + 4
            self.screen.blit(self._sprite("goal_target.png", (pulse_size, pulse_size)), (map_rect.x + gx * CELL_SIZE - (pulse_size - CELL_SIZE) // 2, map_rect.y + gy * CELL_SIZE - (pulse_size - CELL_SIZE) // 2))
        for drone, color in ((self.red_drone, "red"), (self.blue_drone, "blue")):
            if self.mode_index == 1 and drone not in active:
                continue
            dx, dy = drone.current
            frame = self.tick_count // 3 % 2
            image_name = "crash_marker.png" if drone.crashed else f"drone_{color}_{frame}.png"
            image_size = (CELL_SIZE + 6, CELL_SIZE + 6)
            image = self._sprite(image_name, image_size)
            self.screen.blit(image, (map_rect.x + dx * CELL_SIZE - 3, map_rect.y + dy * CELL_SIZE - 3))
        self._draw_drone_hud(self.blue_drone, True)
        self._draw_drone_hud(self.red_drone, False)
        self._text(self.mode_names[self.mode_index], (self.screen.get_width() // 2, 710), color=(235, 220, 162), center=True)
        for drone in (self.red_drone, self.blue_drone):
            if drone.pending_shot_id is None:
                continue
            target = next(
                (entity for entity in self.world.entities if entity.entity_id == drone.pending_shot_id),
                None,
            )
            if target is None:
                continue
            start = (
                map_rect.x + drone.current[0] * CELL_SIZE + CELL_SIZE // 2,
                map_rect.y + drone.current[1] * CELL_SIZE + CELL_SIZE // 2,
            )
            end = (
                map_rect.x + target.position[0] * CELL_SIZE + CELL_SIZE // 2,
                map_rect.y + target.position[1] * CELL_SIZE + CELL_SIZE // 2,
            )
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = max(12, int((dx * dx + dy * dy) ** 0.5))
            beam = self._sprite("laser_beam.png", (length, 16))
            beam = pygame.transform.rotate(beam, -pygame.math.Vector2(1, 0).angle_to((dx, dy)))
            self.screen.blit(beam, beam.get_rect(center=((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)))
        for position, created in self.effects:
            age = self.tick_count - created
            if 0 <= age < 16:
                px, py = position
                frame = min(7, age // 2)
                self.screen.blit(self._sprite(f"explosion_{frame:02}.png", (44, 44)), (map_rect.x + px * CELL_SIZE - 8, map_rect.y + py * CELL_SIZE - 8))
        self.effects = [(position, created) for position, created in self.effects if self.tick_count - created < 16]
        pygame.display.flip()

    def _draw_results(self) -> None:
        self.screen.blit(self._sprite("background_city.png", self.screen.get_size()), (0, 0))
        self._panel(pygame.Rect(70, 35, 1040, 730))
        self._text("COMPARACIÓN FINAL", (590, 75), font=self.large_font, center=True)
        self._text(self.result_text, (590, 118), color=(244, 228, 178), center=True)
        if self.mode_index == 1:
            red_stats = self.mode2_stats.get("dron_1_bfs", {})
            blue_stats = self.mode2_stats.get("dron_2_dfs", {})
        else:
            red_stats = drone_statistics(self.red_drone) if self.red_drone else {}
            blue_stats = drone_statistics(self.blue_drone) if self.blue_drone else {}
        self._text("Métrica", (160, 164), color=(232, 208, 125), font=self.font)
        self._text("Dron 1 · BFS", (560, 164), color=(236, 105, 93), font=self.font, center=True)
        self._text("Dron 2 · DFS", (840, 164), color=(105, 162, 246), font=self.font, center=True)
        rows = [
            ("Score", "score", False),
            ("Tanques destruidos (TP)", "tanks_destroyed", False),
            ("Vehículos destruidos (FP)", "vehicles_destroyed_wrongly", True),
            ("Falsos negativos (FN)", "false_negatives", True),
            ("Reconocimiento (%)", "recognition_accuracy_pct", False),
            ("Precisión de disparo (%)", "shot_precision_pct", False),
            ("Celdas exploradas", "cells_explored", False),
            ("Nodos expandidos", "nodes_expanded", True),
            ("Pasos reales", "real_steps", True),
            ("Tiempo simulado (s)", "elapsed_time_s", True),
        ]
        if self.mode_index == 2:
            rows.extend(
                [
                    ("Batería restante", "battery", False),
                    ("Munición restante", "ammo", False),
                    ("Recargas", "recharges", False),
                    ("Chocó", "crashed", False),
                ]
            )
        for index, (label, key, lower_is_better) in enumerate(rows):
            y = 199 + index * 32
            left = red_stats.get(key, 0)
            right = blue_stats.get(key, 0)
            left_value = float(left) if isinstance(left, (int, float, bool)) else 0.0
            right_value = float(right) if isinstance(right, (int, float, bool)) else 0.0
            if left_value == right_value:
                left_color = right_color = (236, 224, 178)
            else:
                red_is_better = left_value < right_value if lower_is_better else left_value > right_value
                left_color = (255, 232, 153) if red_is_better else (174, 182, 183)
                right_color = (255, 232, 153) if not red_is_better else (174, 182, 183)
            if isinstance(left, float):
                left = f"{left:.1f}" if key == "elapsed_time_s" else f"{left:.1f}%"
            if isinstance(right, float):
                right = f"{right:.1f}" if key == "elapsed_time_s" else f"{right:.1f}%"
            if isinstance(left, bool):
                left = "SÍ" if left else "NO"
            if isinstance(right, bool):
                right = "SÍ" if right else "NO"
            self._text(label, (135, y), font=self.small_font)
            self._text(str(left), (560, y), color=left_color, font=self.small_font, center=True)
            self._text(str(right), (840, y), color=right_color, font=self.small_font, center=True)
        chart_rect = pygame.Rect(935, 200, 150, 290)
        self.screen.blit(self._sprite("hud_panel.png", chart_rect.size), chart_rect)
        self._text("PUNTUACIÓN", (chart_rect.centerx, chart_rect.y + 24), color=(235, 218, 164), font=self.small_font, center=True)
        score_red = max(0.0, float(red_stats.get("score", 0)))
        score_blue = max(0.0, float(blue_stats.get("score", 0)))
        scale = max(score_red, score_blue, 1.0)
        baseline = chart_rect.bottom - 42
        chart_top = chart_rect.y + 58
        max_height = baseline - chart_top
        pygame.draw.line(self.screen, (183, 197, 194), (chart_rect.x + 14, baseline), (chart_rect.right - 14, baseline), 2)
        for index, (score, sprite_name, label) in enumerate((
            (score_red, "bar_score_red.png", "BFS"),
            (score_blue, "bar_score_blue.png", "DFS"),
        )):
            height = int(max_height * score / scale)
            bar = pygame.Rect(chart_rect.x + 30 + index * 56, baseline - height, 34, height)
            if height:
                self.screen.blit(self._sprite(sprite_name, bar.size), bar)
            self._text(f"{score:g}", (bar.centerx, max(chart_top - 4, bar.top - 18)), font=self.small_font, center=True)
            self._text(label, (bar.centerx, baseline + 18), font=self.small_font, center=True)
        self.buttons = []
        self._button(pygame.Rect(250, 700, 270, 52), "JUGAR DE NUEVO", "modes", "icon_drone.png")
        self._button(pygame.Rect(545, 700, 190, 52), "MENÚ", "menu", "icon_score.png")
        self._button(pygame.Rect(770, 700, 190, 52), "SALIR", "quit", "icon_error.png")
        pygame.display.flip()

    def _draw_info(self) -> None:
        self.screen.fill((26, 40, 48))
        self._panel(pygame.Rect(190, 90, 800, 600))
        self._text("INFORMACIÓN DEL MODELO", (590, 145), font=self.large_font, center=True)
        values = (
            f"Backend activo: {self.model_info.get('pipeline', 'No disponible')}",
            f"Estado: {self.model_info.get('status', 'disponible')}",
            f"Precisión: {self.model_metrics.get('accuracy', self.model_info.get('accuracy', 'n/d'))}",
            f"Archivo: {MODEL_FILE.name}",
        )
        for index, line in enumerate(values):
            self._text(line, (270, 250 + index * 52))
        self.buttons = []
        self._button(pygame.Rect(330, 570, 270, 60), "REENTRENAR MODELO", "retrain", "icon_nodes.png")
        self._button(pygame.Rect(620, 570, 220, 60), "VOLVER", "menu", "icon_error.png")
        pygame.display.flip()

    def _draw_state(self) -> None:
        if self.state == "loading":
            self._draw_loading()
        elif self.state == "model_summary":
            self._draw_model_summary()
        elif self.state == "menu":
            self._draw_menu()
        elif self.state == "modes":
            self._draw_modes()
        elif self.state == "drone_select":
            self._draw_drone_select()
        elif self.state == "match":
            self.advance_match()
            if self.state == "match":
                self._draw_match()
            else:
                self._draw_state()
        elif self.state == "results":
            self._draw_results()
        elif self.state == "info":
            self._draw_info()
        elif self.state == "error":
            self._draw_error()

    def _save_screenshot(self, name: str) -> None:
        pygame.event.pump()
        pygame.display.flip()
        pygame.image.save(self.screen, str(SCREENSHOT_DIR / name))

    def capture_evidence(self) -> None:
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        if self.vision is None:
            raise RuntimeError("No hay clasificador real disponible para capturar evidencia.")
        previous_results_dir = self.results_dir
        with tempfile.TemporaryDirectory(prefix="drone-game-captures-") as directory:
            self.results_dir = Path(directory)
            self.loading_progress = 1.0
            self.loading_step = "Modelo y sprites cargados"
            self.loading_messages = [
                f"Backend: {self.vision.backend}",
                "Imágenes procedurales y descargas con licencia verificada",
            ]
            self._draw_loading()
            self._save_screenshot("01-preparando-juego.png")
            self._draw_model_summary()
            self._save_screenshot("02-resumen-modelo.png")
            self._draw_menu()
            self._save_screenshot("03-menu-principal.png")
            self._draw_modes()
            self._save_screenshot("04-seleccion-modos.png")

            self.start_match(0, seed=2401)
            self._draw_match()
            self._save_screenshot("05-modo-1-inicio-neblina.png")
            laser_captured = False
            destroyed_captured = False
            destroyed_effect_tick: Optional[int] = None
            for _ in range(10000):
                if self.state != "match":
                    break
                if not laser_captured and any(
                    drone.pending_shot_id is not None
                    for drone in (self.red_drone, self.blue_drone)
                    if drone
                ):
                    self._draw_match()
                    self._save_screenshot("06-modo-1-disparo-laser.png")
                    laser_captured = True
                before_destroyed = sum(
                    entity.kind == "tank" and not entity.alive
                    for entity in self.world.entities
                )
                self.advance_match()
                after_destroyed = sum(
                    entity.kind == "tank" and not entity.alive
                    for entity in self.world.entities
                )
                if not destroyed_captured and after_destroyed > before_destroyed:
                    self._draw_match()
                    self._save_screenshot("07-modo-1-explosion-destruido.png")
                    destroyed_captured = True
                    destroyed_effect_tick = self.tick_count
                    for frame in range(8):
                        while (
                            self.state == "match"
                            and self.tick_count < destroyed_effect_tick + frame * 2
                        ):
                            self.advance_match()
                        self._draw_match()
                        self._save_screenshot(f"animacion-explosion-{frame:02}.png")
                if self.world.goal_exists:
                    break
            if not laser_captured:
                raise RuntimeError("La partida real no produjo un disparo para la captura.")
            if self.world is None or not self.world.goal_exists:
                for _ in range(10000):
                    if self.state != "match" or self.world.goal_exists:
                        break
                    self.advance_match()
            if self.world is None or not self.world.goal_exists:
                raise RuntimeError("La partida real no produjo una meta para la captura.")
            tanks_remaining = len(self.world.tanks_alive())
            if tanks_remaining and max(
                self.red_drone.sweeps_completed,
                self.blue_drone.sweeps_completed,
            ) < MAX_SWEEPS:
                raise RuntimeError("La meta se activó antes de despejar el mapa o agotar barridos.")
            self._draw_match()
            self._save_screenshot("08-modo-1-meta.png")
            for _ in range(10000):
                if self.state != "match":
                    break
                self.advance_match()
            if self.state != "results":
                raise RuntimeError("La partida real del modo 1 no llegó a resultados.")
            self._draw_results()
            self._save_screenshot("12-resultados-grafica-veredicto.png")

            self.mode2_first = "blue"
            self._draw_drone_select()
            self._save_screenshot("09-modo-2-seleccion.png")
            self.start_match(1, seed=2402)
            self._draw_match()
            self._save_screenshot("10-modo-2-primer-turno.png")
            for _ in range(10000):
                if self.state != "match" or self.mode2_turn_index == 1:
                    break
                self.advance_match()
            if self.state != "match" or self.mode2_turn_index != 1:
                raise RuntimeError("La partida real del modo 2 no alcanzó el segundo turno.")
            self._draw_match()
            self._save_screenshot("11-modo-2-segundo-turno.png")
            for _ in range(10000):
                if self.state != "match":
                    break
                self.advance_match()
            if self.state != "results":
                raise RuntimeError("El modo 2 real no llegó a resultados.")

            self.start_match(2, seed=2403)
            self._draw_match()
            self._save_screenshot("13-modo-3-estaciones-recursos.png")
            for _ in range(10000):
                if self.state != "match":
                    break
                self.advance_match()
            if self.state != "results":
                raise RuntimeError("El modo 3 real no llegó a resultados.")

            self.error_text = "IndexError: list index out of range"
            self.error_type = "IndexError"
            self.error_message = "list index out of range"
            self.error_frame = "main.py:495 (_active_drones)"
            self.state = "error"
            self._draw_error()
            self._save_screenshot("14-pantalla-de-error.png")

            self.start_match(0, seed=2404)
            captured_movements = 0
            previous_positions = (self.red_drone.current, self.blue_drone.current)
            for _ in range(1000):
                self.advance_match()
                current_positions = (self.red_drone.current, self.blue_drone.current)
                moved_adjacent = any(
                    before != after and abs(before[0] - after[0]) + abs(before[1] - after[1]) == 1
                    for before, after in zip(previous_positions, current_positions)
                )
                if moved_adjacent:
                    self._draw_match()
                    self._save_screenshot(f"secuencia-movimiento-{captured_movements:02}.png")
                    captured_movements += 1
                    if captured_movements == 6:
                        break
                previous_positions = current_positions
            if captured_movements != 6:
                raise RuntimeError("No se pudieron capturar seis movimientos reales de celda adyacente.")
            print(
                "CAPTURAS_OK: menús, partidas reales con CNN, turnos, recursos, resultados, "
                "error y secuencias de explosión/movimiento guardadas."
            )
        self.results_dir = previous_results_dir

    def run_smoke(self) -> None:
        with tempfile.TemporaryDirectory(prefix="drone-game-smoke-") as directory:
            self.results_dir = Path(directory)
            self._run_smoke()
        self.results_dir = RESULTS_DIR

    def _run_smoke(self) -> None:
        class SmokeVision:
            def classify(self, image: Any) -> Tuple[str, float]:
                if not isinstance(image, np.ndarray):
                    raise TypeError("El clasificador de smoke recibió algo distinto de una imagen.")
                return "tank", 0.99

        self.vision = SmokeVision()  # type: ignore[assignment]
        self.model_info = {"pipeline": "Clasificador de prueba por imagen", "status": "smoke-test"}
        self.model_metrics = {}
        self.state = "model_summary"
        self._draw_model_summary()
        self._activate_button("menu")
        if self.state != "menu":
            raise RuntimeError("El botón CONTINUAR no abrió el menú.")
        self._draw_menu()
        self._activate_button("play")
        if self.state != "modes":
            raise RuntimeError("El botón JUGAR no abrió la selección de modos.")
        self._draw_modes()
        self._activate_button("mode_1")
        if self.state != "drone_select":
            raise RuntimeError("La selección de drones del modo 2 no se abrió.")
        self._draw_drone_select()
        self._save_screenshot("p0-modo2-seleccion.png")
        self._activate_button("first_blue")
        if self.mode2_first != "blue":
            raise RuntimeError("La selección del primer turno no eligió DFS.")
        self._draw_drone_select()
        self._activate_button("start_mode_2")
        if self.state != "match":
            raise RuntimeError("El botón INICIAR COMPARACIÓN no inició el modo 2.")
        self.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE}))
        if not self.running:
            raise RuntimeError("ESC cerró una partida en vez de limitarse al menú.")
        for mode in range(3):
            self.start_match(mode, seed=12 + mode)
            if self.world is None or self.world.goal is not None or self.world.goal_exists:
                raise RuntimeError(f"El modo {mode + 1} creó la meta antes de destruir los tanques.")
            if mode in (0, 2):
                self._draw_match()
                self._save_screenshot(
                    "p0-modo1-inicio-sin-meta.png" if mode == 0 else "p0-modo3-estaciones-hud.png"
                )
            ticks = 0
            captured_shot = False
            captured_destroyed = False
            captured_goal = False
            captured_second_turn = False
            while self.state == "match" and ticks < 20000:
                self.advance_match()
                ticks += 1
                if mode == 0 and not captured_shot and any(
                    drone.pending_shot_id is not None
                    for drone in (self.red_drone, self.blue_drone)
                    if drone
                ):
                    self._draw_match()
                    self._save_screenshot("p0-modo1-laser.png")
                    captured_shot = True
                if mode == 0 and not captured_destroyed and any(
                    entity.kind == "tank" and not entity.alive
                    for entity in self.world.entities
                ):
                    self._draw_match()
                    self._save_screenshot("p0-modo1-tanque-destruido.png")
                    captured_destroyed = True
                if mode == 0 and self.world.goal_exists and not captured_goal:
                    if self.world.tanks_alive() and not (
                        self.red_drone.sweeps_completed >= MAX_SWEEPS
                        and self.blue_drone.sweeps_completed >= MAX_SWEEPS
                    ):
                        raise RuntimeError("La meta de modo 1 apareció antes de despejar el mapa o agotar barridos.")
                    self._draw_match()
                    self._save_screenshot("p0-modo1-meta-activada.png")
                    captured_goal = True
                if mode == 1 and self.mode2_turn_index == 1 and not captured_second_turn:
                    if self.world.snapshot() != self.world_snapshot:
                        raise RuntimeError("El segundo turno no restauró el mundo original exactamente.")
                    second_drone = self._active_drones()[0]
                    expected_start = (
                        self.world.red_start if self.mode2_order[1] == "red" else self.world.blue_start
                    )
                    if second_drone.current != expected_start or second_drone.seed != self.match_seed:
                        raise RuntimeError("El segundo turno no conservó la posición o semilla original.")
                    self._draw_match()
                    self._save_screenshot("p0-modo2-segundo-turno.png")
                    captured_second_turn = True
            if self.state != "results":
                raise RuntimeError(f"El modo {mode + 1} no llegó a resultados tras {ticks} ciclos.")
            if self.results_path is None or not self.results_path.is_file():
                raise RuntimeError(f"El modo {mode + 1} no guardó sus resultados.")
            if not self.world or not self.world.goal_exists:
                raise RuntimeError(f"El modo {mode + 1} no activó la meta según sus reglas.")
            if not any(not entity.alive and entity.sprite_name.endswith("_destroyed.png") for entity in self.world.entities):
                raise RuntimeError(f"El modo {mode + 1} no dejó un objetivo destruido en el mapa.")
            for entity in self.world.entities:
                if not entity.alive and (
                    entity.position in self.red_drone.detections
                    or entity.position in self.blue_drone.detections
                ):
                    raise RuntimeError(
                        f"Un objetivo destruido siguió en la memoria en modo {mode + 1}, "
                        f"celda {entity.position}."
                    )
            saved = json.loads(self.results_path.read_text(encoding="utf-8"))
            for stats in saved["drones"].values():
                if stats["score"] != stats["tanks_destroyed"] - stats["vehicles_destroyed_wrongly"]:
                    raise RuntimeError("El resultado guardado no cumple score = TP - FP.")
                if stats["elapsed_time_s"] != stats["elapsed_ticks"] * TICK_SECONDS:
                    raise RuntimeError("El cronómetro guardado no corresponde a los ticks simulados.")
                for key in (
                    "false_negatives",
                    "recognition_accuracy_pct",
                    "shot_precision_pct",
                    "cells_explored",
                    "nodes_expanded",
                    "real_steps",
                ):
                    if key not in stats:
                        raise RuntimeError(f"Falta la estadística obligatoria {key}.")
            if self.mode_index == 1:
                last_side = self.mode2_order[-1]
                owners = [
                    (
                        "Dron 1 · Rojo",
                        "dron_1_bfs",
                    )
                    if last_side == "red"
                    else ("Dron 2 · Azul", "dron_2_dfs")
                ]
            else:
                owners = [
                    ("Dron 1 · Rojo", "dron_1_bfs"),
                    ("Dron 2 · Azul", "dron_2_dfs"),
                ]
            for owner_name, stats_key in owners:
                owner_stats = saved["drones"][stats_key]
                owned = [
                    entity
                    for entity in self.world.entities
                    if not entity.alive and entity.destroyed_by == owner_name
                ]
                if sum(entity.kind == "tank" for entity in owned) != owner_stats["tanks_destroyed"]:
                    raise RuntimeError("La puntuación de tanques no pertenece al dron que disparó.")
                if sum(entity.kind != "tank" for entity in owned) != owner_stats["vehicles_destroyed_wrongly"]:
                    raise RuntimeError("La penalización por vehículo no pertenece al dron que disparó.")
            if self.mode_index == 1:
                records = saved["drones"]
                if set(records) != {"dron_1_bfs", "dron_2_dfs"}:
                    raise RuntimeError("Modo 2 no guardó los dos turnos independientes.")
                self._draw_results()
            else:
                if self.red_drone is None or self.blue_drone is None:
                    raise RuntimeError("Faltan estadísticas de drones al finalizar.")
                self._draw_results()
            self._save_screenshot(f"p0-modo{mode + 1}-resultados.png")
            print(
                f"MODO_{mode + 1}_RESULTADOS: OK · ticks={ticks} · "
                f"meta={self.world.goal_exists} · resultados={self.results_path.name}"
            )
        self._activate_button("menu")
        if self.state != "menu":
            raise RuntimeError("El botón MENÚ no volvió al menú principal.")
        self.state = "error"
        self.error_text = "Error de prueba"
        self._draw_error()
        self._activate_button("menu")
        if self.state != "menu":
            raise RuntimeError("VOLVER AL MENÚ no funcionó desde el panel de error.")
        self._draw_menu()
        self._activate_button("quit")
        if self.running:
            raise RuntimeError("El botón SALIR no cerró el bucle.")
        self.running = True
        self.state = "match"
        self.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE}))
        if not self.running:
            raise RuntimeError("ESC cerró una partida.")
        self.handle_event(pygame.event.Event(pygame.QUIT))
        if self.running:
            raise RuntimeError("Cerrar la ventana no cerró el juego.")
        self.state = "menu"
        self.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE}))
        if self.running:
            raise RuntimeError("ESC no salió del menú principal.")
        self.running = True
        print("SMOKE_OK: partidas completas, render, botones, ESC y cierre de ventana comprobados.")

    def run_selfcheck(self, pause: bool = True) -> List[str]:
        failures: List[str] = []

        def report(name: str, ok: bool, detail: str, fix: str = "") -> None:
            status = "OK" if ok else "FALLO"
            print(f"{status} · {name}: {detail}" + (f" · Solución: {fix}" if not ok else ""))
            if not ok:
                failures.append(name)

        report("Python", sys.version_info >= (3, 10), f"{platform.python_version()} · {platform.machine()}", "instala Python 3.10–3.12 de 64 bits.")
        for package, import_name in (("pygame", "pygame"), ("numpy", "numpy"), ("opencv-python-headless", "cv2"), ("scikit-learn", "sklearn"), ("torch", "torch"), ("Pillow", "PIL"), ("certifi", "certifi")):
            try:
                __import__(import_name)
                version = importlib.metadata.version(package)
                report(f"Paquete {package}", True, version)
            except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
                report(f"Paquete {package}", False, str(exc), f"{sys.executable} -m pip install -r requirements.txt")
        report("SDL", bool(pygame.display.get_driver()), str(pygame.display.get_driver()))
        report("Ventana y dibujo real", self.screen is not None, f"set_mode {self.screen.get_size() if self.screen else 'no disponible'}")
        report("Fuentes", self.fonts_ready, "fuente alternativa local o integrada")
        manifest_names: List[str] = []
        if MANIFEST_FILE.exists():
            manifest_names = re.findall(r"`([^`]+\.png)`", MANIFEST_FILE.read_text(encoding="utf-8"))
        missing_assets = []
        for name in manifest_names:
            try:
                pygame.image.load(str(ASSET_DIR / name))
            except (pygame.error, OSError):
                missing_assets.append(name)
        report("Sprites del manifiesto", not missing_assets and bool(manifest_names), f"{len(manifest_names) - len(missing_assets)}/{len(manifest_names)} cargados", "vuelve a ejecutar main.py para regenerar sprites.")
        model_ok = False
        model_detail = "archivo ausente"
        try:
            model = VisionPipeline(MODEL_FILE, FALLBACK_MODEL_FILE)
            samples = (
                render_cell_image(
                    self.sprites["ground_street.png"],
                    self.sprites["tank_01_normal.png"],
                    73.0,
                    CELL_SIZE,
                ),
                render_cell_image(self.sprites["ground_street.png"], None, 0.0, CELL_SIZE),
            )
            predictions = [model.classify(sample) for sample in samples]
            model_ok = model.backend != "unavailable" and all(0 <= confidence <= 1 for _, confidence in predictions)
            model_detail = f"{model.backend} · celdas clasificadas={len(predictions)} · etiquetas={[label for label, _ in predictions]}"
        except (OSError, ValueError, EOFError, TypeError, AttributeError, RuntimeError) as exc:
            model_detail = str(exc)
        report("Modelo e inferencia", model_ok, model_detail, f"{sys.executable} train_vision.py")
        try:
            integrity_detail = _algorithm_integrity_check()
            report("Integridad de algoritmos", True, integrity_detail)
        except (AssertionError, KeyError, ValueError, RuntimeError) as exc:
            report("Integridad de algoritmos", False, str(exc), "compara search.py con reference/reference_algorithms.py.")
        for path in (LOG_DIR, RESULTS_DIR, MODEL_DIR):
            try:
                probe = path / ".selfcheck-write-test"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
                report(f"Escritura {path.name}", True, str(path))
            except OSError as exc:
                report(f"Escritura {path.name}", False, str(exc), "corrige permisos de escritura del proyecto.")
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
            with urlopen("https://pypi.org", context=context, timeout=4) as response:
                report("Internet/SSL", response.status == 200, f"HTTPS verificado · HTTP {response.status}")
        except (OSError, ssl.SSLError, TimeoutError) as exc:
            print(f"AVISO · Internet/SSL: {exc} · no fatal; continúa con activos locales.")

        self.screen.fill((28, 45, 54))
        self.screen.blit(self._sprite("logo.png", (600, 118)), (290, 90))
        self._text("DRONE WARS", (650, 149), font=self.large_font, center=True)
        self._text("AUTODIAGNÓSTICO DEL JUEGO", (590, 238), font=self.large_font, center=True)
        self._text(f"{len(failures)} fallo(s) · ventana real · {pygame.display.get_driver()}", (590, 310), center=True)
        self._text("Captura creada por pygame.display y el render real.", (590, 370), color=(203, 222, 210), center=True)
        pygame.display.flip()
        self._save_screenshot("selfcheck.png")
        if failures:
            print("FALLOS: " + ", ".join(failures))
        else:
            print("TODO OK")
        if pause and sys.stdin.isatty():
            try:
                input("Presiona ENTER para cerrar el autodiagnóstico...")
            except EOFError:
                print("No hay entrada de teclado; se cierra el autodiagnóstico.")
        return failures

    def run_stress(self, seeds_per_mode: int = 200) -> None:
        class EmptyVision:
            def classify(self, image: np.ndarray) -> Tuple[str, float]:
                return "empty", 1.0

        class ClosedWorld:
            goal_exists = False

            @staticmethod
            def tanks_alive() -> List[Entity]:
                return []

        self.vision = EmptyVision()
        completed = {mode: 0 for mode in range(3)}
        failures: List[str] = []
        for mode_index in range(3):
            for seed_offset in range(seeds_per_mode):
                seed = 20261106 + mode_index * 100_000 + seed_offset
                try:
                    self.start_match(mode_index, seed=seed)
                    ticks = 0
                    while self.state == "match" and ticks < 30001:
                        self.advance_match()
                        ticks += 1
                    if self.state != "results":
                        raise RuntimeError(
                            f"Modo {mode_index + 1}, semilla {seed}: "
                            f"no llegó a resultados ({self.state}, {ticks} ticks)."
                        )
                    self._draw_results()
                    completed[mode_index] += 1
                except Exception as exc:
                    details = f"{type(exc).__name__}: {exc}"
                    failures.append(f"modo {mode_index + 1}, semilla {seed}: {details}")
                    print(f"STRESS_FALLO: {failures[-1]}")
                    traceback.print_exc()

        edge_cases = 0
        try:
            import config
            import world as world_module

            original_config_size = config.GRID_SIZE
            original_config_buildings = config.BUILDING_COUNT
            original_world_size = world_module.GRID_SIZE
            original_world_buildings = world_module.BUILDING_COUNT
            try:
                config.GRID_SIZE = 9
                config.BUILDING_COUNT = 40
                world_module.GRID_SIZE = 9
                world_module.BUILDING_COUNT = 40
                dense_world = World(20261107, include_stations=True)
                if dense_world.size != 9 or sum(sum(row) for row in dense_world.grid) != 40:
                    raise AssertionError("El mapa mínimo denso no respetó la configuración permitida.")
                edge_cases += 1
                self.start_match(2, seed=20261107)
                dense_ticks = 0
                while self.state == "match" and dense_ticks < 30001:
                    self.advance_match()
                    dense_ticks += 1
                if self.state != "results":
                    raise RuntimeError(
                        f"El mundo mínimo denso no llegó a resultados ({self.state}, {dense_ticks} ticks)."
                    )
                self._draw_results()
                edge_cases += 1
            finally:
                config.GRID_SIZE = original_config_size
                config.BUILDING_COUNT = original_config_buildings
                world_module.GRID_SIZE = original_world_size
                world_module.BUILDING_COUNT = original_world_buildings

            rng = random.Random(20261109)
            for _ in range(60):
                size = rng.randint(2, 9)
                density = rng.choice((0.0, 0.45, 0.7, 0.85))
                grid = [
                    [int(rng.random() < density) for _ in range(size)]
                    for _ in range(size)
                ]
                grid[rng.randrange(size)][rng.randrange(size)] = 0
                adjacency = arbol_desde_grid(grid)
                start = rng.choice(tuple(adjacency))
                goal = start if rng.randrange(3) == 0 else rng.choice(tuple(adjacency))
                for algorithm, oracle in (("bfs", oracle_bfs), ("dfs", oracle_dfs)):
                    expected, found = oracle(start, goal, adjacency)
                    actual, actual_found = _run_search(start, goal, adjacency, algorithm)
                    if (actual, actual_found) != (expected, found):
                        raise AssertionError(f"Falló la cuadrícula aleatoria {algorithm.upper()}.")
                    edge_cases += 1

            isolated = {(1, 1): []}
            for algorithm, oracle in (("bfs", oracle_bfs), ("dfs", oracle_dfs)):
                for goal in ((1, 1), (8, 8)):
                    expected, found = oracle((1, 1), goal, isolated)
                    actual, actual_found = _run_search(
                        (1, 1), goal, isolated, algorithm
                    )
                    if (actual, actual_found) != (expected, found):
                        raise AssertionError("Falló un caso de meta igual al inicio o inalcanzable.")
                    edge_cases += 1

            enclosed_drone = Drone(
                "Dron encerrado",
                (0, 0, 0),
                (1, 1),
                "bfs",
                None,
                isolated,
                vision=None,
            )
            enclosed_drone.advance(ClosedWorld())
            enclosed_drone.advance(ClosedWorld())
            if not enclosed_drone.done_searching:
                raise AssertionError("El dron encerrado no terminó su búsqueda.")
            edge_cases += 1

            missing_sprite = "icon_error.png"
            self.sprites.pop(missing_sprite, None)
            self._sprite(missing_sprite)
            if missing_sprite not in self.sprites:
                raise AssertionError("El reemplazo de sprite ausente no se registró.")
            edge_cases += 1

            if self.world is None:
                raise AssertionError("No hay un mundo para probar un sprite de entidad ausente.")
            tank = next(entity for entity in self.world.entities if entity.kind == "tank")
            nearby = next(
                position
                for position in self.world.adjacency
                if abs(position[0] - tank.position[0]) + abs(position[1] - tank.position[1]) == 1
            )
            self.sprites.pop(tank.sprite_name, None)
            probe = Drone(
                "Dron de prueba de sprite",
                (0, 0, 0),
                nearby,
                "bfs",
                None,
                self.world.adjacency,
                EmptyVision(),
            )
            probe.observe_world(self.world, self.sprites, CELL_SIZE)
            if tank.sprite_name not in self.sprites:
                raise AssertionError("El sprite de entidad ausente no se regeneró.")
            edge_cases += 1

            self.vision = None
            self.start_match(0, seed=20261108)
            if self.state != "error":
                raise AssertionError("La falta del modelo no produjo un estado de error controlado.")
            self._draw_error()
            edge_cases += 1
        except Exception as exc:
            failures.append(f"casos límite: {type(exc).__name__}: {exc}")
            print(f"STRESS_FALLO: {failures[-1]}")
            traceback.print_exc()

        total_seeds = sum(completed.values())
        print(
            "STRESS_SUMMARY: "
            f"semillas={total_seeds}/{seeds_per_mode * 3} "
            f"(modo1={completed[0]}, modo2={completed[1]}, modo3={completed[2]}); "
            f"casos_límite={edge_cases}; excepciones_no_controladas={len(failures)}; "
            "fallos_conocidos_corregidos=1"
        )
        if failures:
            raise RuntimeError(f"El stress test encontró {len(failures)} fallo(s).")

    def _handle_click(self, position: Tuple[int, int]) -> None:
        for button in self.buttons:
            if not button.rect.collidepoint(position):
                continue
            if button.action == "quit":
                self.running = False
            elif button.action == "cancel":
                self.cancel_preparation.set()
                self.loading_done = True
                fallback = VisionPipeline(None, FALLBACK_MODEL_FILE)
                self.vision = fallback if fallback.backend != "unavailable" else None
                self.model_info = {"pipeline": fallback.backend, "status": "fallback"}
                self.model_metrics = {}
                self.state = "model_summary"
            elif button.action == "menu":
                self.state = "menu"
            elif button.action == "play":
                self.state = "modes"
            elif button.action == "modes":
                self.state = "modes"
            elif button.action == "info":
                self.state = "info"
            elif button.action == "retrain":
                self.cancel_preparation.clear()
                self.loading_done = False
                self.force_retrain = True
                self.state = "loading"
                self.prepare_worker()
            elif button.action.startswith("mode_"):
                mode = int(button.action.split("_")[1])
                if mode == 1:
                    self.state = "drone_select"
                else:
                    self.start_match(mode)
            elif button.action == "first_blue":
                self.mode2_first = "blue"
                self.state = "drone_select"
            elif button.action == "first_red":
                self.mode2_first = "red"
                self.state = "drone_select"
            elif button.action == "start_mode_2":
                self.start_match(1)
            elif button.action == "retry":
                self.error_text = ""
                self.cancel_preparation.clear()
                self.loading_done = False
                self.state = "loading"
                self.prepare_worker()
            break

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and self.state == "menu":
            self.running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._handle_click(event.pos)

    def _activate_button(self, action: str) -> None:
        button = next(button for button in self.buttons if button.action == action)
        event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": button.rect.center, "button": 1})
        self.handle_event(event)

    def _write_error(self, exc: BaseException) -> None:
        self.error_text = f"{type(exc).__name__}: {exc}"
        self.error_traceback = traceback.format_exc()
        self.error_type = type(exc).__name__
        self.error_message = str(exc)
        frames = traceback.extract_tb(exc.__traceback__)
        if frames:
            frame = frames[-1]
            self.error_frame = f"{Path(frame.filename).name}:{frame.lineno} ({frame.name})"
        else:
            self.error_frame = "Ubicación no disponible"
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with (LOG_DIR / "error.log").open("a", encoding="utf-8") as output:
                output.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n{self.error_traceback}\n")
            self.log_warning = ""
        except OSError as log_error:
            self.log_warning = f"No se pudo escribir logs/error.log: {log_error}"
            print(self.log_warning)
        print(f"ERROR: Ocurrió un error: {self.error_text}")
        print(self.error_traceback, end="")
        print(f"Consulta el detalle en {LOG_DIR / 'error.log'}")
        self.state = "error"

    def show_fatal_error(self, exc: Exception) -> bool:
        self._write_error(exc)
        try:
            if self.screen is None:
                self._init_display()
            self._ensure_error_sprites()
        except (pygame.error, OSError) as display_error:
            print(f"No se pudo abrir la pantalla de error: {display_error}")
            return False
        if self.screen is None:
            return False
        self.running = True
        while self.running:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self._handle_click(event.pos)
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and self.state == "menu":
                        self.running = False
                self._read_loading_messages()
                self._draw_state()
                self.clock.tick(20)
            except Exception as display_error:
                print(f"No se pudo dibujar la pantalla de recuperación: {display_error}")
                return False
        pygame.quit()
        return True

    def run(self, smoke: bool = False, capture: bool = False, stress: bool = False) -> None:
        try:
            self._make_dirs()
            self._init_display()
            self._ensure_sprites()
        except Exception as exc:
            if smoke or capture:
                raise
            self._write_error(exc)
            if self.screen is None:
                raise
        if smoke:
            try:
                self.vision = VisionPipeline(MODEL_FILE, FALLBACK_MODEL_FILE)
                self.run_smoke()
            finally:
                pygame.quit()
            return
        if capture:
            try:
                self.vision = VisionPipeline(MODEL_FILE, FALLBACK_MODEL_FILE)
                self.model_info = json.loads(MODEL_INFO_FILE.read_text(encoding="utf-8")) if MODEL_INFO_FILE.exists() else {}
                self.model_metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8")) if METRICS_FILE.exists() else {}
                self.capture_evidence()
            finally:
                pygame.quit()
            return
        if stress:
            previous_results_dir = self.results_dir
            try:
                with tempfile.TemporaryDirectory(prefix="drone-game-stress-") as directory:
                    self.results_dir = Path(directory)
                    self.run_stress()
            finally:
                self.results_dir = previous_results_dir
                pygame.quit()
            return

        self.prepare_worker()
        while self.running:
            try:
                for event in pygame.event.get():
                    self.handle_event(event)
                self._read_loading_messages()
                self._draw_state()
                self.clock.tick(20)
            except Exception as exc:
                self._write_error(exc)
        pygame.quit()


def _selfcheck(debug: bool = False) -> int:
    app = GameApp(debug)
    app._make_dirs()
    app._init_display()
    app._ensure_sprites()
    failures = app.run_selfcheck()
    pygame.quit()
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="DRONE WARS · CIUDAD DE JUGUETE")
    parser.add_argument("--debug", action="store_true", help="Muestra detalles de diagnóstico.")
    parser.add_argument("--smoke", action="store_true", help="Completa y dibuja una partida real de cada modo.")
    parser.add_argument("--selfcheck", action="store_true", help="Diagnostica dependencias, activos, modelo y SDL.")
    parser.add_argument("--capture-evidence", action="store_true", help="Guarda capturas reales de los estados del juego.")
    parser.add_argument("--stress", action="store_true", help="Ejecuta 200 semillas por modo y casos límite.")
    args = parser.parse_args()
    try:
        print(f"Inicio del juego · Python {platform.python_version()} · {platform.machine()} · {BASE_DIR}")
        if args.selfcheck:
            return _selfcheck(args.debug)
        app = GameApp(args.debug)
        app.run(smoke=args.smoke, capture=args.capture_evidence, stress=args.stress)
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        if not (args.selfcheck or args.smoke or args.capture_evidence or args.stress) and "app" in locals():
            if app.show_fatal_error(exc):
                return 1
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with (LOG_DIR / "error.log").open("a", encoding="utf-8") as output:
                output.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n{trace}\n")
        except OSError as log_exc:
            print(f"No se pudo escribir el log: {log_exc}")
        print(f"Ocurrió un error: {type(exc).__name__}: {exc}")
        print(trace)
        print(f"Consulta el log en: {LOG_DIR / 'error.log'}")
        if sys.stdin.isatty():
            try:
                input("Presiona ENTER para salir...")
            except EOFError:
                print("No hay entrada de teclado; se cierra el programa.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
