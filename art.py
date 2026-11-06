from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pygame

Size = Tuple[int, int]


def sprite_catalog() -> Dict[str, str]:
    catalog = {
        "background_city.png": "Fondo ilustrado de ciudad para menús",
        "logo.png": "Logotipo del juego",
        "ground_street.png": "Textura de calle con marcas viales",
        "ground_grass.png": "Textura de césped",
        "fog.png": "Niebla de guerra texturada",
        "goal_target.png": "Meta con anillos y bandera",
        "charging_station.png": "Estación de carga",
        "charging_0.png": "Animación de carga, fotograma 1",
        "charging_1.png": "Animación de carga, fotograma 2",
        "laser_beam.png": "Rayo láser",
        "smoke_00.png": "Humo de objetivo destruido, fotograma 1",
        "smoke_01.png": "Humo de objetivo destruido, fotograma 2",
        "crash_marker.png": "Indicador de dron accidentado",
        "hud_panel.png": "Marco del HUD",
        "button.png": "Botón con marco e iluminación",
        "map_frame.png": "Marco texturado del mapa",
        "bar_track.png": "Pista de barra de progreso",
        "bar_progress.png": "Relleno de barra de carga",
        "bar_battery.png": "Relleno de barra de batería",
        "bar_ammo.png": "Relleno de barra de munición",
        "bar_score_red.png": "Barra de puntuación BFS",
        "bar_score_blue.png": "Barra de puntuación DFS",
        "medal.png": "Medalla de resultados",
        "drone_red_0.png": "Dron rojo, rotores fotograma 1",
        "drone_red_1.png": "Dron rojo, rotores fotograma 2",
        "drone_blue_0.png": "Dron azul, rotores fotograma 1",
        "drone_blue_1.png": "Dron azul, rotores fotograma 2",
        "mode_01.png": "Icono del modo 1",
        "mode_02.png": "Icono del modo 2",
        "mode_03.png": "Icono del modo 3",
    }
    for index in range(1, 7):
        catalog[f"building_{index:02}.png"] = f"Edificio urbano, variante {index}"
    for index in range(1, 5):
        catalog[f"tank_{index:02}_normal.png"] = f"Tanque normal, variante {index}"
        catalog[f"tank_{index:02}_destroyed.png"] = f"Tanque destruido, variante {index}"
    vehicle_names = {"car": "Auto", "pickup": "Camioneta", "truck": "Camión", "helicopter": "Helicóptero"}
    for kind, label in vehicle_names.items():
        for index in range(1, 4):
            catalog[f"{kind}_{index:02}_normal.png"] = f"{label} normal, variante {index}"
            catalog[f"{kind}_{index:02}_destroyed.png"] = f"{label} destruido, variante {index}"
    for index in range(8):
        catalog[f"explosion_{index:02}.png"] = f"Explosión animada, fotograma {index + 1}"
    for name, description in {
        "icon_battery.png": "Icono de batería",
        "icon_ammo.png": "Icono de munición",
        "icon_nodes.png": "Icono de nodos",
        "icon_steps.png": "Icono de pasos",
        "icon_timer.png": "Icono de cronómetro",
        "icon_score.png": "Icono de puntuación",
        "icon_tank.png": "Icono de tanque",
        "icon_vehicle.png": "Icono de vehículo",
        "icon_error.png": "Icono de error",
        "icon_cancel.png": "Icono de cancelar",
        "icon_drone.png": "Icono de dron",
    }.items():
        catalog[name] = description
    return catalog


def sprite_size(name: str) -> Size:
    if name == "background_city.png":
        return (1180, 800)
    if name == "logo.png":
        return (760, 150)
    if name == "hud_panel.png":
        return (400, 150)
    if name == "button.png":
        return (320, 72)
    return (96, 96)


def render_cell_image(
    street_texture: pygame.Surface,
    entity_sprite: pygame.Surface | None,
    orientation: float,
    size: int,
) -> np.ndarray:
    """Renderiza una celda como la ve el mapa y devuelve sus píxeles RGB."""
    cell = pygame.transform.smoothscale(street_texture, (size, size)).copy()
    if entity_sprite is not None:
        sprite = pygame.transform.smoothscale(entity_sprite, (size, size))
        sprite = pygame.transform.rotate(sprite, -orientation)
        cell.blit(sprite, sprite.get_rect(center=(size // 2, size // 2)))
    return pygame.surfarray.array3d(cell).transpose(1, 0, 2).copy()


def perturb_cell_image(
    image: np.ndarray,
    seed: int,
    noise_std: float,
    brightness_range: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    brightness = rng.uniform(-brightness_range, brightness_range)
    noise = rng.normal(0, noise_std, image.shape)
    return np.clip(image.astype(np.float32) + brightness + noise, 0, 255).astype(np.uint8)


def _rounded(surface: pygame.Surface, color: Tuple[int, int, int], rect: pygame.Rect, radius: int = 12) -> None:
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    pygame.draw.rect(surface, tuple(max(0, c - 38) for c in color), rect, width=3, border_radius=radius)
    pygame.draw.line(surface, tuple(min(255, c + 35) for c in color), (rect.left + radius, rect.top + 4), (rect.right - radius, rect.top + 4), 2)


def make_sprite(name: str, size: Size = (96, 96)) -> pygame.Surface:
    width, height = size
    surface = pygame.Surface(size, pygame.SRCALPHA)
    surface.fill((0, 0, 0, 0))
    center = (width // 2, height // 2)
    basename = Path(name).stem
    if name == "background_city.png":
        surface.fill((29, 56, 76))
        for y in range(260, height, 170):
            pygame.draw.rect(surface, (63, 92, 102), (0, y, width, 105))
            for x in range(20, width, 92):
                building_height = 75 + ((x * 7 + y) % 70)
                rect = pygame.Rect(x, y - building_height, 68, building_height)
                _rounded(surface, (80 + (x % 38), 104 + (y % 30), 112), rect, 8)
                for wx in range(rect.x + 12, rect.right - 8, 19):
                    for wy in range(rect.y + 13, rect.bottom - 8, 22):
                        pygame.draw.rect(surface, (244, 202, 112), (wx, wy, 8, 11), border_radius=2)
        pygame.draw.rect(surface, (68, 80, 84), (0, 560, width, 240))
        pygame.draw.line(surface, (236, 214, 143), (0, 680), (width, 680), 6)
        return surface
    if name == "logo.png":
        _rounded(surface, (26, 44, 58), pygame.Rect(2, 2, width - 4, height - 4), 28)
        pygame.draw.circle(surface, (239, 183, 68), (84, height // 2), 49)
        pygame.draw.circle(surface, (255, 227, 151), (84, height // 2), 35, 4)
        pygame.draw.line(surface, (235, 243, 244), (57, 75), (111, 75), 6)
        pygame.draw.line(surface, (235, 243, 244), (84, 48), (84, 102), 6)
        return surface
    if name in {"ground_street.png", "ground_grass.png"}:
        if "grass" in name:
            surface.fill((81, 137, 79))
            for x in range(7, width, 17):
                for y in range(8, height, 19):
                    pygame.draw.circle(surface, (104, 158, 85), (x + (y % 7), y), 3)
            return surface
        surface.fill((72, 83, 91))
        pygame.draw.rect(surface, (139, 143, 133), (0, 0, width, 8))
        pygame.draw.rect(surface, (139, 143, 133), (0, height - 8, width, 8))
        pygame.draw.line(surface, (229, 208, 131), (width // 2, 0), (width // 2, 20), 4)
        pygame.draw.line(surface, (229, 208, 131), (width // 2, 76), (width // 2, height), 4)
        for x in range(5, width, 19):
            pygame.draw.line(surface, (87, 100, 105), (x, 13), (x + 5, 28), 1)
        return surface
    if name == "fog.png":
        surface.fill((35, 43, 53, 206))
        for x in range(0, width, 16):
            for y in range(0, height, 15):
                shade = 55 + ((x * 3 + y * 7) % 28)
                pygame.draw.circle(surface, (shade, shade + 5, shade + 10, 100), (x + 5, y + 6), 5)
        return surface
    if name == "hud_panel.png":
        _rounded(surface, (27, 44, 57), pygame.Rect(2, 2, width - 4, height - 4), 16)
        for y in range(20, height, 24):
            pygame.draw.line(surface, (51, 72, 82), (16, y), (width - 16, y), 1)
        return surface
    if name == "button.png":
        _rounded(surface, (50, 107, 129), pygame.Rect(2, 2, width - 4, height - 4), 18)
        pygame.draw.circle(surface, (240, 197, 103), (25, height // 2), 8)
        return surface
    if name == "map_frame.png":
        _rounded(surface, (32, 46, 52), pygame.Rect(2, 2, width - 4, height - 4), 14)
        for offset in range(12, width, 34):
            pygame.draw.line(surface, (48, 65, 68), (offset, 8), (offset + 18, 8), 2)
        return surface
    bar_colors = {
        "bar_track.png": (43, 56, 61),
        "bar_progress.png": (87, 190, 139),
        "bar_battery.png": (82, 188, 128),
        "bar_ammo.png": (225, 160, 76),
        "bar_score_red.png": (221, 88, 78),
        "bar_score_blue.png": (76, 137, 231),
    }
    if name in bar_colors:
        color = bar_colors[name]
        pygame.draw.rect(surface, (*color, 255), pygame.Rect(0, 0, width, height), border_radius=max(2, height // 3))
        for x in range(0, width, 12):
            pygame.draw.line(surface, tuple(min(255, channel + 25) for channel in color), (x, 3), (x + 8, max(3, height - 3)), 1)
        return surface
    if name == "goal_target.png":
        for radius, color in ((41, (245, 204, 73)), (32, (42, 87, 101)), (24, (251, 228, 145)), (14, (226, 95, 63))):
            pygame.draw.circle(surface, color, center, radius, 6 if radius != 14 else 0)
        pygame.draw.line(surface, (243, 245, 230), (67, 25), (67, 67), 4)
        pygame.draw.polygon(surface, (229, 77, 65), [(69, 27), (88, 34), (69, 41)])
        return surface
    if name.startswith("drone_"):
        is_red = "red" in name
        body = (221, 63, 57) if is_red else (66, 131, 242)
        accent = (255, 205, 179) if is_red else (205, 228, 255)
        phase = int(basename[-1])
        for dx, dy in ((-25, -25), (25, -25), (-25, 25), (25, 25)):
            pygame.draw.line(surface, (48, 56, 66), center, (center[0] + dx, center[1] + dy), 7)
            rotor_center = (center[0] + dx, center[1] + dy)
            pygame.draw.circle(surface, accent, rotor_center, 11, 3)
            angle = 0 if phase == 0 else 45
            start = (rotor_center[0] - 8, rotor_center[1] + angle // 5)
            end = (rotor_center[0] + 8, rotor_center[1] - angle // 5)
            pygame.draw.line(surface, (238, 242, 245), start, end, 3)
        _rounded(surface, body, pygame.Rect(32, 33, 32, 30), 10)
        pygame.draw.circle(surface, accent, center, 5)
        pygame.draw.circle(surface, (248, 243, 218), (center[0], center[1] - 5), 3)
        return surface
    if basename.startswith("building_"):
        idx = int(basename.split("_")[1])
        colors = ((194, 139, 98), (127, 154, 171), (208, 179, 110), (149, 120, 111), (118, 148, 129), (164, 139, 178))
        base = colors[(idx - 1) % len(colors)]
        pygame.draw.rect(surface, (37, 52, 59), (7, 9, 82, 82), border_radius=10)
        pygame.draw.rect(surface, base, (11, 6, 74, 80), border_radius=9)
        pygame.draw.rect(surface, tuple(min(255, c + 32) for c in base), (16, 11, 64, 12), border_radius=4)
        for wx in (21, 43, 65):
            for wy in (31, 51):
                pygame.draw.rect(surface, (62, 111, 133), (wx, wy, 12, 12), border_radius=3)
                pygame.draw.line(surface, (183, 219, 220), (wx + 2, wy + 3), (wx + 8, wy + 3), 2)
        pygame.draw.rect(surface, (84, 73, 62), (42, 68, 15, 18), border_radius=4)
        return surface
    if basename.startswith("tank_"):
        destroyed = basename.endswith("destroyed")
        color_index = int(basename.split("_")[1])
        palette = ((89, 128, 77), (105, 117, 92), (128, 113, 78), (72, 119, 105))
        color = (85, 66, 58) if destroyed else palette[(color_index - 1) % 4]
        for y in (22, 67):
            pygame.draw.rect(surface, (40, 45, 43), (12, y, 72, 9), border_radius=4)
            for x in range(19, 82, 13):
                pygame.draw.circle(surface, (115, 123, 108), (x, y + 4), 3)
        _rounded(surface, color, pygame.Rect(20, 25, 56, 48), 12)
        pygame.draw.ellipse(surface, tuple(min(255, c + 25) for c in color), (31, 29, 35, 32))
        pygame.draw.rect(surface, (55, 62, 56), (57, 43, 30, 8), border_radius=4)
        if destroyed:
            pygame.draw.ellipse(surface, (43, 43, 42), (23, 18, 19, 12))
            pygame.draw.circle(surface, (222, 113, 55), (64, 22), 6)
        return surface
    vehicle_parts = basename.split("_")
    if vehicle_parts[0] in {"car", "pickup", "truck", "helicopter"}:
        kind = vehicle_parts[0]
        variant = int(vehicle_parts[1])
        destroyed = vehicle_parts[2] == "destroyed"
        colors = ((54, 145, 183), (219, 139, 62), (141, 91, 156))
        body = (80, 70, 67) if destroyed else colors[(variant - 1) % 3]
        if kind == "helicopter":
            pygame.draw.line(surface, (43, 52, 57), (49, 18), (49, 78), 5)
            pygame.draw.line(surface, (43, 52, 57), (20, 48), (78, 48), 4)
            pygame.draw.ellipse(surface, body, (26, 34, 46, 28))
            pygame.draw.circle(surface, (103, 167, 187), (59, 48), 10)
            pygame.draw.line(surface, (39, 49, 54), (20, 48), (10, 40), 4)
        else:
            width_body = 42 if kind == "car" else 54 if kind == "pickup" else 66
            x = (96 - width_body) // 2
            pygame.draw.rect(surface, (42, 46, 48), (x - 3, 25, width_body + 6, 46), border_radius=12)
            pygame.draw.rect(surface, body, (x, 22, width_body, 50), border_radius=12)
            pygame.draw.rect(surface, (93, 158, 183), (x + 6, 30, width_body - 12, 15), border_radius=5)
            pygame.draw.rect(surface, (63, 105, 120), (x + 6, 49, width_body - 12, 11), border_radius=4)
            for wy in (31, 57):
                pygame.draw.ellipse(surface, (31, 35, 38), (x - 5, wy, 8, 10))
                pygame.draw.ellipse(surface, (31, 35, 38), (x + width_body - 3, wy, 8, 10))
            if kind in {"pickup", "truck"}:
                pygame.draw.line(surface, (240, 224, 172), (x + 7, 67), (x + width_body - 6, 67), 3)
        if destroyed:
            pygame.draw.circle(surface, (44, 43, 43), (30, 21), 9)
            pygame.draw.circle(surface, (218, 112, 52), (66, 18), 7)
        return surface
    if name.startswith("charging"):
        pygame.draw.circle(surface, (244, 202, 73), center, 37, 6)
        pygame.draw.circle(surface, (48, 119, 143), center, 27)
        pygame.draw.polygon(surface, (240, 247, 235), [(53, 20), (37, 53), (49, 53), (42, 76), (63, 43), (51, 43)])
        if name == "charging_1.png":
            pygame.draw.circle(surface, (255, 231, 138), center, 44, 2)
        return surface
    if name == "laser_beam.png":
        pygame.draw.line(surface, (255, 201, 88, 65), (8, 48), (88, 48), 18)
        pygame.draw.line(surface, (255, 245, 181), (8, 48), (88, 48), 6)
        pygame.draw.circle(surface, (255, 255, 235), (48, 48), 9)
        return surface
    if name.startswith("smoke_"):
        phase = int(basename[-2:])
        cloud_color = (96 + phase * 12, 104 + phase * 10, 105 + phase * 8, 170)
        for index, (x, y, radius) in enumerate(((28, 59, 18), (44, 43, 21), (63, 54, 18), (72, 39, 13))):
            pygame.draw.circle(surface, (*cloud_color[:3], max(35, cloud_color[3] - index * 24)), (x, y - phase * 5), radius + phase * 2)
        return surface
    if name == "crash_marker.png":
        _rounded(surface, (90, 42, 42), pygame.Rect(7, 7, 82, 82), 20)
        pygame.draw.polygon(surface, (248, 224, 159), [(48, 20), (76, 72), (20, 72)])
        pygame.draw.line(surface, (130, 48, 43), (48, 37), (48, 56), 7)
        pygame.draw.circle(surface, (130, 48, 43), (48, 64), 4)
        return surface
    if basename.startswith("explosion_"):
        frame = int(basename.split("_")[1])
        radius = max(8, 39 - frame * 3)
        pygame.draw.circle(surface, (117, 48, 42, 125), center, min(46, radius + 8))
        pygame.draw.circle(surface, (236, 78 + min(100, frame * 15), 43), center, radius)
        pygame.draw.circle(surface, (255, 218, 108), center, max(4, radius // 2))
        for index in range(7):
            angle = (frame * 13 + index * 51) % 360
            endpoint = (center[0] + int(radius * 1.3 * pygame.math.Vector2(1, 0).rotate(angle).x),
                        center[1] + int(radius * 1.3 * pygame.math.Vector2(1, 0).rotate(angle).y))
            pygame.draw.line(surface, (255, 223, 125), center, endpoint, 3)
        return surface
    if name.startswith("mode_"):
        _rounded(surface, (46, 79, 91), pygame.Rect(4, 4, 88, 88), 20)
        index = int(basename[-2:])
        if index == 1:
            for x, color in ((31, (230, 82, 70)), (65, (69, 137, 242))):
                pygame.draw.circle(surface, color, (x, 48), 16)
                pygame.draw.circle(surface, (245, 241, 220), (x, 48), 6)
        elif index == 2:
            pygame.draw.circle(surface, (229, 89, 71), (32, 48), 19)
            pygame.draw.circle(surface, (70, 137, 236), (64, 48), 19)
            pygame.draw.line(surface, (245, 238, 208), (37, 48), (59, 48), 4)
        else:
            pygame.draw.circle(surface, (236, 192, 69), center, 29, 5)
            pygame.draw.polygon(surface, (248, 246, 222), [(54, 20), (37, 52), (51, 52), (43, 75), (66, 42), (52, 42)])
        return surface
    if name == "medal.png":
        pygame.draw.polygon(surface, (66, 127, 201), [(28, 52), (43, 52), (37, 91), (22, 81)])
        pygame.draw.polygon(surface, (218, 72, 69), [(53, 52), (68, 52), (75, 82), (59, 90)])
        pygame.draw.circle(surface, (248, 198, 71), center, 30)
        pygame.draw.circle(surface, (255, 228, 135), center, 21, 4)
        pygame.draw.polygon(surface, (255, 248, 210), [(48, 28), (53, 42), (68, 42), (56, 51), (61, 66), (48, 57), (35, 66), (40, 51), (28, 42), (43, 42)])
        return surface
    if name.startswith("icon_"):
        color = (232, 198, 93)
        if "battery" in name:
            pygame.draw.rect(surface, color, (23, 30, 48, 36), border_radius=7)
            pygame.draw.rect(surface, color, (71, 40, 7, 16), border_radius=2)
            pygame.draw.rect(surface, (43, 133, 103), (29, 36, 27, 24), border_radius=4)
        elif "ammo" in name:
            pygame.draw.rect(surface, (220, 207, 165), (37, 24, 23, 48), border_radius=9)
            pygame.draw.circle(surface, (237, 129, 66), (48, 25), 11)
        elif "nodes" in name or "steps" in name:
            points = ((24, 67), (48, 27), (73, 66))
            pygame.draw.lines(surface, color, False, points, 5)
            for point in points:
                pygame.draw.circle(surface, (241, 237, 218), point, 8)
        elif "timer" in name:
            pygame.draw.circle(surface, color, center, 31, 6)
            pygame.draw.line(surface, (242, 240, 219), center, (48, 28), 4)
            pygame.draw.line(surface, (242, 240, 219), center, (67, 53), 4)
        elif "score" in name or "medal" in name:
            pygame.draw.circle(surface, color, center, 30)
            pygame.draw.polygon(surface, (255, 247, 216), [(48, 25), (54, 42), (71, 42), (57, 53), (62, 70), (48, 60), (34, 70), (39, 53), (25, 42), (42, 42)])
        elif "tank" in name:
            pygame.draw.rect(surface, (47, 54, 48), (19, 35, 57, 28), border_radius=8)
            pygame.draw.ellipse(surface, (114, 148, 81), (30, 32, 34, 30))
            pygame.draw.line(surface, (47, 54, 48), (59, 46), (82, 46), 5)
        elif "vehicle" in name:
            pygame.draw.rect(surface, (53, 127, 164), (23, 30, 50, 39), border_radius=12)
            pygame.draw.rect(surface, (169, 210, 218), (32, 35, 32, 13), border_radius=4)
            pygame.draw.circle(surface, (37, 40, 42), (33, 69), 6)
            pygame.draw.circle(surface, (37, 40, 42), (65, 69), 6)
        elif "error" in name:
            pygame.draw.polygon(surface, (231, 95, 70), [(48, 13), (88, 80), (8, 80)])
            pygame.draw.line(surface, (255, 246, 221), (48, 35), (48, 58), 6)
            pygame.draw.circle(surface, (255, 246, 221), (48, 69), 3)
        elif "cancel" in name:
            pygame.draw.circle(surface, (229, 85, 75), center, 33)
            pygame.draw.line(surface, (255, 244, 224), (34, 34), (62, 62), 7)
            pygame.draw.line(surface, (255, 244, 224), (62, 34), (34, 62), 7)
        else:
            pygame.draw.circle(surface, (70, 140, 226), center, 29)
            pygame.draw.circle(surface, (236, 241, 233), center, 12)
        return surface
    pygame.draw.circle(surface, (220, 80, 220), center, 39)
    pygame.draw.line(surface, (255, 255, 255), (14, 14), (82, 82), 5)
    return surface


def asset_names() -> Iterable[str]:
    return sprite_catalog().keys()
