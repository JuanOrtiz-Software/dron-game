from __future__ import annotations

import inspect
import ast
import os
from pathlib import Path

import numpy as np
import pygame
import pytest
import cv2

import config
from art import make_sprite, perturb_cell_image, render_cell_image
from art import sprite_catalog
from drone import Drone
from entities import Entity
from search import LiveSearch, transit_path
from stats import drone_statistics, save_match_result
from vision import VisionPipeline
from world import World
from assets_manager import process_tank_image
from download_assets import _license_key


@pytest.fixture(scope="module", autouse=True)
def pygame_headless():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    yield
    pygame.quit()


def test_world_minimums_connectivity_unique_positions_and_seed_determinism():
    first = World(52, include_stations=True)
    second = World(52, include_stations=True)

    kinds = [entity.kind for entity in first.entities]
    occupied = [first.red_start, first.blue_start] + [entity.position for entity in first.entities]
    component = first._flood_fill(first.red_start)
    assert kinds.count("tank") >= 20
    assert sum(kind in {"car", "pickup", "truck", "helicopter"} for kind in kinds) >= 10
    assert kinds.count("charging_station") == 3
    assert sum(cell for row in first.grid for cell in row) >= 10
    assert len(occupied) == len(set(occupied))
    assert set(occupied) <= component
    assert first.goal is None
    assert not first.goal_exists
    assert [
        (entity.kind, entity.position, entity.variant, entity.orientation)
        for entity in first.entities
    ] == [
        (entity.kind, entity.position, entity.variant, entity.orientation)
        for entity in second.entities
    ]
    assert first.grid == second.grid
    assert first.red_start == second.red_start
    assert first.blue_start == second.blue_start
    assert all(0 <= entity.orientation < 360 for entity in first.entities)


def test_goal_is_created_after_tanks_clear_and_never_overlaps_entities():
    world = World(8)
    assert world.goal is None
    assert not world.goal_exists
    for tank in world.tanks_alive():
        assert tank.destroy("Prueba", 1)
    world.activate_goal()
    assert world.goal_exists
    assert world.goal is not None
    assert world.goal not in {world.red_start, world.blue_start}
    assert world.goal not in {entity.position for entity in world.entities}
    assert world.goal in world._flood_fill(world.red_start)


def test_world_snapshot_restore_reverts_lifecycle_and_goal():
    world = World(119, include_stations=True)
    snapshot = world.snapshot()
    original = [
        (entity.kind, entity.position, entity.variant, entity.orientation, entity.alive)
        for entity in world.entities
    ]
    world.entities[0].destroy("Dron 1", 12)
    world.activate_goal()
    world.restore(snapshot)
    assert not world.goal_exists
    assert world.goal is None
    assert [
        (entity.kind, entity.position, entity.variant, entity.orientation, entity.alive)
        for entity in world.entities
    ] == original


def test_bfs_and_dfs_order_and_visit_timing():
    adjacency = {
        (0, 0): [(1, 0), (0, 1)],
        (1, 0): [(0, 0), (1, 1)],
        (0, 1): [(0, 0), (1, 1)],
        (1, 1): [(1, 0), (0, 1)],
    }
    bfs = LiveSearch((0, 0), None, adjacency, "bfs")
    assert bfs.explored == {(0, 0)}
    assert bfs.step() == (0, 0)
    assert bfs.explored == {(0, 0), (1, 0), (0, 1)}

    dfs = LiveSearch((0, 0), None, adjacency, "dfs")
    assert not dfs.explored
    assert dfs.step() == (0, 0)
    assert dfs.explored == {(0, 0)}
    assert dfs.step() == (0, 1)
    assert dfs.explored == {(0, 0), (0, 1)}


def test_resource_search_parent_paths_follow_the_original_bfs_and_dfs_order():
    start, branch, shortcut, deep, target = (
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (2, 1),
    )
    adjacency = {
        start: [branch, shortcut],
        branch: [start, deep],
        shortcut: [start, target],
        deep: [branch, target],
        target: [shortcut, deep],
    }
    paths = {}
    for algorithm in ("bfs", "dfs"):
        search = LiveSearch(start, target, adjacency, algorithm)
        while not search.finished:
            search.step()
        paths[algorithm] = search.path_to(target)
    assert len(paths["bfs"]) - 1 == 2
    assert paths["dfs"] == [start, shortcut, target]


def test_shortest_path_and_physical_steps_are_adjacent():
    world = World(41)
    drone = Drone("Prueba", (1, 2, 3), world.red_start, "dfs", None, world.adjacency)
    previous = drone.current
    for _ in range(25):
        drone.advance(world)
        current = drone.current
        assert abs(current[0] - previous[0]) + abs(current[1] - previous[1]) <= 1
        previous = current

    walkable = {(0, 0), (1, 0), (1, 1), (2, 1)}
    route = transit_path((0, 0), (2, 1), walkable)
    assert route[0] == (0, 0)
    assert route[-1] == (2, 1)
    assert all(
        abs(left[0] - right[0]) + abs(left[1] - right[1]) == 1
        for left, right in zip(route, route[1:])
    )


def test_destroyed_entity_keeps_destroyed_sprite_and_cannot_be_destroyed_twice():
    entity = Entity(7, "tank", (3, 5), variant=2, orientation=124.0)
    assert entity.sprite_name == "tank_02_normal.png"
    assert entity.destroy("Dron 1", 9)
    assert entity.sprite_name == "tank_02_destroyed.png"
    assert entity.destroyed_by == "Dron 1"
    assert entity.destroyed_tick == 9
    assert not entity.destroy("Dron 2", 10)
    assert entity.destroyed_by == "Dron 1"


def test_tanks_and_all_vehicle_kinds_have_normal_and_destroyed_sprites():
    assets = Path(__file__).resolve().parents[1] / "assets"
    for kind in ("tank", "car", "pickup", "truck", "helicopter"):
        for variant in range(1, 5 if kind == "tank" else 4):
            for state in ("normal", "destroyed"):
                name = f"{kind}_{variant:02}_{state}.png"
                assert name in sprite_catalog()
                assert (assets / name).is_file()


def test_cell_image_is_composed_from_rendered_sprite_and_classifier_accepts_only_pixels():
    street = make_sprite("ground_street.png")
    tank = make_sprite("tank_01_normal.png")
    road_image = render_cell_image(street, None, 0, 32)
    tank_image = render_cell_image(street, tank, 73, 32)
    assert road_image.shape == (32, 32, 3)
    assert tank_image.shape == (32, 32, 3)
    assert not np.array_equal(road_image, tank_image)
    first_sweep = perturb_cell_image(tank_image, 99, 3, 6)
    assert np.array_equal(first_sweep, perturb_cell_image(tank_image, 99, 3, 6))
    assert not np.array_equal(first_sweep, perturb_cell_image(tank_image, 100, 3, 6))
    assert list(inspect.signature(VisionPipeline.classify).parameters) == ["self", "image"]
    classifier = VisionPipeline()
    with pytest.raises(RuntimeError, match="No hay modelo visual cargado"):
        classifier.classify(tank_image)
    with pytest.raises(TypeError):
        classifier.classify("tank")  # type: ignore[arg-type]


def test_saved_cnn_is_the_primary_cpu_backend_and_classifies_rendered_cells():
    project = Path(__file__).resolve().parents[1]
    classifier = VisionPipeline(project / "models" / "vision.pt", project / "models" / "vision_sklearn.joblib")
    assert classifier.backend == "CNN (PyTorch, CPU)"
    image = render_cell_image(
        make_sprite("ground_street.png"),
        make_sprite("tank_01_normal.png"),
        73,
        64,
    )
    label, confidence = classifier.classify(image)
    assert label in classifier.classes
    assert 0 <= confidence <= 1


def test_drone_passes_the_rendered_perturbed_cell_array_to_vision_only():
    world = World(777)
    tank = next(entity for entity in world.entities if entity.kind == "tank")
    current = next(
        position
        for position in world.adjacency
        if abs(position[0] - tank.position[0]) + abs(position[1] - tank.position[1]) == 1
    )
    tank_sprite = make_sprite(tank.sprite_name)
    sprites = {
        "ground_street.png": make_sprite("ground_street.png"),
    }
    sprites.update({entity.sprite_name: make_sprite(entity.sprite_name) for entity in world.entities})
    tank_sprite = sprites[tank.sprite_name]

    class ImageRecorder:
        def __init__(self):
            self.received = []

        def classify(self, image):
            self.received.append(image)
            return "empty", 1.0

    vision = ImageRecorder()
    drone = Drone("Prueba", (1, 2, 3), current, "bfs", None, world.adjacency, vision, seed=13)
    drone.observe_world(world, sprites, 28)
    perturb_seed = 13 + 1_000_003 + tank.position[0] * world.size + tank.position[1]
    expected = perturb_cell_image(
        render_cell_image(sprites["ground_street.png"], tank_sprite, tank.orientation, 28),
        perturb_seed,
        config.VISION_NOISE_STD,
        config.VISION_BRIGHTNESS_RANGE,
    )
    assert any(np.array_equal(received, expected) for received in vision.received)
    assert all(isinstance(image, np.ndarray) for image in vision.received)


def test_background_worker_does_not_call_pygame_display_apis():
    source = Path(__file__).resolve().parents[1] / "main.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nested_functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "worker"
    }
    assert set(nested_functions) == {"worker"}
    for function in nested_functions.values():
        for node in ast.walk(function):
            if isinstance(node, ast.Attribute) and node.attr == "display":
                assert not (
                    isinstance(node.value, ast.Name) and node.value.id == "pygame"
                )


def test_tank_photo_processor_writes_rgba_normal_and_destroyed_images(tmp_path):
    source = tmp_path / "toy.jpg"
    normal = tmp_path / "processed" / "tank_normal.png"
    destroyed = tmp_path / "processed" / "tank_destroyed.png"
    photo = np.full((100, 140, 3), 236, dtype=np.uint8)
    photo[25:76, 35:112] = (60, 110, 75)
    cv2.imwrite(str(source), cv2.cvtColor(photo, cv2.COLOR_RGB2BGR))
    process_tank_image(source, normal, destroyed)
    normal_image = pygame.image.load(str(normal))
    destroyed_image = pygame.image.load(str(destroyed))
    assert normal_image.get_size() == (96, 96)
    assert destroyed_image.get_size() == (96, 96)
    assert normal_image.get_masks()[3] != 0
    assert destroyed_image.get_masks()[3] != 0


@pytest.mark.parametrize(
    ("license_name", "permitted"),
    [
        ("CC0", True),
        ("Public domain", True),
        ("CC BY 4.0", True),
        ("CC BY-SA 4.0", False),
        ("CC BY-NC 4.0", False),
        ("CC BY-ND 4.0", False),
    ],
)
def test_commons_license_filter_accepts_only_allowed_license_families(license_name, permitted):
    from download_assets import ALLOWED_LICENSES

    assert (_license_key(license_name) in ALLOWED_LICENSES) is permitted


def test_simulated_stopwatch_uses_ticks():
    world = World(7)
    drone = Drone("Cronómetro", (1, 2, 3), world.red_start, "bfs", None, world.adjacency)
    for _ in range(4):
        drone.advance(world)
    stats = drone_statistics(drone)
    assert stats["elapsed_ticks"] == 4
    assert stats["elapsed_time_s"] == pytest.approx(4 * config.TICK_SECONDS)


@pytest.mark.parametrize("algorithm", ["bfs", "dfs"])
def test_mode_three_recharges_refills_ammo_and_crashes_at_zero(algorithm):
    world = World(902, include_stations=True)
    drone = Drone("Carga", (1, 2, 3), world.red_start, algorithm, None, world.adjacency, vision_radius=1)
    suspended_search = drone.search
    drone.battery = 141
    drone.ammo = 1
    suspended_position = None
    for _ in range(1200):
        drone.advance(world, resource_mode=True)
        if suspended_position is None and drone.suspended_position is not None:
            suspended_position = drone.suspended_position
        if drone.recharges and drone.resource_phase is None:
            break
    assert drone.recharges == 1
    assert drone.battery > 0
    assert drone.ammo == config.AMMO_MAX
    assert drone.search is suspended_search
    assert drone.current == suspended_position

    no_station_world = World(902, include_stations=False)
    crashing = Drone("Choque", (1, 2, 3), no_station_world.red_start, "bfs", None, no_station_world.adjacency)
    crashing.battery = 1
    for _ in range(5):
        crashing.advance(no_station_world, resource_mode=True)
        if crashing.crashed:
            break
    assert crashing.crashed
    assert crashing.stopped_at_ticks is not None


def test_config_rejects_below_minimum_counts(monkeypatch):
    monkeypatch.setattr(config, "TANK_COUNT", 19)
    with pytest.raises(ValueError, match="TANK_COUNT"):
        config.validate_config()


def test_saved_match_result_contains_exact_statistics_and_ticks(tmp_path: Path):
    world = World(31)
    drone = Drone("Datos", (1, 2, 3), world.red_start, "bfs", None, world.adjacency)
    for _ in range(3):
        drone.advance(world)
    saved = save_match_result(
        tmp_path,
        "BUSCAR Y DESTRUIR",
        31,
        {"grid_size": config.GRID_SIZE},
        {"dron_1_bfs": drone_statistics(drone)},
        "Dron 1",
    )
    result = __import__("json").loads(saved.read_text(encoding="utf-8"))
    stats = result["drones"]["dron_1_bfs"]
    assert stats["score"] == stats["tanks_destroyed"] - stats["vehicles_destroyed_wrongly"]
    assert stats["elapsed_ticks"] == 3
    assert stats["elapsed_time_s"] == pytest.approx(3 * config.TICK_SECONDS)
    assert result["seed"] == 31
