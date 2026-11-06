from __future__ import annotations

import ast
import os
import random
from pathlib import Path

import pytest
import pygame

from drone import Drone
from reference.reference_algorithms import arbol_desde_grid, bfs as oracle_bfs, dfs as oracle_dfs
from search import LiveSearch, transit_path
from world import World


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def pygame_headless():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    yield
    pygame.quit()


def _game_result(start, goal, adjacency, algorithm):
    search = LiveSearch(start, goal, adjacency, algorithm)
    while not search.finished:
        search.step()
    return search.history, search.result, search


def test_incremental_bfs_and_dfs_match_oracles_on_300_random_grids():
    rng = random.Random(20261106)
    compared = 0
    same_start_goal = 0
    unreachable_goals = 0
    for grid_index in range(300):
        size = rng.randint(2, 14)
        density = rng.choice((0.0, 0.15, 0.35, 0.55, 0.75, rng.uniform(0.0, 0.85)))
        grid = [
            [int(rng.random() < density) for _ in range(size)]
            for _ in range(size)
        ]
        if grid_index == 0:
            grid = [[1 for _ in range(size)] for _ in range(size)]
            grid[0][0] = 0
            grid[-1][-1] = 0
        else:
            grid[rng.randrange(size)][rng.randrange(size)] = 0
        adjacency = arbol_desde_grid(grid)
        if grid_index == 0:
            start, goal = (0, 0), (size - 1, size - 1)
        else:
            start = rng.choice(tuple(adjacency))
            goal = start if grid_index == 1 or rng.randrange(5) == 0 else rng.choice(tuple(adjacency))
        same_start_goal += int(start == goal)

        for algorithm, oracle in (("bfs", oracle_bfs), ("dfs", oracle_dfs)):
            expected_nodes, expected_found = oracle(start, goal, adjacency)
            actual_nodes, actual_found, search = _game_result(
                start, goal, adjacency, algorithm
            )
            assert actual_nodes == expected_nodes
            assert actual_found is expected_found
            assert search.nodes_expanded == len(expected_nodes)
            unreachable_goals += int(not expected_found)
            compared += 1

    assert compared == 600
    assert same_start_goal > 0
    assert unreachable_goals >= 2


def test_bfs_marks_neighbors_when_enqueued_and_dfs_only_when_popped():
    grid = [[0, 0], [0, 0]]
    adjacency = arbol_desde_grid(grid)
    start = (0, 0)
    bfs = LiveSearch(start, None, adjacency, "bfs")
    dfs = LiveSearch(start, None, adjacency, "dfs")

    assert bfs.step() == start
    assert bfs.visited_set == {start, (1, 0), (0, 1)}
    assert dfs.step() == start
    assert dfs.visited_set == {start}
    assert dfs.frontier[-1] == (0, 1)
    assert dfs.step() == (0, 1)
    assert dfs.visited_set == {start, (0, 1)}


@pytest.mark.parametrize(("algorithm", "oracle"), [("bfs", oracle_bfs), ("dfs", oracle_dfs)])
def test_each_drone_exploration_order_matches_oracle_without_vision_or_shooting(
    algorithm, oracle
):
    world = World(78)
    start = world.red_start if algorithm == "bfs" else world.blue_start
    drone = Drone("Prueba", (1, 2, 3), start, algorithm, None, world.adjacency, vision=None)
    expected_nodes, found = oracle(start, None, world.adjacency)
    while not drone.search.finished:
        drone.search.step()
    assert drone.vision is None
    assert drone.pending_shot_id is None
    assert drone.search.history == expected_nodes
    assert drone.search.result is found is False


def test_drone_search_uses_only_original_generators_and_transit_is_movement_only():
    source = (PROJECT / "drone.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_from_search = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "search"
        for alias in node.names
    }
    assert imported_from_search == {"LiveSearch", "transit_path"}
    assert "shortest_path" not in source.lower()
    assert not any(word in source.lower() for word in ("astar", "dijkstra", "heuristic"))

    generator_call_sites = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "LiveSearch"
            for child in ast.walk(node)
        )
    }
    assert {
        "__init__",
        "_begin_goal_phase",
        "_begin_sweep",
        "_begin_resource_trip",
        "_advance_resource_trip",
    } <= generator_call_sites
    assert "path_to" in source

    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "LiveSearch" in calls
    assert "transit_path" in calls

    transit_source = (PROJECT / "search.py").read_text(encoding="utf-8")
    transit_tree = ast.parse(transit_source)
    transit_function = next(
        node
        for node in transit_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "transit_path"
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"step", "send"}
        for node in ast.walk(transit_function)
    )
    route = transit_path((0, 0), (1, 1), {(0, 0), (1, 0), (1, 1)})
    assert route == [(0, 0), (1, 0), (1, 1)]


@pytest.mark.parametrize("algorithm", ["bfs", "dfs"])
def test_goal_appearing_during_old_transit_starts_a_fresh_goal_search(algorithm):
    world = World(93)
    start = world.red_start
    neighbor = world.adjacency[start][0]
    drone = Drone("Prueba", (1, 2, 3), start, algorithm, None, world.adjacency)
    drone.transit_route = [start, neighbor]
    drone.transit_index = 1
    world.goal = start
    world.goal_exists = True

    drone.advance(world)

    assert drone.goal_phase
    assert drone.finished
    assert drone.reached_goal
    assert drone.current == start
    assert drone.transit_route is None


def test_goal_during_mode3_trip_discards_suspended_exploration_transit():
    world = World(100)
    start = world.red_start
    old_transit_next = world.adjacency[start][0]
    drone = Drone("Prueba", (1, 2, 3), start, "bfs", None, world.adjacency)
    drone.resource_phase = "returning"
    drone.suspended_position = start
    drone.suspended_search = drone.search
    drone.suspended_transit_route = [start, old_transit_next]
    drone.suspended_transit_index = 1
    world.goal = start
    world.goal_exists = True

    drone.advance(world, resource_mode=True)
    assert drone.goal_phase
    assert drone.suspended_search is not None
    assert drone.suspended_search.start == start
    assert drone.suspended_transit_route is None

    drone.current = start
    drone._finish_resource_leg()
    drone.advance(world, resource_mode=True)

    assert drone.transit_route is None
    assert drone.reached_goal


def test_goal_is_created_at_tick_zero_after_last_tank_is_already_destroyed():
    world = World(205)
    for tank in world.tanks_alive():
        tank.destroy("Prueba", 0)
    world.activate_goal()
    drone = Drone(
        "Dron",
        (1, 2, 3),
        world.red_start,
        "bfs",
        None,
        world.adjacency,
    )

    drone.advance(world)

    assert world.goal_exists
    assert drone.goal_phase
    assert drone.search.goal == world.goal


def test_transit_rejects_disconnected_or_unexplored_endpoints():
    with pytest.raises(ValueError, match="ya explorados"):
        transit_path((0, 0), (2, 2), {(0, 0), (1, 0)})


def test_missing_visible_entity_sprite_is_reported_and_regenerated():
    from art import make_sprite, sprite_catalog

    world = World(104)
    tank = next(entity for entity in world.entities if entity.kind == "tank")
    start = next(
        position
        for position in world.adjacency
        if abs(position[0] - tank.position[0]) + abs(position[1] - tank.position[1]) == 1
    )
    sprites = {name: make_sprite(name) for name in sprite_catalog()}
    sprites.pop(tank.sprite_name)
    drone = Drone("Prueba", (1, 2, 3), start, "bfs", None, world.adjacency)

    drone.observe_world(world, sprites, 28)

    assert tank.sprite_name in sprites


def test_error_screen_shows_exception_message_and_failing_frame(tmp_path, monkeypatch, capsys):
    import main
    from art import make_sprite, sprite_catalog, sprite_size

    app = main.GameApp()
    app.screen = pygame.display.set_mode(main.SCREEN_SIZE)
    app.font = app._font(20)
    app.small_font = app._font(15)
    app.large_font = app._font(31, bold=True)
    app.sprites = {name: make_sprite(name, sprite_size(name)) for name in sprite_catalog()}
    monkeypatch.setattr(main, "LOG_DIR", tmp_path / "logs")

    try:
        [][0]
    except IndexError as exc:
        app._write_error(exc)

    assert app.error_type == "IndexError"
    assert app.error_message == "list index out of range"
    assert ".py:" in app.error_frame
    assert "test_error_screen_shows_exception_message_and_failing_frame" in app.error_frame
    assert "IndexError" in (tmp_path / "logs" / "error.log").read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in capsys.readouterr().out
    app._draw_error()
    assert app.state == "error"


def test_reproduced_mode2_final_frame_transitions_to_results_without_indexing_turn_list(
    tmp_path,
):
    from art import make_sprite, sprite_catalog, sprite_size
    from main import GameApp, SCREEN_SIZE
    from stats import drone_statistics

    class EmptyVision:
        def classify(self, image):
            return "empty", 1.0

    app = GameApp()
    app.screen = pygame.display.set_mode(SCREEN_SIZE)
    app.font = app._font(20)
    app.small_font = app._font(15)
    app.large_font = app._font(31, bold=True)
    app.sprites = {
        name: make_sprite(name, sprite_size(name))
        for name in sprite_catalog()
    }
    app.vision = EmptyVision()
    app.results_dir = tmp_path
    app.start_match(1, seed=13)
    assert app.red_drone is not None and app.blue_drone is not None
    app.mode2_stats["dron_1_bfs"] = drone_statistics(app.red_drone)
    app.mode2_turn_index = 1
    final_drone = app.blue_drone if app.mode2_order[1] == "blue" else app.red_drone
    final_drone.finished = True
    final_drone.done_searching = True

    app._draw_state()

    assert app.state == "results"
    assert app.mode2_turn_index == len(app.mode2_order)
    assert app.results_path is not None and app.results_path.is_file()
    assert app.buttons
