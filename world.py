from __future__ import annotations

import copy
import random
from typing import Any

from config import (
    BUILDING_COUNT,
    CHARGING_STATION_COUNT,
    GRID_SIZE,
    TANK_COUNT,
    VEHICLE_COUNT,
    validate_config,
)
from entities import Entity, Position


class World:
    def __init__(self, seed: int = 7, include_stations: bool = False):
        validate_config()
        self.seed = seed
        self.size = GRID_SIZE
        self.include_stations = include_stations
        self.grid: list[list[int]] = []
        self.goal: Position | None = None
        self.goal_exists = False
        self.red_start: Position = (0, 0)
        self.blue_start: Position = (0, 0)
        self.entities: list[Entity] = []
        self.adjacency: dict[Position, list[Position]] = {}
        self.generate()

    def _flood_fill(self, start: Position) -> set[Position]:
        pending = [start]
        seen = {start}
        while pending:
            x, y = pending.pop()
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                candidate = (x + dx, y + dy)
                nx, ny = candidate
                if (
                    0 <= nx < self.size
                    and 0 <= ny < self.size
                    and not self.grid[ny][nx]
                    and candidate not in seen
                ):
                    seen.add(candidate)
                    pending.append(candidate)
        return seen

    def _generate_connected_grid(self, rng: random.Random) -> list[Position]:
        self.grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        candidates = [(x, y) for y in range(self.size) for x in range(self.size)]
        rng.shuffle(candidates)
        placed = 0
        for x, y in candidates:
            if placed == BUILDING_COUNT:
                break
            self.grid[y][x] = 1
            walkable = [
                (cx, cy)
                for cy in range(self.size)
                for cx in range(self.size)
                if not self.grid[cy][cx]
            ]
            if len(self._flood_fill(walkable[0])) != len(walkable):
                self.grid[y][x] = 0
                continue
            placed += 1
        if placed != BUILDING_COUNT:
            raise RuntimeError("No fue posible generar edificios sin desconectar la ciudad.")
        return [
            (x, y)
            for y in range(self.size)
            for x in range(self.size)
            if not self.grid[y][x]
        ]

    def generate(self) -> None:
        rng = random.Random(self.seed)
        open_cells = self._generate_connected_grid(rng)
        rng.shuffle(open_cells)
        self.red_start, self.blue_start = open_cells[:2]
        selected = open_cells[2:]

        vehicle_types = ("car", "pickup", "truck", "helicopter")
        entity_specs = [("tank", (index % 4) + 1) for index in range(TANK_COUNT)]
        entity_specs.extend(
            (vehicle_types[index % len(vehicle_types)], (index % 3) + 1)
            for index in range(VEHICLE_COUNT)
        )
        if self.include_stations:
            entity_specs.extend(("charging_station", 1) for _ in range(CHARGING_STATION_COUNT))
        if len(selected) < len(entity_specs):
            raise ValueError("La cuadrícula no tiene suficientes celdas para las entidades.")

        self.entities = [
            Entity(
                entity_id=index,
                kind=kind,
                position=selected[index],
                variant=variant,
                orientation=rng.uniform(0.0, 360.0),
            )
            for index, (kind, variant) in enumerate(entity_specs)
        ]
        self.goal_exists = False
        self.adjacency = {
            (x, y): [
                (x + dx, y + dy)
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= x + dx < self.size
                and 0 <= y + dy < self.size
                and not self.grid[y + dy][x + dx]
            ]
            for y in range(self.size)
            for x in range(self.size)
            if not self.grid[y][x]
        }
        component = self._flood_fill(self.red_start)
        required = {self.red_start, self.blue_start}
        required.update(entity.position for entity in self.entities)
        if not required <= component:
            raise RuntimeError("La generación dejó entidades fuera del componente conectado.")
        if len(required) != len(self.entities) + 2:
            raise RuntimeError("La generación colocó más de una entidad en una celda.")

    def activate_goal(self) -> None:
        if self.goal_exists:
            return
        occupied = {self.red_start, self.blue_start}
        occupied.update(entity.position for entity in self.entities)
        available = sorted(set(self.adjacency) - occupied)
        if not available:
            raise RuntimeError("No hay una celda libre conectada para crear la meta.")
        self.goal = random.Random(self.seed ^ 0x5EED).choice(available)
        component = self._flood_fill(self.red_start)
        if self.goal not in component:
            raise RuntimeError("La meta quedó fuera del componente conectado.")
        self.goal_exists = True

    def tanks_alive(self) -> list[Entity]:
        return [entity for entity in self.entities if entity.kind == "tank" and entity.alive]

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "seed": self.seed,
                "include_stations": self.include_stations,
                "grid": self.grid,
                "goal": self.goal,
                "goal_exists": self.goal_exists,
                "red_start": self.red_start,
                "blue_start": self.blue_start,
                "entities": self.entities,
                "adjacency": self.adjacency,
            }
        )

    def restore(self, snapshot: dict[str, Any]) -> None:
        expected = {
            "seed",
            "include_stations",
            "grid",
            "goal",
            "goal_exists",
            "red_start",
            "blue_start",
            "entities",
            "adjacency",
        }
        if set(snapshot) != expected:
            raise ValueError("La instantánea del mundo tiene una estructura inválida.")
        restored = copy.deepcopy(snapshot)
        self.seed = restored["seed"]
        self.include_stations = restored["include_stations"]
        self.grid = restored["grid"]
        self.goal = restored["goal"]
        self.goal_exists = restored["goal_exists"]
        self.red_start = restored["red_start"]
        self.blue_start = restored["blue_start"]
        self.entities = restored["entities"]
        self.adjacency = restored["adjacency"]
