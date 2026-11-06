from __future__ import annotations

from typing import Dict, Optional, Set

import pygame
import numpy as np

from art import make_sprite, perturb_cell_image, render_cell_image, sprite_catalog, sprite_size
from config import (
    AMMO_MAX,
    BATTERY_MAX,
    CONFIDENCE_THRESHOLD,
    LOW_BATTERY_PCT,
    MAX_SWEEPS,
    MODE_3_VISION_RADIUS,
    RECHARGE_TICKS,
    SHOT_BATTERY_COST,
    SHOT_TICKS,
    VISION_RADIUS,
    VISION_BRIGHTNESS_RANGE,
    VISION_NOISE_STD,
)
from entities import Entity, Position
from search import LiveSearch, transit_path
from world import World


class Drone:
    def __init__(
        self,
        name: str,
        color: tuple[int, int, int],
        start: Position,
        algorithm: str,
        goal: Position | None,
        adjacency: Dict[Position, list[Position]],
        vision: object | None = None,
        vision_radius: int = VISION_RADIUS,
        seed: int = 0,
    ):
        self.name = name
        self.color = color
        self.start = start
        self.current = start
        self.algorithm = algorithm.lower()
        self.goal = goal
        self.adjacency = adjacency
        self.vision = vision
        self.vision_radius = vision_radius
        self.seed = seed
        self.visible: Set[Position] = set()
        self.explored: Set[Position] = {start}
        self.known_positions: Set[Position] = set()
        self.detections: dict[Position, tuple[str, float]] = {}
        self.search = LiveSearch(start, None, adjacency, self.algorithm)
        self.goal_phase = False
        self.reached_goal = False
        self.finished = False
        self.done_searching = False
        self.crashed = False
        self.sweeps_completed = 1
        self.nodes_expanded = 0
        self.real_steps = 0
        self.tanks_destroyed = 0
        self.vehicles_destroyed_wrongly = 0
        self.false_negative_ids: Set[int] = set()
        self.classifications_total = 0
        self.classifications_correct = 0
        self.cells_explored: Set[Position] = set()
        self.elapsed_ticks = 0
        self.stopped_at_ticks: Optional[int] = None
        self.battery = BATTERY_MAX
        self.ammo = AMMO_MAX
        self.recharges = 0
        self.pending_shot_id: Optional[int] = None
        self.shot_ticks_remaining = 0
        self.shot_elapsed_ticks = 0
        self.suspended_search: Optional[LiveSearch] = None
        self.resource_search: Optional[LiveSearch] = None
        self.resource_path: Optional[list[Position]] = None
        self.resource_path_index = 1
        self.resource_phase: Optional[str] = None
        self.suspended_position: Optional[Position] = None
        self.transit_route: Optional[list[Position]] = None
        self.transit_index = 1
        self.suspended_transit_route: Optional[list[Position]] = None
        self.suspended_transit_index = 1
        self.recharge_ticks_remaining = 0

    @property
    def score(self) -> int:
        return self.tanks_destroyed - self.vehicles_destroyed_wrongly

    @property
    def false_negatives(self) -> int:
        return len(self.false_negative_ids)

    @property
    def elapsed_time_s(self) -> float:
        ticks = self.stopped_at_ticks if self.stopped_at_ticks is not None else self.elapsed_ticks
        from config import TICK_SECONDS

        return ticks * TICK_SECONDS

    @property
    def recognition_accuracy_pct(self) -> float:
        if not self.classifications_total:
            return 0.0
        return 100.0 * self.classifications_correct / self.classifications_total

    @property
    def shot_precision_pct(self) -> float:
        total_shots = self.tanks_destroyed + self.vehicles_destroyed_wrongly
        if total_shots == 0:
            return 0.0
        return 100.0 * self.tanks_destroyed / total_shots

    @property
    def active(self) -> bool:
        return not self.finished and not self.crashed

    def observe_world(self, world: World, sprites: Dict[str, pygame.Surface], cell_size: int) -> None:
        x0, y0 = self.current
        self.visible = {
            (x, y)
            for y in range(y0 - self.vision_radius, y0 + self.vision_radius + 1)
            for x in range(x0 - self.vision_radius, x0 + self.vision_radius + 1)
            if 0 <= x < world.size
            and 0 <= y < world.size
            and not world.grid[y][x]
        }
        entities_by_position = {entity.position: entity for entity in world.entities}
        for position in sorted(self.visible - self.known_positions):
            entity = entities_by_position.get(position)
            if entity and (entity.kind == "charging_station" or not entity.alive):
                self.known_positions.add(position)
                self.cells_explored.add(position)
                continue
            sprite = None
            orientation = 0.0
            if entity is not None:
                if entity.sprite_name not in sprites:
                    if entity.sprite_name not in sprite_catalog():
                        raise KeyError(f"El sprite de entidad {entity.sprite_name!r} no pertenece al catálogo.")
                    print(f"AVISO: falta el sprite {entity.sprite_name}; se genera un reemplazo procedural.")
                    sprites[entity.sprite_name] = make_sprite(
                        entity.sprite_name,
                        sprite_size(entity.sprite_name),
                    )
                sprite = sprites[entity.sprite_name]
                orientation = entity.orientation
            image = render_cell_image(sprites["ground_street.png"], sprite, orientation, cell_size)
            perturbation_seed = (
                self.seed
                + self.sweeps_completed * 1_000_003
                + position[0] * world.size
                + position[1]
            )
            image = perturb_cell_image(
                image,
                perturbation_seed,
                VISION_NOISE_STD,
                VISION_BRIGHTNESS_RANGE,
            )
            if self.vision is not None:
                label, confidence = self.vision.classify(image)
            else:
                label, confidence = ("empty", 1.0)
            self.detections[position] = (label, confidence)
            self.known_positions.add(position)
            self.cells_explored.add(position)
            if entity is not None:
                expected = "tank" if entity.kind == "tank" else "vehicle"
                self.classifications_total += 1
                if label == expected:
                    self.classifications_correct += 1
                if entity.kind == "tank" and (
                    label != "tank" or confidence < CONFIDENCE_THRESHOLD
                ):
                    self.false_negative_ids.add(entity.entity_id)

    def target_candidates(self, world: World) -> list[Entity]:
        if self.goal_phase or self.resource_phase is not None or world.goal_exists:
            return []
        candidates: list[Entity] = []
        by_position = {entity.position: entity for entity in world.entities}
        for position in sorted(self.visible):
            detection = self.detections.get(position)
            entity = by_position.get(position)
            if (
                detection is not None
                and detection[0] == "tank"
                and detection[1] >= CONFIDENCE_THRESHOLD
                and entity is not None
                and entity.alive
                and entity.kind != "charging_station"
            ):
                candidates.append(entity)
        return candidates

    def forget_entity(self, entity: Entity) -> None:
        self.detections.pop(entity.position, None)

    def begin_shot(self, entity: Entity, resource_mode: bool) -> bool:
        if not self.active or self.pending_shot_id is not None or self.ammo <= 0:
            return False
        self.pending_shot_id = entity.entity_id
        self.shot_ticks_remaining = SHOT_TICKS
        self.shot_elapsed_ticks = 0
        self.ammo -= 1
        if resource_mode:
            self.battery = max(0, self.battery - SHOT_BATTERY_COST)
        return True

    def advance_shot(self) -> Optional[int]:
        if self.pending_shot_id is None:
            return None
        self.shot_ticks_remaining -= 1
        self.shot_elapsed_ticks += 1
        if self.shot_ticks_remaining > 0:
            return None
        completed = self.pending_shot_id
        self.pending_shot_id = None
        self.shot_elapsed_ticks = 0
        return completed

    def _begin_goal_phase(self) -> None:
        if self.goal is None:
            raise RuntimeError("No se puede iniciar la fase de meta antes de crearla.")
        self.goal_phase = True
        self.done_searching = False
        goal_start = (
            self.suspended_position
            if self.resource_phase is not None and self.suspended_position is not None
            else self.current
        )
        goal_search = LiveSearch(goal_start, self.goal, self.adjacency, self.algorithm)
        if self.resource_phase is not None:
            self.suspended_search = goal_search
            self.suspended_transit_route = None
            self.suspended_transit_index = 1
        else:
            self.search = goal_search
            self.transit_route = None
            self.transit_index = 1

    def _begin_sweep(self, world: World) -> None:
        self.sweeps_completed += 1
        self.known_positions = {
            entity.position for entity in world.entities if not entity.alive
        }
        self.detections.clear()
        self.visible.clear()
        self.search = LiveSearch(self.current, None, self.adjacency, self.algorithm)
        self.transit_route = None
        self.transit_index = 1

    def _begin_resource_trip(self, world: World) -> None:
        stations = [
            entity.position
            for entity in world.entities
            if entity.kind == "charging_station"
        ]
        if not stations:
            return
        station = min(
            stations,
            key=lambda position: (
                abs(position[0] - self.current[0]) + abs(position[1] - self.current[1]),
                position,
            ),
        )
        self.suspended_position = self.current
        self.suspended_search = self.search
        self.suspended_transit_route = self.transit_route
        self.suspended_transit_index = self.transit_index
        self.transit_route = None
        self.transit_index = 1
        self.resource_search = LiveSearch(self.current, station, self.adjacency, self.algorithm)
        self.resource_path = None
        self.resource_path_index = 1
        self.resource_phase = "to_station"

    def _finish_resource_leg(self) -> None:
        if self.resource_phase == "to_station":
            self.resource_phase = "recharge"
            self.recharge_ticks_remaining = RECHARGE_TICKS
        elif self.resource_phase == "returning":
            self.search = self.suspended_search or self.search
            self.suspended_search = None
            self.resource_search = None
            self.resource_path = None
            self.resource_phase = None
            self.suspended_position = None
            self.transit_route = self.suspended_transit_route
            self.transit_index = self.suspended_transit_index
            self.suspended_transit_route = None
            self.suspended_transit_index = 1

    def _cancel_resource_trip(self) -> None:
        self.search = self.suspended_search or self.search
        self.suspended_search = None
        self.resource_search = None
        self.resource_path = None
        self.resource_phase = None
        self.suspended_position = None
        self.transit_route = self.suspended_transit_route
        self.transit_index = self.suspended_transit_index
        self.suspended_transit_route = None
        self.suspended_transit_index = 1

    def _advance_transit(self, resource_mode: bool) -> bool:
        if self.transit_route is None:
            return False
        if self.transit_index >= len(self.transit_route):
            self.transit_route = None
            self.transit_index = 1
            return False

        next_position = self.transit_route[self.transit_index]
        if abs(next_position[0] - self.current[0]) + abs(next_position[1] - self.current[1]) != 1:
            raise RuntimeError("La ruta de tránsito contiene un movimiento no adyacente.")
        self.current = next_position
        self.transit_index += 1
        self.real_steps += 1
        self.explored.add(self.current)
        if resource_mode:
            self.battery = max(0, self.battery - 1)
            if self.battery <= 0:
                self.crashed = True
                self.finished = True
                self.stopped_at_ticks = self.elapsed_ticks
        if self.transit_index >= len(self.transit_route):
            self.transit_route = None
            self.transit_index = 1
        return True

    def _advance_resource_trip(self) -> bool:
        if self.resource_phase == "recharge":
            self.recharge_ticks_remaining -= 1
            if self.recharge_ticks_remaining <= 0:
                self.battery = BATTERY_MAX
                self.ammo = AMMO_MAX
                self.recharges += 1
                assert self.suspended_position is not None
                self.resource_search = LiveSearch(
                    self.current, self.suspended_position, self.adjacency, self.algorithm
                )
                self.resource_path = None
                self.resource_path_index = 1
                self.resource_phase = "returning"
            return True
        if self.resource_search is None:
            return False
        if self.resource_path is None:
            node = self.resource_search.step()
            if node is None:
                if self.resource_search.finished and not self.resource_search.result:
                    self._cancel_resource_trip()
                return True
            self.nodes_expanded += 1
            if node == self.resource_search.goal:
                self.resource_path = self.resource_search.path_to(node)
                if not self.resource_path:
                    raise RuntimeError("La búsqueda llegó a la estación sin registrar su trayectoria.")
                self.resource_path_index = 1
                if self.current == node:
                    self._finish_resource_leg()
            return True
        if self.resource_path_index < len(self.resource_path):
            next_position = self.resource_path[self.resource_path_index]
            if abs(next_position[0] - self.current[0]) + abs(next_position[1] - self.current[1]) != 1:
                raise RuntimeError("La ruta de recursos contiene un movimiento no adyacente.")
            self.current = next_position
            self.resource_path_index += 1
            self.real_steps += 1
            self.battery = max(0, self.battery - 1)
            if self.battery <= 0:
                self.crashed = True
                self.finished = True
                self.stopped_at_ticks = self.elapsed_ticks
            elif self.resource_path_index == len(self.resource_path):
                self._finish_resource_leg()
            return True
        return True

    def advance(self, world: World, resource_mode: bool = False) -> None:
        if not self.active:
            return
        self.elapsed_ticks += 1
        if self.pending_shot_id is not None:
            return
        if resource_mode and self.battery <= 0:
            self.crashed = True
            self.finished = True
            self.stopped_at_ticks = self.elapsed_ticks
            return
        if world.goal_exists and not self.goal_phase:
            self.goal = world.goal
            self._begin_goal_phase()
        if self.done_searching and not self.goal_phase:
            return
        if (
            resource_mode
            and self.resource_phase is None
            and self.battery < BATTERY_MAX * LOW_BATTERY_PCT / 100
        ):
            self._begin_resource_trip(world)
        if self.resource_phase is not None:
            self._advance_resource_trip()
            return

        if self._advance_transit(resource_mode):
            if self.crashed:
                return
            if self.goal_phase and self.current == self.goal:
                self.finished = True
                self.reached_goal = True
                self.stopped_at_ticks = self.elapsed_ticks
            return

        node = self.search.step()
        if node is None:
            if self.goal_phase:
                self.done_searching = True
                self.finished = True
                self.stopped_at_ticks = self.elapsed_ticks
            elif world.tanks_alive() and self.sweeps_completed < MAX_SWEEPS:
                self._begin_sweep(world)
            else:
                self.done_searching = True
            return

        self.nodes_expanded += 1
        self.explored.update(self.search.visited_set)
        self.transit_route = transit_path(
            self.current,
            node,
            set(self.search.visited_set),
        )
        self.transit_index = 1
        if self.current == node:
            self.transit_route = None
        else:
            self._advance_transit(resource_mode)
        if self.crashed:
            return
        if self.goal_phase and self.current == self.goal:
            self.finished = True
            self.reached_goal = True
            self.stopped_at_ticks = self.elapsed_ticks
