from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Position = Tuple[int, int]


@dataclass
class Entity:
    entity_id: int
    kind: str
    position: Position
    variant: int = 1
    orientation: float = 0.0
    alive: bool = True
    destroyed_by: str | None = None
    destroyed_tick: int | None = None

    @property
    def sprite_name(self) -> str:
        if self.kind == "charging_station":
            return "charging_station.png"
        suffix = "normal" if self.alive else "destroyed"
        return f"{self.kind}_{self.variant:02}_{suffix}.png"

    def destroy(self, drone_name: str, tick: int | None = None) -> bool:
        if not self.alive or self.kind == "charging_station":
            return False
        self.alive = False
        self.destroyed_by = drone_name
        self.destroyed_tick = tick
        return True
