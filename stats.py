from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from config import TICK_SECONDS
from drone import Drone


def drone_statistics(drone: Drone) -> dict[str, int | float | bool]:
    ticks = drone.stopped_at_ticks if drone.stopped_at_ticks is not None else drone.elapsed_ticks
    shot_count = drone.tanks_destroyed + drone.vehicles_destroyed_wrongly
    return {
        "tanks_destroyed": drone.tanks_destroyed,
        "vehicles_destroyed_wrongly": drone.vehicles_destroyed_wrongly,
        "false_negatives": drone.false_negatives,
        "score": drone.score,
        "recognition_accuracy_pct": drone.recognition_accuracy_pct,
        "shot_precision_pct": drone.shot_precision_pct if shot_count else 0.0,
        "cells_explored": len(drone.cells_explored),
        "nodes_expanded": drone.nodes_expanded,
        "real_steps": drone.real_steps,
        "elapsed_time_s": ticks * TICK_SECONDS,
        "elapsed_ticks": ticks,
        "battery": drone.battery,
        "ammo": drone.ammo,
        "recharges": drone.recharges,
        "crashed": drone.crashed,
    }


def save_match_result(
    results_dir: Path,
    mode: str,
    seed: int,
    parameters: dict[str, int | float],
    drones: dict[str, dict[str, int | float | bool]],
    winner: str,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    destination = results_dir / f"match-{time.time_ns()}.json"
    result = {
        "date": timestamp,
        "mode": mode,
        "seed": seed,
        "parameters": parameters,
        "drones": drones,
        "winner": winner,
    }
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
