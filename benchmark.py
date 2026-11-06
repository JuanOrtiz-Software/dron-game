from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Dict, Iterable

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from main import GameApp, MODEL_FILE, FALLBACK_MODEL_FILE
from vision import VisionPipeline


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return statistics.fmean(items) if items else 0.0


def run_benchmarks(mode2_seeds: int, mode3_seeds: int, output: Path, first_seed: int) -> dict[str, object]:
    app = GameApp()
    app._make_dirs()
    app._init_display()
    app._ensure_sprites()
    app.vision = VisionPipeline(MODEL_FILE, FALLBACK_MODEL_FILE)
    if app.vision.backend == "unavailable":
        raise RuntimeError("No hay un modelo de visión cargado para las corridas de balance.")

    mode2_rows: list[dict[str, float | int]] = []
    mode3_rows: list[dict[str, float | int | str | bool]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="drone-game-benchmark-") as directory:
            app.results_dir = Path(directory)
            for index in range(mode2_seeds):
                seed = first_seed + index
                app.mode2_first = "red" if index % 2 == 0 else "blue"
                app.start_match(1, seed)
                for _ in range(30001):
                    if app.state != "match":
                        break
                    app.advance_match()
                if app.state != "results":
                    raise RuntimeError(f"Modo 2 no terminó con semilla {seed}.")
                for key, stats in app.mode2_stats.items():
                    mode2_rows.append(
                        {
                            "seed": seed,
                            "algorithm": "BFS" if key == "dron_1_bfs" else "DFS",
                            "elapsed_time_s": float(stats["elapsed_time_s"]),
                            "real_steps": int(stats["real_steps"]),
                            "nodes_expanded": int(stats["nodes_expanded"]),
                            "score": int(stats["score"]),
                            "fp": int(stats["vehicles_destroyed_wrongly"]),
                            "fn": int(stats["false_negatives"]),
                        }
                    )

            for index in range(mode3_seeds):
                seed = first_seed + index
                app.start_match(2, seed)
                for _ in range(30001):
                    if app.state != "match":
                        break
                    app.advance_match()
                if app.state != "results":
                    raise RuntimeError(f"Modo 3 no terminó con semilla {seed}.")
                for key, stats in app._drone_records().items():
                    mode3_rows.append(
                        {
                            "seed": seed,
                            "algorithm": "BFS" if key == "dron_1_bfs" else "DFS",
                            "winner": app.winner,
                            "win": app.winner.startswith("Dron 1") if key == "dron_1_bfs" else app.winner.startswith("Dron 2"),
                            "crashed": bool(stats["crashed"]),
                            "score": int(stats["score"]),
                            "elapsed_time_s": float(stats["elapsed_time_s"]),
                            "real_steps": int(stats["real_steps"]),
                            "nodes_expanded": int(stats["nodes_expanded"]),
                            "fp": int(stats["vehicles_destroyed_wrongly"]),
                            "fn": int(stats["false_negatives"]),
                            "recharges": int(stats["recharges"]),
                        }
                    )
        output.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "backend": app.vision.backend,
            "mode2_seed_count": mode2_seeds,
            "mode2_by_algorithm": {
                algorithm: {
                    metric: _mean(float(row[metric]) for row in mode2_rows if row["algorithm"] == algorithm)
                    for metric in ("elapsed_time_s", "real_steps", "nodes_expanded", "score", "fp", "fn")
                }
                for algorithm in ("BFS", "DFS")
            },
            "mode3_seed_count": mode3_seeds,
            "mode3_by_algorithm": {
                algorithm: {
                    "win_rate": _mean(float(row["win"]) for row in mode3_rows if row["algorithm"] == algorithm),
                    "crash_rate": _mean(float(row["crashed"]) for row in mode3_rows if row["algorithm"] == algorithm),
                    "score": _mean(float(row["score"]) for row in mode3_rows if row["algorithm"] == algorithm),
                    "elapsed_time_s": _mean(float(row["elapsed_time_s"]) for row in mode3_rows if row["algorithm"] == algorithm),
                    "real_steps": _mean(float(row["real_steps"]) for row in mode3_rows if row["algorithm"] == algorithm),
                    "nodes_expanded": _mean(float(row["nodes_expanded"]) for row in mode3_rows if row["algorithm"] == algorithm),
                    "fp": _mean(float(row["fp"]) for row in mode3_rows if row["algorithm"] == algorithm),
                    "fn": _mean(float(row["fn"]) for row in mode3_rows if row["algorithm"] == algorithm),
                    "recharges": _mean(float(row["recharges"]) for row in mode3_rows if row["algorithm"] == algorithm),
                }
                for algorithm in ("BFS", "DFS")
            },
        }
        output.write_text(
            json.dumps(
                {"summary": summary, "mode2_runs": mode2_rows, "mode3_runs": mode3_rows},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        with output.with_name(f"{output.stem}_mode2.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=mode2_rows[0].keys() if mode2_rows else [])
            if mode2_rows:
                writer.writeheader()
                writer.writerows(mode2_rows)
        with output.with_name(f"{output.stem}_mode3.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=mode3_rows[0].keys() if mode3_rows else [])
            if mode3_rows:
                writer.writeheader()
                writer.writerows(mode3_rows)
        print(f"Backend: {summary['backend']}")
        print(f"Modo 2: {mode2_seeds} semillas")
        for algorithm, values in summary["mode2_by_algorithm"].items():
            print(f"{algorithm}: {values}")
        print(f"Modo 3: {mode3_seeds} semillas")
        for algorithm, values in summary["mode3_by_algorithm"].items():
            print(f"{algorithm}: {values}")
        print(f"Detalle guardado en {output}")
        return summary
    finally:
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mide BFS/DFS con la visión CNN activa.")
    parser.add_argument("--mode2-seeds", type=int, default=30)
    parser.add_argument("--mode3-seeds", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    args = parser.parse_args()
    if args.mode2_seeds < 1 or args.mode3_seeds < 1:
        parser.error("Las cantidades de semillas deben ser positivas.")
    run_benchmarks(args.mode2_seeds, args.mode3_seeds, args.output, args.first_seed)


if __name__ == "__main__":
    main()
