"""Generate benchboard/data/benchmark_axis_weights.json from a Benchpress run.

The builder's capability/taxonomy buttons are the run's learned vocab tags
(fold*/final/selected_vocab.json), and each benchmark's per-tag weights come from
the run's T_star.json. Cost/time are carried over from the previous axes file by
benchmark name where available.

Usage: python scripts/gen_benchboard_axes.py [RUN_DIR] [FOLD]
Defaults: RUN_DIR=results/part2_experiment/run_cv_20260703_182036, FOLD=fold2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO / "results" / "part2_experiment" / "run_cv_20260703_182036"
OUT_PATH = REPO / "benchboard" / "data" / "benchmark_axis_weights.json"
OLD_AXES_PATH = OUT_PATH  # reuse existing file for cost/time lookup by name

RELATED_WEIGHT_MIN = 0.75
RELATED_MAX = 6
DEFAULT_COST = 1.5
DEFAULT_TIME_MINUTES = 15


def slug(name: str) -> str:
    return "-".join(name.lower().split())


def load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def cost_time_by_name(old_axes: dict) -> dict[str, tuple[float, int]]:
    lookup: dict[str, tuple[float, int]] = {}
    for bench in old_axes.get("benchmarks", []):
        lookup[bench["name"]] = (
            float(bench.get("cost", DEFAULT_COST)),
            int(bench.get("time_minutes", DEFAULT_TIME_MINUTES)),
        )
    return lookup


def build_axes(vocab: list[dict], tag_vectors: dict[str, dict]) -> list[dict]:
    axes: list[dict] = []
    for tag in vocab:
        tag_id = tag["id"]
        ranked = sorted(
            ((name, vec.get(tag_id, 0.0)) for name, vec in tag_vectors.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        related = [slug(name) for name, weight in ranked if weight >= RELATED_WEIGHT_MIN][:RELATED_MAX]
        axes.append(
            {
                "id": tag_id,
                "name": tag["name"],
                "description": tag.get("definition", ""),
                "source": "learned",
                "stability": 0.85,
                "confidence": 0.9,
                "lineage_status": "new",
                "related_benchmarks": related,
                "high_performing_models": [],
            }
        )
    return axes


def build_benchmarks(tag_vectors: dict[str, dict], cost_time: dict[str, tuple[float, int]]) -> list[dict]:
    benchmarks: list[dict] = []
    for name, vector in tag_vectors.items():
        cost, minutes = cost_time.get(name, (DEFAULT_COST, DEFAULT_TIME_MINUTES))
        benchmarks.append(
            {
                "id": slug(name),
                "name": name,
                "cost": cost,
                "time_minutes": minutes,
                "weights": {tag_id: round(float(weight), 3) for tag_id, weight in vector.items()},
                "rationale": "",
            }
        )
    return benchmarks


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUN
    fold = sys.argv[2] if len(sys.argv) > 2 else "fold2"
    final_dir = run_dir / fold / "final"

    vocab = load_json(final_dir / "selected_vocab.json")
    tag_vectors = load_json(final_dir / "T_star.json")
    cost_time = cost_time_by_name(load_json(OLD_AXES_PATH)) if OLD_AXES_PATH.exists() else {}

    payload = {
        "run_id": run_dir.name,
        "axes": build_axes(vocab, tag_vectors),
        "benchmarks": build_benchmarks(tag_vectors, cost_time),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH} - {len(payload['axes'])} tags, {len(payload['benchmarks'])} benchmarks")


if __name__ == "__main__":
    main()
