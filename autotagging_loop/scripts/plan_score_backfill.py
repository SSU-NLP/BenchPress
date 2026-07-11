"""Plan score cells needed for strict seen/held model validation.

The strict v-loop requires two independent score-pattern matrices:

* F_seen drives train/dev selection and benchmark test reporting.
* F_held is used only for held_model_test.

With min_common=6, a sufficient unbiased data plan is to choose six seen-core
models and six disjoint held-core models, then ensure every active benchmark
has scores for every core model. This script searches current score coverage
for the cheapest such core assignment and prints the exact missing cells.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autotagging_loop.runner.config import load_config
from autotagging_loop.runner.corpus import (
    load_corpus,
    load_leaderboard_scores,
    load_score_sources,
    merge_score_sources,
)


def _load_scores_from_config(
    *,
    include_hf_report: bool,
    curated_score_backfill_path: str | None = None,
    use_curated_score_backfill: bool = True,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    config = load_config()
    aliases = config.get("model_aliases") or {}
    if curated_score_backfill_path:
        config["curated_score_backfill_path"] = curated_score_backfill_path
    config["use_curated_score_backfill"] = bool(use_curated_score_backfill)
    scores = load_score_sources(config)
    leaderboard_scores = load_leaderboard_scores(
        config["leaderboard_path"],
        exclude=config.get("exclude", []),
        model_aliases=aliases,
    )
    active_model_axis = {
        model
        for row in leaderboard_scores.values()
        for model in row
    }
    sources = [config["leaderboard_path"]]
    aai_path = config.get("aai_scores_path")
    if config.get("use_aai_scores", True) and aai_path and Path(aai_path).exists():
        sources.append(aai_path)
    curated_path = config.get("curated_score_backfill_path")
    if (
        config.get("use_curated_score_backfill", True)
        and curated_path
        and Path(curated_path).exists()
    ):
        sources.append(curated_path)
    if include_hf_report:
        hf_path = "data/hf_benchmark_report_scores.json"
        if Path(hf_path).exists():
            hf_scores = load_leaderboard_scores(
                hf_path,
                exclude=config.get("exclude", []),
                model_aliases=aliases,
            )
            hf_scores = {
                benchmark: {
                    model: score
                    for model, score in row.items()
                    if model in active_model_axis
                }
                for benchmark, row in hf_scores.items()
            }
            scores = merge_score_sources(scores, hf_scores, model_aliases=aliases)
            sources.append(hf_path)
    # Reuse load_corpus only to get the active benchmark/document filter.
    active = load_corpus(config).benchmark_names
    return {bench: scores.get(bench, {}) for bench in active}, sources


def _missing_cells(
    scores: dict[str, dict[str, float]],
    benchmarks: list[str],
    models: tuple[str, ...],
) -> list[tuple[str, str]]:
    missing: list[tuple[str, str]] = []
    for bench in benchmarks:
        row = scores.get(bench, {})
        for model in models:
            if model not in row:
                missing.append((bench, model))
    return missing


def _top_core_plans(
    scores: dict[str, dict[str, float]],
    *,
    core_size: int,
    top_k: int,
) -> list[dict[str, Any]]:
    benchmarks = sorted(scores)
    models = sorted({model for row in scores.values() for model in row})
    plans: list[dict[str, Any]] = []
    if len(models) < core_size * 2:
        return [{
            "error": (
                f"need at least {core_size * 2} distinct models for disjoint "
                f"seen/held cores; got {len(models)}"
            )
        }]

    for seen_core in combinations(models, core_size):
        remaining = [model for model in models if model not in set(seen_core)]
        seen_missing = _missing_cells(scores, benchmarks, seen_core)
        for held_core in combinations(remaining, core_size):
            held_missing = _missing_cells(scores, benchmarks, held_core)
            total_missing = len(seen_missing) + len(held_missing)
            plans.append({
                "seen_core": list(seen_core),
                "held_core": list(held_core),
                "missing_seen": seen_missing,
                "missing_held": held_missing,
                "missing_seen_count": len(seen_missing),
                "missing_held_count": len(held_missing),
                "total_missing_count": total_missing,
                "benchmark_count": len(benchmarks),
                "model_count": len(models),
                "guaranteed_common_models": core_size,
            })
    plans.sort(
        key=lambda plan: (
            int(plan["total_missing_count"]),
            int(plan["missing_held_count"]),
            int(plan["missing_seen_count"]),
            plan["seen_core"],
            plan["held_core"],
        )
    )
    return plans[:top_k]


def _missing_by_benchmark(cells: list[tuple[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for bench, _model in cells:
        counts[bench] = counts.get(bench, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _write_csv(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scope", "benchmark", "model"])
        writer.writeheader()
        for scope, key in (("seen", "missing_seen"), ("held", "missing_held")):
            for bench, model in plan.get(key, []):
                writer.writerow({"scope": scope, "benchmark": bench, "model": model})


def _jsonable_plan(plan: dict[str, Any]) -> dict[str, Any]:
    out = dict(plan)
    out["missing_seen"] = [
        {"benchmark": bench, "model": model}
        for bench, model in plan.get("missing_seen", [])
    ]
    out["missing_held"] = [
        {"benchmark": bench, "model": model}
        for bench, model in plan.get("missing_held", [])
    ]
    out["missing_seen_by_benchmark"] = _missing_by_benchmark(plan.get("missing_seen", []))
    out["missing_held_by_benchmark"] = _missing_by_benchmark(plan.get("missing_held", []))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-hf-report", action="store_true")
    parser.add_argument("--curated-score-backfill-path")
    parser.add_argument("--no-curated-score-backfill", action="store_true")
    parser.add_argument("--core-size", type=int, default=6)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--write-csv", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    scores, sources = _load_scores_from_config(
        include_hf_report=args.include_hf_report,
        curated_score_backfill_path=args.curated_score_backfill_path,
        use_curated_score_backfill=not args.no_curated_score_backfill,
    )
    plans = _top_core_plans(
        scores,
        core_size=max(1, int(args.core_size)),
        top_k=max(1, int(args.top)),
    )
    payload = {
        "sources": sources,
        "plans": [_jsonable_plan(plan) if "error" not in plan else plan for plan in plans],
    }
    if args.write_csv and plans and "error" not in plans[0]:
        _write_csv(args.write_csv, plans[0])
        payload["csv_path"] = str(args.write_csv)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "[score-backfill] "
            f"sources={sources}, core_size={args.core_size}, active_benchmarks={len(scores)}"
        )
        for idx, plan in enumerate(plans, start=1):
            if "error" in plan:
                print(f"  ERROR {plan['error']}")
                continue
            print(
                f"  #{idx} total_missing={plan['total_missing_count']} "
                f"seen_missing={plan['missing_seen_count']} "
                f"held_missing={plan['missing_held_count']}"
            )
            print(f"     seen_core={plan['seen_core']}")
            print(f"     held_core={plan['held_core']}")
            print(
                "     top_missing_seen="
                f"{_missing_by_benchmark(plan['missing_seen'])}"
            )
            print(
                "     top_missing_held="
                f"{_missing_by_benchmark(plan['missing_held'])}"
            )
        if args.write_csv and payload.get("csv_path"):
            print(f"  wrote {payload['csv_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
