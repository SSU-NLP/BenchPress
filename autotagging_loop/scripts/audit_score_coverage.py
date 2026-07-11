"""Audit Part 2 score-source coverage and held-model feasibility."""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autotagging_loop.experiment.score_matrix import normalize_matrix, spearman_pair_matrix
from autotagging_loop.experiment.split_diagnostics import benchmark_split_from_config, split_valid_pair_counts
from autotagging_loop.experiment.splits import split_models
from autotagging_loop.runner.config import load_config
from autotagging_loop.runner.corpus import (
    load_corpus,
    load_leaderboard_scores,
    load_score_sources,
    merge_score_sources,
)


def _parse_ratio(text: str) -> tuple[float, float]:
    left, right = str(text).replace(",", ":").split(":", 1)
    return (float(left), float(right))


def _normalize_ratio(ratio: tuple[float, float]) -> tuple[float, float]:
    total = float(ratio[0]) + float(ratio[1])
    if total <= 0.0:
        raise argparse.ArgumentTypeError(f"ratio must be positive, got {ratio}")
    return (float(ratio[0]) / total, float(ratio[1]) / total)


def _load_score_file(path: str, *, model_aliases: dict[str, str]) -> dict[str, dict[str, float]]:
    return load_leaderboard_scores(path, model_aliases=model_aliases)


def _active_benchmark_names(scores: dict[str, dict[str, float]], *, min_models: int) -> list[str]:
    return sorted(name for name, row in scores.items() if len(row) >= min_models)


def _basic_stats(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"min": 0, "median": 0.0, "max": 0}
    return {
        "min": int(min(values)),
        "median": float(median(values)),
        "max": int(max(values)),
    }


def _top_counts(counts: dict[str, int], *, limit: int) -> list[dict[str, int | str]]:
    return [
        {"name": name, "count": int(count)}
        for name, count in sorted(
            counts.items(),
            key=lambda item: (-int(item[1]), item[0]),
        )[: max(0, int(limit))]
        if count > 0
    ]


def _connected_components(
    benchmark_names: list[str],
    R_raw: dict[tuple[str, str], float | None],
) -> list[list[str]]:
    neighbors = {name: set() for name in benchmark_names}
    for (left, right), value in R_raw.items():
        if value is None:
            continue
        if left not in neighbors or right not in neighbors:
            continue
        neighbors[left].add(right)
        neighbors[right].add(left)

    components: list[list[str]] = []
    seen: set[str] = set()
    for start in sorted(neighbors):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[str] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for nxt in sorted(neighbors[node] - seen):
                seen.add(nxt)
                stack.append(nxt)
        components.append(sorted(component))
    components.sort(key=lambda comp: (-len(comp), comp))
    return components


def _score_coverage_profile(
    scores: dict[str, dict[str, float]],
    *,
    benchmark_names: list[str],
    R_raw: dict[tuple[str, str], float | None],
    common_count: dict[tuple[str, str], int],
    top_missing: int,
) -> dict:
    Y = {name: scores.get(name, {}) for name in benchmark_names}
    model_names = sorted({model for row in Y.values() for model in row})
    total_cells = len(benchmark_names) * len(model_names)
    present_cells = sum(1 for row in Y.values() for model in model_names if model in row)
    missing_by_benchmark = {
        benchmark: sum(1 for model in model_names if model not in Y.get(benchmark, {}))
        for benchmark in benchmark_names
    }
    missing_by_model = {
        model: sum(1 for benchmark in benchmark_names if model not in Y.get(benchmark, {}))
        for model in model_names
    }
    components = _connected_components(benchmark_names, R_raw)
    return {
        "total_cells": total_cells,
        "present_cells": present_cells,
        "missing_cells": total_cells - present_cells,
        "density": (present_cells / total_cells) if total_cells else 0.0,
        "row_count_stats": _basic_stats([len(Y.get(benchmark, {})) for benchmark in benchmark_names]),
        "model_count_stats": _basic_stats([
            sum(1 for benchmark in benchmark_names if model in Y.get(benchmark, {}))
            for model in model_names
        ]),
        "common_model_count_stats": _basic_stats(list(common_count.values())),
        "top_missing_benchmarks": _top_counts(
            missing_by_benchmark,
            limit=top_missing,
        ),
        "top_missing_models": _top_counts(
            missing_by_model,
            limit=top_missing,
        ),
        "comparable_components": [
            {"size": len(component), "benchmarks": component}
            for component in components
        ],
    }


def _coverage_summary(
    scores: dict[str, dict[str, float]],
    *,
    benchmark_names: list[str],
    normalize: str,
    min_common: int,
    splits_cfg: dict,
    model_ratios: tuple[float, float],
    model_seed: int,
    min_held_test_pairs: int,
    top_missing: int,
) -> dict:
    Y = {name: scores[name] for name in benchmark_names if name in scores}
    Y_norm = normalize_matrix(Y, method=normalize)
    R_raw, common_count = spearman_pair_matrix(
        Y_norm,
        benchmark_names,
        min_common=min_common,
        warn_below=10**9,
    )
    comparable_pairs = sum(1 for value in R_raw.values() if value is not None)
    model_names = sorted({model for row in Y.values() for model in row})
    model_split = split_models(model_names, ratios=model_ratios, seed=model_seed)
    coverage_profile = _score_coverage_profile(
        scores,
        benchmark_names=benchmark_names,
        R_raw=R_raw,
        common_count=common_count,
        top_missing=top_missing,
    )

    held_failures: list[str] = []
    held_test_counts: list[int] = []
    if len(model_split.held) < min_common:
        held_failures.append(f"held_models:{len(model_split.held)}<{min_common}")
    else:
        held_set = set(model_split.held)
        Y_held = {
            bench: {model: score for model, score in row.items() if model in held_set}
            for bench, row in Y_norm.items()
        }
        R_held, _ = spearman_pair_matrix(
            Y_held,
            benchmark_names,
            min_common=min_common,
            warn_below=10**9,
        )
        cv_folds = int(splits_cfg.get("cv_folds", 1))
        if cv_folds > 1:
            for fold in range(cv_folds):
                split = benchmark_split_from_config(
                    benchmark_names,
                    {**splits_cfg, "fold": fold},
                )
                held_test_counts.append(
                    int(split_valid_pair_counts(R_held, split)["test"])
                )
            for fold, count in enumerate(held_test_counts):
                if count < min_held_test_pairs:
                    held_failures.append(
                        f"fold{fold}:held_test:{count}<{min_held_test_pairs}"
                    )

    return {
        "benchmarks": len(benchmark_names),
        "models": len(model_names),
        "comparable_pairs": comparable_pairs,
        "model_seen": len(model_split.seen),
        "model_held": len(model_split.held),
        "held_test_pairs": held_test_counts,
        "held_failures": held_failures,
        "min_held_test_pairs": int(min_held_test_pairs),
        "min_models_per_benchmark": min(len(Y[name]) for name in benchmark_names)
        if benchmark_names else 0,
        "coverage": coverage_profile,
    }


def _split_counts_for_models(
    scores: dict[str, dict[str, float]],
    *,
    benchmark_names: list[str],
    model_subset: set[str],
    normalize: str,
    min_common: int,
    splits_cfg: dict,
) -> list[dict[str, int]]:
    Y = {
        bench: {
            model: score
            for model, score in scores.get(bench, {}).items()
            if model in model_subset
        }
        for bench in benchmark_names
    }
    R_raw, _ = spearman_pair_matrix(
        normalize_matrix(Y, method=normalize),
        benchmark_names,
        min_common=min_common,
        warn_below=10**9,
    )
    cv_folds = int(splits_cfg.get("cv_folds", 1))
    if cv_folds <= 1:
        return []
    rows: list[dict[str, int]] = []
    for fold in range(cv_folds):
        split = benchmark_split_from_config(
            benchmark_names,
            {**splits_cfg, "fold": fold},
        )
        rows.append(split_valid_pair_counts(R_raw, split))
    return rows


def _min_counts(rows: list[dict[str, int]]) -> dict[str, int]:
    if not rows:
        return {"train": 0, "dev": 0, "test": 0}
    return {
        name: min(int(row.get(name, 0)) for row in rows)
        for name in ("train", "dev", "test")
    }


def _search_model_splits(
    scores: dict[str, dict[str, float]],
    *,
    benchmark_names: list[str],
    normalize: str,
    min_common: int,
    splits_cfg: dict,
    top_k: int,
) -> list[dict]:
    model_names = sorted({
        model
        for bench in benchmark_names
        for model in scores.get(bench, {})
    })
    rows: list[dict] = []
    if len(model_names) > 20:
        return [{
            "skipped": f"too_many_models_for_exhaustive_search:{len(model_names)}"
        }]
    model_set = set(model_names)
    for held_size in range(min_common, len(model_names) - min_common + 1):
        for held_tuple in combinations(model_names, held_size):
            held = set(held_tuple)
            seen = model_set - held
            seen_counts = _split_counts_for_models(
                scores,
                benchmark_names=benchmark_names,
                model_subset=seen,
                normalize=normalize,
                min_common=min_common,
                splits_cfg=splits_cfg,
            )
            held_counts = _split_counts_for_models(
                scores,
                benchmark_names=benchmark_names,
                model_subset=held,
                normalize=normalize,
                min_common=min_common,
                splits_cfg=splits_cfg,
            )
            seen_min = _min_counts(seen_counts)
            held_min = _min_counts(held_counts)
            rows.append({
                "seen_n": len(seen),
                "held_n": len(held),
                "seen_models": sorted(seen),
                "held_models": sorted(held),
                "seen_min_pairs": seen_min,
                "held_min_pairs": held_min,
                "score": [
                    min(min(seen_min.values()), int(held_min["test"])),
                    int(held_min["test"]),
                    min(seen_min.values()),
                    int(held_min["test"]),
                    int(seen_min["dev"]),
                    int(seen_min["train"]),
                    int(seen_min["test"]),
                ],
            })
    rows.sort(key=lambda row: tuple(row["score"]), reverse=True)
    return rows[:top_k]


def _print_summary(label: str, summary: dict, *, top_missing: int) -> None:
    held_counts = ",".join(str(v) for v in summary["held_test_pairs"]) or "-"
    failures = ",".join(summary["held_failures"]) or "-"
    coverage = summary.get("coverage") or {}
    components = coverage.get("comparable_components") or []
    largest_component = int(components[0]["size"]) if components else 0
    print(
        f"{label:<18} "
        f"benches={summary['benchmarks']:<2} "
        f"models={summary['models']:<2} "
        f"pairs={summary['comparable_pairs']:<3} "
        f"density={coverage.get('density', 0.0):.3f} "
        f"components={len(components)} "
        f"largest={largest_component:<2} "
        f"seen/held={summary['model_seen']}/{summary['model_held']} "
        f"held_test={held_counts:<8} "
        f"failures={failures}"
    )
    if top_missing <= 0:
        return
    print(
        "  row_counts="
        f"{coverage.get('row_count_stats', {})} "
        "model_counts="
        f"{coverage.get('model_count_stats', {})} "
        "common_models="
        f"{coverage.get('common_model_count_stats', {})}"
    )
    print(
        "  top_missing_benchmarks="
        f"{coverage.get('top_missing_benchmarks', [])}"
    )
    print(
        "  top_missing_models="
        f"{coverage.get('top_missing_models', [])}"
    )


def _print_search(label: str, rows: list[dict]) -> None:
    if not rows:
        return
    print()
    print(f"[model-search] {label}")
    if rows and rows[0].get("skipped"):
        print(f"  skipped={rows[0]['skipped']}")
        return
    for idx, row in enumerate(rows, start=1):
        print(
            f"  #{idx} seen={row['seen_n']} held={row['held_n']} "
            f"seen_min={row['seen_min_pairs']} "
            f"held_min={row['held_min_pairs']} "
            f"held={row['held_models']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extra-source",
        action="append",
        default=[],
        help="Additional score JSON to merge after canonicalizing model aliases.",
    )
    parser.add_argument(
        "--include-hf-report",
        action="store_true",
        help="Also evaluate data/hf_benchmark_report_scores.json as an opt-in source.",
    )
    parser.add_argument(
        "--model-ratio",
        type=_parse_ratio,
        help="Override seen:held model ratio, e.g. 1:1 or 4:1.",
    )
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--min-held-test-pairs", type=int)
    parser.add_argument(
        "--search-model-splits",
        action="store_true",
        help="Exhaustively search held model subsets for feasible seen/held coverage.",
    )
    parser.add_argument("--top-search", type=int, default=5)
    parser.add_argument(
        "--top-missing",
        type=int,
        default=8,
        help="Rows to show for missing-by-benchmark/model diagnostics.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = load_config()
    aliases = config.get("model_aliases") or {}
    min_models = int(config.get("min_common_models", 6))
    normalize = str(config.get("normalize", "rank"))
    splits_cfg = config.get("splits", {}) or {}
    model_ratios = _normalize_ratio(
        args.model_ratio or tuple(splits_cfg.get("model_ratios", (0.8, 0.2)))
    )
    model_seed = int(args.model_seed if args.model_seed is not None else splits_cfg.get("model_seed", 0))
    min_held_test_pairs = int(
        args.min_held_test_pairs
        if args.min_held_test_pairs is not None
        else config.get("v_loop_min_test_valid_pairs", 1)
    )

    base = load_score_sources(config)
    benchmark_names = load_corpus(config).benchmark_names
    summaries = {
        "configured": _coverage_summary(
            base,
            benchmark_names=benchmark_names,
            normalize=normalize,
            min_common=min_models,
            splits_cfg=splits_cfg,
            model_ratios=model_ratios,
            model_seed=model_seed,
            min_held_test_pairs=min_held_test_pairs,
            top_missing=args.top_missing,
        )
    }
    searches: dict[str, list[dict]] = {}
    if args.search_model_splits:
        searches["configured"] = _search_model_splits(
            base,
            benchmark_names=benchmark_names,
            normalize=normalize,
            min_common=min_models,
            splits_cfg=splits_cfg,
            top_k=max(1, int(args.top_search)),
        )

    candidates: list[tuple[str, str]] = []
    if args.include_hf_report:
        candidates.append(("hf_report", "data/hf_benchmark_report_scores.json"))
    for idx, source in enumerate(args.extra_source, start=1):
        candidates.append((f"extra_{idx}", source))

    merged = base
    for label, source in candidates:
        path = Path(source)
        if not path.exists():
            summaries[label] = {"error": f"missing:{source}"}
            continue
        source_scores = _load_score_file(str(path), model_aliases=aliases)
        merged = merge_score_sources(merged, source_scores, model_aliases=aliases)
        merged_names = _active_benchmark_names(merged, min_models=min_models)
        summaries[f"configured+{label}"] = _coverage_summary(
            merged,
            benchmark_names=merged_names,
            normalize=normalize,
            min_common=min_models,
            splits_cfg=splits_cfg,
            model_ratios=model_ratios,
            model_seed=model_seed,
            min_held_test_pairs=min_held_test_pairs,
            top_missing=args.top_missing,
        )
        if args.search_model_splits:
            searches[f"configured+{label}"] = _search_model_splits(
                merged,
                benchmark_names=merged_names,
                normalize=normalize,
                min_common=min_models,
                splits_cfg=splits_cfg,
                top_k=max(1, int(args.top_search)),
            )

    if args.json:
        print(json.dumps({"summaries": summaries, "model_search": searches}, indent=2, sort_keys=True))
    else:
        print(
            "[score-coverage] "
            f"min_common={min_models}, normalize={normalize}, "
            f"model_ratios={model_ratios}, model_seed={model_seed}, "
            f"min_held_test_pairs={min_held_test_pairs}"
        )
        for label, summary in summaries.items():
            if "error" in summary:
                print(f"{label:<18} {summary['error']}")
            else:
                _print_summary(label, summary, top_missing=args.top_missing)
        for label, rows in searches.items():
            _print_search(label, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
