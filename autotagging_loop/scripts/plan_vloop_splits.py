"""Plan v-loop benchmark splits without making LLM calls.

The script checks only score-matrix coverage: how many benchmark pairs remain
inside each train/dev/test bucket after min_common filtering. It is intended as
a preflight before expensive K-fold runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autotagging_loop.experiment.config import load_experiment_config
from autotagging_loop.experiment.score_matrix import normalize_matrix, spearman_pair_matrix
from autotagging_loop.experiment.split_diagnostics import (
    benchmark_split_from_config,
    split_pair_count_failures,
    split_coverage_pair_counts,
    split_valid_pair_counts,
)
from autotagging_loop.experiment.splits import parse_dev_train_split
from autotagging_loop.experiment.splits import split_models
from autotagging_loop.runner.config import load_config
from autotagging_loop.runner.corpus import load_corpus
from autotagging_loop.runner.run import _build_v3_overrides


def parse_ratio(text: str) -> tuple[float, float]:
    raw = str(text).strip().replace(",", ":")
    parts = raw.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"expected dev:train ratio like 1:3, got {text!r}"
        )
    try:
        return parse_dev_train_split((float(parts[0]), float(parts[1])))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_seed_range(text: str) -> list[int]:
    raw = str(text).strip()
    if ":" not in raw:
        return [int(raw)]
    start_s, end_s = raw.split(":", 1)
    start = int(start_s)
    end = int(end_s)
    if end < start:
        raise argparse.ArgumentTypeError(
            f"seed range end must be >= start, got {text!r}"
        )
    return list(range(start, end + 1))


def format_ratio(ratio: tuple[float, float]) -> str:
    def fmt(v: float) -> str:
        if float(v).is_integer():
            return str(int(v))
        return f"{v:.3g}"

    return f"{fmt(ratio[0])}:{fmt(ratio[1])}"


def format_count_pair(valid: int, coverage: int, *, width: int) -> str:
    return f"{int(valid)}/{int(coverage)}".ljust(width)


def normalized_ratio(ratio: tuple[float, float]) -> tuple[float, float]:
    total = float(ratio[0]) + float(ratio[1])
    if total <= 0.0:
        return (0.0, 0.0)
    return (float(ratio[0]) / total, float(ratio[1]) / total)


def unique_ratios(ratios: list[tuple[float, float]]) -> list[tuple[float, float]]:
    seen: set[tuple[float, float]] = set()
    out: list[tuple[float, float]] = []
    for ratio in ratios:
        key = (float(ratio[0]), float(ratio[1]))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def fold_diagnostics(
    *,
    benchmark_names: list[str],
    R_raw: dict,
    common_count: dict | None = None,
    cv_folds: int,
    seed: int,
    stratified: bool,
    dev_train_split: tuple[float, float],
    thresholds: dict[str, int],
    min_common: int | None = None,
) -> dict:
    folds: list[dict] = []
    for fold in range(cv_folds):
        split = benchmark_split_from_config(
            benchmark_names,
            {
                "cv_folds": cv_folds,
                "fold": fold,
                "benchmark_seed": seed,
                "stratified": stratified,
                "dev_train_split": list(dev_train_split),
            },
        )
        counts = split_valid_pair_counts(R_raw, split)
        coverage_counts = (
            split_coverage_pair_counts(
                common_count,
                split,
                min_common=int(min_common),
            )
            if common_count is not None and min_common is not None
            else dict(counts)
        )
        degenerate_counts = {
            name: max(0, int(coverage_counts[name]) - int(counts[name]))
            for name in ("train", "dev", "test")
        }
        failures = split_pair_count_failures(counts, thresholds)
        folds.append(
            {
                "fold": fold,
                "sizes": {
                    "train": len(split.train),
                    "dev": len(split.dev),
                    "test": len(split.test),
                },
                "valid_pairs": counts,
                "coverage_pairs": coverage_counts,
                "degenerate_pairs": degenerate_counts,
                "failures": failures,
                "train": split.train,
                "dev": split.dev,
                "test": split.test,
            }
        )
    min_pairs = {
        name: min(f["valid_pairs"][name] for f in folds)
        for name in ("train", "dev", "test")
    }
    total_pairs = {
        name: sum(f["valid_pairs"][name] for f in folds)
        for name in ("train", "dev", "test")
    }
    min_coverage_pairs = {
        name: min(f["coverage_pairs"][name] for f in folds)
        for name in ("train", "dev", "test")
    }
    total_degenerate_pairs = {
        name: sum(f["degenerate_pairs"][name] for f in folds)
        for name in ("train", "dev", "test")
    }
    failures = [
        f"fold{f['fold']}:{failure}"
        for f in folds
        for failure in f["failures"]
    ]
    return {
        "status": "PASS" if not failures else "FAIL",
        "cv_folds": cv_folds,
        "seed": seed,
        "stratified": stratified,
        "dev_train_split": list(dev_train_split),
        "min_valid_pairs": min_pairs,
        "total_valid_pairs": total_pairs,
        "min_coverage_pairs": min_coverage_pairs,
        "total_degenerate_pairs": total_degenerate_pairs,
        "failures": failures,
        "folds": folds,
    }


def sort_key(row: dict) -> tuple:
    passed = row["status"] == "PASS"
    min_pairs = row["min_valid_pairs"]
    min_coverage = row.get("min_coverage_pairs") or min_pairs
    total_pairs = row["total_valid_pairs"]
    return (
        0 if passed else 1,
        -int(min_coverage["dev"]),
        -int(min_coverage["test"]),
        -int(min_coverage["train"]),
        -int(min_pairs["dev"]),
        -int(min_pairs["test"]),
        -int(min_pairs["train"]),
        -int(total_pairs["test"]),
        int(row["cv_folds"]),
        int(row["seed"]),
        format_ratio(tuple(row["dev_train_split"])),
    )


def model_fold_diagnostics(
    *,
    benchmark_names: list[str],
    model_names: list[str],
    Y_norm: dict,
    cv_folds: int,
    benchmark_seed: int,
    stratified: bool,
    dev_train_split: tuple[float, float],
    model_ratio: tuple[float, float],
    model_seed: int,
    min_common: int,
    min_held_test_pairs: int,
) -> dict:
    split_ratio = normalized_ratio(model_ratio)
    model_split = split_models(model_names, ratios=split_ratio, seed=model_seed)
    failures: list[str] = []
    held_test_counts: list[int] = []
    held_test_coverage_counts: list[int] = []

    if len(model_split.held) < min_common:
        failures.append(
            f"held_models:{len(model_split.held)}<{min_common}"
        )
    else:
        Y_held = {
            bench: {
                model: score
                for model, score in scores.items()
                if model in set(model_split.held)
            }
            for bench, scores in Y_norm.items()
        }
        R_held, common_held = spearman_pair_matrix(
            Y_held,
            benchmark_names,
            min_common=min_common,
            warn_below=10**9,
        )
        for fold in range(cv_folds):
            split = benchmark_split_from_config(
                benchmark_names,
                {
                    "cv_folds": cv_folds,
                    "fold": fold,
                    "benchmark_seed": benchmark_seed,
                    "stratified": stratified,
                    "dev_train_split": list(dev_train_split),
                },
            )
            counts = split_valid_pair_counts(R_held, split)
            coverage_counts = split_coverage_pair_counts(
                common_held,
                split,
                min_common=min_common,
            )
            held_test_counts.append(int(counts["test"]))
            held_test_coverage_counts.append(int(coverage_counts["test"]))
        below = [
            f"fold{fold}:held_test:{count}<{min_held_test_pairs}"
            for fold, count in enumerate(held_test_counts)
            if count < min_held_test_pairs
        ]
        failures.extend(below)

    return {
        "status": "PASS" if not failures else "FAIL",
        "model_seed": int(model_seed),
        "model_ratio": list(model_ratio),
        "model_ratio_normalized": list(split_ratio),
        "n_seen": len(model_split.seen),
        "n_held": len(model_split.held),
        "min_common": int(min_common),
        "min_held_test_pairs": int(min_held_test_pairs),
        "held_test_pairs": held_test_counts,
        "held_test_coverage_pairs": held_test_coverage_counts,
        "held_test_degenerate_pairs": [
            max(0, int(cov) - int(valid))
            for cov, valid in zip(held_test_coverage_counts, held_test_counts)
        ],
        "failures": failures,
    }


def _model_pair_matrices(
    *,
    benchmark_names: list[str],
    model_names: list[str],
    Y_norm: dict,
    model_ratio: tuple[float, float],
    model_seed: int,
    min_common: int,
) -> dict:
    split_ratio = normalized_ratio(model_ratio)
    model_split = split_models(model_names, ratios=split_ratio, seed=model_seed)
    seen_set = set(model_split.seen)
    held_set = set(model_split.held)
    failures: list[str] = []

    if len(model_split.seen) < min_common:
        failures.append(f"seen_models:{len(model_split.seen)}<{min_common}")
        R_seen = {}
        common_seen = {}
    else:
        Y_seen = {
            bench: {
                model: score
                for model, score in scores.items()
                if model in seen_set
            }
            for bench, scores in Y_norm.items()
        }
        R_seen, common_seen = spearman_pair_matrix(
            Y_seen,
            benchmark_names,
            min_common=min_common,
            warn_below=10**9,
        )

    if len(model_split.held) < min_common:
        failures.append(f"held_models:{len(model_split.held)}<{min_common}")
        R_held = {}
        common_held = {}
    else:
        Y_held = {
            bench: {
                model: score
                for model, score in scores.items()
                if model in held_set
            }
            for bench, scores in Y_norm.items()
        }
        R_held, common_held = spearman_pair_matrix(
            Y_held,
            benchmark_names,
            min_common=min_common,
            warn_below=10**9,
        )

    return {
        "model_split": model_split,
        "model_ratio_normalized": split_ratio,
        "R_seen": R_seen,
        "common_seen": common_seen,
        "R_held": R_held,
        "common_held": common_held,
        "failures": failures,
    }


def joint_fold_diagnostics(
    *,
    benchmark_names: list[str],
    model_names: list[str],
    Y_norm: dict,
    cv_folds: int,
    benchmark_seed: int,
    stratified: bool,
    dev_train_split: tuple[float, float],
    model_ratio: tuple[float, float],
    model_seed: int,
    min_common: int,
    thresholds: dict[str, int],
    min_held_test_pairs: int,
    model_pair_matrices: dict | None = None,
) -> dict:
    """Evaluate the exact strict split contract for one benchmark/model combo.

    Unlike the separate benchmark/model tables, this computes benchmark
    train/dev/test pair counts on F_seen and held-model test counts on F_held
    for the same candidate model split. This is the preflight shape used by the
    research-grade v-loop when ``v_loop_score_model_scope=seen``.
    """

    matrices = model_pair_matrices or _model_pair_matrices(
        benchmark_names=benchmark_names,
        model_names=model_names,
        Y_norm=Y_norm,
        model_ratio=model_ratio,
        model_seed=model_seed,
        min_common=min_common,
    )
    model_split = matrices["model_split"]
    split_ratio = matrices["model_ratio_normalized"]
    R_seen = matrices["R_seen"]
    R_held = matrices["R_held"]
    common_seen = matrices.get("common_seen") or {}
    common_held = matrices.get("common_held") or {}
    failures: list[str] = list(matrices["failures"])

    fold_rows: list[dict] = []
    held_test_counts: list[int] = []
    for fold in range(cv_folds):
        split = benchmark_split_from_config(
            benchmark_names,
            {
                "cv_folds": cv_folds,
                "fold": fold,
                "benchmark_seed": benchmark_seed,
                "stratified": stratified,
                "dev_train_split": list(dev_train_split),
            },
        )
        seen_counts = split_valid_pair_counts(R_seen, split) if R_seen else {
            "train": 0,
            "dev": 0,
            "test": 0,
        }
        seen_coverage_counts = (
            split_coverage_pair_counts(
                common_seen,
                split,
                min_common=min_common,
            )
            if common_seen
            else {"train": 0, "dev": 0, "test": 0}
        )
        seen_degenerate_counts = {
            name: max(0, int(seen_coverage_counts[name]) - int(seen_counts[name]))
            for name in ("train", "dev", "test")
        }
        fold_failures = split_pair_count_failures(seen_counts, thresholds)
        failures.extend(f"fold{fold}:seen_{failure}" for failure in fold_failures)

        held_count = (
            split_valid_pair_counts(R_held, split)["test"]
            if R_held
            else 0
        )
        held_coverage_count = (
            split_coverage_pair_counts(
                common_held,
                split,
                min_common=min_common,
            )["test"]
            if common_held
            else 0
        )
        held_test_counts.append(int(held_count))
        if held_count < min_held_test_pairs:
            failures.append(
                f"fold{fold}:held_test:{held_count}<{min_held_test_pairs}"
            )
        fold_rows.append(
            {
                "fold": fold,
                "sizes": {
                    "train": len(split.train),
                    "dev": len(split.dev),
                    "test": len(split.test),
                },
                "seen_valid_pairs": seen_counts,
                "seen_coverage_pairs": seen_coverage_counts,
                "seen_degenerate_pairs": seen_degenerate_counts,
                "held_test_pairs": int(held_count),
                "held_test_coverage_pairs": int(held_coverage_count),
                "held_test_degenerate_pairs": max(
                    0,
                    int(held_coverage_count) - int(held_count),
                ),
                "train": split.train,
                "dev": split.dev,
                "test": split.test,
            }
        )

    min_seen_pairs = {
        name: min(row["seen_valid_pairs"][name] for row in fold_rows)
        for name in ("train", "dev", "test")
    }
    total_seen_pairs = {
        name: sum(row["seen_valid_pairs"][name] for row in fold_rows)
        for name in ("train", "dev", "test")
    }
    min_seen_coverage_pairs = {
        name: min(row["seen_coverage_pairs"][name] for row in fold_rows)
        for name in ("train", "dev", "test")
    }
    total_seen_degenerate_pairs = {
        name: sum(row["seen_degenerate_pairs"][name] for row in fold_rows)
        for name in ("train", "dev", "test")
    }
    min_held_test = min(held_test_counts) if held_test_counts else 0
    held_test_coverage_counts = [
        int(row["held_test_coverage_pairs"]) for row in fold_rows
    ]
    min_held_test_coverage = (
        min(held_test_coverage_counts) if held_test_coverage_counts else 0
    )
    return {
        "status": "PASS" if not failures else "FAIL",
        "cv_folds": int(cv_folds),
        "benchmark_seed": int(benchmark_seed),
        "stratified": bool(stratified),
        "dev_train_split": list(dev_train_split),
        "model_seed": int(model_seed),
        "model_ratio": list(model_ratio),
        "model_ratio_normalized": list(split_ratio),
        "n_seen": len(model_split.seen),
        "n_held": len(model_split.held),
        "min_common": int(min_common),
        "thresholds": dict(thresholds),
        "min_held_test_pairs": int(min_held_test_pairs),
        "min_seen_valid_pairs": min_seen_pairs,
        "total_seen_valid_pairs": total_seen_pairs,
        "min_seen_coverage_pairs": min_seen_coverage_pairs,
        "total_seen_degenerate_pairs": total_seen_degenerate_pairs,
        "held_test_pairs": held_test_counts,
        "min_held_test_pairs_observed": int(min_held_test),
        "held_test_coverage_pairs": held_test_coverage_counts,
        "min_held_test_coverage_pairs_observed": int(min_held_test_coverage),
        "held_test_degenerate_pairs": [
            max(0, int(cov) - int(valid))
            for cov, valid in zip(held_test_coverage_counts, held_test_counts)
        ],
        "failures": failures,
        "folds": fold_rows,
    }


def print_model_table(rows: list[dict], *, top: int) -> None:
    if not rows:
        return
    rows = sorted(
        rows,
        key=lambda row: (
            0 if row["status"] == "PASS" else 1,
            -min(row["held_test_pairs"] or [0]),
            -int(row["n_held"]),
            int(row["model_seed"]),
            format_ratio(tuple(row["model_ratio"])),
        ),
    )
    print()
    header = (
        "model_status seed seen:held n_seen n_held min_held_test "
        "held_test_pairs failures"
    )
    print(header)
    print("-" * len(header))
    for row in rows[:top]:
        failures = ",".join(row["failures"][:3])
        if len(row["failures"]) > 3:
            failures += f",+{len(row['failures']) - 3}"
        held_counts = ",".join(str(v) for v in row["held_test_pairs"]) or "-"
        print(
            f"{row['status']:<12} "
            f"{row['model_seed']:<4} "
            f"{format_ratio(tuple(row['model_ratio'])):<9} "
            f"{row['n_seen']:<6} "
            f"{row['n_held']:<6} "
            f"{row['min_held_test_pairs']:<13} "
            f"{held_counts:<15} "
            f"{failures or '-'}"
        )


def joint_sort_key(row: dict) -> tuple:
    passed = row["status"] == "PASS"
    min_seen = row["min_seen_valid_pairs"]
    min_seen_coverage = row.get("min_seen_coverage_pairs") or min_seen
    total_seen = row["total_seen_valid_pairs"]
    return (
        0 if passed else 1,
        -int(row.get("min_held_test_coverage_pairs_observed", 0)),
        -int(row["min_held_test_pairs_observed"]),
        -int(min_seen_coverage["dev"]),
        -int(min_seen_coverage["test"]),
        -int(min_seen_coverage["train"]),
        -int(min_seen["dev"]),
        -int(min_seen["test"]),
        -int(min_seen["train"]),
        -int(total_seen["test"]),
        int(row["cv_folds"]),
        int(row["benchmark_seed"]),
        int(row["model_seed"]),
        format_ratio(tuple(row["dev_train_split"])),
        format_ratio(tuple(row["model_ratio"])),
    )


def print_joint_table(rows: list[dict], *, top: int) -> None:
    if not rows:
        return
    print()
    header = (
        "joint_status cv strat bench_seed model_seed dev:train seen:held "
        "seen_train seen_dev seen_test held_test failures"
    )
    print(header)
    print("-" * len(header))
    for row in rows[:top]:
        failures = ",".join(row["failures"][:3])
        if len(row["failures"]) > 3:
            failures += f",+{len(row['failures']) - 3}"
        print(
            f"{row['status']:<12} "
            f"{row['cv_folds']:<2} "
            f"{str(row['stratified']).lower():<5} "
            f"{row['benchmark_seed']:<10} "
            f"{row['model_seed']:<10} "
            f"{format_ratio(tuple(row['dev_train_split'])):<9} "
            f"{format_ratio(tuple(row['model_ratio'])):<9} "
            f"{format_count_pair(row['min_seen_valid_pairs']['train'], row['min_seen_coverage_pairs']['train'], width=10)} "
            f"{format_count_pair(row['min_seen_valid_pairs']['dev'], row['min_seen_coverage_pairs']['dev'], width=8)} "
            f"{format_count_pair(row['min_seen_valid_pairs']['test'], row['min_seen_coverage_pairs']['test'], width=9)} "
            f"{format_count_pair(row['min_held_test_pairs_observed'], row['min_held_test_coverage_pairs_observed'], width=9)} "
            f"{failures or '-'}"
        )


def print_table(rows: list[dict], *, top: int) -> None:
    header = (
        "status cv strat seed dev:train min_train min_dev min_test "
        "total_test failures"
    )
    print(header)
    print("-" * len(header))
    for row in rows[:top]:
        failures = ",".join(row["failures"][:3])
        if len(row["failures"]) > 3:
            failures += f",+{len(row['failures']) - 3}"
        print(
            f"{row['status']:<6} "
            f"{row['cv_folds']:<2} "
            f"{str(row['stratified']).lower():<5} "
            f"{row['seed']:<4} "
            f"{format_ratio(tuple(row['dev_train_split'])):<9} "
            f"{format_count_pair(row['min_valid_pairs']['train'], row['min_coverage_pairs']['train'], width=7)} "
            f"{format_count_pair(row['min_valid_pairs']['dev'], row['min_coverage_pairs']['dev'], width=5)} "
            f"{format_count_pair(row['min_valid_pairs']['test'], row['min_coverage_pairs']['test'], width=6)} "
            f"{row['total_valid_pairs']['test']:<10} "
            f"{failures or '-'}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-folds", type=int, nargs="+")
    parser.add_argument("--seed-range", default=None, help="Seed or inclusive range, e.g. 44 or 0:99.")
    parser.add_argument(
        "--dev-train-split",
        type=parse_ratio,
        action="append",
        help="Candidate dev:train ratio, e.g. 1:3. Repeatable.",
    )
    parser.add_argument("--include-plain", action="store_true")
    parser.add_argument("--include-stratified", action="store_true")
    parser.add_argument(
        "--model-ratio",
        type=parse_ratio,
        action="append",
        help="Candidate seen:held model ratio, e.g. 4:1 or 1:1. Repeatable.",
    )
    parser.add_argument(
        "--model-seed-range",
        default=None,
        help="Model split seed or inclusive range, e.g. 0 or 0:20.",
    )
    parser.add_argument("--min-held-test-pairs", type=int)
    parser.add_argument(
        "--min-common-models",
        type=int,
        help="Override min_common_models for split-planning sensitivity checks.",
    )
    parser.add_argument(
        "--joint",
        action="store_true",
        help=(
            "Also search benchmark/model split combinations together. This is "
            "the strict preflight shape when v_loop_score_model_scope=seen."
        ),
    )
    parser.add_argument("--top-model", type=int, default=12)
    parser.add_argument("--top-joint", type=int, default=12)
    parser.add_argument("--min-train-pairs", type=int)
    parser.add_argument("--min-dev-pairs", type=int)
    parser.add_argument("--min-test-pairs", type=int)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    part2_overrides = {}
    if args.min_common_models is not None:
        part2_overrides["min_common_models"] = int(args.min_common_models)
    part2_config = load_config(part2_overrides or None)
    exp_config = load_experiment_config(_build_v3_overrides(part2_config))
    splits_cfg = exp_config.get("splits", {}) or {}
    current_cv = int(splits_cfg.get("cv_folds", 1))
    current_seed = int(splits_cfg.get("benchmark_seed", 0))
    current_stratified = bool(splits_cfg.get("stratified", False))
    current_ratio = parse_dev_train_split(splits_cfg.get("dev_train_split"))
    current_model_ratio = parse_dev_train_split(splits_cfg.get("model_ratios", (0.8, 0.2)))
    current_model_seed = int(splits_cfg.get("model_seed", 0))
    score_model_scope = str(exp_config.get("v_loop_score_model_scope", "all")).strip().lower()
    if bool(exp_config.get("v_loop_require_held_model_test", False)):
        score_model_scope = "seen"
    if score_model_scope not in {"all", "seen"}:
        raise ValueError(
            "v_loop_score_model_scope must be one of {'all', 'seen'}, "
            f"got {score_model_scope!r}"
        )

    cv_values = args.cv_folds or [current_cv]
    seed_values = parse_seed_range(args.seed_range) if args.seed_range else [current_seed]
    ratios = unique_ratios(
        [current_ratio]
        + (args.dev_train_split or [])
        + [(1.0, 3.0), (1.0, 2.0), (1.0, 1.0), (2.0, 1.0)]
    )
    stratified_values = []
    if args.include_plain:
        stratified_values.append(False)
    if args.include_stratified:
        stratified_values.append(True)
    if not stratified_values:
        stratified_values = [current_stratified]

    thresholds = {
        "train": int(args.min_train_pairs if args.min_train_pairs is not None else exp_config.get("v_loop_min_train_valid_pairs", 1)),
        "dev": int(args.min_dev_pairs if args.min_dev_pairs is not None else exp_config.get("v_loop_min_dev_valid_pairs", 1)),
        "test": int(args.min_test_pairs if args.min_test_pairs is not None else exp_config.get("v_loop_min_test_valid_pairs", 1)),
    }
    min_common = int(
        args.min_common_models
        if args.min_common_models is not None
        else exp_config.get("min_common_models", 6)
    )
    min_held_test_pairs = int(
        args.min_held_test_pairs
        if args.min_held_test_pairs is not None
        else thresholds["test"]
    )

    corpus = load_corpus(part2_config)
    Y_norm_full = normalize_matrix(corpus.Y, method=exp_config.get("normalize", "rank"))
    score_Y = corpus.Y
    if score_model_scope == "seen":
        model_split = split_models(
            corpus.model_names,
            ratios=tuple(splits_cfg.get("model_ratios", (0.8, 0.2))),
            seed=current_model_seed,
        )
        seen = set(model_split.seen)
        score_Y = {
            bench: {
                model: score
                for model, score in scores.items()
                if model in seen
            }
            for bench, scores in corpus.Y.items()
        }
    Y_norm = normalize_matrix(score_Y, method=exp_config.get("normalize", "rank"))
    R_raw, common_count = spearman_pair_matrix(
        Y_norm,
        corpus.benchmark_names,
        min_common=min_common,
        warn_below=10**9,
    )
    comparable_pairs = sum(1 for value in R_raw.values() if value is not None)
    print(
        "[split-plan] "
        f"benchmarks={len(corpus.benchmark_names)}, "
        f"models={len(corpus.model_names)}, "
        f"comparable_pairs={comparable_pairs}, "
        f"min_common={min_common}, "
        f"thresholds={thresholds}, "
        f"score_model_scope={score_model_scope}"
    )

    rows: list[dict] = []
    for cv_folds in cv_values:
        if cv_folds <= 1:
            continue
        for seed in seed_values:
            for stratified in stratified_values:
                for ratio in ratios:
                    rows.append(
                        fold_diagnostics(
                            benchmark_names=corpus.benchmark_names,
                            R_raw=R_raw,
                            common_count=common_count,
                            cv_folds=int(cv_folds),
                            seed=int(seed),
                            stratified=bool(stratified),
                            dev_train_split=ratio,
                            thresholds=thresholds,
                            min_common=min_common,
                        )
                    )
    rows.sort(key=sort_key)
    print_table(rows, top=max(1, int(args.top)))
    model_ratios = unique_ratios(
        [current_model_ratio]
        + (args.model_ratio or [])
        + [(4.0, 1.0), (3.0, 2.0), (1.0, 1.0)]
    )
    model_seed_values = (
        parse_seed_range(args.model_seed_range)
        if args.model_seed_range
        else [current_model_seed]
    )
    model_rows = [
        model_fold_diagnostics(
            benchmark_names=corpus.benchmark_names,
            model_names=corpus.model_names,
            Y_norm=Y_norm_full,
            cv_folds=current_cv,
            benchmark_seed=current_seed,
            stratified=current_stratified,
            dev_train_split=current_ratio,
            model_ratio=ratio,
            model_seed=int(model_seed),
            min_common=min_common,
            min_held_test_pairs=min_held_test_pairs,
        )
        for model_seed in model_seed_values
        for ratio in model_ratios
    ]
    print_model_table(model_rows, top=max(1, int(args.top_model)))
    joint_rows: list[dict] = []
    if args.joint:
        model_matrix_cache = {
            (int(model_seed), tuple(float(v) for v in model_ratio)): _model_pair_matrices(
                benchmark_names=corpus.benchmark_names,
                model_names=corpus.model_names,
                Y_norm=Y_norm_full,
                model_ratio=model_ratio,
                model_seed=int(model_seed),
                min_common=min_common,
            )
            for model_seed in model_seed_values
            for model_ratio in model_ratios
        }
        joint_rows = [
            joint_fold_diagnostics(
                benchmark_names=corpus.benchmark_names,
                model_names=corpus.model_names,
                Y_norm=Y_norm_full,
                cv_folds=int(cv_folds),
                benchmark_seed=int(seed),
                stratified=bool(stratified),
                dev_train_split=ratio,
                model_ratio=model_ratio,
                model_seed=int(model_seed),
                min_common=min_common,
                thresholds=thresholds,
                min_held_test_pairs=min_held_test_pairs,
                model_pair_matrices=model_matrix_cache[
                    (int(model_seed), tuple(float(v) for v in model_ratio))
                ],
            )
            for cv_folds in cv_values
            if int(cv_folds) > 1
            for seed in seed_values
            for stratified in stratified_values
            for ratio in ratios
            for model_seed in model_seed_values
            for model_ratio in model_ratios
        ]
        joint_rows.sort(key=joint_sort_key)
        print_joint_table(joint_rows, top=max(1, int(args.top_joint)))
    if args.json:
        print(json.dumps(
            {
                "benchmark_splits": rows,
                "model_splits": model_rows,
                "joint_splits": joint_rows,
            },
            indent=2,
            sort_keys=True,
        ))
    benchmark_ok = any(row["status"] == "PASS" for row in rows)
    model_ok = any(row["status"] == "PASS" for row in model_rows)
    joint_ok = any(row["status"] == "PASS" for row in joint_rows) if args.joint else True
    return 0 if benchmark_ok and model_ok and joint_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
