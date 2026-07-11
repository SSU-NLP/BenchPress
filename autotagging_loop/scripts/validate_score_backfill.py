"""Validate curated score backfill data before running the v-loop.

This script is intentionally stricter than the generic leaderboard loader:
manual score cells must carry per-cell provenance, use an explicit 0-1 scale,
and avoid duplicate cells after benchmark/model alias normalization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autotagging_loop.experiment.config import load_experiment_config
from autotagging_loop.experiment.score_matrix import normalize_matrix, spearman_pair_matrix
from autotagging_loop.experiment.split_diagnostics import (
    benchmark_split_from_config,
    split_coverage_pair_counts,
    split_pair_count_failures,
)
from autotagging_loop.experiment.splits import split_models
from autotagging_loop.runner.config import load_config
from autotagging_loop.runner.corpus import (
    _looks_placeholder,
    _validate_source_date,
    _validate_source_url,
    load_corpus,
    load_curated_score_backfill,
    name_key,
    normalize_model_name,
)
from autotagging_loop.runner.run import _build_v3_overrides
from autotagging_loop.scripts.plan_score_backfill import _jsonable_plan, _load_scores_from_config, _top_core_plans
from autotagging_loop.scripts.plan_vloop_splits import fold_diagnostics, model_fold_diagnostics


DEFAULT_MISSING_CSV = Path("data") / "score_backfill_missing.csv"


def load_missing_cell_plan(
    path: str | Path,
    *,
    model_aliases: dict[str, str] | None = None,
) -> dict[tuple[str, str], dict[str, str]]:
    """Load the committed missing-cell plan keyed by normalized benchmark/model."""

    plan_path = Path(path)
    if not plan_path.exists():
        raise FileNotFoundError(str(plan_path))
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with open(plan_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"scope", "benchmark", "model"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "missing-cell CSV missing required columns: "
                + ",".join(sorted(missing))
            )
        for index, row in enumerate(reader, start=2):
            scope = str(row.get("scope") or "").strip()
            benchmark = str(row.get("benchmark") or "").strip()
            model = normalize_model_name(
                str(row.get("model") or "").strip(),
                model_aliases,
            )
            if scope not in {"seen", "held"}:
                raise ValueError(f"missing-cell CSV line {index} invalid scope: {scope!r}")
            if not benchmark or not model:
                raise ValueError(f"missing-cell CSV line {index} has empty benchmark/model")
            key = (name_key(benchmark), model)
            if key in rows:
                prev = rows[key]
                raise ValueError(
                    "duplicate missing-cell CSV entry after normalization: "
                    f"{prev['benchmark']}/{prev['model']} and {benchmark}/{model}"
                )
            rows[key] = {"scope": scope, "benchmark": benchmark, "model": model}
    return rows


def curated_cells_outside_plan(
    curated: dict[str, dict[str, float]],
    expected: dict[tuple[str, str], dict[str, str]],
) -> list[str]:
    outside: list[str] = []
    for benchmark, scores in curated.items():
        for model in scores:
            if (name_key(benchmark), model) not in expected:
                outside.append(f"{benchmark}/{model}")
    return sorted(outside)


def _plan_cell_label(cell: tuple[str, str, str]) -> str:
    scope, benchmark_key, model = cell
    return f"{scope}/{benchmark_key}/{model}"


def _expected_plan_cells(
    expected: dict[tuple[str, str], dict[str, str]],
) -> set[tuple[str, str, str]]:
    return {
        (row["scope"], name_key(row["benchmark"]), row["model"])
        for row in expected.values()
    }


def _generated_plan_cells(
    plan: dict[str, Any],
    *,
    model_aliases: dict[str, str] | None = None,
) -> set[tuple[str, str, str]]:
    cells: set[tuple[str, str, str]] = set()
    for scope, key in (("seen", "missing_seen"), ("held", "missing_held")):
        for item in plan.get(key, []) or []:
            if isinstance(item, dict):
                benchmark = str(item.get("benchmark") or "").strip()
                model = str(item.get("model") or "").strip()
            else:
                try:
                    benchmark, model = item
                except (TypeError, ValueError):
                    continue
                benchmark = str(benchmark or "").strip()
                model = str(model or "").strip()
            if not benchmark or not model:
                continue
            cells.add((scope, name_key(benchmark), normalize_model_name(model, model_aliases)))
    return cells


def missing_plan_drift(
    expected: dict[tuple[str, str], dict[str, str]],
    generated_plan: dict[str, Any],
    *,
    model_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compare committed missing-cell CSV against the current generated plan."""

    if not isinstance(generated_plan, dict) or "error" in generated_plan:
        error = str((generated_plan or {}).get("error") or "missing_generated_plan")
        return {
            "status": "UNKNOWN",
            "expected_cells": len(expected),
            "generated_cells": 0,
            "extra_in_csv": [],
            "missing_from_csv": [],
            "failures": [f"generated_plan_error:{error}"],
        }
    expected_cells = _expected_plan_cells(expected)
    generated_cells = _generated_plan_cells(
        generated_plan,
        model_aliases=model_aliases,
    )
    extra_in_csv = sorted(expected_cells - generated_cells)
    missing_from_csv = sorted(generated_cells - expected_cells)
    failures: list[str] = []
    if extra_in_csv:
        failures.append(f"missing_plan_extra_in_csv:{len(extra_in_csv)}")
    if missing_from_csv:
        failures.append(f"missing_plan_missing_from_csv:{len(missing_from_csv)}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "expected_cells": len(expected_cells),
        "generated_cells": len(generated_cells),
        "extra_in_csv": [_plan_cell_label(cell) for cell in extra_in_csv],
        "missing_from_csv": [_plan_cell_label(cell) for cell in missing_from_csv],
        "failures": failures,
    }


def _cell_label(row: dict[str, str]) -> str:
    return f"{row['benchmark']}/{row['model']}"


def _is_complete_curated_record(record: dict[str, Any]) -> bool:
    try:
        score = float(record.get("score"))
    except (TypeError, ValueError):
        return False
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        return False
    if str(record.get("scale") or "").strip() != "0-1":
        return False
    metric = str(record.get("metric") or "").strip()
    if not metric or _looks_placeholder(metric):
        return False
    source = record.get("source")
    if not isinstance(source, dict):
        return False
    title = str(source.get("title") or "").strip()
    url = str(source.get("url") or "").strip()
    if not title or not url or _looks_placeholder(title):
        return False
    try:
        _validate_source_url(url)
    except ValueError:
        return False
    try:
        if source.get("date"):
            _validate_source_date(source.get("date"), field="date")
        elif source.get("retrieved_at"):
            _validate_source_date(source.get("retrieved_at"), field="retrieved_at")
        else:
            return False
    except ValueError:
        return False
    return True


def curated_backfill_progress(
    path: str | Path,
    expected: dict[tuple[str, str], dict[str, str]],
    *,
    model_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a lax progress report for partial score curation.

    Unlike `load_curated_score_backfill`, this intentionally accepts TODO
    skeleton records so humans can see which planned cells remain incomplete.
    It must not be used as an experiment data source.
    """

    progress_path = Path(path)
    if not progress_path.exists():
        return {
            "path": str(progress_path),
            "expected_cells": len(expected),
            "present_cells": 0,
            "complete_cells": 0,
            "incomplete_cells": 0,
            "missing_cells": len(expected),
            "outside_plan": [],
            "duplicate_cells": [],
            "incomplete": [],
            "missing": [_cell_label(row) for row in expected.values()],
        }
    with open(progress_path, encoding="utf-8") as f:
        raw = json.load(f)
    records = raw.get("scores") if isinstance(raw, dict) else None
    if not isinstance(records, list):
        raise ValueError("curated score backfill must contain a 'scores' list")

    states: dict[tuple[str, str], bool] = {}
    outside: list[str] = []
    duplicates: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        benchmark = str(record.get("benchmark") or "").strip()
        model = normalize_model_name(
            str(record.get("model") or "").strip(),
            model_aliases,
        )
        if not benchmark or not model:
            continue
        key = (name_key(benchmark), model)
        label = f"{benchmark}/{model}"
        if key not in expected:
            outside.append(label)
            continue
        complete = _is_complete_curated_record(record)
        if key in states:
            duplicates.append(_cell_label(expected[key]))
            states[key] = states[key] and complete
        else:
            states[key] = complete

    incomplete = [
        _cell_label(expected[key])
        for key, complete in states.items()
        if not complete
    ]
    missing = [
        _cell_label(row)
        for key, row in expected.items()
        if key not in states
    ]
    return {
        "path": str(progress_path),
        "expected_cells": len(expected),
        "present_cells": len(states),
        "complete_cells": sum(1 for complete in states.values() if complete),
        "incomplete_cells": len(incomplete),
        "missing_cells": len(missing),
        "outside_plan": sorted(outside),
        "duplicate_cells": sorted(set(duplicates)),
        "incomplete": sorted(incomplete),
        "missing": missing,
    }


def _project_scores_with_missing_plan(
    scores: dict[str, dict[str, float]],
    expected: dict[tuple[str, str], dict[str, str]],
) -> dict[str, dict[str, float]]:
    """Add planned missing cells as presence-only dummy scores.

    The dummy values are not used as evidence. Projection consumers only read
    common-model coverage counts, which depend on cell presence, not score
    magnitude.
    """

    projected = {
        benchmark: dict(row)
        for benchmark, row in scores.items()
    }
    by_key = {name_key(benchmark): benchmark for benchmark in projected}
    for row in expected.values():
        target = by_key.get(name_key(row["benchmark"]))
        if target is None:
            continue
        projected.setdefault(target, {})
        projected[target].setdefault(row["model"], 0.5)
    return projected


def find_model_seed_for_seen_core(
    model_names: list[str],
    *,
    model_ratio: tuple[float, float],
    seen_core: list[str],
    max_seed: int = 100_000,
) -> int | None:
    target = set(seen_core)
    for seed in range(max(0, int(max_seed))):
        split = split_models(model_names, ratios=model_ratio, seed=seed)
        if set(split.seen) == target:
            return seed
    return None


def _coverage_projection_for_config(
    *,
    config: dict[str, Any],
    expected_missing: dict[tuple[str, str], dict[str, str]],
    include_hf_report: bool,
    model_seed_override: int | None = None,
) -> dict[str, Any]:
    """Project whether the planned missing cells are coverage-sufficient.

    This is intentionally missingness-only: it answers whether all planned
    cells, if filled with real values, provide enough common-model support for
    the configured split contract. It does not claim final Spearman pairs are
    non-degenerate.
    """

    base_scores, sources = _load_scores_from_config(
        include_hf_report=include_hf_report,
        curated_score_backfill_path=config.get("curated_score_backfill_path"),
        use_curated_score_backfill=False,
    )
    projected_scores = _project_scores_with_missing_plan(base_scores, expected_missing)
    exp_config = load_experiment_config(_build_v3_overrides(config))
    splits_cfg = dict(exp_config.get("splits", {}) or {})
    if model_seed_override is not None:
        splits_cfg["model_seed"] = int(model_seed_override)
        exp_config["splits"] = splits_cfg

    benchmark_names = sorted(projected_scores)
    model_names = sorted({
        model
        for row in projected_scores.values()
        for model in row
    })
    min_common = int(exp_config.get("min_common_models", 6))
    thresholds = {
        "train": int(exp_config.get("v_loop_min_train_valid_pairs", 1)),
        "dev": int(exp_config.get("v_loop_min_dev_valid_pairs", 1)),
        "test": int(exp_config.get("v_loop_min_test_valid_pairs", 1)),
    }
    model_ratios = tuple(splits_cfg.get("model_ratios", (0.8, 0.2)))
    model_seed = int(splits_cfg.get("model_seed", 0))
    model_split = split_models(model_names, ratios=model_ratios, seed=model_seed)
    score_model_scope = str(exp_config.get("v_loop_score_model_scope", "all")).strip().lower()
    if bool(exp_config.get("v_loop_require_held_model_test", False)):
        score_model_scope = "seen"
    if score_model_scope not in {"all", "seen"}:
        raise ValueError(
            "v_loop_score_model_scope must be one of {'all', 'seen'}, "
            f"got {score_model_scope!r}"
        )

    score_scores = projected_scores
    if score_model_scope == "seen":
        seen = set(model_split.seen)
        score_scores = {
            benchmark: {
                model: score
                for model, score in row.items()
                if model in seen
            }
            for benchmark, row in projected_scores.items()
        }
    _, common_score = spearman_pair_matrix(
        normalize_matrix(score_scores, method=exp_config.get("normalize", "rank")),
        benchmark_names,
        min_common=min_common,
        warn_below=10**9,
    )

    held_common = {}
    held_failures: list[str] = []
    if len(model_split.held) < min_common:
        held_failures.append(f"held_models:{len(model_split.held)}<{min_common}")
    else:
        held = set(model_split.held)
        held_scores = {
            benchmark: {
                model: score
                for model, score in row.items()
                if model in held
            }
            for benchmark, row in projected_scores.items()
        }
        _, held_common = spearman_pair_matrix(
            normalize_matrix(held_scores, method=exp_config.get("normalize", "rank")),
            benchmark_names,
            min_common=min_common,
            warn_below=10**9,
        )

    cv_folds = int(splits_cfg.get("cv_folds", 1))
    fold_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    held_test_coverage: list[int] = []
    if cv_folds > 1:
        for fold in range(cv_folds):
            split = benchmark_split_from_config(
                benchmark_names,
                {**splits_cfg, "fold": fold},
            )
            coverage_counts = split_coverage_pair_counts(
                common_score,
                split,
                min_common=min_common,
            )
            fold_failures = split_pair_count_failures(coverage_counts, thresholds)
            failures.extend(
                f"fold{fold}:coverage_{failure}"
                for failure in fold_failures
            )
            held_count = (
                split_coverage_pair_counts(
                    held_common,
                    split,
                    min_common=min_common,
                )["test"]
                if held_common
                else 0
            )
            held_test_coverage.append(int(held_count))
            if held_count < thresholds["test"]:
                held_failures.append(
                    f"fold{fold}:held_test_coverage:{held_count}<{thresholds['test']}"
                )
            fold_rows.append({
                "fold": fold,
                "coverage_pairs": coverage_counts,
                "held_test_coverage_pairs": int(held_count),
            })

    min_coverage = {
        name: min((row["coverage_pairs"][name] for row in fold_rows), default=0)
        for name in ("train", "dev", "test")
    }
    all_failures = failures + held_failures
    return {
        "status": "PASS" if not all_failures else "FAIL",
        "sources": sources,
        "model_seed": model_seed,
        "model_ratio": list(model_ratios),
        "model_seen": model_split.seen,
        "model_held": model_split.held,
        "score_model_scope": score_model_scope,
        "thresholds": thresholds,
        "min_common": min_common,
        "min_coverage_pairs": min_coverage,
        "held_test_coverage_pairs": held_test_coverage,
        "failures": all_failures,
        "folds": fold_rows,
    }


def _planned_backfill_projection(
    *,
    config: dict[str, Any],
    expected_missing: dict[tuple[str, str], dict[str, str]],
    best_plan: dict[str, Any],
    include_hf_report: bool,
) -> dict[str, Any]:
    configured = _coverage_projection_for_config(
        config=config,
        expected_missing=expected_missing,
        include_hf_report=include_hf_report,
    )
    recommended_seed = None
    recommended = None
    if "error" not in best_plan:
        projected_scores, _ = _load_scores_from_config(
            include_hf_report=include_hf_report,
            curated_score_backfill_path=config.get("curated_score_backfill_path"),
            use_curated_score_backfill=False,
        )
        projected_scores = _project_scores_with_missing_plan(
            projected_scores,
            expected_missing,
        )
        model_names = sorted({
            model
            for row in projected_scores.values()
            for model in row
        })
        splits_cfg = (config.get("splits") or {})
        model_ratio = tuple(splits_cfg.get("model_ratios", (0.8, 0.2)))
        recommended_seed = find_model_seed_for_seen_core(
            model_names,
            model_ratio=model_ratio,
            seen_core=list(best_plan.get("seen_core") or []),
        )
        if recommended_seed is not None:
            recommended = _coverage_projection_for_config(
                config=config,
                expected_missing=expected_missing,
                include_hf_report=include_hf_report,
                model_seed_override=recommended_seed,
            )
    return {
        "configured": configured,
        "recommended_model_seed": recommended_seed,
        "recommended": recommended,
    }


def planned_projection_failures(projection: dict[str, Any]) -> list[str]:
    configured = projection.get("configured")
    if not isinstance(configured, dict):
        return ["planned_projection_configured_missing"]
    if configured.get("status") == "PASS":
        return []
    failures = ["planned_projection_configured_failed"]
    recommended = projection.get("recommended")
    recommended_seed = projection.get("recommended_model_seed")
    if isinstance(recommended, dict) and recommended.get("status") == "PASS":
        failures.append(f"planned_projection_recommended_model_seed:{recommended_seed}")
    else:
        failures.append("planned_projection_no_passing_recommendation")
    for failure in (configured.get("failures") or [])[:5]:
        failures.append(f"configured:{failure}")
    return failures


def _current_strict_diagnostics(config: dict[str, Any]) -> dict[str, Any]:
    exp_config = load_experiment_config(_build_v3_overrides(config))
    splits_cfg = exp_config.get("splits", {}) or {}
    corpus = load_corpus(config)
    benchmark_names = corpus.benchmark_names
    model_names = corpus.model_names
    normalize = exp_config.get("normalize", "rank")
    Y_norm_full = normalize_matrix(corpus.Y, method=normalize)

    model_ratios = tuple(splits_cfg.get("model_ratios", (0.8, 0.2)))
    model_seed = int(splits_cfg.get("model_seed", 0))
    model_split = split_models(model_names, ratios=model_ratios, seed=model_seed)

    score_model_scope = str(exp_config.get("v_loop_score_model_scope", "all")).strip().lower()
    if bool(exp_config.get("v_loop_require_held_model_test", False)):
        score_model_scope = "seen"
    if score_model_scope not in {"all", "seen"}:
        raise ValueError(
            "v_loop_score_model_scope must be one of {'all', 'seen'}, "
            f"got {score_model_scope!r}"
        )

    score_Y = corpus.Y
    if score_model_scope == "seen":
        seen = set(model_split.seen)
        score_Y = {
            benchmark: {
                model: score
                for model, score in scores.items()
                if model in seen
            }
            for benchmark, scores in corpus.Y.items()
        }
    Y_norm_score = normalize_matrix(score_Y, method=normalize)
    R_raw, common_count = spearman_pair_matrix(
        Y_norm_score,
        benchmark_names,
        min_common=int(exp_config.get("min_common_models", 6)),
        warn_below=10**9,
    )
    comparable_pairs = sum(1 for value in R_raw.values() if value is not None)

    thresholds = {
        "train": int(exp_config.get("v_loop_min_train_valid_pairs", 1)),
        "dev": int(exp_config.get("v_loop_min_dev_valid_pairs", 1)),
        "test": int(exp_config.get("v_loop_min_test_valid_pairs", 1)),
    }
    cv_folds = int(splits_cfg.get("cv_folds", 1))
    benchmark_seed = int(splits_cfg.get("benchmark_seed", 0))
    stratified = bool(splits_cfg.get("stratified", False))
    dev_train_split = tuple(splits_cfg.get("dev_train_split", (0.25, 0.75)))
    min_common = int(exp_config.get("min_common_models", 6))

    benchmark_diag = fold_diagnostics(
        benchmark_names=benchmark_names,
        R_raw=R_raw,
        common_count=common_count,
        cv_folds=cv_folds,
        seed=benchmark_seed,
        stratified=stratified,
        dev_train_split=dev_train_split,
        thresholds=thresholds,
        min_common=min_common,
    )
    held_diag = model_fold_diagnostics(
        benchmark_names=benchmark_names,
        model_names=model_names,
        Y_norm=Y_norm_full,
        cv_folds=cv_folds,
        benchmark_seed=benchmark_seed,
        stratified=stratified,
        dev_train_split=dev_train_split,
        model_ratio=model_ratios,
        model_seed=model_seed,
        min_common=min_common,
        min_held_test_pairs=thresholds["test"],
    )
    return {
        "status": (
            "PASS"
            if benchmark_diag["status"] == "PASS" and held_diag["status"] == "PASS"
            else "FAIL"
        ),
        "score_model_scope": score_model_scope,
        "benchmarks": len(benchmark_names),
        "models": len(model_names),
        "comparable_pairs": comparable_pairs,
        "thresholds": thresholds,
        "model_split": {
            "seen": len(model_split.seen),
            "held": len(model_split.held),
            "ratios": list(model_ratios),
            "seed": model_seed,
        },
        "benchmark_splits": benchmark_diag,
        "held_model_test": held_diag,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Curated score backfill JSON path.")
    parser.add_argument(
        "--missing-csv",
        default=str(DEFAULT_MISSING_CSV),
        help="Allowed missing-cell CSV plan. Defaults to data/score_backfill_missing.csv.",
    )
    parser.add_argument("--include-hf-report", action="store_true")
    parser.add_argument("--core-size", type=int, default=6)
    parser.add_argument(
        "--progress",
        action="store_true",
        help=(
            "Lax progress report for partial TODO files. Does not validate "
            "the file for experiment use."
        ),
    )
    parser.add_argument(
        "--require-projection-pass",
        action="store_true",
        help=(
            "Fail if the committed missing-cell plan is not coverage-sufficient "
            "for the currently configured model split."
        ),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = load_config()
    path = args.path or config.get("curated_score_backfill_path")
    if not path:
        raise SystemExit("curated_score_backfill_path is not configured")
    config["curated_score_backfill_path"] = path
    config["use_curated_score_backfill"] = True

    try:
        expected_missing = load_missing_cell_plan(
            args.missing_csv,
            model_aliases=config.get("model_aliases") or {},
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"missing-cell plan validation failed: {exc}") from exc

    if args.progress:
        try:
            progress = curated_backfill_progress(
                path,
                expected_missing,
                model_aliases=config.get("model_aliases") or {},
            )
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"curated score backfill progress failed: {exc}") from exc
        projection_scores, _projection_sources = _load_scores_from_config(
            include_hf_report=args.include_hf_report,
            curated_score_backfill_path=path,
            use_curated_score_backfill=False,
        )
        projection_plans = _top_core_plans(
            projection_scores,
            core_size=max(1, int(args.core_size)),
            top_k=1,
        )
        projection_best_plan = (
            projection_plans[0]
            if projection_plans
            else {"error": "no score plan generated"}
        )
        plan_drift = missing_plan_drift(
            expected_missing,
            projection_best_plan,
            model_aliases=config.get("model_aliases") or {},
        )
        planned_projection = _planned_backfill_projection(
            config=config,
            expected_missing=expected_missing,
            best_plan=projection_best_plan,
            include_hf_report=args.include_hf_report,
        )
        projection_failures = planned_projection_failures(planned_projection)
        if plan_drift.get("status") != "PASS":
            projection_failures.extend(plan_drift.get("failures") or [])
        progress["missing_plan_drift"] = plan_drift
        progress["planned_backfill_projection"] = planned_projection
        progress["planned_projection_failures"] = projection_failures
        if args.json:
            print(json.dumps(progress, indent=2, sort_keys=True))
        else:
            print(
                "[score-backfill-progress] "
                f"path={progress['path']}, "
                f"expected={progress['expected_cells']}, "
                f"present={progress['present_cells']}, "
                f"complete={progress['complete_cells']}, "
                f"incomplete={progress['incomplete_cells']}, "
                f"missing={progress['missing_cells']}, "
                f"outside={len(progress['outside_plan'])}, "
                f"duplicates={len(progress['duplicate_cells'])}"
            )
            if progress["incomplete"]:
                print(f"  incomplete={progress['incomplete'][:8]}")
            if progress["missing"]:
                print(f"  missing={progress['missing'][:8]}")
            if progress["outside_plan"]:
                print(f"  outside_plan={progress['outside_plan'][:8]}")
            if progress["duplicate_cells"]:
                print(f"  duplicate_cells={progress['duplicate_cells'][:8]}")
            print(
                "  missing_plan_drift="
                f"{plan_drift['status']} "
                f"expected={plan_drift['expected_cells']} "
                f"generated={plan_drift['generated_cells']}"
            )
            if plan_drift.get("failures"):
                print(f"  missing_plan_drift_failures={plan_drift['failures'][:8]}")
            configured = planned_projection["configured"]
            print(
                "  planned_projection_configured="
                f"{configured['status']} "
                f"model_seed={configured['model_seed']} "
                f"min_coverage={configured['min_coverage_pairs']} "
                f"held_coverage={configured['held_test_coverage_pairs']}"
            )
            recommended = planned_projection.get("recommended")
            if recommended is not None:
                print(
                    "  planned_projection_recommended="
                    f"{recommended['status']} "
                    f"model_seed={planned_projection['recommended_model_seed']} "
                    f"min_coverage={recommended['min_coverage_pairs']} "
                    f"held_coverage={recommended['held_test_coverage_pairs']}"
                )
            if projection_failures:
                print(f"  planned_projection_failures={projection_failures[:8]}")
        if args.require_projection_pass and projection_failures:
            return 2
        return 0

    try:
        curated = load_curated_score_backfill(
            path,
            exclude=config.get("exclude", []),
            model_aliases=config.get("model_aliases") or {},
            require_exists=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            f"curated score backfill file not found: {exc}. "
            "Create data/curated_score_backfill.json from "
            "data/curated_score_backfill.example.json."
        ) from exc
    except ValueError as exc:
        raise SystemExit(f"curated score backfill validation failed: {exc}") from exc
    outside_plan = curated_cells_outside_plan(curated, expected_missing)
    if outside_plan:
        preview = ", ".join(outside_plan[:8])
        suffix = "" if len(outside_plan) <= 8 else f", +{len(outside_plan) - 8} more"
        raise SystemExit(
            "curated score backfill contains cells outside the committed "
            f"missing-cell plan {args.missing_csv}: {preview}{suffix}"
        )
    scores, sources = _load_scores_from_config(
        include_hf_report=args.include_hf_report,
        curated_score_backfill_path=path,
        use_curated_score_backfill=True,
    )
    plans = _top_core_plans(
        scores,
        core_size=max(1, int(args.core_size)),
        top_k=1,
    )
    best_plan = plans[0] if plans else {"error": "no score plan generated"}
    base_scores, _base_sources = _load_scores_from_config(
        include_hf_report=args.include_hf_report,
        curated_score_backfill_path=path,
        use_curated_score_backfill=False,
    )
    base_plans = _top_core_plans(
        base_scores,
        core_size=max(1, int(args.core_size)),
        top_k=1,
    )
    base_best_plan = (
        base_plans[0]
        if base_plans
        else {"error": "no score plan generated before curated backfill"}
    )
    plan_drift = missing_plan_drift(
        expected_missing,
        base_best_plan,
        model_aliases=config.get("model_aliases") or {},
    )
    strict = _current_strict_diagnostics(config)
    planned_projection = _planned_backfill_projection(
        config=config,
        expected_missing=expected_missing,
        best_plan=base_best_plan,
        include_hf_report=args.include_hf_report,
    )
    projection_failures = planned_projection_failures(planned_projection)
    if plan_drift.get("status") != "PASS":
        projection_failures.extend(plan_drift.get("failures") or [])
    payload = {
        "path": str(Path(path)),
        "sources": sources,
        "missing_csv": str(Path(args.missing_csv)),
        "expected_missing_cells": len(expected_missing),
        "curated_benchmarks": len(curated),
        "curated_cells": sum(len(row) for row in curated.values()),
        "best_core_plan": (
            _jsonable_plan(best_plan) if "error" not in best_plan else best_plan
        ),
        "planned_missing_core_plan": (
            _jsonable_plan(base_best_plan)
            if "error" not in base_best_plan
            else base_best_plan
        ),
        "missing_plan_drift": plan_drift,
        "planned_backfill_projection": planned_projection,
        "planned_projection_failures": projection_failures,
        "strict_preflight": strict,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "[score-backfill-validate] "
            f"path={payload['path']}, "
            f"missing_csv={payload['missing_csv']}, "
            f"expected_missing_cells={payload['expected_missing_cells']}, "
            f"curated_cells={payload['curated_cells']}, "
            f"sources={sources}"
        )
        if "error" in best_plan:
            print(f"  plan=ERROR {best_plan['error']}")
        else:
            print(
                "  plan="
                f"total_missing={best_plan['total_missing_count']} "
                f"seen_missing={best_plan['missing_seen_count']} "
                f"held_missing={best_plan['missing_held_count']}"
            )
        print(
            "  missing_plan_drift="
            f"{plan_drift['status']} "
            f"expected={plan_drift['expected_cells']} "
            f"generated={plan_drift['generated_cells']}"
        )
        if plan_drift.get("failures"):
            print(f"  missing_plan_drift_failures={plan_drift['failures'][:8]}")
        print(
            "  strict_preflight="
            f"{strict['status']} "
            f"pairs={strict['comparable_pairs']} "
            f"model_split={strict['model_split']['seen']}/"
            f"{strict['model_split']['held']}"
        )
        bench_failures = strict["benchmark_splits"].get("failures", [])
        held_failures = strict["held_model_test"].get("failures", [])
        if bench_failures:
            print(f"  benchmark_failures={bench_failures[:8]}")
        if held_failures:
            print(f"  held_failures={held_failures[:8]}")
        configured = planned_projection["configured"]
        print(
            "  planned_projection_configured="
            f"{configured['status']} "
            f"model_seed={configured['model_seed']} "
            f"min_coverage={configured['min_coverage_pairs']} "
            f"held_coverage={configured['held_test_coverage_pairs']}"
        )
        recommended = planned_projection.get("recommended")
        if recommended is not None:
            print(
                "  planned_projection_recommended="
                f"{recommended['status']} "
                f"model_seed={planned_projection['recommended_model_seed']} "
                f"min_coverage={recommended['min_coverage_pairs']} "
                f"held_coverage={recommended['held_test_coverage_pairs']}"
            )
        if projection_failures:
            print(f"  planned_projection_failures={projection_failures[:8]}")
    ok_plan = "error" not in best_plan and int(best_plan["total_missing_count"]) == 0
    if args.require_projection_pass and projection_failures:
        return 2
    return 0 if ok_plan and strict["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
