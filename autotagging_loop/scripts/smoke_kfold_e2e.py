"""End-to-end smoke for the v3 K-fold pipeline.

Loads `benchpress_config.json` and calls `run_part2` with `max_iter=3` override.
Verifies the 4 fold subdirs + `agg/permutation_test.json` are written under the
parent `cv_<UTC>` run dir.

Triggers ~12 real LLM calls per fold (3 iters × ~4 role-calls). Mapper cache
should hit when re-running. Total wall ≈ 8–10 min on the self-hosted server.

Usage:
    python scripts/smoke_kfold_e2e.py [--max-iter 3]
    python scripts/smoke_kfold_e2e.py --max-iter 10 --cv-folds 2 --require-significant
    python scripts/smoke_kfold_e2e.py --max-iter 1 --cv-folds 2 --benchmark-seed 7 --dev-train-split 1:1 --min-common-models 5 --train-pairs 1 --dev-pairs 1 --test-pairs 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autotagging_loop.runner.config import load_config
from autotagging_loop.runner.run import run_part2


def parse_pair_ratio(text: str) -> list[float]:
    raw = str(text).strip().replace(",", ":")
    parts = raw.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"expected ratio like 1:1, got {text!r}"
        )
    try:
        left = float(parts[0])
        right = float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"ratio values must be numeric, got {text!r}"
        ) from exc
    if left < 0.0 or right < 0.0:
        raise argparse.ArgumentTypeError(
            f"ratio values must be non-negative, got {text!r}"
        )
    return [left, right]


def parse_name_list(values: list[str] | None) -> list[str] | None:
    names: list[str] = []
    for value in values or []:
        for item in str(value).split(","):
            name = item.strip()
            if name:
                names.append(name)
    return names or None


def quality_gate_failures(
    agg: dict,
    *,
    alpha: float,
    min_rho_s: float,
) -> list[str]:
    pooled = (agg or {}).get("pooled") or {}
    rho = pooled.get("rho_spearman") or {}
    failures: list[str] = []
    try:
        observed = float(rho["observed"])
        p_two = float(rho["p_two_sided"])
        n_pairs = int(pooled["n_pairs"])
    except (KeyError, TypeError, ValueError):
        return ["missing_pooled_rho_spearman"]

    if not math.isfinite(observed):
        failures.append("pooled_rho_s_not_finite")
    elif observed < float(min_rho_s):
        failures.append(f"pooled_rho_s_below_floor:{observed:.4f}<{float(min_rho_s):.4f}")

    if not math.isfinite(p_two):
        failures.append("pooled_p_two_not_finite")
    elif p_two > float(alpha):
        failures.append(f"pooled_p_two_above_alpha:{p_two:.4f}>{float(alpha):.4f}")

    if n_pairs <= 0:
        failures.append("pooled_n_pairs_empty")
    return failures


def _float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _selection_split_metric_failures(
    *,
    fold_name: str,
    selection: dict,
    split_block: dict,
    tolerance: float = 1e-9,
) -> list[str]:
    failures: list[str] = []
    for key in (
        "L_align",
        "L_align_01",
        "rho_align_pearson",
        "rho_align_spearman",
        "delta_tag",
    ):
        sel_v = _float_or_nan(selection.get(key))
        split_v = _float_or_nan(split_block.get(key))
        if not math.isfinite(sel_v) and not math.isfinite(split_v):
            continue
        if not math.isfinite(sel_v) or not math.isfinite(split_v):
            failures.append(f"{fold_name}_selection_split_{key}_finite_mismatch")
            continue
        if abs(sel_v - split_v) > tolerance:
            failures.append(
                f"{fold_name}_selection_split_{key}_mismatch:"
                f"{sel_v:.6g}!={split_v:.6g}"
            )
    return failures


def _metrics_path_for_scope(iter_dir: Path, scope: str | None) -> Path:
    if scope in {"train", "dev", "test"}:
        scoped = iter_dir / f"metrics_{scope}.json"
        if scoped.exists():
            return scoped
    return iter_dir / "metrics.json"


def _selected_iteration_gate_failures(
    *,
    fold_dir: Path,
    selected_label: str,
    selection_scope: str | None,
    min_delta_tag: float | None,
) -> list[str]:
    if min_delta_tag is None:
        return []
    if not selected_label:
        return [f"{fold_dir.name}_selected_iter_missing"]
    iter_dir = fold_dir / selected_label
    metrics_path = _metrics_path_for_scope(iter_dir, selection_scope)
    if not metrics_path.exists():
        return [f"{fold_dir.name}_selected_iter_metrics_missing:{selected_label}"]
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    delta = _float_or_nan(metrics.get("delta_tag"))
    if not math.isfinite(delta):
        return [f"{fold_dir.name}_selected_iter_delta_tag_not_finite:{selected_label}"]
    if delta <= float(min_delta_tag):
        return [
            f"{fold_dir.name}_selected_iter_delta_tag_below_gate:"
            f"{delta:.4f}<={float(min_delta_tag):.4f}:{selected_label}"
        ]
    return []


def fold_summary_quality_failures(
    fold_summaries: list[dict],
    *,
    min_delta_tag: float | None = None,
) -> list[str]:
    failures: list[str] = []
    for fold in fold_summaries or []:
        fold_id = fold.get("fold", "?")
        l_align = _float_or_nan(fold.get("L_align"))
        if not math.isfinite(l_align):
            failures.append(f"fold{fold_id}_selection_L_align_not_finite")
        if not fold.get("best_label"):
            failures.append(f"fold{fold_id}_best_label_missing")
        if min_delta_tag is not None:
            delta = _float_or_nan(fold.get("delta_tag"))
            if not math.isfinite(delta):
                failures.append(f"fold{fold_id}_selection_delta_tag_not_finite")
            elif delta <= float(min_delta_tag):
                failures.append(
                    f"fold{fold_id}_selection_delta_tag_below_gate:"
                    f"{delta:.4f}<={float(min_delta_tag):.4f}"
                )
    return failures


def parent_run_quality_failures(
    parent_run_dir: str,
    *,
    min_delta_tag: float | None = None,
    require_held_model_test: bool = False,
) -> list[str]:
    parent = Path(parent_run_dir)
    failures: list[str] = []
    fold_dirs = sorted(p for p in parent.glob("fold*") if p.is_dir())
    if not fold_dirs:
        return [f"missing_fold_dirs:{parent}"]
    for fold_dir in fold_dirs:
        metrics_path = fold_dir / "final" / "metrics_with_bootstrap.json"
        if not metrics_path.exists():
            failures.append(f"{fold_dir.name}_missing_final_metrics")
            continue
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        selection = metrics.get("selection") or metrics
        l_align = _float_or_nan(selection.get("L_align"))
        if not math.isfinite(l_align):
            failures.append(f"{fold_dir.name}_selection_L_align_not_finite")
        if "n_pairs" in selection and int(selection.get("n_pairs") or 0) <= 0:
            failures.append(f"{fold_dir.name}_selection_n_pairs_empty")
        if min_delta_tag is not None:
            delta = _float_or_nan(selection.get("delta_tag"))
            if not math.isfinite(delta):
                failures.append(f"{fold_dir.name}_selection_delta_tag_not_finite")
            elif delta <= float(min_delta_tag):
                failures.append(
                    f"{fold_dir.name}_selection_delta_tag_below_gate:"
                    f"{delta:.4f}<={float(min_delta_tag):.4f}"
                )
        selection_scope = metrics.get("selection_scope")
        best_path = fold_dir / "final" / "best_iter.txt"
        if best_path.exists() or selection_scope in {"train", "dev", "test"}:
            selected_label = best_path.read_text(encoding="utf-8").strip() if best_path.exists() else ""
            failures.extend(
                _selected_iteration_gate_failures(
                    fold_dir=fold_dir,
                    selected_label=selected_label,
                    selection_scope=selection_scope,
                    min_delta_tag=min_delta_tag,
                )
            )
        if selection_scope in {"train", "dev", "test"}:
            split_path = fold_dir / "final" / "split_metrics.json"
            if not split_path.exists():
                failures.append(f"{fold_dir.name}_missing_final_split_metrics")
                continue
            with open(split_path, encoding="utf-8") as f:
                split_metrics = json.load(f)
            split_block = split_metrics.get(selection_scope) or {}
            failures.extend(
                _selection_split_metric_failures(
                    fold_name=fold_dir.name,
                    selection=selection,
                    split_block=split_block,
                )
            )
            if require_held_model_test:
                held = split_metrics.get("held_model_test") or {}
                if not held:
                    failures.append(f"{fold_dir.name}_held_model_test_missing")
                elif held.get("skipped"):
                    failures.append(
                        f"{fold_dir.name}_held_model_test_skipped:{held.get('skipped')}"
                    )
                elif int(held.get("n_pairs") or 0) <= 0:
                    failures.append(f"{fold_dir.name}_held_model_test_n_pairs_empty")
                elif not math.isfinite(_float_or_nan(held.get("rho_align_spearman"))):
                    failures.append(f"{fold_dir.name}_held_model_test_rho_s_not_finite")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument(
        "--cv-folds",
        type=int,
        help="Override part2_experiment.splits.cv_folds for this smoke run.",
    )
    parser.add_argument(
        "--split-pairs",
        type=int,
        help=(
            "Override v_loop_min_train/dev/test_valid_pairs for this smoke run. "
            "Use 0 only for backend exercise; it is not research evidence."
        ),
    )
    parser.add_argument(
        "--train-pairs",
        type=int,
        help="Override v_loop_min_train_valid_pairs for this smoke run.",
    )
    parser.add_argument(
        "--dev-pairs",
        type=int,
        help="Override v_loop_min_dev_valid_pairs for this smoke run.",
    )
    parser.add_argument(
        "--test-pairs",
        type=int,
        help="Override v_loop_min_test_valid_pairs for this smoke run.",
    )
    parser.add_argument(
        "--min-common-models",
        type=int,
        help="Override min_common_models for this smoke run.",
    )
    parser.add_argument(
        "--benchmark-seed",
        type=int,
        help="Override part2_experiment.splits.benchmark_seed for this smoke run.",
    )
    parser.add_argument(
        "--dev-train-split",
        type=parse_pair_ratio,
        help="Override part2_experiment.splits.dev_train_split, e.g. 1:1.",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        help="Override part2_experiment.splits.model_seed for this smoke run.",
    )
    parser.add_argument(
        "--model-ratio",
        type=parse_pair_ratio,
        help="Override part2_experiment.splits.model_ratios, e.g. 1:1.",
    )
    parser.add_argument(
        "--stratified",
        dest="stratified",
        action="store_true",
        default=None,
        help="Force stratified benchmark folds for this smoke run.",
    )
    parser.add_argument(
        "--plain",
        dest="stratified",
        action="store_false",
        help="Force plain benchmark folds for this smoke run.",
    )
    parser.add_argument(
        "--include-benchmark",
        action="append",
        help="Restrict to benchmark names. Repeat or pass comma-separated names.",
    )
    parser.add_argument(
        "--include-model",
        action="append",
        help="Restrict to model names. Repeat or pass comma-separated names.",
    )
    parser.add_argument(
        "--exclude-model",
        action="append",
        help="Drop model names. Repeat or pass comma-separated names.",
    )
    parser.add_argument(
        "--no-seed-taxonomy",
        action="store_true",
        help="Induce a no-seed taxonomy before the v-loop for diagnostic runs.",
    )
    parser.add_argument(
        "--no-seed-min-tags",
        type=int,
        help="Override no_seed_taxonomy_min_tags for this smoke run.",
    )
    parser.add_argument(
        "--no-seed-max-tags",
        type=int,
        help="Override no_seed_taxonomy_max_tags for this smoke run.",
    )
    parser.add_argument(
        "--no-seed-max-attempts",
        type=int,
        help="Override no_seed_taxonomy_max_attempts for this smoke run.",
    )
    parser.add_argument(
        "--llm-request-timeout-s",
        type=float,
        help="Override per-request LLM timeout seconds for this smoke run.",
    )
    parser.add_argument(
        "--llm-sdk-exception-retries",
        type=int,
        help="Override retries after SDK timeout/read exceptions for this smoke run.",
    )
    parser.add_argument(
        "--agg-path",
        help="Validate an existing agg/permutation_test.json instead of running a new smoke.",
    )
    parser.add_argument(
        "--parent-run-dir",
        help="Validate an existing k-fold parent run dir, including fold final selection metrics.",
    )
    parser.add_argument(
        "--require-significant",
        action="store_true",
        help="Fail unless pooled rho_s is positive enough and p_two <= alpha.",
    )
    parser.add_argument(
        "--require-held-model-test",
        action="store_true",
        help="Fail unless every fold has a non-skipped held-model test block.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-rho-s", type=float, default=0.20)
    parser.add_argument(
        "--min-fold-delta-tag",
        type=float,
        default=-0.10,
        help="Reject fold selections whose delta_tag is <= this gate.",
    )
    parser.add_argument(
        "--delta-tag-threshold",
        type=float,
        help="Override experiment delta_tag_threshold for diagnostic smoke runs.",
    )
    args = parser.parse_args()

    if args.parent_run_dir and not args.agg_path:
        args.agg_path = os.path.join(args.parent_run_dir, "agg", "permutation_test.json")

    if args.agg_path:
        with open(args.agg_path, encoding="utf-8") as f:
            agg = json.load(f)
        pooled = (agg or {}).get("pooled") or {}
        rho = pooled.get("rho_spearman") or {}
        observed = _float_or_nan(rho.get("observed"))
        p_two = _float_or_nan(rho.get("p_two_sided"))
        print(
            f"[smoke-e2e] existing agg: n_pairs={pooled.get('n_pairs')}, "
            f"rho_s={observed:+.4f}, "
            f"p_two={p_two:.4f}"
        )
        failures: list[str] = []
        warnings = quality_gate_failures(
            agg,
            alpha=args.alpha,
            min_rho_s=args.min_rho_s,
        )
        if args.require_significant:
            failures.extend(warnings)
            warnings = []
        if args.parent_run_dir:
            parent_failures = parent_run_quality_failures(
                args.parent_run_dir,
                min_delta_tag=(
                    args.min_fold_delta_tag if args.require_significant else None
                ),
                require_held_model_test=args.require_held_model_test,
            )
            if args.require_significant or args.require_held_model_test:
                failures.extend(parent_failures)
            else:
                warnings.extend(parent_failures)
        if failures:
            print(
                "[smoke-e2e] FAIL — quality gate failed: "
                + ", ".join(failures),
                file=sys.stderr,
            )
            return 2
        if warnings:
            print("[smoke-e2e] quality gate warnings: " + ", ".join(warnings))
        else:
            print("[smoke-e2e] PASS — pooled quality gate passed")
        return 0

    overrides = {"max_iter": args.max_iter}
    split_overrides: dict = {}
    if args.cv_folds is not None:
        split_overrides["cv_folds"] = args.cv_folds
    if args.benchmark_seed is not None:
        split_overrides["benchmark_seed"] = args.benchmark_seed
    if args.dev_train_split is not None:
        split_overrides["dev_train_split"] = args.dev_train_split
    if args.model_seed is not None:
        split_overrides["model_seed"] = args.model_seed
    if args.model_ratio is not None:
        split_overrides["model_ratios"] = args.model_ratio
    if args.stratified is not None:
        split_overrides["stratified"] = bool(args.stratified)
    if split_overrides:
        overrides["splits"] = split_overrides
    if args.split_pairs is not None:
        split_pairs = int(args.split_pairs)
        overrides.update({
            "v_loop_min_train_valid_pairs": split_pairs,
            "v_loop_min_dev_valid_pairs": split_pairs,
            "v_loop_min_test_valid_pairs": split_pairs,
        })
    if args.train_pairs is not None:
        overrides["v_loop_min_train_valid_pairs"] = int(args.train_pairs)
    if args.dev_pairs is not None:
        overrides["v_loop_min_dev_valid_pairs"] = int(args.dev_pairs)
    if args.test_pairs is not None:
        overrides["v_loop_min_test_valid_pairs"] = int(args.test_pairs)
    if args.min_common_models is not None:
        overrides["min_common_models"] = int(args.min_common_models)
    include_benchmarks = parse_name_list(args.include_benchmark)
    include_models = parse_name_list(args.include_model)
    exclude_models = parse_name_list(args.exclude_model)
    if include_benchmarks is not None:
        overrides["include_benchmarks"] = include_benchmarks
    if include_models is not None:
        overrides["include_models"] = include_models
    if exclude_models is not None:
        overrides["exclude_models"] = exclude_models
    if args.no_seed_taxonomy:
        overrides["no_seed_taxonomy_enabled"] = True
    if args.no_seed_min_tags is not None:
        overrides["no_seed_taxonomy_min_tags"] = int(args.no_seed_min_tags)
    if args.no_seed_max_tags is not None:
        overrides["no_seed_taxonomy_max_tags"] = int(args.no_seed_max_tags)
    if args.no_seed_max_attempts is not None:
        overrides["no_seed_taxonomy_max_attempts"] = int(args.no_seed_max_attempts)
    if args.llm_request_timeout_s is not None:
        overrides["llm_request_timeout_s"] = float(args.llm_request_timeout_s)
    if args.llm_sdk_exception_retries is not None:
        overrides["llm_sdk_exception_retries"] = int(args.llm_sdk_exception_retries)
    if args.delta_tag_threshold is not None:
        overrides["delta_tag_threshold"] = float(args.delta_tag_threshold)
    config = load_config(overrides)
    splits = config.get("splits", {}) or {}
    cv_folds = int(splits.get("cv_folds", 1))
    print(
        f"[smoke-e2e] cv_folds={cv_folds}, max_iter={config['max_iter']}, "
        f"split_pair_thresholds="
        f"{config.get('v_loop_min_train_valid_pairs')}/"
        f"{config.get('v_loop_min_dev_valid_pairs')}/"
        f"{config.get('v_loop_min_test_valid_pairs')}, "
        "best_iter_selection in benchpress_config.json"
    )
    if cv_folds <= 1:
        print("[smoke-e2e] WARN: cv_folds<=1 in config; smoke will fall back to single-split mode", file=sys.stderr)

    result = run_part2(config)
    print()
    print("[smoke-e2e] run_part2 returned:")
    print(f"  mode = {result.get('mode')}")
    if result.get("mode") == "v3_kfold":
        parent = result["parent_run_dir"]
        print(f"  parent_run_dir = {parent}")
        for fs in result["fold_summaries"]:
            print(
                f"  fold{fs['fold']}: iters={fs['iterations']}, "
                f"best={fs['best_label']}, L_align={fs['L_align']:.4f}, "
                f"ρ_p={fs['rho_align_pearson']:.4f}"
            )
        agg = result.get("agg", {})
        if "pooled" in agg:
            p = agg["pooled"]
            if "rho_spearman" in p:
                print(
                    f"  pooled: n_pairs={p['n_pairs']}, "
                    f"ρ_s={p['rho_spearman']['observed']:+.4f}, "
                    f"p_two={p['rho_spearman']['p_two_sided']:.4f}"
                )
        for k in range(int(result["cv_folds"])):
            fd = os.path.join(parent, f"fold{k}")
            assert os.path.isdir(fd), f"missing fold dir: {fd}"
            assert os.path.exists(os.path.join(fd, "final", "split_metrics.json")), f"missing split_metrics.json in {fd}"
        assert os.path.exists(os.path.join(parent, "agg", "permutation_test.json"))
        if args.require_significant or args.require_held_model_test:
            failures: list[str] = []
            if args.require_significant:
                failures.extend(quality_gate_failures(
                    agg,
                    alpha=args.alpha,
                    min_rho_s=args.min_rho_s,
                ))
                failures.extend(
                    fold_summary_quality_failures(
                        result.get("fold_summaries", []),
                        min_delta_tag=args.min_fold_delta_tag,
                    )
                )
            failures.extend(
                parent_run_quality_failures(
                    parent,
                    min_delta_tag=(
                        args.min_fold_delta_tag
                        if args.require_significant
                        else None
                    ),
                    require_held_model_test=args.require_held_model_test,
                )
            )
            if failures:
                print(
                    "[smoke-e2e] FAIL — quality gate failed: "
                    + ", ".join(failures),
                    file=sys.stderr,
                )
                return 2
        print("[smoke-e2e] PASS — all fold dirs + agg/permutation_test.json present")
    else:
        print(f"[smoke-e2e] WARN: expected mode=v3_kfold, got {result.get('mode')}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
