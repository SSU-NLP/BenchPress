"""Run the Part 2 main experiment.

Two paths:

* Legacy single-shot (default): load corpus → fixed-vocab tag → metrics.
  Backward-compatible behavior. Active when ``enable_v_loop`` is False.

* v3 main loop: when ``enable_v_loop=True`` is set in the part2_experiment
  config section, ``run_part2`` delegates to ``experiment.loop.run_part1``
  with the Part 2 corpus. This activates the full v3 pipeline
  (Executer→Maker→split-aware metrics→Improver) on ``data/labels_part2``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from autotagging_loop.runner.alignment import compute_metrics
from autotagging_loop.runner.config import load_config
from autotagging_loop.runner.corpus import load_corpus
from autotagging_loop.runner.mapreduce import build_tag_vectors, load_vocab
from autotagging_loop.runner.score_matrix import normalize_matrix, spearman_pair_matrix
from autotagging_loop.runner.storage import make_run_dir, write_json


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float | int]:
    flat: dict[str, float | int] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, prefix=f"{name}/"))
        elif isinstance(value, (int, float)):
            flat[name] = value
    return flat


def init_wandb(config: dict, run_dir: str):
    if not config.get("wandb"):
        return None
    try:
        import wandb
        from datetime import datetime

        mode = config.get("wandb_mode")
        init_kwargs = {
            "project": config.get("wandb_project") or "bench experiment",
            "name": f"part2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "tags": ["part2", "main", "seed-vocab", "mapreduce"],
            "config": config,
        }
        if config.get("wandb_entity"):
            init_kwargs["entity"] = config["wandb_entity"]
        if mode:
            init_kwargs["mode"] = mode
        run = wandb.init(**init_kwargs)
        run.summary["run_dir"] = run_dir
        return run
    except Exception as exc:
        print(f"  [part2][wandb] init failed: {exc}")
        return None


def log_wandb(wandb_run, payload: dict) -> None:
    if wandb_run is None:
        return
    try:
        wandb_run.log(payload)
    except Exception as exc:
        print(f"  [part2][wandb] log failed: {exc}")


def _to_experiment_corpus(part2_corpus):
    """Repack the Part 2 corpus dataclass as ``experiment.corpus.Corpus``.

    Both classes carry the same fields; ``experiment.loop.run_part1`` does
    light duck-typing but emitting the same class avoids any surprises.
    """
    from autotagging_loop.experiment.corpus import Corpus as ExperimentCorpus

    return ExperimentCorpus(
        benchmark_names=list(part2_corpus.benchmark_names),
        model_names=list(part2_corpus.model_names),
        Y=dict(part2_corpus.Y),
        descriptions=dict(part2_corpus.descriptions),
        documents=dict(part2_corpus.documents),
        drop_log=dict(part2_corpus.drop_log),
    )


def _build_v3_overrides(config: dict) -> dict:
    """Translate Part 2 config keys → experiment.config keys.

    The v_loop seed is always ``experiment/prompts/I_exec_seed.txt`` —
    a schema-agnostic vocabulary-criteria prompt that Executer/Maker/Improver
    share. ``part1_best_prompt_path`` (if present in legacy configs) is
    intentionally ignored: Part 1's ``I_star.txt`` is a fixed-vocab tagging
    prompt and steers Executer into emitting weights{} instead of vocab[].
    """
    prompt_seed = str(
        Path(__file__).resolve().parent.parent.parent
        / "experiment" / "prompts" / "I_exec_seed.txt"
    )

    overrides: dict = {
        "vocab_path": config["vocab_path"],
        "prompt_i0_path": prompt_seed,
        "results_dir": config["results_dir"],
        "labels_dir": config["labels_dir"],
        "leaderboard_path": config["leaderboard_path"],
        "min_common_models": int(config.get("min_common_models", 6)),
        "min_common_models_warn": int(config.get("min_common_models_warn", 5)),
        "exclude": list(config.get("exclude", [])),
        "normalize": config.get("normalize", "rank"),
        "bootstrap_B": int(config.get("bootstrap_B", 200)),
        "seed": int(config.get("seed", 42)),
        "prompt_examples_per_benchmark": int(
            config.get("prompt_examples_per_benchmark", 20)
        ),
        "max_prompt_chars_per_benchmark": int(
            config.get("max_prompt_chars_per_benchmark", 24000)
        ),
        "max_iter": int(config.get("max_iter", 5)),
        "enable_v_loop": True,
    }
    for key in (
        "mapreduce_cache_enabled",
        "mapreduce_cache_dir",
        "mapreduce_cache_schema_version",
        "splits",
        "v_loop_min_train_valid_pairs",
        "v_loop_min_dev_valid_pairs",
        "v_loop_min_test_valid_pairs",
        "v_loop_min_train_effective_benchmarks",
        "v_loop_min_dev_effective_benchmarks",
        "v_loop_min_test_effective_benchmarks",
        "v_loop_require_held_model_test",
        "v_loop_score_model_scope",
        "executer_candidate_counts",
        "llm_request_timeout_s",
        "llm_sdk_exception_retries",
        "llm_empty_content_retries",
        "llm_debug_dump_dir",
        "llm_reasoning",
        "delta_tag_threshold",
        "best_iter_selection",
        "best_iter_dev_rho_floor",
        "best_iter_dev_rho_drop_tolerance",
        "best_iter_train_l_increase_tolerance",
        "best_iter_train_rho_drop_tolerance",
        "best_iter_train_rho_floor",
        "best_iter_stability_rho_weight",
        "best_iter_model_probe_enabled",
        "best_iter_model_probe_min_common",
        "best_iter_model_probe_dev_rho_floor",
        "best_iter_model_probe_dev_rho_drop_tolerance",
        "best_iter_model_probe_dev_l_increase_tolerance",
    ):
        if key in config:
            overrides[key] = config[key]
    for key, value in config.items():
        if (
            key.startswith("no_seed_taxonomy_")
            or key.startswith("taxonomy_refinement_")
            or key.startswith("taxonomy_selection_")
        ):
            overrides[key] = value
    if "wandb" in config:
        overrides["wandb"] = bool(config["wandb"])
    return overrides


def _init_wandb_v3(config: dict):
    if not config.get("wandb"):
        return None
    try:
        import wandb
        from datetime import datetime

        init_kwargs = {
            "project": config.get("wandb_project") or "bench experiment",
            "name": f"part2_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "tags": ["part2", "main", "v3"],
            "config": config,
        }
        if config.get("wandb_entity"):
            init_kwargs["entity"] = config["wandb_entity"]
        if config.get("wandb_mode"):
            init_kwargs["mode"] = config["wandb_mode"]
        return wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"  [part2][wandb] init failed: {exc}")
        return None


class _FoldSummaryProxy:
    """Wraps a wandb Run.summary so writes are namespaced under `fold/{k}/`.

    The v3 loop calls `wandb_run.summary["best_label"] = ...` at end-of-run.
    Without prefixing, four folds would overwrite each other's summary fields.
    """

    def __init__(self, base_summary, fold: int) -> None:
        self._base = base_summary
        self._fold = fold

    def __setitem__(self, key, value):  # noqa: D401
        self._base[f"fold/{self._fold}/{key}"] = value

    def __getitem__(self, key):
        return self._base[f"fold/{self._fold}/{key}"]


class _FoldWandbProxy:
    """Wraps a wandb Run so all `.log()` and `.summary[...]` writes carry a
    `fold/{k}/` prefix. Lets multiple K-fold runs share one wandb Run while
    keeping metrics independently scoped per fold.

    `finish()` is a no-op — the orchestrator owns the Run lifecycle. Step
    arguments are dropped so wandb auto-increments globally across folds
    (avoids cross-fold step collisions overwriting each other).
    """

    def __init__(self, base_run, fold: int) -> None:
        self._base = base_run
        self._fold = fold
        self.summary = _FoldSummaryProxy(base_run.summary, fold)

    def log(self, payload, step=None, **kwargs):  # noqa: D401
        prefixed = {f"fold/{self._fold}/{k}": v for k, v in payload.items()}
        self._base.log(prefixed, **kwargs)

    def finish(self):
        return None

    def __getattr__(self, name):
        return getattr(self._base, name)


def _run_part2_v3(config: dict) -> dict:
    """v3 main loop on labels_part2 — delegates to experiment.loop.run_part1.

    When `config["splits"]["cv_folds"]` > 1, fans out to K disjoint folds
    sharing a single parent run dir and a single wandb Run (fold-prefixed).
    Aggregated permutation test runs after the last fold.
    """
    from autotagging_loop.experiment.config import load_experiment_config
    from autotagging_loop.experiment.loop import run_part1

    print("  [part2] enable_v_loop=True → delegating to v3 main loop")
    part2_corpus = load_corpus(config)
    if not part2_corpus.benchmark_names:
        raise ValueError("Part 2 corpus is empty. Build labels_part2 first.")
    exp_corpus = _to_experiment_corpus(part2_corpus)

    overrides = _build_v3_overrides(config)
    exp_config = load_experiment_config(overrides)
    splits_cfg = exp_config.get("splits", {}) or {}
    cv_folds = int(splits_cfg.get("cv_folds", 1))

    if cv_folds > 1:
        _preflight_v3_kfold_splits(exp_config, exp_corpus, cv_folds=cv_folds)
        return _run_part2_v3_kfold(
            config=config,
            exp_config=exp_config,
            exp_corpus=exp_corpus,
            cv_folds=cv_folds,
            run_part1=run_part1,
        )

    wandb_run = _init_wandb_v3(config)
    try:
        history, best = run_part1(exp_config, corpus=exp_corpus, wandb_run=wandb_run)
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception as exc:
                print(f"  [part2][wandb] finish failed: {exc}")
    print(
        f"  [part2] v3 loop done: iterations={len(history)}, "
        f"best={best.label}, L_align={best.L_align:.4f}"
    )
    return {
        "mode": "v3",
        "iterations": len(history),
        "best_label": best.label,
        "L_align": best.L_align,
        "metrics": {
            "L_align": best.L_align,
            "rho_align_pearson": best.rho_align_pearson,
            "delta_tag": best.delta_tag,
        },
    }


def _preflight_v3_kfold_splits(exp_config: dict, exp_corpus, *, cv_folds: int) -> list[dict]:
    """Validate every K-fold split before any expensive LLM-backed fold starts."""
    from autotagging_loop.experiment.score_matrix import normalize_matrix as exp_normalize_matrix
    from autotagging_loop.experiment.score_matrix import spearman_pair_matrix as exp_spearman_pair_matrix
    from autotagging_loop.experiment.split_diagnostics import (
        benchmark_split_from_config,
        split_pair_count_failures,
        split_effective_benchmark_counts,
        split_coverage_pair_counts,
        split_valid_pair_counts,
    )
    from autotagging_loop.experiment.splits import split_models

    splits_cfg = exp_config.get("splits", {}) or {}
    min_common = int(exp_config.get("min_common_models", 6))
    thresholds = {
        "train": int(exp_config.get("v_loop_min_train_valid_pairs", 1)),
        "dev": int(exp_config.get("v_loop_min_dev_valid_pairs", 1)),
        "test": int(exp_config.get("v_loop_min_test_valid_pairs", 1)),
    }
    effective_thresholds = {
        "train": int(exp_config.get("v_loop_min_train_effective_benchmarks", 0)),
        "dev": int(exp_config.get("v_loop_min_dev_effective_benchmarks", 0)),
        "test": int(exp_config.get("v_loop_min_test_effective_benchmarks", 0)),
    }
    Y_norm_full = exp_normalize_matrix(
        exp_corpus.Y,
        method=exp_config.get("normalize", "rank"),
    )
    score_model_scope = str(exp_config.get("v_loop_score_model_scope", "all")).strip().lower()
    if bool(exp_config.get("v_loop_require_held_model_test", False)):
        score_model_scope = "seen"
    if score_model_scope not in {"all", "seen"}:
        raise ValueError(
            "v_loop_score_model_scope must be one of {'all', 'seen'}, "
            f"got {score_model_scope!r}"
        )
    model_split = split_models(
        exp_corpus.model_names,
        ratios=tuple(splits_cfg.get("model_ratios", (0.8, 0.2))),
        seed=int(splits_cfg.get("model_seed", 0)),
        strategy=splits_cfg.get("model_split_strategy", "random"),
    )
    if score_model_scope == "seen":
        seen = set(model_split.seen)
        Y_for_score = {
            bench: {
                model: score
                for model, score in scores.items()
                if model in seen
            }
            for bench, scores in exp_corpus.Y.items()
        }
        Y_norm_score = exp_normalize_matrix(
            Y_for_score,
            method=exp_config.get("normalize", "rank"),
        )
    else:
        Y_norm_score = Y_norm_full
    R_raw, common_count = exp_spearman_pair_matrix(
        Y_norm_score,
        exp_corpus.benchmark_names,
        min_common=min_common,
        warn_below=10**9,
    )
    diagnostics: list[dict] = []
    failures: list[str] = []
    held_failures: list[str] = []
    held_test_counts: list[int] = []
    held_test_coverage_counts: list[int] = []
    R_held: dict | None = None
    common_held: dict | None = None
    if len(model_split.held) < min_common:
        held_failures.append(f"held_models:{len(model_split.held)}<{min_common}")
    else:
        Y_held = {
            bench: {
                model: score
                for model, score in scores.items()
                if model in set(model_split.held)
            }
            for bench, scores in Y_norm_full.items()
        }
        R_held, common_held = exp_spearman_pair_matrix(
            Y_held,
            exp_corpus.benchmark_names,
            min_common=min_common,
            warn_below=10**9,
        )
    for fold in range(cv_folds):
        fold_splits = {**splits_cfg, "fold": fold}
        split = benchmark_split_from_config(
            exp_corpus.benchmark_names,
            fold_splits,
            score_pair_dict=R_raw,
            required_pair_dicts=[pair_dict for pair_dict in (R_raw, R_held) if pair_dict is not None],
            min_test_valid_pairs=thresholds["test"],
            min_test_effective_benchmarks=effective_thresholds["test"],
        )
        counts = split_valid_pair_counts(R_raw, split)
        effective_counts = split_effective_benchmark_counts(R_raw, split)
        coverage_counts = split_coverage_pair_counts(
            common_count,
            split,
            min_common=min_common,
        )
        degenerate_counts = {
            name: max(0, int(coverage_counts[name]) - int(counts[name]))
            for name in ("train", "dev", "test")
        }
        fold_failures = split_pair_count_failures(counts, thresholds)
        for name in ("train", "dev", "test"):
            threshold = int(effective_thresholds.get(name, 0))
            if threshold > 0 and int(effective_counts.get(name, 0)) < threshold:
                fold_failures.append(
                    f"{name}_effective_benchmarks:"
                    f"{int(effective_counts.get(name, 0))}<{threshold}"
                )
        if R_held is not None:
            held_count = split_valid_pair_counts(R_held, split)["test"]
            held_effective_count = split_effective_benchmark_counts(R_held, split)["test"]
            held_coverage_count = (
                split_coverage_pair_counts(
                    common_held or {},
                    split,
                    min_common=min_common,
                )["test"]
                if common_held is not None
                else 0
            )
            held_test_counts.append(int(held_count))
            held_test_coverage_counts.append(int(held_coverage_count))
            if held_count < thresholds["test"]:
                held_failures.append(
                    f"fold{fold}:held_test:{held_count}<{thresholds['test']}"
                )
            held_effective_threshold = int(effective_thresholds.get("test", 0))
            if held_effective_threshold > 0 and held_effective_count < held_effective_threshold:
                held_failures.append(
                    f"fold{fold}:held_test_effective_benchmarks:"
                    f"{held_effective_count}<{held_effective_threshold}"
                )
        diagnostics.append(
            {
                "fold": fold,
                "sizes": {
                    "train": len(split.train),
                    "dev": len(split.dev),
                    "test": len(split.test),
                },
                "valid_pairs": counts,
                "effective_benchmarks": effective_counts,
                "coverage_pairs": coverage_counts,
                "degenerate_pairs": degenerate_counts,
                "failures": fold_failures,
            }
        )
        failures.extend(f"fold{fold}:{failure}" for failure in fold_failures)

    min_counts = {
        name: min(row["valid_pairs"][name] for row in diagnostics)
        for name in ("train", "dev", "test")
    }
    min_coverage_counts = {
        name: min(row["coverage_pairs"][name] for row in diagnostics)
        for name in ("train", "dev", "test")
    }
    min_effective_counts = {
        name: min(row["effective_benchmarks"][name] for row in diagnostics)
        for name in ("train", "dev", "test")
    }
    print(
        "  [part2] k-fold split preflight: "
        f"thresholds={thresholds}, min_valid_pairs={min_counts}, "
        f"effective_thresholds={effective_thresholds}, "
        f"min_effective_benchmarks={min_effective_counts}, "
        f"min_coverage_pairs={min_coverage_counts}, "
        f"score_model_scope={score_model_scope}"
    )
    held_min = min(held_test_counts) if held_test_counts else 0
    held_coverage_min = min(held_test_coverage_counts) if held_test_coverage_counts else 0
    held_status = "PASS" if not held_failures else "WARN"
    print(
        "  [part2] held-model preflight: "
        f"status={held_status}, seen={len(model_split.seen)}, "
        f"held={len(model_split.held)}, min_common={min_common}, "
        f"min_held_test_pairs={held_min}, "
        f"min_held_test_coverage_pairs={held_coverage_min}, "
        f"failures={held_failures[:3]}"
        f"{'...' if len(held_failures) > 3 else ''}"
    )
    if failures:
        raise ValueError(
            "k-fold split preflight failed before LLM calls: "
            + ", ".join(failures)
        )
    if held_failures and bool(exp_config.get("v_loop_require_held_model_test", False)):
        raise ValueError(
            "k-fold held-model preflight failed before LLM calls: "
            + ", ".join(held_failures)
        )
    return diagnostics


def _score_backfill_readiness_failures(
    config: dict,
    *,
    missing_csv: str | Path = Path("data") / "score_backfill_missing.csv",
) -> list[str]:
    """Check that the planned curated score backfill is ready for strict runs.

    This guard is intentionally about data readiness, not scoring behavior.
    It keeps research-grade runs from silently falling back to the sparse
    built-in score matrix when the hand-curated cells are still missing.
    """

    path_raw = config.get("curated_score_backfill_path")
    use_curated = config.get("use_curated_score_backfill")
    if not path_raw:
        if use_curated is True:
            return ["research_preflight_curated_score_backfill_path_missing"]
        return []
    if use_curated is False:
        return []

    try:
        from autotagging_loop.scripts.validate_score_backfill import (
            _load_scores_from_config,
            _planned_backfill_projection,
            _top_core_plans,
            curated_backfill_progress,
            load_missing_cell_plan,
            missing_plan_drift,
            planned_projection_failures,
        )
    except Exception as exc:  # pragma: no cover - defensive import guard
        return [f"research_preflight_score_backfill_check_error:{type(exc).__name__}:{exc}"]

    failures: list[str] = []
    model_aliases = config.get("model_aliases") or {}
    try:
        expected = load_missing_cell_plan(
            missing_csv,
            model_aliases=model_aliases,
        )
    except FileNotFoundError:
        return [f"research_preflight_missing_cell_plan_missing:{missing_csv}"]
    except ValueError as exc:
        return [f"research_preflight_missing_cell_plan_invalid:{exc}"]

    path = Path(str(path_raw))
    try:
        progress = curated_backfill_progress(
            path,
            expected,
            model_aliases=model_aliases,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        return [f"research_preflight_curated_score_backfill_invalid:{exc}"]

    if not path.exists():
        failures.append(f"research_preflight_curated_score_backfill_missing:{path}")

    expected_count = int(progress.get("expected_cells") or 0)
    complete_count = int(progress.get("complete_cells") or 0)
    missing_count = int(progress.get("missing_cells") or 0)
    incomplete_count = int(progress.get("incomplete_cells") or 0)
    if complete_count < expected_count:
        failures.append(
            "research_preflight_curated_score_backfill_incomplete:"
            f"{complete_count}/{expected_count}"
            f"(missing={missing_count},incomplete={incomplete_count})"
        )

    outside = progress.get("outside_plan") or []
    if outside:
        preview = ",".join(str(item) for item in outside[:3])
        failures.append(
            "research_preflight_curated_score_backfill_outside_plan:"
            f"{len(outside)}:{preview}"
        )

    duplicates = progress.get("duplicate_cells") or []
    if duplicates:
        preview = ",".join(str(item) for item in duplicates[:3])
        failures.append(
            "research_preflight_curated_score_backfill_duplicates:"
            f"{len(duplicates)}:{preview}"
        )

    if failures:
        return failures

    default_missing_csv = (Path("data") / "score_backfill_missing.csv").resolve()
    if Path(missing_csv).resolve() != default_missing_csv:
        return failures

    try:
        projection_scores, _projection_sources = _load_scores_from_config(
            include_hf_report=True,
            curated_score_backfill_path=str(path),
            use_curated_score_backfill=False,
        )
        projection_plans = _top_core_plans(
            projection_scores,
            core_size=max(1, int(config.get("min_common_models", 6))),
            top_k=1,
        )
        projection_best_plan = (
            projection_plans[0]
            if projection_plans
            else {"error": "no score plan generated"}
        )
        plan_drift = missing_plan_drift(
            expected,
            projection_best_plan,
            model_aliases=model_aliases,
        )
        if plan_drift.get("status") != "PASS":
            for failure in plan_drift.get("failures") or ["status_not_pass"]:
                failures.append(f"research_preflight_missing_plan_drift:{failure}")

        planned_projection = _planned_backfill_projection(
            config=config,
            expected_missing=expected,
            best_plan=projection_best_plan,
            include_hf_report=True,
        )
        for failure in planned_projection_failures(planned_projection):
            failures.append(f"research_preflight_{failure}")
    except Exception as exc:
        failures.append(
            "research_preflight_score_backfill_projection_check_error:"
            f"{type(exc).__name__}:{exc}"
        )

    return failures


def preflight_research_grade(config: dict) -> dict[str, Any]:
    """Validate that a run is eligible to be treated as research evidence.

    This is a fail-fast guard for CLI/API callers. It does not change the
    scoring method; it checks the strict config shape and split coverage before
    any model-backed fold starts.
    """

    from autotagging_loop.experiment.config import load_experiment_config
    from autotagging_loop.scripts.audit_kfold_run import strict_config_failures

    failures: list[str] = []
    diagnostics: list[dict[str, Any]] = []

    if not bool(config.get("enable_v_loop")):
        failures.append("research_preflight_enable_v_loop_not_true")

    score_backfill_failures = _score_backfill_readiness_failures(config)
    failures.extend(score_backfill_failures)

    part2_corpus = None
    if not score_backfill_failures:
        try:
            part2_corpus = load_corpus(config)
        except Exception as exc:
            failures.append(f"research_preflight_corpus_load_failed:{exc}")
        if part2_corpus is not None and not part2_corpus.benchmark_names:
            failures.append("research_preflight_empty_corpus")

    exp_config = load_experiment_config(_build_v3_overrides(config))
    failures.extend(strict_config_failures({"experiment": exp_config}))

    splits_cfg = exp_config.get("splits", {}) or {}
    cv_folds = int(splits_cfg.get("cv_folds", 1))
    if cv_folds <= 1:
        failures.append(f"research_preflight_cv_folds:{cv_folds}<2")
    elif part2_corpus is not None and part2_corpus.benchmark_names:
        exp_corpus = _to_experiment_corpus(part2_corpus)
        try:
            diagnostics = _preflight_v3_kfold_splits(
                exp_config,
                exp_corpus,
                cv_folds=cv_folds,
            )
        except ValueError as exc:
            failures.append(f"research_preflight_split_failed:{exc}")

    return {
        "ok": not failures,
        "failures": failures,
        "cv_folds": cv_folds,
        "diagnostics": diagnostics,
    }


def _raise_research_grade_preflight(report: dict[str, Any]) -> None:
    failures = report.get("failures") or []
    preview = "; ".join(str(failure) for failure in failures[:8])
    if len(failures) > 8:
        preview += f"; ... ({len(failures)} total)"
    raise ValueError(
        "research-grade preflight failed before LLM calls: "
        + (preview or "unknown failure")
    )


def _raise_research_grade_quality(result: dict[str, Any]) -> None:
    quality_gate = result.get("quality_gate") if isinstance(result, dict) else None
    if not isinstance(quality_gate, dict):
        raise RuntimeError(
            "research-grade quality gate missing after run; "
            "only K-fold v-loop runs can be accepted as research evidence"
        )
    if quality_gate.get("research_grade") is True:
        return
    failures = quality_gate.get("failures") or []
    preview = "; ".join(str(failure) for failure in failures[:8])
    if len(failures) > 8:
        preview += f"; ... ({len(failures)} total)"
    raise RuntimeError(
        "research-grade quality gate failed after run: "
        + (preview or "unknown failure")
    )


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _completed_fold_summary(fold_dir: Path, fold: int) -> dict[str, Any] | None:
    final_dir = fold_dir / "final"
    required = (
        fold_dir / "selection.json",
        final_dir / "metrics_raw.json",
        final_dir / "metrics_with_bootstrap.json",
        final_dir / "split_metrics.json",
        final_dir / "llm_fallbacks.json",
    )
    if not all(path.exists() for path in required):
        return None

    metrics = _read_json_dict(final_dir / "metrics_raw.json")
    required_metrics = ("L_align", "rho_align_pearson", "rho_align_spearman", "delta_tag")
    if not all(isinstance(metrics.get(key), (int, float)) for key in required_metrics):
        return None

    selection = _read_json_dict(fold_dir / "selection.json")
    selected_candidate = selection.get("selected_candidate")
    if not isinstance(selected_candidate, dict):
        selected_candidate = {}
    actual_sources: list[str] | None = None
    vocab_meta = _read_json_dict(final_dir / "vocab_star_metadata.json")
    if isinstance(vocab_meta.get("source_benchmarks"), list):
        actual_sources = vocab_meta["source_benchmarks"]

    return {
        "fold": fold,
        "run_dir": str(fold_dir),
        "iterations": len(list(fold_dir.glob("iter_*"))),
        "best_label": (
            selected_candidate.get("label")
            or selection.get("selected_label")
            or _read_best_iter(final_dir)
        ),
        "L_align": metrics["L_align"],
        "rho_align_pearson": metrics["rho_align_pearson"],
        "rho_align_spearman": metrics["rho_align_spearman"],
        "delta_tag": metrics["delta_tag"],
        "source_benchmarks": actual_sources,
        "resumed": True,
    }


def _read_best_iter(final_dir: Path) -> str | None:
    path = final_dir / "best_iter.txt"
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except Exception:
        return None


def _write_fold_summaries(parent_dir: str, fold_summaries: list[dict]) -> None:
    agg_dir = os.path.join(parent_dir, "agg")
    os.makedirs(agg_dir, exist_ok=True)
    with open(os.path.join(agg_dir, "fold_summaries.json"), "w", encoding="utf-8") as fh:
        json.dump(fold_summaries, fh, indent=2, sort_keys=True)


def _build_kfold_quality_gate(
    *,
    parent_dir: str,
    exp_config: dict,
    agg_result: dict,
    fold_dirs: list[str],
) -> dict[str, Any]:
    """Summarize whether a completed K-fold run is research-grade.

    This does not change scoring or selection. It records the acceptance gate
    that downstream analysis should use before treating a run as evidence.
    """

    from autotagging_loop.scripts.audit_kfold_run import (
        DEFAULT_ALPHA,
        DEFAULT_MIN_FOLD_RHO_S,
        DEFAULT_MIN_RHO_S,
        STRICT_MIN_EFFECTIVE_BENCHMARKS,
        STRICT_MIN_COMMON_MODELS,
        STRICT_MIN_SPLIT_VALID_PAIRS,
        fold_quality_failures,
        pooled_quality_failures,
        role_output_quality_failures,
        strict_config_failures,
    )

    failures: list[str] = []
    failures.extend(strict_config_failures({"experiment": exp_config}))

    if isinstance(agg_result, dict) and "pooled" in agg_result:
        failures.extend(
            pooled_quality_failures(
                agg_result,
                alpha=DEFAULT_ALPHA,
                min_rho_s=DEFAULT_MIN_RHO_S,
            )
        )
    else:
        failures.append("quality_missing:agg/permutation_test.json")

    fold_reports: list[dict[str, Any]] = []
    for fold_dir_raw in fold_dirs:
        fold_dir = Path(fold_dir_raw)
        final_dir = fold_dir / "final"
        metrics = _read_json_dict(final_dir / "metrics_with_bootstrap.json")
        split_metrics = _read_json_dict(final_dir / "split_metrics.json")
        fallbacks = _read_json_dict(final_dir / "llm_fallbacks.json")
        selection_scope = metrics.get("selection_scope")
        fold_failures = fold_quality_failures(
            split_metrics,
            selection_scope=str(selection_scope) if selection_scope else None,
            min_pairs=STRICT_MIN_SPLIT_VALID_PAIRS,
            min_rho_s=DEFAULT_MIN_FOLD_RHO_S,
        )
        fallback_total = fallbacks.get("total")
        if fallback_total not in (0, None):
            fold_failures.append(f"quality_llm_fallbacks:{fallback_total}")
        role_failures = role_output_quality_failures(fold_dir)
        fold_failures.extend(role_failures)
        failures.extend(f"{fold_dir.name}:{failure}" for failure in fold_failures)
        fold_reports.append(
            {
                "fold": fold_dir.name,
                "selection_scope": selection_scope,
                "test_n_pairs": (split_metrics.get("test") or {}).get("n_pairs"),
                "test_n_effective_benchmarks": (
                    split_metrics.get("test") or {}
                ).get("n_effective_benchmarks"),
                "held_model_test_n_pairs": (
                    split_metrics.get("held_model_test") or {}
                ).get("n_pairs"),
                "held_model_test_n_effective_benchmarks": (
                    split_metrics.get("held_model_test") or {}
                ).get("n_effective_benchmarks"),
                "llm_fallbacks": fallback_total,
                "role_quality_failures": len(role_failures),
                "failures": fold_failures,
            }
        )

    pooled = (agg_result or {}).get("pooled") if isinstance(agg_result, dict) else {}
    payload = {
        "status": "pass" if not failures else "fail",
        "research_grade": not failures,
        "failures": failures,
        "thresholds": {
            "alpha": DEFAULT_ALPHA,
            "min_rho_s": DEFAULT_MIN_RHO_S,
            "min_fold_pairs": STRICT_MIN_SPLIT_VALID_PAIRS,
            "min_effective_benchmarks": STRICT_MIN_EFFECTIVE_BENCHMARKS,
            "min_fold_rho_s": DEFAULT_MIN_FOLD_RHO_S,
            "strict_min_common_models": STRICT_MIN_COMMON_MODELS,
            "role_quality_required": True,
        },
        "pooled": {
            "n_pairs": (pooled or {}).get("n_pairs"),
            "rho_spearman": ((pooled or {}).get("rho_spearman") or {}).get("observed"),
            "rho_spearman_p_two_sided": (
                ((pooled or {}).get("rho_spearman") or {}).get("p_two_sided")
            ),
        },
        "folds": fold_reports,
    }
    agg_dir = Path(parent_dir) / "agg"
    agg_dir.mkdir(parents=True, exist_ok=True)
    with open(agg_dir / "quality_gate.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    return payload


def _run_part2_v3_kfold(
    *,
    config: dict,
    exp_config: dict,
    exp_corpus,
    cv_folds: int,
    run_part1,
) -> dict:
    """K-fold orchestration: K disjoint test partitions, one parent dir, one wandb Run."""
    import copy
    import os
    from datetime import datetime

    resume_run_dir = config.get("resume_run_dir")
    if resume_run_dir:
        parent_dir = str(Path(resume_run_dir))
    else:
        parent_uid = datetime.now().strftime("cv_%Y%m%d_%H%M%S")
        parent_dir = os.path.join(exp_config["results_dir"], f"run_{parent_uid}")
    os.makedirs(parent_dir, exist_ok=True)
    print(f"  [part2] cv_folds={cv_folds} → parent run_dir={parent_dir}")
    if resume_run_dir:
        print("  [part2] resume enabled: completed folds will be skipped")

    with open(os.path.join(parent_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"part2": config, "experiment": exp_config}, fh, indent=2, sort_keys=True, default=str)

    wandb_run = _init_wandb_v3(config)
    fold_summaries: list[dict] = []
    fold_dirs: list[str] = []
    try:
        for k in range(cv_folds):
            fold_dir = os.path.join(parent_dir, f"fold{k}")
            os.makedirs(fold_dir, exist_ok=True)
            fold_dirs.append(fold_dir)
            completed = _completed_fold_summary(Path(fold_dir), k)
            if completed is not None:
                fold_summaries.append(completed)
                print(
                    f"  [part2][fold {k}/{cv_folds - 1}] resume skip: "
                    f"best={completed['best_label']}, L_align={completed['L_align']:.4f}"
                )
                _write_fold_summaries(parent_dir, fold_summaries)
                continue
            fold_config = copy.deepcopy(exp_config)
            fold_config.setdefault("splits", {})
            fold_config["splits"]["fold"] = k
            proxy = _FoldWandbProxy(wandb_run, k) if wandb_run is not None else None
            print(f"  [part2][fold {k}/{cv_folds - 1}] starting at {fold_dir}")
            history, best = run_part1(
                fold_config,
                corpus=exp_corpus,
                run_dir=fold_dir,
                wandb_run=proxy,
            )
            print(
                f"  [part2][fold {k}] done: iterations={len(history)}, "
                f"best={best.label}, L_align={best.L_align:.4f}"
            )
            actual_sources: list[str] | None = None
            vocab_meta_path = os.path.join(fold_dir, "final", "vocab_star_metadata.json")
            if os.path.exists(vocab_meta_path):
                try:
                    vocab_meta = json.load(open(vocab_meta_path))
                    actual_sources = vocab_meta.get("source_benchmarks")
                except Exception:
                    pass
            fold_summaries.append({
                "fold": k,
                "run_dir": fold_dir,
                "iterations": len(history),
                "best_label": best.label,
                "L_align": best.L_align,
                "rho_align_pearson": best.rho_align_pearson,
                "rho_align_spearman": best.rho_align_spearman,
                "delta_tag": best.delta_tag,
                "source_benchmarks": actual_sources,
            })
            _write_fold_summaries(parent_dir, fold_summaries)
            if wandb_run is not None:
                try:
                    wandb_run.summary[f"fold/{k}/best_L_align"] = best.L_align
                    wandb_run.summary[f"fold/{k}/best_rho_p"] = best.rho_align_pearson
                except Exception as exc:
                    print(f"  [part2][wandb] summary write failed: {exc}")
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception as exc:
                print(f"  [part2][wandb] finish failed: {exc}")

    agg_dir = os.path.join(parent_dir, "agg")
    os.makedirs(agg_dir, exist_ok=True)
    _write_fold_summaries(parent_dir, fold_summaries)

    print(f"  [part2] all {cv_folds} folds complete. Running pooled permutation test.")
    try:
        from autotagging_loop.scripts.permutation_test_run import run_pooled
        agg_result = run_pooled(fold_dirs=fold_dirs, out_path=os.path.join(agg_dir, "permutation_test.json"))
        print(
            f"  [part2] pooled test ρ_s={agg_result['pooled']['rho_spearman']['observed']:+.4f}, "
            f"p_two={agg_result['pooled']['rho_spearman']['p_two_sided']:.4f}, "
            f"n_pairs={agg_result['pooled']['n_pairs']}"
        )
    except Exception as exc:
        print(f"  [part2] WARN: pooled permutation test failed: {exc}")
        agg_result = {"error": str(exc)}

    quality_gate = _build_kfold_quality_gate(
        parent_dir=parent_dir,
        exp_config=exp_config,
        agg_result=agg_result,
        fold_dirs=fold_dirs,
    )
    print(
        "  [part2] quality gate "
        f"status={quality_gate['status']} "
        f"research_grade={quality_gate['research_grade']} "
        f"failures={len(quality_gate['failures'])}"
    )

    pooled = (agg_result or {}).get("pooled") or {}
    pooled_rho_s = pooled.get("rho_spearman") or {}
    pooled_rho_p = pooled.get("rho_pearson") or {}
    pooled_L = pooled.get("L_align") or {}
    n_folds_done = len(fold_summaries)
    mean_L = (
        sum(fs["L_align"] for fs in fold_summaries) / n_folds_done
        if n_folds_done else float("nan")
    )
    mean_rho_p = (
        sum(fs["rho_align_pearson"] for fs in fold_summaries) / n_folds_done
        if n_folds_done else float("nan")
    )
    metrics = {
        "mean_L_align_across_folds": mean_L,
        "mean_rho_align_pearson_across_folds": mean_rho_p,
        "pooled_n_pairs": pooled.get("n_pairs"),
        "pooled_rho_spearman": pooled_rho_s.get("observed"),
        "pooled_rho_spearman_p_two_sided": pooled_rho_s.get("p_two_sided"),
        "pooled_rho_pearson": pooled_rho_p.get("observed"),
        "pooled_L_align": pooled_L.get("observed"),
        "quality_gate_research_grade": bool(quality_gate.get("research_grade")),
    }
    return {
        "mode": "v3_kfold",
        "cv_folds": cv_folds,
        "parent_run_dir": parent_dir,
        "fold_summaries": fold_summaries,
        "agg": agg_result,
        "quality_gate": quality_gate,
        "metrics": metrics,
    }


def run_part2(config: dict, *, require_research_grade: bool = False) -> dict:
    if require_research_grade:
        report = preflight_research_grade(config)
        if not report.get("ok"):
            _raise_research_grade_preflight(report)

    if bool(config.get("enable_v_loop")):
        result = _run_part2_v3(config)
        if require_research_grade:
            _raise_research_grade_quality(result)
        return result

    if require_research_grade:
        _raise_research_grade_preflight(
            {
                "failures": [
                    "research_preflight_enable_v_loop_not_true",
                    "research_preflight_missing_kfold_quality_gate",
                ]
            }
        )

    run_dir = make_run_dir(config["results_dir"])
    print(f"  [part2] run_dir={run_dir}")
    wandb_run = init_wandb(config, str(run_dir))
    corpus = load_corpus(config)
    if not corpus.benchmark_names:
        raise ValueError("Part 2 corpus is empty. Build labels_part2 first.")
    vocab = load_vocab(config["vocab_path"])
    print(
        f"  [part2] corpus: benchmarks={len(corpus.benchmark_names)}, "
        f"models={len(corpus.model_names)}, docs={len(corpus.documents)}, tags={len(vocab)}"
    )
    log_wandb(wandb_run, {
        "corpus/benchmarks": len(corpus.benchmark_names),
        "corpus/models": len(corpus.model_names),
        "corpus/docs": len(corpus.documents),
        "corpus/tags": len(vocab),
    })

    Y_norm = normalize_matrix(corpus.Y, method=config.get("normalize", "rank"))
    R_raw, common_count = spearman_pair_matrix(
        Y_norm,
        corpus.benchmark_names,
        min_common=int(config.get("min_common_models", 6)),
        warn_below=int(config.get("min_common_models_warn", 5)),
    )
    T, tag_metadata = build_tag_vectors(corpus.documents, vocab, config)
    metrics, S, residuals = compute_metrics(
        T,
        corpus.benchmark_names,
        R_raw,
        bootstrap_B=int(config.get("bootstrap_B", 200)),
        seed=int(config.get("seed", 42)),
    )

    write_json(run_dir / "config.json", config)
    write_json(run_dir / "corpus.json", {
        "benchmark_names": corpus.benchmark_names,
        "model_names": corpus.model_names,
        "drop_log": corpus.drop_log,
        "document_counts": {
            name: corpus.documents.get(name, {}).get("reviewed_rows")
            for name in corpus.benchmark_names
        },
    })
    write_json(run_dir / "score_similarity.json", {f"{p}||{q}": v for (p, q), v in R_raw.items()})
    write_json(run_dir / "common_model_counts.json", {f"{p}||{q}": v for (p, q), v in common_count.items()})
    write_json(run_dir / "tag_similarity.json", {f"{p}||{q}": v for (p, q), v in S.items()})
    write_json(run_dir / "T_star.json", T)
    write_json(run_dir / "tag_weight_metadata.json", tag_metadata)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "residual_report.json", residuals)
    log_wandb(wandb_run, {f"final/{key}": value for key, value in flatten_metrics(metrics).items()})
    if wandb_run is not None:
        try:
            wandb_run.summary["L_align"] = metrics.get("L_align")
            wandb_run.summary["rho_align_pearson"] = metrics.get("rho_align_pearson")
            wandb_run.summary["rho_align_spearman"] = metrics.get("rho_align_spearman")
            wandb_run.summary["delta_tag"] = metrics.get("delta_tag")
            wandb_run.summary["residual_max"] = metrics.get("residual_max")
            wandb_run.finish()
        except Exception as exc:
            print(f"  [part2][wandb] finish failed: {exc}")
    print(
        "  [part2] metrics: "
        f"L_align={metrics['L_align']:.4f}, "
        f"rho_p={metrics['rho_align_pearson']:.4f}, "
        f"rho_s={metrics['rho_align_spearman']:.4f}, "
        f"delta_tag={metrics['delta_tag']:.4f}"
    )
    return {"run_dir": str(run_dir), "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BenchPress Part 2 main experiment")
    parser.add_argument("--config", default=None, help="optional JSON config path")
    parser.add_argument("--labels-dir", default=None, help="override labels_dir")
    parser.add_argument("--bootstrap-B", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", help="enable W&B logging")
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument(
        "--require-research-grade",
        action="store_true",
        help=(
            "Fail before model calls unless strict score/split preflight passes, "
            "and fail after the run unless agg/quality_gate.json is research-grade."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.labels_dir:
        overrides["labels_dir"] = args.labels_dir
    if args.bootstrap_B is not None:
        overrides["bootstrap_B"] = args.bootstrap_B
    if args.wandb:
        overrides["wandb"] = True
    if args.wandb_mode:
        overrides["wandb_mode"] = args.wandb_mode
    config = load_config(overrides, config_path=args.config)
    try:
        result = run_part2(config, require_research_grade=args.require_research_grade)
    except (RuntimeError, ValueError) as exc:
        if args.require_research_grade and str(exc).startswith("research-grade "):
            raise SystemExit(str(exc)) from exc
        raise
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
