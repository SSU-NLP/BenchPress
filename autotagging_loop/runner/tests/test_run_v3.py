"""Tests for the v3 main loop opt-in path in part2_experiment.run."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from autotagging_loop.runner import run as part2_run


@pytest.fixture
def part2_v3_config(tmp_path):
    """Build a Part 2 config with enable_v_loop=True against a tiny fixture."""
    labels_dir = tmp_path / "labels"
    results_dir = tmp_path / "results"
    leaderboard_path = tmp_path / "leaderboard_scores.json"
    vocab_path = tmp_path / "vocab.json"

    bench_names = [f"Bench{i:02d}" for i in range(8)]
    model_names = [f"m{i}" for i in range(1, 8)]
    Y = {
        b: {
            m: 0.5 + 0.4 * math.sin((idx + 1) * (i + 1) * 0.7)
            for i, m in enumerate(model_names)
        }
        for idx, b in enumerate(bench_names)
    }
    leaderboard_path.write_text(json.dumps(Y), encoding="utf-8")
    vocab_path.write_text(
        json.dumps([
            {"id": "deductive_reasoning", "name": "D", "definition": "rules"},
            {"id": "long_term_knowledge_recall", "name": "R", "definition": "facts"},
        ]),
        encoding="utf-8",
    )

    labels_dir.mkdir()
    for b in bench_names:
        slug_dir = labels_dir / b.lower()
        slug_dir.mkdir()
        rows = [{
            "benchmark": b, "reviewer_status": "reviewed",
            "question": f"{b} q", "answer": "x",
            "gt_topic": "math", "gt_reasoning_depth": "shallow",
            "gt_answer_format": "free",
        }]
        with open(slug_dir / "tasks.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    return {
        "leaderboard_path": str(leaderboard_path),
        "labels_dir": str(labels_dir),
        "vocab_path": str(vocab_path),
        "results_dir": str(results_dir),
        "use_aai_scores": False,
        "min_common_models": 6,
        "min_common_models_warn": 5,
        "exclude": [],
        "normalize": "rank",
        "bootstrap_B": 5,
        "seed": 0,
        "max_iter": 1,
        "prompt_examples_per_benchmark": 2,
        "max_prompt_chars_per_benchmark": 2000,
        "enable_v_loop": True,
        # v3-only knobs the experiment loop reads:
        "splits": {"benchmark_ratios": (0.5, 0.25, 0.25), "benchmark_seed": 0},
        "use_mapreduce_evidence": False,
        "mapreduce_chunk_examples": 1,
        "mapreduce_max_chunk_chars": 2000,
        "mapreduce_max_evidence_chars": 4000,
        "mapreduce_max_workers": 1,
        "mapreduce_cache_enabled": False,
        "run_baseline": False,
        "early_stop_consecutive": 999,
        "taxonomy_refinement_enabled": False,
        "wandb": False,
    }


def _make_iter_result(label: str = "iter_001", L_align: float = 0.0):
    from autotagging_loop.experiment.loop import IterationResult
    return IterationResult(
        label=label, iter=0, prompt="x",
        T={}, S={}, L_align=L_align, L_align_01=0.0,
        rho_align_pearson=0.0, rho_align_spearman=0.0,
        delta_tag=0.0, bootstrap={}, error_report_size=0,
    )


def _strict_exp_config() -> dict:
    return {
        "min_common_models": 6,
        "v_loop_min_train_valid_pairs": 10,
        "v_loop_min_dev_valid_pairs": 10,
        "v_loop_min_test_valid_pairs": 10,
        "v_loop_min_train_effective_benchmarks": 6,
        "v_loop_min_dev_effective_benchmarks": 6,
        "v_loop_min_test_effective_benchmarks": 6,
        "v_loop_require_held_model_test": True,
        "v_loop_score_model_scope": "seen",
        "executer_fallback_to_seed": False,
        "tag_generator_allow_uniform_fallback": False,
        "llm_json_contract_strict": True,
    }


def _write_fold_quality_artifacts(
    parent: Path,
    *,
    fold: str = "fold0",
    dev_pairs: int = 10,
    test_pairs: int = 11,
    held_pairs: int = 12,
    effective_benchmarks: int = 6,
    rho_s: float = 0.31,
    fallbacks: int = 0,
) -> Path:
    final = parent / fold / "final"
    final.mkdir(parents=True)
    (final / "metrics_with_bootstrap.json").write_text(
        json.dumps({"selection_scope": "dev"}),
        encoding="utf-8",
    )
    split_metrics = {
        "dev": {
            "n_pairs": dev_pairs,
            "n_effective_benchmarks": effective_benchmarks,
            "L_align": 0.04,
            "rho_align_spearman": rho_s,
        },
        "test": {
            "n_pairs": test_pairs,
            "n_effective_benchmarks": effective_benchmarks,
            "L_align": 0.05,
            "rho_align_spearman": rho_s,
        },
        "held_model_test": {
            "n_pairs": held_pairs,
            "n_effective_benchmarks": effective_benchmarks,
            "L_align": 0.06,
            "rho_align_spearman": rho_s,
        },
    }
    (final / "split_metrics.json").write_text(
        json.dumps(split_metrics),
        encoding="utf-8",
    )
    (final / "llm_fallbacks.json").write_text(
        json.dumps({"total": fallbacks, "counts": {}}),
        encoding="utf-8",
    )
    return parent / fold


def _write_completed_fold(
    parent: Path,
    *,
    fold: str = "fold0",
    label: str = "iter_001",
    L_align: float = 0.12,
) -> Path:
    fold_dir = _write_fold_quality_artifacts(parent, fold=fold)
    final = fold_dir / "final"
    metrics = {
        "L_align": L_align,
        "rho_align_pearson": 0.34,
        "rho_align_spearman": 0.32,
        "delta_tag": 0.21,
    }
    (final / "metrics_raw.json").write_text(json.dumps(metrics), encoding="utf-8")
    (final / "best_iter.txt").write_text(label, encoding="utf-8")
    (fold_dir / "selection.json").write_text(
        json.dumps({"selected_candidate": {"label": label}}),
        encoding="utf-8",
    )
    (fold_dir / label).mkdir()
    return fold_dir


def _significant_agg() -> dict:
    return {
        "pooled": {
            "n_pairs": 25,
            "rho_spearman": {
                "observed": 0.31,
                "p_two_sided": 0.03,
            },
        }
    }


def test_kfold_quality_gate_artifact_marks_research_grade_pass(tmp_path):
    fold_dir = _write_fold_quality_artifacts(tmp_path)

    gate = part2_run._build_kfold_quality_gate(
        parent_dir=str(tmp_path),
        exp_config=_strict_exp_config(),
        agg_result=_significant_agg(),
        fold_dirs=[str(fold_dir)],
    )

    saved = json.loads((tmp_path / "agg" / "quality_gate.json").read_text())
    assert gate["status"] == "pass"
    assert gate["research_grade"] is True
    assert gate["failures"] == []
    assert saved["research_grade"] is True


def test_completed_fold_summary_reads_final_artifacts(tmp_path):
    fold_dir = _write_completed_fold(
        tmp_path,
        label="iter_004",
        L_align=0.087,
    )

    summary = part2_run._completed_fold_summary(fold_dir, 0)

    assert summary is not None
    assert summary["fold"] == 0
    assert summary["best_label"] == "iter_004"
    assert summary["L_align"] == 0.087
    assert summary["iterations"] == 1
    assert summary["resumed"] is True


def test_run_part2_v3_kfold_resume_skips_completed_fold(tmp_path, monkeypatch):
    parent = tmp_path / "run_cv_resume"
    _write_completed_fold(parent, fold="fold0", label="iter_000_baseline_static")
    called_folds: list[int] = []

    def fake_run_part1(cfg, *, run_dir=None, **_kwargs):
        fold = int(cfg["splits"]["fold"])
        called_folds.append(fold)
        _write_completed_fold(parent, fold=Path(run_dir).name, label="iter_001", L_align=0.22)
        result = _make_iter_result(label="iter_001", L_align=0.22)
        result.rho_align_pearson = 0.34
        result.rho_align_spearman = 0.32
        result.delta_tag = 0.21
        return [result], result

    from autotagging_loop.scripts import permutation_test_run

    def fake_run_pooled(*, fold_dirs, out_path):
        payload = _significant_agg()
        Path(out_path).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(part2_run, "_init_wandb_v3", lambda _config: None)
    monkeypatch.setattr(permutation_test_run, "run_pooled", fake_run_pooled)
    result = part2_run._run_part2_v3_kfold(
        config={"wandb": False, "resume_run_dir": str(parent)},
        exp_config={**_strict_exp_config(), "results_dir": str(tmp_path)},
        exp_corpus=None,
        cv_folds=2,
        run_part1=fake_run_part1,
    )

    assert called_folds == [1]
    assert [row["fold"] for row in result["fold_summaries"]] == [0, 1]
    assert result["fold_summaries"][0]["resumed"] is True
    assert (parent / "agg" / "fold_summaries.json").exists()


def test_kfold_quality_gate_artifact_rejects_relaxed_thin_fold(tmp_path):
    fold_dir = _write_fold_quality_artifacts(
        tmp_path,
        dev_pairs=1,
        test_pairs=4,
        rho_s=float("nan"),
        fallbacks=2,
    )
    config = _strict_exp_config()
    config.update(
        {
            "min_common_models": 4,
            "v_loop_min_dev_valid_pairs": 1,
            "v_loop_min_test_valid_pairs": 1,
        }
    )

    gate = part2_run._build_kfold_quality_gate(
        parent_dir=str(tmp_path),
        exp_config=config,
        agg_result={"pooled": {"n_pairs": 10, "rho_spearman": {"observed": 0.1, "p_two_sided": 0.8}}},
        fold_dirs=[str(fold_dir)],
    )

    assert gate["status"] == "fail"
    assert gate["research_grade"] is False
    assert "strict_config_min_common_models:4<6" in gate["failures"]
    assert "quality_pooled_rho_s_below_floor:0.1000<0.2000" in gate["failures"]
    assert "fold0:quality_selection_dev_n_pairs:1<10" in gate["failures"]
    assert "fold0:quality_selection_dev_rho_s_not_finite" in gate["failures"]
    assert "fold0:quality_test_n_pairs:4<10" in gate["failures"]
    assert "fold0:quality_llm_fallbacks:2" in gate["failures"]


def test_kfold_quality_gate_rejects_low_fold_rho_with_enough_pairs(tmp_path):
    fold_dir = _write_fold_quality_artifacts(
        tmp_path,
        dev_pairs=10,
        test_pairs=11,
        held_pairs=12,
        rho_s=-0.01,
    )

    gate = part2_run._build_kfold_quality_gate(
        parent_dir=str(tmp_path),
        exp_config=_strict_exp_config(),
        agg_result=_significant_agg(),
        fold_dirs=[str(fold_dir)],
    )

    assert gate["status"] == "fail"
    assert gate["research_grade"] is False
    assert gate["thresholds"]["min_fold_rho_s"] == 0.0


def test_kfold_quality_gate_rejects_low_effective_benchmark_coverage(tmp_path):
    fold_dir = _write_fold_quality_artifacts(
        tmp_path,
        dev_pairs=10,
        test_pairs=11,
        held_pairs=12,
        effective_benchmarks=5,
    )

    gate = part2_run._build_kfold_quality_gate(
        parent_dir=str(tmp_path),
        exp_config=_strict_exp_config(),
        agg_result=_significant_agg(),
        fold_dirs=[str(fold_dir)],
    )

    assert gate["status"] == "fail"
    assert gate["research_grade"] is False
    assert gate["thresholds"]["min_effective_benchmarks"] == 6
    assert (
        "fold0:quality_test_n_effective_benchmarks:5<6"
        in gate["failures"]
    )


def test_kfold_quality_gate_rejects_role_output_leakage(tmp_path):
    fold_dir = _write_fold_quality_artifacts(tmp_path)
    (fold_dir / "corpus.json").write_text(
        json.dumps({"benchmark_names": ["MMLU-Pro"]}),
        encoding="utf-8",
    )
    vocab_path = fold_dir / "iter_001" / "V.json"
    vocab_path.parent.mkdir(parents=True)
    vocab_path.write_text(
        json.dumps({
            "vocab": [
                {
                    "id": "mmlu_pro_difficulty",
                    "name": "MMLU Pro Difficulty",
                    "definition": "Groups benchmark items by leaderboard difficulty.",
                }
            ]
        }),
        encoding="utf-8",
    )

    gate = part2_run._build_kfold_quality_gate(
        parent_dir=str(tmp_path),
        exp_config=_strict_exp_config(),
        agg_result=_significant_agg(),
        fold_dirs=[str(fold_dir)],
    )

    assert gate["status"] == "fail"
    assert gate["research_grade"] is False
    assert gate["thresholds"]["role_quality_required"] is True
    assert gate["folds"][0]["role_quality_failures"] >= 1
    assert any("fold0:role_quality_vocab" in failure for failure in gate["failures"])


def test_run_part2_legacy_path_unchanged_when_v_loop_disabled(part2_v3_config, monkeypatch):
    """Legacy single-shot path is selected when enable_v_loop=False."""
    config = dict(part2_v3_config)
    config["enable_v_loop"] = False

    monkeypatch.setattr(
        part2_run, "_run_part2_v3",
        lambda _c: pytest.fail("v3 path should NOT run when enable_v_loop=False"),
    )
    # Stub the legacy body just enough to confirm the legacy branch is taken.
    sentinel = {"legacy_called": False}
    def fake_legacy_make_run_dir(_results_dir):
        sentinel["legacy_called"] = True
        raise RuntimeError("stop legacy here")
    monkeypatch.setattr(part2_run, "make_run_dir", fake_legacy_make_run_dir)

    with pytest.raises(RuntimeError, match="stop legacy here"):
        part2_run.run_part2(config)
    assert sentinel["legacy_called"] is True


def test_run_part2_v3_path_invokes_experiment_loop(part2_v3_config, monkeypatch):
    """enable_v_loop=True dispatches to experiment.loop.run_part1."""
    part2_v3_config["llm_request_timeout_s"] = 12.5
    part2_v3_config["llm_sdk_exception_retries"] = 1
    part2_v3_config["no_seed_taxonomy_enabled"] = True
    part2_v3_config["no_seed_taxonomy_min_tags"] = 6
    part2_v3_config["taxonomy_refinement_enabled"] = True
    part2_v3_config["taxonomy_refinement_retain_seed_tags"] = False
    part2_v3_config["delta_tag_threshold"] = -1.0
    part2_v3_config["best_iter_selection"] = "dev_l_align"
    part2_v3_config["best_iter_dev_rho_drop_tolerance"] = 0.2
    part2_v3_config["best_iter_train_l_increase_tolerance"] = 0.01
    part2_v3_config["best_iter_train_rho_drop_tolerance"] = 0.10
    part2_v3_config["best_iter_train_rho_floor"] = 0.2
    part2_v3_config["best_iter_stability_rho_weight"] = 0.42
    part2_v3_config["best_iter_model_probe_enabled"] = True
    part2_v3_config["best_iter_model_probe_min_common"] = 5
    part2_v3_config["best_iter_model_probe_dev_rho_floor"] = 0.0
    part2_v3_config["best_iter_model_probe_dev_rho_drop_tolerance"] = 0.30
    part2_v3_config["best_iter_model_probe_dev_l_increase_tolerance"] = 0.05
    part2_v3_config["taxonomy_selection_enabled"] = True
    part2_v3_config["taxonomy_selection_min_tags"] = 4
    part2_v3_config["v_loop_min_train_effective_benchmarks"] = 6
    part2_v3_config["v_loop_min_dev_effective_benchmarks"] = 6
    part2_v3_config["v_loop_min_test_effective_benchmarks"] = 6
    part2_v3_config["executer_candidate_counts"] = [6, 8, 10, 12]
    captured = {}

    def fake_run_part1(cfg, corpus=None, **_kw):
        captured["cfg"] = cfg
        captured["corpus"] = corpus
        result = _make_iter_result(label="iter_001", L_align=0.123)
        return [result], result

    monkeypatch.setattr("autotagging_loop.experiment.loop.run_part1", fake_run_part1)

    out = part2_run.run_part2(part2_v3_config)
    assert out["mode"] == "v3"
    assert out["best_label"] == "iter_001"
    assert captured["corpus"] is not None
    cfg = captured["cfg"]
    assert cfg["enable_v_loop"] is True
    assert "executer_source_benchmark" not in cfg, (
        "executer_source_benchmark is obsolete — source is the train split"
    )
    assert cfg["vocab_path"] == part2_v3_config["vocab_path"]
    assert cfg["labels_dir"] == part2_v3_config["labels_dir"]
    assert cfg["llm_request_timeout_s"] == 12.5
    assert cfg["llm_sdk_exception_retries"] == 1
    assert cfg["no_seed_taxonomy_enabled"] is True
    assert cfg["no_seed_taxonomy_min_tags"] == 6
    assert cfg["taxonomy_refinement_enabled"] is True
    assert cfg["taxonomy_refinement_retain_seed_tags"] is False
    assert cfg["delta_tag_threshold"] == -1.0
    assert cfg["best_iter_selection"] == "dev_l_align"
    assert cfg["best_iter_dev_rho_drop_tolerance"] == 0.2
    assert cfg["best_iter_train_l_increase_tolerance"] == 0.01
    assert cfg["best_iter_train_rho_drop_tolerance"] == 0.10
    assert cfg["best_iter_train_rho_floor"] == 0.2
    assert cfg["best_iter_stability_rho_weight"] == 0.42
    assert cfg["best_iter_model_probe_enabled"] is True
    assert cfg["best_iter_model_probe_min_common"] == 5
    assert cfg["best_iter_model_probe_dev_rho_floor"] == 0.0
    assert cfg["best_iter_model_probe_dev_rho_drop_tolerance"] == 0.30
    assert cfg["best_iter_model_probe_dev_l_increase_tolerance"] == 0.05
    assert cfg["taxonomy_selection_enabled"] is True
    assert cfg["taxonomy_selection_min_tags"] == 4
    assert cfg["v_loop_min_train_effective_benchmarks"] == 6
    assert cfg["v_loop_min_dev_effective_benchmarks"] == 6
    assert cfg["v_loop_min_test_effective_benchmarks"] == 6
    assert cfg["executer_candidate_counts"] == [6, 8, 10, 12]
    # prompt_i0_path falls back to experiment/prompts/I_exec_seed.txt when Part 1 product missing.
    # Never I0.txt or tag_mapper.txt — those are fixed-vocab tagger prompts.
    assert os.path.basename(cfg["prompt_i0_path"]) == "I_exec_seed.txt"


def test_run_part2_v3_ignores_part1_best_prompt_path(part2_v3_config, monkeypatch, tmp_path):
    """Part 1's I_star.txt is a fixed-vocab tagging prompt and produces
    weights{} responses from the Executer. _build_v3_overrides must not honour
    a stale ``part1_best_prompt_path`` even when the file exists on disk —
    the v_loop seed is always experiment/prompts/I_exec_seed.txt."""
    stale = tmp_path / "stale_I_star.txt"
    stale.write_text(
        "Tag this benchmark and return JSON with weights for each ability id.",
        encoding="utf-8",
    )

    config = dict(part2_v3_config)
    config["part1_best_prompt_path"] = str(stale)

    captured = {}

    def fake_run_part1(cfg, corpus=None, **_kw):
        captured["cfg"] = cfg
        return [_make_iter_result()], _make_iter_result()

    monkeypatch.setattr("autotagging_loop.experiment.loop.run_part1", fake_run_part1)
    part2_run.run_part2(config)

    cfg = captured["cfg"]
    assert os.path.basename(cfg["prompt_i0_path"]) == "I_exec_seed.txt"
    assert str(stale) not in cfg["prompt_i0_path"]


def test_run_part2_v3_translates_corpus_class(part2_v3_config, monkeypatch):
    """Adapter must hand experiment.corpus.Corpus to run_part1, not Part 2's class."""
    from autotagging_loop.experiment.corpus import Corpus as ExpCorpus

    received = {}

    def fake_run_part1(cfg, corpus=None, **_kw):
        received["corpus_type"] = type(corpus).__name__
        received["corpus_module"] = type(corpus).__module__
        return [_make_iter_result()], _make_iter_result()

    monkeypatch.setattr("autotagging_loop.experiment.loop.run_part1", fake_run_part1)
    part2_run.run_part2(part2_v3_config)

    assert received["corpus_type"] == "Corpus"
    assert received["corpus_module"] == ExpCorpus.__module__


def test_run_part2_v3_kfold_preflight_fails_before_llm_calls(part2_v3_config, monkeypatch):
    config = dict(part2_v3_config)
    config["splits"] = {
        "cv_folds": 2,
        "fold": 0,
        "benchmark_seed": 0,
        "dev_train_split": [1.0, 1.0],
    }
    config["v_loop_min_train_valid_pairs"] = 999
    config["v_loop_min_dev_valid_pairs"] = 999
    config["v_loop_min_test_valid_pairs"] = 999

    def fail_run_part1(*_args, **_kwargs):  # pragma: no cover - regression assertion
        raise AssertionError("run_part1 should not be called after split preflight failure")

    monkeypatch.setattr("autotagging_loop.experiment.loop.run_part1", fail_run_part1)

    with pytest.raises(ValueError, match="k-fold split preflight failed before LLM calls"):
        part2_run.run_part2(config)


def test_run_part2_v3_kfold_held_model_preflight_can_be_required(part2_v3_config, monkeypatch):
    config = dict(part2_v3_config)
    config["splits"] = {
        "cv_folds": 2,
        "fold": 0,
        "benchmark_seed": 0,
        "dev_train_split": [1.0, 1.0],
    }
    config["v_loop_require_held_model_test"] = True

    def fail_run_part1(*_args, **_kwargs):  # pragma: no cover - regression assertion
        raise AssertionError("run_part1 should not be called after held-model preflight failure")

    monkeypatch.setattr("autotagging_loop.experiment.loop.run_part1", fail_run_part1)

    with pytest.raises(ValueError, match="k-fold held-model preflight failed before LLM calls"):
        part2_run.run_part2(config)


def test_research_grade_preflight_rejects_relaxed_config(part2_v3_config):
    config = dict(part2_v3_config)
    config["splits"] = {
        "cv_folds": 2,
        "fold": 0,
        "benchmark_seed": 0,
        "dev_train_split": [1.0, 1.0],
    }
    config["min_common_models"] = 4
    config["v_loop_min_train_valid_pairs"] = 1
    config["v_loop_min_dev_valid_pairs"] = 1
    config["v_loop_min_test_valid_pairs"] = 1
    config["v_loop_min_train_effective_benchmarks"] = 1
    config["v_loop_min_dev_effective_benchmarks"] = 1
    config["v_loop_min_test_effective_benchmarks"] = 1

    report = part2_run.preflight_research_grade(config)

    assert report["ok"] is False
    assert "strict_config_min_common_models:4<6" in report["failures"]
    assert "strict_config_v_loop_min_train_valid_pairs:1<10" in report["failures"]
    assert (
        "strict_config_v_loop_min_train_effective_benchmarks:1<6"
        in report["failures"]
    )


def test_score_backfill_readiness_reports_missing_and_incomplete(tmp_path):
    missing_csv = tmp_path / "missing.csv"
    missing_csv.write_text(
        "scope,benchmark,model\nseen,Bench A,Model A\n",
        encoding="utf-8",
    )
    curated_path = tmp_path / "curated_score_backfill.json"

    failures = part2_run._score_backfill_readiness_failures(
        {
            "curated_score_backfill_path": str(curated_path),
            "use_curated_score_backfill": True,
        },
        missing_csv=missing_csv,
    )

    assert f"research_preflight_curated_score_backfill_missing:{curated_path}" in failures
    assert (
        "research_preflight_curated_score_backfill_incomplete:"
        "0/1(missing=1,incomplete=0)"
    ) in failures


def test_score_backfill_readiness_accepts_disabled_curated_backfill(tmp_path):
    curated_path = tmp_path / "curated_score_backfill.json"

    failures = part2_run._score_backfill_readiness_failures(
        {
            "curated_score_backfill_path": str(curated_path),
            "use_curated_score_backfill": False,
        },
        missing_csv=tmp_path / "missing.csv",
    )

    assert failures == []


def test_score_backfill_readiness_accepts_complete_planned_cell(tmp_path):
    missing_csv = tmp_path / "missing.csv"
    missing_csv.write_text(
        "scope,benchmark,model\nseen,Bench A,Model A\n",
        encoding="utf-8",
    )
    curated_path = tmp_path / "curated_score_backfill.json"
    curated_path.write_text(
        json.dumps({
            "scores": [
                {
                    "benchmark": "Bench A",
                    "model": "Model A",
                    "score": 0.73,
                    "metric": "accuracy",
                    "scale": "0-1",
                    "source": {
                        "title": "Benchmark Report",
                        "url": "https://reports.example.org/bench-a",
                        "date": "2026-06-01",
                    },
                }
            ]
        }),
        encoding="utf-8",
    )

    failures = part2_run._score_backfill_readiness_failures(
        {
            "curated_score_backfill_path": str(curated_path),
            "use_curated_score_backfill": True,
        },
        missing_csv=missing_csv,
    )

    assert failures == []


def test_score_backfill_readiness_reports_outside_and_duplicate_cells(tmp_path):
    missing_csv = tmp_path / "missing.csv"
    missing_csv.write_text(
        "scope,benchmark,model\nseen,Bench A,Model A\n",
        encoding="utf-8",
    )
    curated_path = tmp_path / "curated_score_backfill.json"
    valid_record = {
        "benchmark": "Bench A",
        "model": "Model A",
        "score": 0.73,
        "metric": "accuracy",
        "scale": "0-1",
        "source": {
            "title": "Benchmark Report",
            "url": "https://reports.example.org/bench-a",
            "date": "2026-06-01",
        },
    }
    outside_record = dict(valid_record)
    outside_record["benchmark"] = "Bench B"
    curated_path.write_text(
        json.dumps({"scores": [valid_record, valid_record, outside_record]}),
        encoding="utf-8",
    )

    failures = part2_run._score_backfill_readiness_failures(
        {
            "curated_score_backfill_path": str(curated_path),
            "use_curated_score_backfill": True,
        },
        missing_csv=missing_csv,
    )

    assert any(
        failure.startswith("research_preflight_curated_score_backfill_outside_plan:1:")
        for failure in failures
    )
    assert any(
        failure.startswith("research_preflight_curated_score_backfill_duplicates:1:")
        for failure in failures
    )


def test_score_backfill_readiness_reports_plan_drift_and_projection(
    tmp_path,
    monkeypatch,
):
    from autotagging_loop.scripts import validate_score_backfill as validator

    curated_path = tmp_path / "curated_score_backfill.json"
    curated_path.write_text(json.dumps({"scores": []}), encoding="utf-8")
    expected = {("bencha", "Model A"): {"scope": "seen", "benchmark": "Bench A", "model": "Model A"}}

    monkeypatch.setattr(
        validator,
        "load_missing_cell_plan",
        lambda *_args, **_kwargs: expected,
    )
    monkeypatch.setattr(
        validator,
        "curated_backfill_progress",
        lambda *_args, **_kwargs: {
            "expected_cells": 1,
            "present_cells": 1,
            "complete_cells": 1,
            "incomplete_cells": 0,
            "missing_cells": 0,
            "outside_plan": [],
            "duplicate_cells": [],
        },
    )
    monkeypatch.setattr(
        validator,
        "_load_scores_from_config",
        lambda **_kwargs: ({"Bench A": {"Model A": 0.73}}, []),
    )
    monkeypatch.setattr(
        validator,
        "_top_core_plans",
        lambda *_args, **_kwargs: [{"missing_seen": [], "missing_held": []}],
    )
    monkeypatch.setattr(
        validator,
        "missing_plan_drift",
        lambda *_args, **_kwargs: {
            "status": "FAIL",
            "failures": ["missing_plan_extra_in_csv:1"],
        },
    )
    monkeypatch.setattr(
        validator,
        "_planned_backfill_projection",
        lambda **_kwargs: {"configured": {"status": "FAIL"}},
    )
    monkeypatch.setattr(
        validator,
        "planned_projection_failures",
        lambda _projection: ["planned_projection_configured_failed"],
    )

    failures = part2_run._score_backfill_readiness_failures(
        {
            "curated_score_backfill_path": str(curated_path),
            "use_curated_score_backfill": True,
            "min_common_models": 6,
        },
        missing_csv=Path("data") / "score_backfill_missing.csv",
    )

    assert (
        "research_preflight_missing_plan_drift:missing_plan_extra_in_csv:1"
        in failures
    )
    assert "research_preflight_planned_projection_configured_failed" in failures


def test_run_part2_require_research_grade_fails_before_llm_calls(
    part2_v3_config,
    monkeypatch,
):
    config = dict(part2_v3_config)
    config["splits"] = {
        "cv_folds": 2,
        "fold": 0,
        "benchmark_seed": 0,
        "dev_train_split": [1.0, 1.0],
    }
    config["min_common_models"] = 4

    def fail_run_part1(*_args, **_kwargs):  # pragma: no cover - regression assertion
        raise AssertionError("run_part1 should not be called after research preflight failure")

    monkeypatch.setattr("autotagging_loop.experiment.loop.run_part1", fail_run_part1)

    with pytest.raises(ValueError, match="research-grade preflight failed before LLM calls"):
        part2_run.run_part2(config, require_research_grade=True)


def test_run_part2_require_research_grade_rejects_failed_quality_gate(monkeypatch):
    config = {"enable_v_loop": True}
    monkeypatch.setattr(
        part2_run,
        "preflight_research_grade",
        lambda _config: {"ok": True, "failures": []},
    )
    monkeypatch.setattr(
        part2_run,
        "_run_part2_v3",
        lambda _config: {
            "mode": "v3_kfold",
            "quality_gate": {
                "research_grade": False,
                "failures": ["quality_pooled_p_two_above_alpha:0.50>0.05"],
            },
        },
    )

    with pytest.raises(RuntimeError, match="research-grade quality gate failed after run"):
        part2_run.run_part2(config, require_research_grade=True)


def test_run_part2_require_research_grade_accepts_passing_quality_gate(monkeypatch):
    config = {"enable_v_loop": True}
    expected = {
        "mode": "v3_kfold",
        "quality_gate": {"research_grade": True, "failures": []},
    }
    monkeypatch.setattr(
        part2_run,
        "preflight_research_grade",
        lambda _config: {"ok": True, "failures": []},
    )
    monkeypatch.setattr(part2_run, "_run_part2_v3", lambda _config: expected)

    assert part2_run.run_part2(config, require_research_grade=True) is expected


def test_kfold_preflight_uses_seen_models_when_held_model_test_required():
    from autotagging_loop.experiment.corpus import Corpus as ExpCorpus

    benches = [f"B{i}" for i in range(6)]
    seen = [f"s{i}" for i in range(6)]
    held = [f"h{i}" for i in range(3)]
    # B0/B1/B2 only have held models plus a single seen model; full-model score
    # pairs pass min_common=3, but F_seen-only pairs must be too sparse.
    Y = {
        "B0": {**{m: float(i) for i, m in enumerate(held)}, "s0": 0.1},
        "B1": {**{m: float(i + 1) for i, m in enumerate(held)}, "s1": 0.2},
        "B2": {**{m: float(i + 2) for i, m in enumerate(held)}, "s2": 0.3},
        "B3": {m: float(i + 3) for i, m in enumerate(seen)},
        "B4": {m: float(i + 4) for i, m in enumerate(seen)},
        "B5": {m: float(i + 5) for i, m in enumerate(seen)},
    }
    corpus = ExpCorpus(
        benchmark_names=benches,
        model_names=seen + held,
        Y=Y,
        descriptions={b: "" for b in benches},
        documents={},
        drop_log={},
    )
    config = {
        "normalize": "rank",
        "min_common_models": 3,
        "min_common_models_warn": 2,
        "v_loop_min_train_valid_pairs": 1,
        "v_loop_min_dev_valid_pairs": 0,
        "v_loop_min_test_valid_pairs": 0,
        "v_loop_require_held_model_test": True,
        "splits": {
            "cv_folds": 2,
            "fold": 0,
            "benchmark_seed": 0,
            "dev_train_split": [0.0, 1.0],
            "model_ratios": [2.0 / 3.0, 1.0 / 3.0],
            "model_seed": 3,
        },
    }

    with pytest.raises(ValueError, match="k-fold split preflight failed before LLM calls"):
        part2_run._preflight_v3_kfold_splits(config, corpus, cv_folds=2)
