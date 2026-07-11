"""Smoke + monotonicity tests for experiment/loop.py."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from autotagging_loop.experiment.config import load_experiment_config
from autotagging_loop.experiment.corpus import Corpus
from autotagging_loop.experiment.json_contract import JSONContractError
from autotagging_loop.experiment.loop import run_part1
from autotagging_loop.experiment.no_seed_taxonomy import NoSeedTaxonomyResult
from autotagging_loop.experiment.prompt_improver import ImproverResult
from autotagging_loop.experiment.splits import split_benchmarks
from autotagging_loop.experiment.tag_generator import TagVector
from autotagging_loop.experiment.taxonomy_refiner import TaxonomyRefinementResult


VOCAB = [
    {"id": "analogical_reasoning"},
    {"id": "deductive_reasoning"},
    {"id": "inductive_reasoning"},
    {"id": "long_term_knowledge_recall"},
    {"id": "quantitative_reasoning"},
]


@pytest.fixture
def tmp_resources(tmp_path):
    """Fake vocab + I_0 + leaderboard inside tmp dir."""
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(VOCAB), encoding="utf-8")

    i0_path = tmp_path / "I0.txt"
    i0_path.write_text(
        "Tag with " + ", ".join(v["id"] for v in VOCAB) + ". Provide weighted scores.",
        encoding="utf-8",
    )

    Y = {
        "BenchA": {"m1": 0.9, "m2": 0.8, "m3": 0.7, "m4": 0.6, "m5": 0.5, "m6": 0.4, "m7": 0.3},
        "BenchB": {"m1": 0.85, "m2": 0.78, "m3": 0.72, "m4": 0.61, "m5": 0.50, "m6": 0.41, "m7": 0.31},
        "BenchC": {"m1": 0.4, "m2": 0.5, "m3": 0.6, "m4": 0.7, "m5": 0.8, "m6": 0.9, "m7": 0.95},
        "BenchD": {"m1": 0.3, "m2": 0.4, "m3": 0.5, "m4": 0.6, "m5": 0.7, "m6": 0.8, "m7": 0.85},
        "BenchE": {"m1": 0.5, "m2": 0.55, "m3": 0.45, "m4": 0.6, "m5": 0.5, "m6": 0.55, "m7": 0.5},
    }
    lb_path = tmp_path / "lb.json"
    lb_path.write_text(json.dumps(Y), encoding="utf-8")

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return {
        "vocab_path": str(vocab_path),
        "prompt_i0_path": str(i0_path),
        "leaderboard_path": str(lb_path),
        "results_dir": str(results_dir),
    }


def _fixed_tag_fn(weights_per_bench: dict[str, dict[str, float]]):
    def fn(benchmark, description, vocab, prompt, version):
        return TagVector(benchmark=benchmark,
                         weights=dict(weights_per_bench[benchmark]),
                         raw_response="{}",
                         prompt_version=version)
    return fn


def _taxonomy_test_weights() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    ids = [v["id"] for v in VOCAB]

    def full(values: dict[str, float]) -> dict[str, float]:
        return {tag_id: float(values.get(tag_id, 0.0)) for tag_id in ids}

    good = {
        "BenchA": full({"analogical_reasoning": 1.0}),
        "BenchB": full({"analogical_reasoning": 0.95, "deductive_reasoning": 0.05}),
        "BenchC": full({"quantitative_reasoning": 1.0}),
        "BenchD": full({"quantitative_reasoning": 0.95, "deductive_reasoning": 0.05}),
        "BenchE": full({
            "analogical_reasoning": 0.3,
            "deductive_reasoning": 0.3,
            "quantitative_reasoning": 0.3,
        }),
    }
    mid = {
        "BenchA": full({"analogical_reasoning": 0.7, "quantitative_reasoning": 0.3}),
        "BenchB": full({"analogical_reasoning": 0.6, "quantitative_reasoning": 0.4}),
        "BenchC": full({"analogical_reasoning": 0.3, "quantitative_reasoning": 0.7}),
        "BenchD": full({"analogical_reasoning": 0.4, "quantitative_reasoning": 0.6}),
        "BenchE": full({
            "analogical_reasoning": 0.3,
            "deductive_reasoning": 0.4,
            "quantitative_reasoning": 0.3,
        }),
    }
    bad = {
        bench: full({
            "analogical_reasoning": (idx + 1) / 5.0,
            "quantitative_reasoning": (5 - idx) / 5.0,
            "deductive_reasoning": 0.2,
        })
        for idx, bench in enumerate(["BenchA", "BenchB", "BenchC", "BenchD", "BenchE"])
    }
    return good, mid, bad


def _accepted_taxonomy_refiner(**kwargs):
    prompt = (
        "Tag with analogical_reasoning, deductive_reasoning, inductive_reasoning, "
        "long_term_knowledge_recall, and quantitative_reasoning. Return JSON weights "
        "for every listed tag id."
    )
    return TaxonomyRefinementResult(
        vocab=VOCAB,
        prompt=prompt,
        accepted=True,
        reasons=[],
        raw_response="{}",
        rationale="residuals remained large",
    )


def test_smoke_baselines_and_one_iter(tmp_resources):
    overrides = {
        **tmp_resources,
        "max_iter": 1,
        "bootstrap_B": 20,
        "min_common_models": 6,
        "min_common_models_warn": 5,
        "optimize_tag_weights": False,
        "taxonomy_refinement_enabled": False,
    }
    config = load_experiment_config(overrides)

    fixed = {
        "BenchA": {"analogical_reasoning": 0.9, "deductive_reasoning": 0.1, "inductive_reasoning": 0.0,
                    "long_term_knowledge_recall": 0.2, "quantitative_reasoning": 0.0},
        "BenchB": {"analogical_reasoning": 0.85, "deductive_reasoning": 0.05, "inductive_reasoning": 0.05,
                    "long_term_knowledge_recall": 0.25, "quantitative_reasoning": 0.0},
        "BenchC": {"analogical_reasoning": 0.0, "deductive_reasoning": 0.7, "inductive_reasoning": 0.6,
                    "long_term_knowledge_recall": 0.1, "quantitative_reasoning": 0.5},
        "BenchD": {"analogical_reasoning": 0.0, "deductive_reasoning": 0.65, "inductive_reasoning": 0.55,
                    "long_term_knowledge_recall": 0.05, "quantitative_reasoning": 0.55},
        "BenchE": {"analogical_reasoning": 0.3, "deductive_reasoning": 0.3, "inductive_reasoning": 0.3,
                    "long_term_knowledge_recall": 0.3, "quantitative_reasoning": 0.3},
    }

    history, best = run_part1(
        config,
        tag_fn=_fixed_tag_fn(fixed),
    )
    assert any(h.label == "iter_000_baseline_static" for h in history)
    assert any(h.label == "iter_000_baseline_random" for h in history)
    assert any(h.label == "iter_001" for h in history)

    # final/ files exist
    run_dirs = os.listdir(tmp_resources["results_dir"])
    assert len(run_dirs) == 1
    final_dir = os.path.join(tmp_resources["results_dir"], run_dirs[0], "final")
    assert os.path.isfile(os.path.join(final_dir, "I_star.txt"))
    assert os.path.isfile(os.path.join(final_dir, "T_star.json"))
    assert os.path.isfile(os.path.join(final_dir, "T_star_raw.json"))
    assert os.path.isfile(os.path.join(final_dir, "metrics_with_bootstrap.json"))
    assert os.path.isfile(os.path.join(final_dir, "metrics_raw.json"))
    assert os.path.isfile(os.path.join(final_dir, "residual_report.json"))
    assert os.path.isfile(os.path.join(final_dir, "profile_support.json"))

    with open(os.path.join(final_dir, "profile_support.json"), encoding="utf-8") as f:
        profile_support = json.load(f)
    assert profile_support["full_profile_formula"] == "P_r = Y_norm[r, :] @ T_star"
    assert profile_support["subsets"]


def test_static_mapreduce_weight_mode_runs_without_direct_tagger(tmp_resources):
    overrides = {
        **tmp_resources,
        "max_iter": 1,
        "bootstrap_B": 5,
        "min_common_models": 6,
        "run_baseline": False,
        "tag_weight_mode": "static_from_mapreduce",
        "static_tag_require_mapreduce": True,
        "static_tag_restrict_to_labeled_benchmarks": True,
        "mapreduce_chunk_examples": 1,
        "mapreduce_max_chunk_chars": 2000,
        "mapreduce_max_evidence_chars": 4000,
        "mapreduce_max_workers": 1,
        "mapreduce_cache_enabled": False,
        "taxonomy_refinement_enabled": False,
    }
    config = load_experiment_config(overrides)
    Y = {
        "BenchA": {"m1": 0.9, "m2": 0.8, "m3": 0.7, "m4": 0.6, "m5": 0.5, "m6": 0.4},
        "BenchB": {"m1": 0.88, "m2": 0.79, "m3": 0.71, "m4": 0.61, "m5": 0.51, "m6": 0.41},
        "BenchC": {"m1": 0.4, "m2": 0.5, "m3": 0.6, "m4": 0.7, "m5": 0.8, "m6": 0.9},
    }
    corpus = Corpus(
        benchmark_names=["BenchA", "BenchB", "BenchC"],
        model_names=[f"m{i}" for i in range(1, 7)],
        Y=Y,
        descriptions={b: "" for b in Y},
        documents={
            b: {
                "reviewed_rows": 2,
                "topic_counts": {},
                "reasoning_depth_counts": {},
                "answer_format_counts": {},
                "examples": [f"{b} question {i}" for i in range(2)],
            }
            for b in Y
        },
    )
    calls = {"map": 0, "reducer": 0}

    def map_chat(_system, user):
        calls["map"] += 1
        return json.dumps({
            "chunk_summary": "synthetic evidence",
            "task_patterns": [],
            "reasoning_patterns": [],
            "justifications": ["synthetic descriptive evidence"],
        })

    def reducer_chat(_system, user):
        calls["reducer"] += 1
        assert "aggregate_chunk_ability_scores" not in user
        assert "weight_formula" not in user
        if "Benchmark: BenchC" in user:
            levels = {
                "analogical_reasoning": "absent",
                "deductive_reasoning": "weak",
                "inductive_reasoning": "weak",
                "long_term_knowledge_recall": "absent",
                "quantitative_reasoning": "dominant",
            }
        else:
            levels = {
                "analogical_reasoning": "dominant",
                "deductive_reasoning": "strong",
                "inductive_reasoning": "medium",
                "long_term_knowledge_recall": "weak",
                "quantitative_reasoning": "absent",
            }
        return json.dumps({
            "benchmark_summary": "benchmark-level synthesis",
            "ability_levels": levels,
            "ability_rationale": {tag["id"]: "reduced evidence" for tag in VOCAB},
        })

    history, best = run_part1(
        config,
        corpus=corpus,
        mapreduce_chat_fn=map_chat,
        mapreduce_reducer_chat_fn=reducer_chat,
    )

    assert calls["map"] == 6
    assert calls["reducer"] == 3
    assert best.label == "iter_001"
    assert history[0].T["BenchA"]["analogical_reasoning"] == 1.0
    assert history[0].T["BenchA"]["deductive_reasoning"] == 0.75
    assert history[0].tag_weight_metadata["method"] == "mapreduce_llm_reducer_static_weights"
    assert history[0].tag_weight_metadata["prompt_drives"].startswith("current prompt drives")

    run_dirs = os.listdir(tmp_resources["results_dir"])
    final_dir = os.path.join(tmp_resources["results_dir"], run_dirs[0], "final")
    assert os.path.isfile(os.path.join(final_dir, "tag_weight_metadata.json"))


def test_static_mapreduce_reuses_fixed_mapper_cache_across_prompt_iterations(tmp_resources):
    overrides = {
        **tmp_resources,
        "max_iter": 2,
        "bootstrap_B": 5,
        "min_common_models": 6,
        "run_baseline": False,
        "early_stop_consecutive": 999,
        "tag_weight_mode": "static_from_mapreduce",
        "static_tag_require_mapreduce": True,
        "static_tag_restrict_to_labeled_benchmarks": True,
        "mapreduce_chunk_examples": 1,
        "mapreduce_max_chunk_chars": 2000,
        "mapreduce_max_evidence_chars": 4000,
        "mapreduce_max_workers": 1,
        "mapreduce_cache_enabled": True,
        "mapreduce_cache_dir": os.path.join(tmp_resources["results_dir"], "_cache", "mapreduce"),
        "taxonomy_refinement_enabled": False,
    }
    config = load_experiment_config(overrides)
    Y = {
        "BenchA": {"m1": 0.9, "m2": 0.8, "m3": 0.7, "m4": 0.6, "m5": 0.5, "m6": 0.4},
        "BenchB": {"m1": 0.88, "m2": 0.79, "m3": 0.71, "m4": 0.61, "m5": 0.51, "m6": 0.41},
        "BenchC": {"m1": 0.4, "m2": 0.5, "m3": 0.6, "m4": 0.7, "m5": 0.8, "m6": 0.9},
    }
    corpus = Corpus(
        benchmark_names=["BenchA", "BenchB", "BenchC"],
        model_names=[f"m{i}" for i in range(1, 7)],
        Y=Y,
        descriptions={b: "" for b in Y},
        documents={
            b: {
                "reviewed_rows": 2,
                "topic_counts": {},
                "reasoning_depth_counts": {},
                "answer_format_counts": {},
                "examples": [f"{b} question {i}" for i in range(2)],
            }
            for b in Y
        },
    )
    calls = {"map": 0, "reducer": 0}

    def map_chat(_system, _user):
        calls["map"] += 1
        return json.dumps({
            "chunk_summary": "cached mapper evidence",
            "task_patterns": [],
            "reasoning_patterns": [],
            "justifications": ["cached descriptive evidence"],
        })

    def reducer_chat(_system, _user):
        calls["reducer"] += 1
        return json.dumps({
            "benchmark_summary": "benchmark-level synthesis",
            "ability_levels": {
                "analogical_reasoning": "strong",
                "deductive_reasoning": "weak",
                "inductive_reasoning": "weak",
                "long_term_knowledge_recall": "absent",
                "quantitative_reasoning": "medium",
            },
            "ability_rationale": {tag["id"]: "reduced evidence" for tag in VOCAB},
        })

    def improver_fn(**kwargs):
        new_prompt = kwargs["prev_prompt"] + "\nKeep the same fixed vocabulary and clarify reducer synthesis."
        return ImproverResult(new_prompt=new_prompt, accepted=True, reasons=[],
                              raw_response="{}", rationale="x")

    history, _best = run_part1(
        config,
        corpus=corpus,
        improver_fn=improver_fn,
        mapreduce_chat_fn=map_chat,
        mapreduce_reducer_chat_fn=reducer_chat,
    )

    assert [h.label for h in history] == ["iter_001", "iter_002"]
    assert calls["map"] == 6
    assert calls["reducer"] == 6
    iter_2 = next(h for h in history if h.label == "iter_002")
    assert iter_2.tag_weight_metadata["mapper_prompt_source"] == "base"
    for bench in iter_2.tag_weight_metadata["benchmarks"].values():
        assert bench["source"] == "mapreduce_llm_reducer_levels"


def test_best_so_far_rollback_keeps_best(tmp_resources):
    """If improver returns regression-causing prompts, best_so_far must hold the first iter."""
    overrides = {
        **tmp_resources,
        "max_iter": 3,
        "bootstrap_B": 10,
        "min_common_models": 6,
        "early_stop_consecutive": 999,  # disable early stop
        "optimize_tag_weights": False,
        "taxonomy_refinement_enabled": False,
    }
    config = load_experiment_config(overrides)

    aligned = {
        "BenchA": {"analogical_reasoning": 1.0, "deductive_reasoning": 0.0, "inductive_reasoning": 0.0,
                    "long_term_knowledge_recall": 0.0, "quantitative_reasoning": 0.0},
        "BenchB": {"analogical_reasoning": 0.95, "deductive_reasoning": 0.0, "inductive_reasoning": 0.0,
                    "long_term_knowledge_recall": 0.0, "quantitative_reasoning": 0.0},
        "BenchC": {"analogical_reasoning": 0.0, "deductive_reasoning": 0.0, "inductive_reasoning": 0.0,
                    "long_term_knowledge_recall": 0.0, "quantitative_reasoning": 1.0},
        "BenchD": {"analogical_reasoning": 0.0, "deductive_reasoning": 0.0, "inductive_reasoning": 0.0,
                    "long_term_knowledge_recall": 0.0, "quantitative_reasoning": 0.95},
        "BenchE": {"analogical_reasoning": 0.3, "deductive_reasoning": 0.3, "inductive_reasoning": 0.3,
                    "long_term_knowledge_recall": 0.3, "quantitative_reasoning": 0.3},
    }
    bad = {b: {tid: 0.5 for tid in next(iter(aligned.values())).keys()} for b in aligned}

    call_count = {"n": 0}

    def tag_fn(benchmark, description, vocab, prompt, version):
        # iter 1: aligned weights (good).
        # iter 2+: bad uniform weights (regression).
        if version == 1:
            return TagVector(benchmark=benchmark, weights=dict(aligned[benchmark]),
                             raw_response="{}", prompt_version=version)
        return TagVector(benchmark=benchmark, weights=dict(bad[benchmark]),
                         raw_response="{}", prompt_version=version)

    def improver_fn(**kwargs):
        # always "accept" a different prompt so we move forward
        call_count["n"] += 1
        new_prompt = kwargs["prev_prompt"] + " ADDED"
        return ImproverResult(new_prompt=new_prompt, accepted=True, reasons=[],
                              raw_response="{}", rationale="x")

    history, best = run_part1(
        config,
        tag_fn=tag_fn,
        improver_fn=improver_fn,
    )

    # Best must be iter_001 (aligned), not later iters.
    assert best.label == "iter_001"
    L1 = next(h.L_align for h in history if h.label == "iter_001")
    L2 = next(h.L_align for h in history if h.label == "iter_002")
    assert L1 < L2


@pytest.mark.parametrize(
    ("improver_case", "expected_status"),
    [
        ("rejected", "improver_rejected"),
        ("no_change", "improver_no_change"),
    ],
)
def test_main_loop_stops_when_improver_has_no_valid_next_prompt(
    tmp_resources,
    improver_case,
    expected_status,
):
    """A rejected/no-op improver result must not trigger duplicate re-evaluation."""
    overrides = {
        **tmp_resources,
        "max_iter": 3,
        "bootstrap_B": 10,
        "min_common_models": 6,
        "run_baseline": False,
        "early_stop_consecutive": 999,
        "optimize_tag_weights": False,
        "taxonomy_refinement_enabled": False,
    }
    config = load_experiment_config(overrides)

    ids = [v["id"] for v in VOCAB]

    def full(active: str) -> dict[str, float]:
        return {tag_id: (1.0 if tag_id == active else 0.0) for tag_id in ids}

    weights = {
        "BenchA": full("analogical_reasoning"),
        "BenchB": full("analogical_reasoning"),
        "BenchC": full("quantitative_reasoning"),
        "BenchD": full("quantitative_reasoning"),
        "BenchE": {tag_id: 0.2 for tag_id in ids},
    }

    improver_calls = {"n": 0}

    def improver_fn(**kwargs):
        improver_calls["n"] += 1
        if improver_case == "no_change":
            return ImproverResult(
                new_prompt=kwargs["prev_prompt"],
                accepted=True,
                reasons=[],
                raw_response="{}",
                rationale="same prompt",
            )
        return ImproverResult(
            new_prompt=kwargs["prev_prompt"] + "\nThis rejected prompt must not be used.",
            accepted=False,
            reasons=["prompt_too_long:3600>3500"],
            raw_response="{}",
            rationale="guard rejected",
        )

    history, _best = run_part1(
        config,
        tag_fn=_fixed_tag_fn(weights),
        improver_fn=improver_fn,
    )

    assert [h.label for h in history] == ["iter_001"]
    assert improver_calls["n"] == 1

    run_dirs = os.listdir(tmp_resources["results_dir"])
    assert len(run_dirs) == 1
    stop_path = os.path.join(
        tmp_resources["results_dir"],
        run_dirs[0],
        "final",
        "stop_reason.json",
    )
    with open(stop_path, encoding="utf-8") as f:
        stop_reason = json.load(f)
    assert stop_reason["status"] == expected_status
    assert stop_reason["details"]["iter"] == 1


def test_weight_calibration_saved_as_ablation_not_primary(tmp_resources):
    overrides = {
        **tmp_resources,
        "max_iter": 1,
        "bootstrap_B": 5,
        "min_common_models": 6,
        "run_baseline": False,
        "run_weight_calibration_ablation": True,
        "weight_l2_lambda": 0.0,
        "weight_max_iter": 20,
        "taxonomy_refinement_enabled": False,
    }
    config = load_experiment_config(overrides)

    fixed = {
        "BenchA": {"analogical_reasoning": 0.9, "deductive_reasoning": 0.1, "inductive_reasoning": 0.0,
                    "long_term_knowledge_recall": 0.2, "quantitative_reasoning": 0.0},
        "BenchB": {"analogical_reasoning": 0.85, "deductive_reasoning": 0.05, "inductive_reasoning": 0.05,
                    "long_term_knowledge_recall": 0.25, "quantitative_reasoning": 0.0},
        "BenchC": {"analogical_reasoning": 0.0, "deductive_reasoning": 0.7, "inductive_reasoning": 0.6,
                    "long_term_knowledge_recall": 0.1, "quantitative_reasoning": 0.5},
        "BenchD": {"analogical_reasoning": 0.0, "deductive_reasoning": 0.65, "inductive_reasoning": 0.55,
                    "long_term_knowledge_recall": 0.05, "quantitative_reasoning": 0.55},
        "BenchE": {"analogical_reasoning": 0.3, "deductive_reasoning": 0.3, "inductive_reasoning": 0.3,
                    "long_term_knowledge_recall": 0.3, "quantitative_reasoning": 0.3},
    }

    history, best = run_part1(config, tag_fn=_fixed_tag_fn(fixed))

    assert best.label == "iter_001"
    assert history[0].T == fixed
    run_dirs = os.listdir(tmp_resources["results_dir"])
    iter_dir = os.path.join(tmp_resources["results_dir"], run_dirs[0], "iter_001")
    final_dir = os.path.join(tmp_resources["results_dir"], run_dirs[0], "final")
    assert os.path.isfile(os.path.join(iter_dir, "T_calibrated.json"))
    assert os.path.isfile(os.path.join(iter_dir, "metrics_calibrated.json"))
    assert os.path.isfile(os.path.join(final_dir, "T_star_raw.json"))
    assert os.path.isfile(os.path.join(final_dir, "T_star_calibrated.json"))


def test_taxonomy_refinement_phase_runs_after_residual_trigger(tmp_resources):
    overrides = {
        **tmp_resources,
        "max_iter": 1,
        "bootstrap_B": 5,
        "min_common_models": 6,
        "run_baseline": False,
        "taxonomy_refinement_enabled": True,
        "taxonomy_refinement_min_pairs": 1,
        "taxonomy_refinement_residual_max_threshold": 0.0,
        "taxonomy_refinement_max_iter": 1,
    }
    config = load_experiment_config(overrides)
    extra_tag = {
        "id": "visual_pattern_recognition",
        "name": "Visual Pattern Recognition",
        "definition": "Recognizing visual or diagrammatic patterns not covered by seed tags.",
    }
    refined_vocab = [*VOCAB, extra_tag]
    refined_prompt = (
        "Tag with analogical_reasoning, deductive_reasoning, inductive_reasoning, "
        "long_term_knowledge_recall, quantitative_reasoning, and visual_pattern_recognition. "
        "Return JSON weights for every listed tag id."
    )

    def tag_fn(benchmark, description, vocab, prompt, version):
        weights = {v["id"]: 0.1 for v in vocab}
        if version >= 10_000:
            weights["visual_pattern_recognition"] = 0.9 if benchmark in {"BenchA", "BenchB"} else 0.0
        return TagVector(benchmark=benchmark, weights=weights, raw_response="{}", prompt_version=version)

    def taxonomy_refiner_fn(**kwargs):
        return TaxonomyRefinementResult(
            vocab=refined_vocab,
            prompt=refined_prompt,
            accepted=True,
            reasons=[],
            raw_response="{}",
            rationale="residuals remained large",
        )

    run_part1(config, tag_fn=tag_fn, taxonomy_refiner_fn=taxonomy_refiner_fn)

    run_dirs = os.listdir(tmp_resources["results_dir"])
    root = os.path.join(tmp_resources["results_dir"], run_dirs[0])
    tax_dir = os.path.join(root, "taxonomy_refinement")
    assert os.path.isfile(os.path.join(tax_dir, "status.json"))
    assert os.path.isfile(os.path.join(tax_dir, "refinement_result.json"))
    assert os.path.isfile(os.path.join(tax_dir, "iter_001", "tag_vectors.json"))
    assert os.path.isfile(os.path.join(tax_dir, "final", "vocab_star.json"))
    assert os.path.isfile(os.path.join(tax_dir, "final", "I_star.txt"))


def test_taxonomy_adoption_rejects_worse_candidate(tmp_resources):
    overrides = {
        **tmp_resources,
        "max_iter": 1,
        "bootstrap_B": 5,
        "min_common_models": 6,
        "run_baseline": False,
        "taxonomy_refinement_enabled": True,
        "taxonomy_refinement_min_pairs": 1,
        "taxonomy_refinement_residual_max_threshold": 0.0,
        "taxonomy_refinement_max_iter": 1,
    }
    config = load_experiment_config(overrides)
    good, _mid, bad = _taxonomy_test_weights()

    def tag_fn(benchmark, description, vocab, prompt, version):
        weights = bad[benchmark] if version >= 10_000 else good[benchmark]
        return TagVector(benchmark=benchmark, weights=dict(weights),
                         raw_response="{}", prompt_version=version)

    run_part1(config, tag_fn=tag_fn, taxonomy_refiner_fn=_accepted_taxonomy_refiner)

    run_dirs = os.listdir(tmp_resources["results_dir"])
    root = os.path.join(tmp_resources["results_dir"], run_dirs[0])
    with open(os.path.join(root, "selection.json"), encoding="utf-8") as f:
        selection = json.load(f)
    with open(os.path.join(root, "taxonomy_refinement", "status.json"), encoding="utf-8") as f:
        status = json.load(f)

    assert selection["selected_source"] == "fixed"
    assert status["adoption"]["adopted"] is False
    assert "L_align_not_improved" in status["adoption"]["reasons"]


def test_taxonomy_adoption_selects_better_candidate(tmp_resources):
    overrides = {
        **tmp_resources,
        "max_iter": 1,
        "bootstrap_B": 5,
        "min_common_models": 6,
        "run_baseline": False,
        "taxonomy_refinement_enabled": True,
        "taxonomy_refinement_min_pairs": 1,
        "taxonomy_refinement_residual_max_threshold": 0.0,
        "taxonomy_refinement_max_iter": 1,
    }
    config = load_experiment_config(overrides)
    good, mid, _bad = _taxonomy_test_weights()
    seen = {"protected_pairs": None}

    def tag_fn(benchmark, description, vocab, prompt, version):
        weights = good[benchmark] if version >= 10_000 else mid[benchmark]
        return TagVector(benchmark=benchmark, weights=dict(weights),
                         raw_response="{}", prompt_version=version)

    def taxonomy_refiner_fn(**kwargs):
        seen["protected_pairs"] = kwargs.get("protected_pairs")
        return _accepted_taxonomy_refiner(**kwargs)

    run_part1(config, tag_fn=tag_fn, taxonomy_refiner_fn=taxonomy_refiner_fn)

    run_dirs = os.listdir(tmp_resources["results_dir"])
    root = os.path.join(tmp_resources["results_dir"], run_dirs[0])
    with open(os.path.join(root, "selection.json"), encoding="utf-8") as f:
        selection = json.load(f)
    with open(os.path.join(root, "taxonomy_refinement", "status.json"), encoding="utf-8") as f:
        status = json.load(f)

    assert seen["protected_pairs"]
    assert seen["protected_pairs"][0]["score_similarity"] >= 0.8
    assert selection["selected_source"] == "taxonomy_refinement"
    assert status["adoption"]["adopted"] is True
    assert status["adoption"]["reasons"] == []


def test_no_seed_taxonomy_replaces_seed_vocab(tmp_resources):
    overrides = {
        **tmp_resources,
        "max_iter": 1,
        "bootstrap_B": 5,
        "min_common_models": 6,
        "run_baseline": False,
        "taxonomy_refinement_enabled": False,
        "no_seed_taxonomy_enabled": True,
        "no_seed_taxonomy_min_tags": 2,
        "no_seed_taxonomy_max_tags": 3,
        "no_seed_taxonomy_max_attempts": 2,
        "llm_empty_content_retries": 4,
        "llm_request_timeout_s": 12.5,
        "llm_sdk_exception_retries": 0,
        "llm_debug_dump_dir": "tmp/debug",
        "llm_reasoning": {"effort": "none", "exclude": True},
    }
    config = load_experiment_config(overrides)
    no_seed_vocab = [
        {
            "id": "symbolic_problem_solving",
            "name": "Symbolic Problem Solving",
            "definition": "Solving abstract symbolic tasks.",
        },
        {
            "id": "world_knowledge_use",
            "name": "World Knowledge Use",
            "definition": "Using external factual knowledge.",
        },
    ]
    no_seed_prompt = (
        "Use only symbolic_problem_solving and world_knowledge_use. "
        "Rate every listed tag id as absent, weak, medium, strong, or dominant."
    )

    seen_no_seed_kwargs = {}

    def no_seed_taxonomy_fn(**kwargs):
        seen_no_seed_kwargs.update(kwargs)
        return NoSeedTaxonomyResult(
            vocab=no_seed_vocab,
            prompt=no_seed_prompt,
            accepted=True,
            reasons=[],
            raw_response="{}",
            rationale="no seed ablation",
        )

    def tag_fn(benchmark, description, vocab, prompt, version):
        assert [v["id"] for v in vocab] == [
            "symbolic_problem_solving",
            "world_knowledge_use",
        ]
        assert prompt == no_seed_prompt
        weights = {
            "symbolic_problem_solving": 0.8 if benchmark in {"BenchA", "BenchB"} else 0.2,
            "world_knowledge_use": 0.2 if benchmark in {"BenchA", "BenchB"} else 0.8,
        }
        return TagVector(benchmark=benchmark, weights=weights,
                         raw_response="{}", prompt_version=version)

    history, best = run_part1(
        config,
        tag_fn=tag_fn,
        no_seed_taxonomy_fn=no_seed_taxonomy_fn,
    )

    assert best.label == "iter_001"
    assert set(history[0].T["BenchA"]) == {
        "symbolic_problem_solving",
        "world_knowledge_use",
    }
    run_dirs = os.listdir(tmp_resources["results_dir"])
    root = os.path.join(tmp_resources["results_dir"], run_dirs[0])
    with open(os.path.join(root, "no_seed_taxonomy", "proposal.json"), encoding="utf-8") as f:
        proposal = json.load(f)
    with open(os.path.join(root, "config.json"), encoding="utf-8") as f:
        saved_config = json.load(f)
    assert proposal["accepted"] is True
    assert saved_config["active_vocab_source"] == "no_seed_taxonomy"
    assert seen_no_seed_kwargs["empty_content_retries"] == 4
    assert seen_no_seed_kwargs["max_attempts"] == 2
    assert seen_no_seed_kwargs["request_timeout_s"] == 12.5
    assert seen_no_seed_kwargs["sdk_exception_retries"] == 0
    assert seen_no_seed_kwargs["debug_dump_dir"] == "tmp/debug"
    assert seen_no_seed_kwargs["extra_body"] == {
        "reasoning": {"effort": "none", "exclude": True}
    }


# --------------------------------------------------------------------------
# Phase I/J — V loop + split-aware metrics tests.
# --------------------------------------------------------------------------


def _v_loop_corpus() -> Corpus:
    """Eight benchmarks so default ratios give clean dev/test splits with pairs."""
    bench_names = [f"Bench{c}" for c in "ABCDEFGH"]
    Y = {
        b: {f"m{i}": 0.9 - 0.05 * i - 0.01 * idx for i in range(1, 8)}
        for idx, b in enumerate(bench_names)
    }
    return Corpus(
        benchmark_names=bench_names,
        model_names=[f"m{i}" for i in range(1, 8)],
        Y=Y,
        descriptions={b: f"description {b}" for b in bench_names},
        documents={
            b: {
                "reviewed_rows": 1,
                "topic_counts": {},
                "reasoning_depth_counts": {},
                "answer_format_counts": {},
                "examples": [f"{b} example"],
            }
            for b in bench_names
        },
    )


def _v_loop_overrides(tmp_resources: dict, **extra) -> dict:
    base = {
        **tmp_resources,
        "max_iter": 2,
        "bootstrap_B": 5,
        "min_common_models": 6,
        "run_baseline": False,
        "early_stop_consecutive": 999,
        "taxonomy_refinement_enabled": False,
        "taxonomy_selection_enabled": False,
        "best_iter_selection": "train_l_align",
        "best_iter_dev_rho_floor": 0.0,
        "best_iter_dev_rho_drop_tolerance": None,
        "best_iter_train_l_increase_tolerance": None,
        "best_iter_train_rho_drop_tolerance": None,
        "best_iter_train_rho_floor": None,
        "best_iter_stability_rho_weight": 0.30,
        "best_iter_model_probe_enabled": False,
        "best_iter_model_probe_min_common": None,
        "best_iter_model_probe_dev_rho_floor": None,
        "best_iter_model_probe_dev_rho_drop_tolerance": None,
        "best_iter_model_probe_dev_l_increase_tolerance": None,
        "executer_candidate_counts": None,
        "enable_v_loop": True,
        "use_mapreduce_evidence": False,
        "mapreduce_chunk_examples": 1,
        "mapreduce_max_chunk_chars": 2000,
        "mapreduce_max_evidence_chars": 4000,
        "mapreduce_max_workers": 1,
        "mapreduce_cache_enabled": False,
        "llm_fallback_fail_threshold": None,
        "v_loop_min_test_valid_pairs": 0,
    }
    base.update(extra)
    return base


def _make_mapper_fn() -> tuple[callable, dict]:
    counter = {"n": 0}

    def fn(_system, user):
        counter["n"] += 1
        return json.dumps({
            "chunk_summary": f"summary for {user[:32]}",
            "task_patterns": ["task pattern"],
            "reasoning_patterns": ["reasoning pattern"],
            "justifications": ["evidence"],
        })

    return fn, counter


def _make_executer_fn(vocabs_by_iter: list[list[dict]]) -> tuple[callable, dict]:
    """Returns vocab[iter_idx] on each call, where iter_idx is the call count."""
    counter = {"n": 0, "prompts_seen": []}

    def fn(_system, user):
        counter["n"] += 1
        counter["prompts_seen"].append(user)
        idx = min(counter["n"] - 1, len(vocabs_by_iter) - 1)
        v = vocabs_by_iter[idx]
        return json.dumps({
            "vocab": v,
            "rationale": f"vocab v{idx}",
        })

    return fn, counter


def _uniform_tag_fn(target_vocab_ids_by_bench: dict[str, list[str]]):
    def fn(benchmark, description, vocab, prompt, version):
        ids = [v["id"] for v in vocab]
        chosen = set(target_vocab_ids_by_bench.get(benchmark, []))
        if not chosen:
            chosen = {ids[0]} if ids else set()
        weights = {tid: (1.0 if tid in chosen else 0.0) for tid in ids}
        return TagVector(
            benchmark=benchmark, weights=weights, raw_response="{}",
            prompt_version=version,
        )
    return fn


def _stepping_improver(prefix: str = "STEP"):
    def fn(**kwargs):
        prev = kwargs["prev_prompt"]
        suffix = f" {prefix}{prev.count(prefix) + 1}"
        return ImproverResult(
            new_prompt=prev + suffix, accepted=True, reasons=[],
            raw_response="{}", rationale="step",
        )
    return fn


def test_v_loop_source_is_train_split(tmp_resources):
    """Source = entire per-fold train split — Executer's source_benchmarks
    must equal sorted(bench_split.train) at iter 1."""
    corpus = _v_loop_corpus()
    bench_split = split_benchmarks(corpus.benchmark_names, ratios=(0.6, 0.2, 0.2), seed=0)

    overrides = _v_loop_overrides(tmp_resources)
    config = load_experiment_config(overrides)

    captured = {"sources": None}

    def exec_fn(_system, user_msg):
        # The user_msg embeds the source-benchmark summary lines we can grep.
        captured["user_msg"] = user_msg
        return json.dumps({
            "vocab": [{"id": "deductive_reasoning", "name": "D", "definition": "d"}],
            "rationale": "x",
        })

    map_fn, _ = _make_mapper_fn()
    history, _best = run_part1(
        config, corpus=corpus,
        tag_fn=_uniform_tag_fn({}),
        mapreduce_chat_fn=map_fn,
        executer_chat_fn=exec_fn,
    )
    # All train benches must appear as `## Benchmark: <name>` sections.
    for bench in bench_split.train:
        assert f"## Benchmark: {bench}" in captured["user_msg"], (
            f"train bench {bench!r} not represented in Executer user_msg"
        )
    # Test benches must NOT appear (test untouched until end).
    for bench in bench_split.test:
        assert f"## Benchmark: {bench}" not in captured["user_msg"], (
            f"test bench {bench!r} leaked into Executer evidence"
        )


def test_v_loop_tags_test_benchmarks_only_in_final_pass(tmp_resources):
    """The Maker/tagger must not see D_test during baseline or loop iterations."""
    corpus = _v_loop_corpus()
    bench_split = split_benchmarks(corpus.benchmark_names, ratios=(0.6, 0.2, 0.2), seed=0)

    overrides = _v_loop_overrides(tmp_resources, max_iter=1, run_baseline=False)
    config = load_experiment_config(overrides)
    v0 = [{"id": "deductive_reasoning", "name": "D", "definition": "d"}]
    map_fn, _ = _make_mapper_fn()
    exec_fn, _ = _make_executer_fn([v0])
    calls: list[str] = []

    def tag_fn(benchmark, description, vocab, prompt, version):
        calls.append(benchmark)
        return TagVector(
            benchmark=benchmark,
            weights={v["id"]: 1.0 for v in vocab},
            raw_response="{}",
            prompt_version=version,
        )

    run_part1(
        config,
        corpus=corpus,
        tag_fn=tag_fn,
        mapreduce_chat_fn=map_fn,
        executer_chat_fn=exec_fn,
    )

    counts = {b: calls.count(b) for b in corpus.benchmark_names}
    for bench in [*bench_split.train, *bench_split.dev]:
        assert counts[bench] == 2, f"{bench} should be tagged in-loop and final"
    for bench in bench_split.test:
        assert counts[bench] == 1, f"{bench} should be tagged only in final"


def test_v_loop_raises_when_train_split_empty(tmp_resources):
    """When train ends up empty (degenerate split), v_loop must fail loudly."""
    overrides = _v_loop_overrides(tmp_resources, best_iter_dev_rho_floor=None)
    # Force a degenerate ratio: train=0, dev=0, test=all.
    overrides["splits"] = {"benchmark_ratios": (0.0, 0.0, 1.0), "benchmark_seed": 0}
    config = load_experiment_config(overrides)
    corpus = _v_loop_corpus()
    map_fn, _ = _make_mapper_fn()
    exec_fn, _ = _make_executer_fn([[{"id": "a", "name": "A", "definition": "a"}]])

    with pytest.raises(ValueError, match="train split"):
        run_part1(
            config, corpus=corpus,
            tag_fn=_uniform_tag_fn({}),
            mapreduce_chat_fn=map_fn,
            executer_chat_fn=exec_fn,
        )


def test_v_loop_raises_when_dev_split_empty(tmp_resources):
    """When dev is empty, v_loop has no selection signal → must fail loudly."""
    overrides = _v_loop_overrides(tmp_resources)
    overrides["splits"] = {"benchmark_ratios": (1.0, 0.0, 0.0), "benchmark_seed": 0}
    config = load_experiment_config(overrides)
    corpus = _v_loop_corpus()
    map_fn, _ = _make_mapper_fn()
    exec_fn, _ = _make_executer_fn([[{"id": "a", "name": "A", "definition": "a"}]])

    with pytest.raises(ValueError, match="dev split"):
        run_part1(
            config, corpus=corpus,
            tag_fn=_uniform_tag_fn({}),
            mapreduce_chat_fn=map_fn,
            executer_chat_fn=exec_fn,
        )


def test_v_loop_raises_when_dev_has_no_valid_pairs(tmp_resources):
    """A non-empty dev split still fails if min_common filtering leaves no pairs."""
    overrides = _v_loop_overrides(tmp_resources)
    overrides["splits"] = {"cv_folds": 2, "fold": 0, "benchmark_seed": 0}
    config = load_experiment_config(overrides)
    corpus = _v_loop_corpus()
    map_fn, _ = _make_mapper_fn()
    exec_fn, _ = _make_executer_fn([[{"id": "a", "name": "A", "definition": "a"}]])

    with pytest.raises(
        ValueError,
        match=r"insufficient score-comparable pairs after min_common filtering: dev:0<1",
    ):
        run_part1(
            config,
            corpus=corpus,
            tag_fn=_uniform_tag_fn({}),
            mapreduce_chat_fn=map_fn,
            executer_chat_fn=exec_fn,
        )


def test_v_loop_honors_dev_train_split_override(tmp_resources):
    """A wider dev split from config must reach the real v-loop splitter."""
    overrides = _v_loop_overrides(tmp_resources, max_iter=1)
    overrides["splits"] = {
        "cv_folds": 2,
        "fold": 0,
        "benchmark_seed": 0,
        "dev_train_split": [1.0, 1.0],
    }
    config = load_experiment_config(overrides)
    corpus = _v_loop_corpus()
    map_fn, _ = _make_mapper_fn()
    exec_fn, _ = _make_executer_fn([[{"id": "a", "name": "A", "definition": "a"}]])

    run_part1(
        config,
        corpus=corpus,
        tag_fn=_uniform_tag_fn({}),
        mapreduce_chat_fn=map_fn,
        executer_chat_fn=exec_fn,
    )

    run_dirs = os.listdir(tmp_resources["results_dir"])
    root = os.path.join(tmp_resources["results_dir"], run_dirs[0])
    with open(os.path.join(root, "final", "vocab_star_metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert len(meta["source_benchmarks"]) == 2


def test_v_loop_raises_when_no_candidate_passes_gate(tmp_resources):
    """v-loop must not select the min-L candidate after every candidate failed the gate."""
    overrides = _v_loop_overrides(
        tmp_resources,
        max_iter=1,
        run_baseline=False,
        delta_tag_threshold=999.0,
    )
    overrides["splits"] = {
        "cv_folds": 2,
        "fold": 0,
        "benchmark_seed": 0,
        "dev_train_split": [1.0, 1.0],
    }
    config = load_experiment_config(overrides)
    corpus = _v_loop_corpus()
    map_fn, _ = _make_mapper_fn()
    exec_fn, _ = _make_executer_fn([[{"id": "a", "name": "A", "definition": "a"}]])

    with pytest.raises(RuntimeError, match="no gate-passing candidate"):
        run_part1(
            config,
            corpus=corpus,
            tag_fn=_uniform_tag_fn({}),
            mapreduce_chat_fn=map_fn,
            executer_chat_fn=exec_fn,
        )

    run_dirs = os.listdir(tmp_resources["results_dir"])
    root = os.path.join(tmp_resources["results_dir"], run_dirs[0])
    with open(os.path.join(root, "final", "stop_reason.json"), encoding="utf-8") as f:
        stop = json.load(f)
    assert stop["status"] == "no_gate_passing_candidate"


def test_v_loop_uses_static_baseline_when_taxonomy_candidates_all_fail(tmp_resources):
    """A fold with no valid v-loop taxonomy should still write fixed-baseline final."""
    overrides = _v_loop_overrides(
        tmp_resources,
        max_iter=1,
        run_baseline=True,
        delta_tag_threshold=999.0,
    )
    overrides["splits"] = {
        "cv_folds": 2,
        "fold": 0,
        "benchmark_seed": 0,
        "dev_train_split": [1.0, 1.0],
    }
    config = load_experiment_config(overrides)
    corpus = _v_loop_corpus()
    map_fn, _ = _make_mapper_fn()
    exec_fn, _ = _make_executer_fn([[{"id": "a", "name": "A", "definition": "a"}]])

    _history, best = run_part1(
        config,
        corpus=corpus,
        tag_fn=_uniform_tag_fn({}),
        mapreduce_chat_fn=map_fn,
        executer_chat_fn=exec_fn,
    )

    assert best.label == "iter_000_baseline_static"
    run_dirs = os.listdir(tmp_resources["results_dir"])
    root = os.path.join(tmp_resources["results_dir"], run_dirs[0])
    with open(os.path.join(root, "final", "stop_reason.json"), encoding="utf-8") as f:
        stop = json.load(f)
    with open(os.path.join(root, "final", "vocab_star_metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert stop["status"] == "no_gate_passing_taxonomy_candidate"
    assert stop["details"]["selected_fixed_baseline"] is True
    assert meta["best_iter_label"] == "iter_000_baseline_static"


def test_v_loop_stops_when_gate_passing_candidates_do_not_improve_static(tmp_resources):
    """A Δ_tag-passing v_loop candidate that is worse than the seeded static
    baseline must count toward early stop. Otherwise bad generated vocabularies
    can burn the full max_iter budget without any selection improvement.
    """
    corpus = _v_loop_corpus()
    overrides = _v_loop_overrides(
        tmp_resources,
        max_iter=5,
        run_baseline=True,
        early_stop_consecutive=2,
        delta_tag_threshold=-1.0,
    )
    overrides["splits"] = {
        "cv_folds": 2,
        "fold": 0,
        "benchmark_seed": 0,
        "dev_train_split": [1.0, 1.0],
    }
    config = load_experiment_config(overrides)
    map_fn, _ = _make_mapper_fn()
    exec_fn, exec_counter = _make_executer_fn([
        [
            {"id": "axis_a", "name": "Axis A", "definition": "a"},
            {"id": "axis_b", "name": "Axis B", "definition": "b"},
        ]
    ])

    def tag_fn(benchmark, description, vocab, prompt, version):
        ids = [v["id"] for v in vocab]
        if version == 0:
            weights = {tid: 1.0 for tid in ids}
        else:
            chosen = ids[0] if ord(benchmark[-1]) % 2 == 0 else ids[-1]
            weights = {tid: (1.0 if tid == chosen else 0.0) for tid in ids}
        return TagVector(
            benchmark=benchmark,
                weights=weights,
                raw_response="{}",
                prompt_version=version,
            )

    improver_calls = {"n": 0}

    def improver_fn(**kwargs):
        improver_calls["n"] += 1
        return _stepping_improver()(**kwargs)

    history, best = run_part1(
        config,
        corpus=corpus,
        tag_fn=tag_fn,
        mapreduce_chat_fn=map_fn,
        executer_chat_fn=exec_fn,
        improver_fn=improver_fn,
    )

    labels = [item.label for item in history]
    assert best.label == "iter_000_baseline_static"
    assert "iter_003" not in labels
    assert exec_counter["n"] == 2
    assert improver_calls["n"] == 0
    assert "STEP" not in exec_counter["prompts_seen"][1]

    run_dirs = os.listdir(tmp_resources["results_dir"])
    root = os.path.join(tmp_resources["results_dir"], run_dirs[0])
    with open(os.path.join(root, "final", "stop_reason.json"), encoding="utf-8") as f:
        stop = json.load(f)
    assert stop["status"] == "stalled_no_improvement"
    assert stop["details"]["reason"] == "not_better_than_current_best"
    assert stop["details"]["gate_pass"] is True


def test_v_loop_selection_uses_dev_signal(tmp_resources):
    """IterationResult.L_align must equal m_dev["L_align"], not m_train."""
    corpus = _v_loop_corpus()
    overrides = _v_loop_overrides(tmp_resources, max_iter=1)
    config = load_experiment_config(overrides)

    v0 = [{"id": "deductive_reasoning", "name": "D", "definition": "d"}]
    map_fn, _ = _make_mapper_fn()
    exec_fn, _ = _make_executer_fn([v0])
    # Tag function that gives different weights per bench so train and dev
    # L_align are not identical by symmetry.
    def varying_tag_fn(benchmark, description, vocab, prompt, version):
        ids = [v["id"] for v in vocab]
        weights = {tid: (sum(ord(c) for c in benchmark) % 5 + i) / 10.0
                   for i, tid in enumerate(ids)}
        return TagVector(
            benchmark=benchmark, weights=weights, raw_response="{}",
            prompt_version=version,
        )

    history, _best = run_part1(
        config, corpus=corpus,
        tag_fn=varying_tag_fn,
        mapreduce_chat_fn=map_fn,
        executer_chat_fn=exec_fn,
    )
    # ir.L_align matches dev_metrics.L_align (selection signal).
    iter_results = [h for h in history if h.label.startswith("iter_001")]
    assert iter_results, "expected at least one iter_001* result"
    ir = iter_results[0]
    assert ir.dev_metrics is not None
    assert ir.L_align == ir.dev_metrics["L_align"]


def test_v_loop_improver_error_report_uses_dev_pairs(tmp_resources):
    """Prompt improvement must chase the same split used for selection."""
    corpus = _v_loop_corpus()
    split = split_benchmarks(corpus.benchmark_names, ratios=(0.6, 0.2, 0.2), seed=0)
    expected_dev_pairs = {
        tuple(sorted((a, b)))
        for i, a in enumerate(split.dev)
        for b in split.dev[i + 1:]
    }
    train_pairs = {
        tuple(sorted((a, b)))
        for i, a in enumerate(split.train)
        for b in split.train[i + 1:]
    }
    overrides = _v_loop_overrides(tmp_resources, max_iter=1)
    config = load_experiment_config(overrides)

    v0 = [{"id": "deductive_reasoning", "name": "D", "definition": "d"}]
    map_fn, _ = _make_mapper_fn()
    exec_fn, _ = _make_executer_fn([v0])

    import autotagging_loop.experiment.pipeline.run as run_mod

    real_build_error_report = run_mod.build_error_report
    seen_pair_sets: list[set[tuple[str, str]]] = []

    def spy_build_error_report(_S, R_raw, *_args, **_kwargs):
        seen_pair_sets.append({tuple(sorted(pair)) for pair in R_raw})
        return []

    run_mod.build_error_report = spy_build_error_report
    try:
        run_part1(
            config,
            corpus=corpus,
            tag_fn=_uniform_tag_fn({}),
            mapreduce_chat_fn=map_fn,
            executer_chat_fn=exec_fn,
        )
    finally:
        run_mod.build_error_report = real_build_error_report

    assert seen_pair_sets
    assert expected_dev_pairs in seen_pair_sets
    assert train_pairs not in seen_pair_sets


def test_v_loop_dev_objective_selects_executer_taxonomy_candidate(tmp_resources):
    """Dev objective, not train loss or seed order, chooses the final V_star."""

    corpus = _v_loop_corpus()
    overrides = _v_loop_overrides(
        tmp_resources,
        max_iter=2,
        best_iter_selection="dev_l_align",
        taxonomy_selection_enabled=True,
        taxonomy_selection_min_tags=1,
        taxonomy_selection_max_tags=4,
        taxonomy_selection_target_tags=2,
        taxonomy_selection_count_penalty=0.0,
        best_iter_model_probe_enabled=True,
        best_iter_model_probe_min_common=5,
        best_iter_model_probe_dev_rho_floor=0.0,
    )
    config = load_experiment_config(overrides)

    v_train_better = [
        {"id": "train_fit", "name": "Train Fit", "definition": "fits train only"},
    ]
    v_dev_better = [
        {"id": "dev_fit", "name": "Dev Fit", "definition": "fits dev"},
        {"id": "cross_domain_reasoning", "name": "Cross Domain", "definition": "generalizes"},
    ]
    map_fn, _ = _make_mapper_fn()
    exec_fn, _ = _make_executer_fn([v_train_better, v_dev_better])
    improver = _stepping_improver()

    import autotagging_loop.experiment.loop as loop_mod
    real_compute = loop_mod._compute_metrics

    def fake_compute(T, benchmark_names, R_raw, R01, *args, **kwargs):
        vocab_ids = tuple(sorted(next(iter(T.values())).keys()))
        names = set(benchmark_names)
        if vocab_ids == ("cross_domain_reasoning", "dev_fit"):
            L = 0.05 if len(names) <= 3 else 0.30
        elif vocab_ids == ("train_fit",):
            L = 0.40 if len(names) <= 3 else 0.01
        else:
            L = 0.90
        n_pairs = max(1, len(names) * (len(names) - 1) // 2)
        S = {
            (a, b): 0.5
            for idx, a in enumerate(sorted(names))
            for b in sorted(names)[idx + 1:]
        }
        metrics = {
            "L_align": L,
            "L_align_01": L,
            "rho_align_pearson": 0.50,
            "rho_align_spearman": 0.50,
            "delta_tag": 0.25,
            "residual_mean": L,
            "residual_max": L,
            "n_pairs": n_pairs,
        }
        boot = {
            key: {"mean": value, "lo": value, "hi": value}
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        return S, metrics, boot

    setattr(loop_mod, "_compute_metrics", fake_compute)
    try:
        _history, best = run_part1(
            config,
            corpus=corpus,
            tag_fn=_uniform_tag_fn({}),
            improver_fn=improver,
            mapreduce_chat_fn=map_fn,
            executer_chat_fn=exec_fn,
        )
    finally:
        setattr(loop_mod, "_compute_metrics", real_compute)

    assert best.label == "iter_002"
    assert best.vocab is not None
    assert [v["id"] for v in best.vocab] == [v["id"] for v in v_dev_better]
    assert best.dev_metrics["L_align"] == 0.05

    run_dirs = os.listdir(tmp_resources["results_dir"])
    root = os.path.join(tmp_resources["results_dir"], run_dirs[0])
    with open(os.path.join(root, "selection.json"), encoding="utf-8") as f:
        selection = json.load(f)
    with open(os.path.join(root, "final", "vocab_star.json"), encoding="utf-8") as f:
        selected_vocab = json.load(f)
    with open(os.path.join(root, "selection_candidates.json"), encoding="utf-8") as f:
        candidates = json.load(f)

    assert selection["selected_source"] == "executer"
    assert selection["selected_label"] == "iter_002"
    assert selection["selected_candidate"]["dev_selection_score"] == 0.05
    assert selection["candidate_history_path"] == "selection_candidates.json"
    assert [v["id"] for v in selected_vocab] == [v["id"] for v in v_dev_better]
    assert candidates["objective_key"] == "dev_selection_score"
    assert [c["label"] for c in candidates["candidates"]] == ["iter_001", "iter_002"]
    assert [c["selected_final"] for c in candidates["candidates"]] == [False, True]
    assert [c["tag_count"] for c in candidates["candidates"]] == [1, 2]
    assert candidates["candidates"][1]["selection_objective_value"] == 0.05
    assert candidates["candidates"][1]["train_L_align"] == 0.30
    assert candidates["candidates"][1]["train_rho_spearman"] == 0.50
    assert candidates["candidates"][1]["model_probe_dev_n_probes"] > 0
    assert candidates["candidates"][1]["model_probe_dev_rho_spearman_min"] == 0.50


def test_v_loop_writes_v_artifacts_and_changes_v_across_iterations(tmp_resources):
    """V_0 ≠ V_1 when Improver mutates the prompt → executer cache key changes."""
    corpus = _v_loop_corpus()
    bench_split = split_benchmarks(corpus.benchmark_names, ratios=(0.6, 0.2, 0.2), seed=0)
    src = bench_split.train[0]

    overrides = _v_loop_overrides(tmp_resources, best_iter_dev_rho_floor=None)
    config = load_experiment_config(overrides)

    v0 = [
        {"id": "deductive_reasoning", "name": "Deductive", "definition": "rule chains"},
        {"id": "long_term_knowledge_recall", "name": "Recall", "definition": "facts"},
    ]
    v1 = [
        {"id": "analogical_reasoning", "name": "Analogical", "definition": "analogies"},
        {"id": "quantitative_reasoning", "name": "Quant", "definition": "numbers"},
        {"id": "long_term_knowledge_recall", "name": "Recall", "definition": "facts"},
    ]

    map_fn, map_counter = _make_mapper_fn()
    exec_fn, exec_counter = _make_executer_fn([v0, v1])
    tag_fn = _uniform_tag_fn(
        {b: ["deductive_reasoning"] for b in corpus.benchmark_names}
    )
    improver = _stepping_improver()

    history, _best = run_part1(
        config, corpus=corpus,
        tag_fn=tag_fn,
        improver_fn=improver,
        mapreduce_chat_fn=map_fn,
        executer_chat_fn=exec_fn,
    )

    # Two iterations → two distinct prompts → two executer calls (no cache hit).
    assert exec_counter["n"] == 2
    assert exec_counter["prompts_seen"][0] != exec_counter["prompts_seen"][1]

    # vocab_hash differs across iters because V differs.
    assert history[0].vocab_hash is not None
    assert history[1].vocab_hash is not None
    assert history[0].vocab_hash != history[1].vocab_hash
    assert [v["id"] for v in history[0].vocab] == [v["id"] for v in v0]
    assert [v["id"] for v in history[1].vocab] == [v["id"] for v in v1]

    # Per-iter V.json + final/vocab_star.json + final/vocab_star_metadata.json all exist.
    run_dirs = os.listdir(tmp_resources["results_dir"])
    root = os.path.join(tmp_resources["results_dir"], run_dirs[0])
    assert os.path.isfile(os.path.join(root, "iter_001", "V.json"))
    assert os.path.isfile(os.path.join(root, "iter_002", "V.json"))
    assert os.path.isfile(os.path.join(root, "final", "vocab_star.json"))
    assert os.path.isfile(os.path.join(root, "final", "vocab_star_metadata.json"))

    with open(os.path.join(root, "iter_001", "V.json"), encoding="utf-8") as f:
        v_iter1 = json.load(f)
    assert [v["id"] for v in v_iter1["vocab"]] == [v["id"] for v in v0]
    assert v_iter1["vocab_hash"] == history[0].vocab_hash

    with open(os.path.join(root, "final", "vocab_star.json"), encoding="utf-8") as f:
        v_star = json.load(f)
    assert isinstance(v_star, list)
    assert [v["id"] for v in v_star] in (
        [v["id"] for v in v0],
        [v["id"] for v in v1],
    )
    with open(os.path.join(root, "final", "vocab_star_metadata.json"), encoding="utf-8") as f:
        v_meta = json.load(f)
    # source = full train split per fold (v3 multi-source).
    assert sorted(v_meta["source_benchmarks"]) == sorted(bench_split.train)
    assert v_meta["best_iter_label"] in {"iter_001", "iter_002"}
    assert v_meta["selected_vocab_source"] == "executer"
    assert v_meta["selected_tag_count"] in {len(v0), len(v1)}

    with open(os.path.join(root, "selection.json"), encoding="utf-8") as f:
        selection = json.load(f)
    assert selection["selected_source"] == "executer"
    assert selection["selected_vocab_source"] == "executer"
    assert selection["selected_tag_count"] in {len(v0), len(v1)}
    assert selection["selected_vocab_path"] == "final/vocab_star.json"
    assert selection["candidate_history_path"] == "selection_candidates.json"


def test_v_loop_seed_prompt_is_schema_agnostic():
    """Regression: the v_loop seed prompt must NOT prescribe an output schema.

    Both Executer and Maker enforce their own JSON schemas via system/user
    messages (executer.py:200-211 → ``{"vocab":[...]}``,
    maker.py:217-225 → ``{"ability_levels":{...}}``). If the seed prompt body
    repeats a schema directive, the LLM follows the later in-prompt directive
    and emits the wrong shape — the original ``vocab_not_list`` failure that
    caused V_i to fall back to the seed every iteration.
    """
    seed_path = (
        Path(__file__).resolve().parent.parent / "prompts" / "I_exec_seed.txt"
    )
    assert seed_path.exists(), "I_exec_seed.txt must ship in experiment/prompts/"
    body = seed_path.read_text(encoding="utf-8")
    lower = body.lower()

    # Schema-fixing directives that would conflict with role-level schemas.
    forbidden_phrases = [
        "return json",
        '"vocab":',
        '"weights":',
        '"ability_levels":',
        '"rationale":',
    ]
    offenders = [p for p in forbidden_phrases if p in lower]
    assert not offenders, (
        f"I_exec_seed.txt re-introduced schema directive(s): {offenders}. "
        "Schemas must be set by role-level executer/maker code, not the prompt body."
    )


def test_v_loop_executer_with_schema_agnostic_seed_returns_vocab(tmp_resources):
    """End-to-end: when the seed prompt is schema-agnostic, Executer's user_msg
    embeds it without competing schema directives, so the LLM (mocked here)
    can respond with the correct ``{"vocab":[...]}`` shape and IterationResult
    records vocab_size > 0 — i.e. no fallback to the seed vocab."""
    seed_text = (
        "You are reasoning about a small reusable cognitive ability vocabulary V. "
        "Each entry should be a domain-general cognitive dimension defined by one "
        "short sentence. Avoid benchmark or dataset names; avoid duplicates. "
        "The structured output schema is determined by the calling role."
    )

    overrides = _v_loop_overrides(tmp_resources)
    overrides["prompt_i0_path"] = str(
        Path(tmp_resources["vocab_path"]).parent / "schema_agnostic_seed.txt"
    )
    Path(overrides["prompt_i0_path"]).write_text(seed_text, encoding="utf-8")

    config = load_experiment_config(overrides)
    corpus = _v_loop_corpus()

    captured = {}
    v_iter = [
        {"id": "deductive_reasoning", "name": "Deductive", "definition": "rule chains"},
        {"id": "long_term_knowledge_recall", "name": "Recall", "definition": "facts"},
    ]

    def exec_fn(system_msg, user_msg):
        captured["system_msg"] = system_msg
        captured["user_msg"] = user_msg
        return json.dumps({"vocab": v_iter, "rationale": "schema-agnostic seed worked"})

    map_fn, _ = _make_mapper_fn()
    tag_fn = _uniform_tag_fn(
        {b: ["deductive_reasoning"] for b in corpus.benchmark_names}
    )

    history, _best = run_part1(
        config, corpus=corpus,
        tag_fn=tag_fn,
        improver_fn=_stepping_improver(),
        mapreduce_chat_fn=map_fn,
        executer_chat_fn=exec_fn,
    )

    # Executer's system message pins the {"vocab":[...]} schema.
    assert '{"vocab":' in captured["system_msg"]
    # The seed embedded in user_msg has no competing schema directive.
    assert "Return JSON" not in captured["user_msg"]
    assert '"weights":' not in captured["user_msg"]

    # No fallback to seed vocab — V_i carries the executer-emitted ids.
    assert history, "iteration history should not be empty"
    assert history[0].vocab is not None
    assert len(history[0].vocab) == len(v_iter)
    assert [v["id"] for v in history[0].vocab] == [v["id"] for v in v_iter]


def test_v_loop_executer_invalid_vocab_fails_run(tmp_resources):
    """Executer schema failures must not silently fall back to the seed vocab."""
    overrides = _v_loop_overrides(tmp_resources, max_iter=1)
    config = load_experiment_config(overrides)
    corpus = _v_loop_corpus()
    map_fn, _ = _make_mapper_fn()

    def bad_exec_fn(_system_msg, _user_msg):
        return json.dumps({"not_vocab": []})

    with pytest.raises(JSONContractError, match="executer:iter_001"):
        run_part1(
            config,
            corpus=corpus,
            tag_fn=_uniform_tag_fn({}),
            mapreduce_chat_fn=map_fn,
            executer_chat_fn=bad_exec_fn,
        )


def test_v_loop_cycles_executer_candidate_counts(tmp_resources):
    overrides = _v_loop_overrides(
        tmp_resources,
        max_iter=2,
        executer_candidate_counts=[1, 2],
    )
    config = load_experiment_config(overrides)
    corpus = _v_loop_corpus()
    map_fn, _ = _make_mapper_fn()
    seen_targets: list[int] = []

    def count_conditioned_exec(_system_msg, user_msg):
        target = 2 if "exactly 2" in user_msg else 1
        seen_targets.append(target)
        vocab = [
            {
                "id": f"target_reasoning_{target}_{idx}",
                "name": f"Target Reasoning {target} {idx}",
                "definition": "Reusable cognitive operation.",
            }
            for idx in range(target)
        ]
        return json.dumps({"vocab": vocab, "rationale": "counted"})

    run_part1(
        config,
        corpus=corpus,
        tag_fn=_uniform_tag_fn({}),
        improver_fn=_stepping_improver(),
        mapreduce_chat_fn=map_fn,
        executer_chat_fn=count_conditioned_exec,
    )

    run_dirs = sorted(Path(config["results_dir"]).glob("run_*"))
    assert len(run_dirs) == 1
    root = str(run_dirs[0])
    assert seen_targets == [1, 2]
    with open(os.path.join(root, "iter_001", "V.json"), encoding="utf-8") as f:
        v1 = json.load(f)
    with open(os.path.join(root, "iter_002", "V.json"), encoding="utf-8") as f:
        v2 = json.load(f)
    assert len(v1["vocab"]) == 1
    assert len(v2["vocab"]) == 2
    assert v1["executer_metadata"]["target_count"] == 1
    assert v2["executer_metadata"]["target_count"] == 2


def test_v_loop_continues_after_improver_rejection_when_candidate_counts_vary(tmp_resources):
    """A rejected prompt rewrite is not a duplicate evaluation when target_count changes."""
    overrides = _v_loop_overrides(
        tmp_resources,
        max_iter=2,
        executer_candidate_counts=[1, 2],
    )
    config = load_experiment_config(overrides)
    corpus = _v_loop_corpus()
    map_fn, _ = _make_mapper_fn()
    seen_targets: list[int] = []

    def count_conditioned_exec(_system_msg, user_msg):
        target = 2 if "exactly 2" in user_msg else 1
        seen_targets.append(target)
        vocab = [
            {
                "id": f"target_reasoning_{target}_{idx}",
                "name": f"Target Reasoning {target} {idx}",
                "definition": "Reusable cognitive operation.",
            }
            for idx in range(target)
        ]
        return json.dumps({"vocab": vocab, "rationale": "counted"})

    def rejected_improver(**kwargs):
        return ImproverResult(
            new_prompt=kwargs["prev_prompt"] + "\nRejected rewrite.",
            accepted=False,
            reasons=["guard_rejected"],
            raw_response="{}",
            rationale="guard rejected",
        )

    run_part1(
        config,
        corpus=corpus,
        tag_fn=_uniform_tag_fn({}),
        improver_fn=rejected_improver,
        mapreduce_chat_fn=map_fn,
        executer_chat_fn=count_conditioned_exec,
    )

    run_dirs = sorted(Path(config["results_dir"]).glob("run_*"))
    assert len(run_dirs) == 1
    root = str(run_dirs[0])
    assert seen_targets == [1, 2]
    assert os.path.exists(os.path.join(root, "iter_001", "V.json"))
    assert os.path.exists(os.path.join(root, "iter_002", "V.json"))


def test_v_loop_improver_receives_active_vocab_not_seed(tmp_resources):
    """Regression: in v_loop, Improver must be called with V_i (active_vocab),
    not the seed vocab. A bug where loop.py passed `vocab=vocab` instead of
    `vocab=active_vocab` would feed Improver stale seed IDs across iterations
    even though tagging used V_i — invalidating the new_prompt validation
    against the active vocabulary."""
    corpus = _v_loop_corpus()
    bench_split = split_benchmarks(corpus.benchmark_names, ratios=(0.6, 0.2, 0.2), seed=0)
    src = bench_split.train[0]

    overrides = _v_loop_overrides(
        tmp_resources,
        max_iter=2,
        best_iter_dev_rho_floor=None,
    )
    config = load_experiment_config(overrides)

    # IDs deliberately chosen to be disjoint from the seed VOCAB constant so we
    # can prove Improver received the executer's V_i and not the loaded seed.
    v0 = [
        {"id": "spatial_reasoning_v3only", "name": "Spatial", "definition": "spatial reasoning"},
        {"id": "metacognitive_planning_v3only", "name": "Plan", "definition": "metacognition"},
    ]
    with open(tmp_resources["vocab_path"], encoding="utf-8") as f:
        seed_vocab_ids = {v["id"] for v in json.load(f)}
    assert seed_vocab_ids.isdisjoint({v["id"] for v in v0}), (
        "fixture invariant: seed vocab and v0 must have distinct IDs so the test discriminates"
    )

    map_fn, _ = _make_mapper_fn()
    exec_fn, _ = _make_executer_fn([v0])
    tag_fn = _uniform_tag_fn({b: ["spatial_reasoning_v3only"] for b in corpus.benchmark_names})

    captured: list[list[dict]] = []

    def capturing_improver(**kwargs):
        captured.append(list(kwargs["vocab"]))
        return ImproverResult(
            new_prompt=kwargs["prev_prompt"] + " STEP",
            accepted=True, reasons=[], raw_response="{}", rationale="x",
        )

    run_part1(
        config, corpus=corpus,
        tag_fn=tag_fn,
        improver_fn=capturing_improver,
        mapreduce_chat_fn=map_fn,
        executer_chat_fn=exec_fn,
    )

    assert captured, "Improver should be called at least once when max_iter >= 2"
    received_ids = [v["id"] for v in captured[0]]
    assert received_ids == [v["id"] for v in v0], (
        f"Improver got vocab={received_ids}, expected V_i={[v['id'] for v in v0]} (active_vocab)."
    )
    assert set(received_ids).isdisjoint(seed_vocab_ids), (
        "Improver received seed vocab IDs — regression of the vocab→active_vocab fix."
    )


def test_v_loop_cache_hit_skips_executer_llm(tmp_resources):
    """Same (I_exec, Z_src, source) → cache hit → zero new executer calls."""
    corpus = _v_loop_corpus()
    bench_split = split_benchmarks(corpus.benchmark_names, ratios=(0.6, 0.2, 0.2), seed=0)
    src = bench_split.train[0]

    overrides = _v_loop_overrides(
        tmp_resources, max_iter=1,
    )
    config = load_experiment_config(overrides)

    v0 = [{"id": "deductive_reasoning", "name": "D", "definition": "d"}]
    map_fn, _ = _make_mapper_fn()
    exec_fn, exec_counter = _make_executer_fn([v0])
    tag_fn = _uniform_tag_fn({})

    # First run populates the executer cache.
    run_part1(
        config, corpus=corpus,
        tag_fn=tag_fn,
        mapreduce_chat_fn=map_fn,
        executer_chat_fn=exec_fn,
    )
    assert exec_counter["n"] == 1

    # Second run with the same run_dir (same prompt + same Z_src) must hit cache.
    run_dirs = os.listdir(tmp_resources["results_dir"])
    assert len(run_dirs) == 1
    run_dir = os.path.join(tmp_resources["results_dir"], run_dirs[0])

    def must_not_call(_system, _user):  # pragma: no cover - regression assertion
        raise AssertionError("Executer should hit cache instead of calling LLM")

    run_part1(
        config, corpus=corpus,
        tag_fn=tag_fn,
        mapreduce_chat_fn=map_fn,
        executer_chat_fn=must_not_call,
        run_dir=run_dir,
    )


def test_v_loop_disabled_keeps_legacy_path_artifacts(tmp_resources):
    """enable_v_loop=False (default) writes no V.json / V_star.json."""
    overrides = {
        **tmp_resources,
        "max_iter": 1,
        "bootstrap_B": 5,
        "min_common_models": 6,
        "run_baseline": False,
        "taxonomy_refinement_enabled": False,
        # enable_v_loop omitted (default False)
    }
    config = load_experiment_config(overrides)

    fixed = {
        b: {tid: 0.5 for tid in (v["id"] for v in VOCAB)}
        for b in ["BenchA", "BenchB", "BenchC", "BenchD", "BenchE"]
    }

    history, _best = run_part1(config, tag_fn=_fixed_tag_fn(fixed))

    assert all(h.vocab is None and h.vocab_hash is None for h in history)
    run_dirs = os.listdir(tmp_resources["results_dir"])
    root = os.path.join(tmp_resources["results_dir"], run_dirs[0])
    assert not os.path.isfile(os.path.join(root, "iter_001", "V.json"))
    # No executer ran → no metadata file. The seed vocab is still written as
    # vocab_star.json by the unified Phase L schema (purely additive vs. legacy).
    assert not os.path.isfile(os.path.join(root, "final", "vocab_star_metadata.json"))
    assert os.path.isfile(os.path.join(root, "final", "vocab_star.json"))


def test_dev_l_align_without_v_loop_normalizes_to_train_l_align(tmp_resources):
    overrides = {
        **tmp_resources,
        "max_iter": 1,
        "bootstrap_B": 5,
        "min_common_models": 6,
        "run_baseline": False,
        "taxonomy_refinement_enabled": False,
        "best_iter_selection": "dev_l_align",
    }
    config = load_experiment_config(overrides)
    fixed = {
        b: {tid: 0.5 for tid in (v["id"] for v in VOCAB)}
        for b in ["BenchA", "BenchB", "BenchC", "BenchD", "BenchE"]
    }

    history, best = run_part1(config, tag_fn=_fixed_tag_fn(fixed))

    assert history
    assert best is not None


def test_v_loop_compute_metrics_only_sees_train_pairs(tmp_resources):
    """codex 2026-05-10 #6: _compute_metrics must never receive split-external pairs."""
    corpus = _v_loop_corpus()
    bench_split = split_benchmarks(corpus.benchmark_names, ratios=(0.6, 0.2, 0.2), seed=0)
    src = bench_split.train[0]
    train_set = set(bench_split.train)
    dev_set = set(bench_split.dev)

    overrides = _v_loop_overrides(
        tmp_resources, max_iter=1,
    )
    config = load_experiment_config(overrides)

    spy_calls: list[set] = []

    import autotagging_loop.experiment.loop as loop_mod
    real_compute = loop_mod._compute_metrics

    def spy(T, benchmark_names, R_raw, R01, *args, **kwargs):
        spy_calls.append(set(benchmark_names))
        return real_compute(T, benchmark_names, R_raw, R01, *args, **kwargs)

    v0 = [{"id": "deductive_reasoning", "name": "D", "definition": "d"}]
    map_fn, _ = _make_mapper_fn()
    exec_fn, _ = _make_executer_fn([v0])

    monkeypatch_target = loop_mod
    setattr(monkeypatch_target, "_compute_metrics", spy)
    try:
        run_part1(
            config, corpus=corpus,
            tag_fn=_uniform_tag_fn({}),
            mapreduce_chat_fn=map_fn,
            executer_chat_fn=exec_fn,
        )
    finally:
        setattr(monkeypatch_target, "_compute_metrics", real_compute)

    # Iter call inside the loop body must use either train or dev split — never
    # full corpus, never test split. A single full-corpus metric call is allowed
    # only after selection, when final/ is written.
    in_loop_calls = [
        names for names in spy_calls
        if names == train_set or names == dev_set
    ]
    assert in_loop_calls, "expected at least one train/dev-only metric call"
    test_set = set(bench_split.test)
    full_set = set(corpus.benchmark_names)
    final_full_calls = [names for names in spy_calls if names == full_set]
    assert len(final_full_calls) == 1
    for names in spy_calls:
        # No call that mixes train+test or dev+test pairs.
        assert not (names & test_set and names not in (test_set, full_set)), (
            f"metrics call leaked test split benchmarks: {names}"
        )


def test_v_loop_determinism_same_seed_same_artifacts(tmp_resources, tmp_path):
    """Same config + same chat fns → identical V_star.json and metrics."""
    corpus = _v_loop_corpus()
    bench_split = split_benchmarks(corpus.benchmark_names, ratios=(0.6, 0.2, 0.2), seed=0)
    src = bench_split.train[0]

    def run_once(results_dir: str) -> dict:
        overrides = {
            **tmp_resources,
            "results_dir": results_dir,
            "max_iter": 1,
            "bootstrap_B": 5,
            "min_common_models": 6,
            "run_baseline": False,
            "taxonomy_refinement_enabled": False,
            "taxonomy_selection_enabled": False,
            "best_iter_selection": "train_l_align",
            "executer_candidate_counts": None,
            "enable_v_loop": True,
            "use_mapreduce_evidence": False,
            "mapreduce_chunk_examples": 1,
            "mapreduce_max_chunk_chars": 2000,
            "mapreduce_max_evidence_chars": 4000,
            "mapreduce_max_workers": 1,
            "mapreduce_cache_enabled": False,
            "v_loop_min_test_valid_pairs": 0,
        }
        config = load_experiment_config(overrides)
        v0 = [
            {"id": "deductive_reasoning", "name": "D", "definition": "d"},
            {"id": "long_term_knowledge_recall", "name": "R", "definition": "r"},
        ]
        map_fn, _ = _make_mapper_fn()
        exec_fn, _ = _make_executer_fn([v0])
        run_part1(
            config, corpus=corpus,
            tag_fn=_uniform_tag_fn({}),
            mapreduce_chat_fn=map_fn,
            executer_chat_fn=exec_fn,
        )
        run_dirs = os.listdir(results_dir)
        root = os.path.join(results_dir, run_dirs[0])
        with open(os.path.join(root, "final", "vocab_star.json"), encoding="utf-8") as f:
            v_star = json.load(f)
        with open(os.path.join(root, "final", "vocab_star_metadata.json"), encoding="utf-8") as f:
            v_meta = json.load(f)
        with open(os.path.join(root, "final", "metrics_with_bootstrap.json"), encoding="utf-8") as f:
            metrics = json.load(f)
        return {
            "vocab": [v["id"] for v in v_star],
            "vocab_hash": v_meta["vocab_hash"],
            "L_align": metrics.get("L_align"),
            "delta_tag": metrics.get("delta_tag"),
        }

    dir_a = str(tmp_path / "run_a")
    dir_b = str(tmp_path / "run_b")
    os.makedirs(dir_a)
    os.makedirs(dir_b)
    a = run_once(dir_a)
    b = run_once(dir_b)
    assert a == b


def test_phase_l_final_schema_symmetric_across_paths(tmp_resources):
    """Phase L — `final/` and `taxonomy_refinement/final/` share the same core schema."""
    overrides = {
        **tmp_resources,
        "max_iter": 1,
        "bootstrap_B": 5,
        "min_common_models": 6,
        "run_baseline": False,
        "taxonomy_refinement_enabled": True,
        "taxonomy_refinement_min_pairs": 1,
        "taxonomy_refinement_residual_max_threshold": 0.0,
        "taxonomy_refinement_max_iter": 1,
    }
    config = load_experiment_config(overrides)
    good, mid, _bad = _taxonomy_test_weights()

    def tag_fn(benchmark, description, vocab, prompt, version):
        weights = good[benchmark] if version >= 10_000 else mid[benchmark]
        return TagVector(benchmark=benchmark, weights=dict(weights),
                         raw_response="{}", prompt_version=version)

    run_part1(config, tag_fn=tag_fn, taxonomy_refiner_fn=_accepted_taxonomy_refiner)

    run_dirs = os.listdir(tmp_resources["results_dir"])
    root = os.path.join(tmp_resources["results_dir"], run_dirs[0])
    fixed_dir = os.path.join(root, "final")
    tax_dir = os.path.join(root, "taxonomy_refinement", "final")

    # Core schema present in both paths.
    core_files = [
        "best_iter.txt",
        "I_star.txt",
        "T_star.json",
        "T_star_raw.json",
        "metrics_with_bootstrap.json",
        "metrics_raw.json",
    ]
    for fname in core_files:
        assert os.path.isfile(os.path.join(fixed_dir, fname)), f"fixed/{fname} missing"
        assert os.path.isfile(os.path.join(tax_dir, fname)), f"taxonomy/{fname} missing"

    # Both paths emit vocab_star.json (taxonomy: refined vocab; fixed: seed).
    assert os.path.isfile(os.path.join(fixed_dir, "vocab_star.json"))
    assert os.path.isfile(os.path.join(tax_dir, "vocab_star.json"))

    # selection.json carries both legacy `selected_*` and unified `mode` /
    # `chosen_iter_label` / `chosen_metrics` keys.
    with open(os.path.join(root, "selection.json"), encoding="utf-8") as f:
        sel = json.load(f)
    assert sel["mode"] in {"fixed", "taxonomy"}
    assert sel["chosen_iter_label"]
    assert sel["chosen_metrics"]
    assert sel["selected_source"] in {"fixed", "executer", "taxonomy_refinement"}
    assert sel["selected_vocab_source"] in {"seed", "executer", "taxonomy_refinement"}
    assert sel["selected_tag_count"] > 0


def test_phase_m_v3_main_loop_end_to_end_smoke(tmp_resources):
    """Phase M — single integration smoke covering all v3 pass criteria.

    Combines pass criteria #1 (artifacts), #2 (split metrics finite),
    #3 (metric ranges), #4 (V loop activated). #5 / #6 / #7 are covered
    by sibling tests; this test guards against regressions where the
    pieces work individually but not together.
    """
    # Phase M corpus — 12 benchmarks with distinct score *patterns* (not just
    # shift) so pairwise score similarities R have variance and ρ metrics are
    # finite. With 0.5/0.25/0.25 ratios → train=6, dev=3, test=3 → ≥3 pairs per
    # split (so ρ is well-defined on every split). _v_loop_corpus uses linearly-
    # shifted scores which collapse R to a constant → NaN ρ.
    bench_names = [f"Bench{i:02d}" for i in range(12)]
    model_names = [f"m{i}" for i in range(1, 8)]
    Y = {}
    for idx, b in enumerate(bench_names):
        Y[b] = {
            m: 0.5 + 0.4 * math.sin((idx + 1) * (i + 1) * 0.7)
            for i, m in enumerate(model_names)
        }
    corpus = Corpus(
        benchmark_names=bench_names,
        model_names=model_names,
        Y=Y,
        descriptions={b: f"description {b}" for b in bench_names},
        documents={
            b: {
                "reviewed_rows": 1,
                "topic_counts": {},
                "reasoning_depth_counts": {},
                "answer_format_counts": {},
                "examples": [f"{b} example"],
            }
            for b in bench_names
        },
    )
    bench_split = split_benchmarks(corpus.benchmark_names, ratios=(0.5, 0.25, 0.25), seed=0)
    src = bench_split.train[0]

    overrides = _v_loop_overrides(
        tmp_resources,
        max_iter=2,
        splits={"benchmark_ratios": (0.5, 0.25, 0.25), "benchmark_seed": 0},
    )
    config = load_experiment_config(overrides)

    v0 = [
        {"id": "deductive_reasoning", "name": "D", "definition": "rules"},
        {"id": "long_term_knowledge_recall", "name": "R", "definition": "facts"},
    ]
    v1 = [
        {"id": "deductive_reasoning", "name": "D", "definition": "rules"},
        {"id": "analogical_reasoning", "name": "A", "definition": "analogies"},
    ]
    # Varying tag function — gives correlation/spearman finite values by spreading
    # per-benchmark weights deterministically so ρ has non-zero variance.
    def varying_tag_fn(benchmark, description, vocab, prompt, version):
        ids = [v["id"] for v in vocab]
        seed_val = sum(ord(c) for c in benchmark) % 7
        weights = {tid: ((seed_val + i + 1) % 5) / 5.0 + 0.1
                   for i, tid in enumerate(ids)}
        return TagVector(
            benchmark=benchmark, weights=weights, raw_response="{}",
            prompt_version=version,
        )

    map_fn, _ = _make_mapper_fn()
    exec_fn, _ = _make_executer_fn([v0, v1])
    history, _best = run_part1(
        config, corpus=corpus,
        tag_fn=varying_tag_fn,
        improver_fn=_stepping_improver(),
        mapreduce_chat_fn=map_fn,
        executer_chat_fn=exec_fn,
    )

    run_dirs = os.listdir(tmp_resources["results_dir"])
    root = os.path.join(tmp_resources["results_dir"], run_dirs[0])
    final_dir = os.path.join(root, "final")

    # Pass criterion #1 — required final artifacts exist.
    for fname in (
        "I_star.txt",
        "T_star.json",
        "T_star_raw.json",
        "metrics_with_bootstrap.json",
        "vocab_star.json",
        "vocab_star_metadata.json",
        "split_metrics.json",
    ):
        assert os.path.isfile(os.path.join(final_dir, fname)), f"missing final/{fname}"
    assert os.path.isfile(os.path.join(root, "selection.json"))

    # Pass criterion #2 + #3 — split metrics finite + within plan ranges.
    with open(os.path.join(final_dir, "split_metrics.json"), encoding="utf-8") as f:
        sm = json.load(f)
    for split_name in ("train", "dev", "test"):
        assert split_name in sm, f"split_metrics missing {split_name!r}"
        m = sm[split_name]
        for key in ("L_align", "L_align_01", "rho_align_pearson",
                    "rho_align_spearman", "delta_tag"):
            assert key in m, f"{split_name}.{key} missing"
            v = m[key]
            assert math.isfinite(v), f"{split_name}.{key} = {v!r} not finite"
        # Ranges per Phase M pass criterion #3. L_align is MSE(S, R_raw) where
        # R_raw ∈ [-1, 1] and S ∈ [0, 1], so worst-case L_align ≤ 4 (synthetic
        # data only — real benchmarks stay well below 1). L_align_01 uses R01.
        assert 0.0 <= m["L_align"] <= 4.0
        assert 0.0 <= m["L_align_01"] <= 1.0
        assert -1.0 <= m["rho_align_pearson"] <= 1.0
        assert -1.0 <= m["rho_align_spearman"] <= 1.0
        assert abs(m["delta_tag"]) <= 2.0

    # Pass criterion #4 — V loop actually activated → vocab_hash changed.
    hashes = [h.vocab_hash for h in history if h.vocab_hash is not None]
    assert len(hashes) >= 2, "expected ≥2 iterations producing vocab_hash"
    assert len(set(hashes)) >= 2, (
        "V loop did not change vocab across iterations — Improver may not have "
        "mutated the prompt"
    )


def test_v_loop_writes_test_split_metrics_exactly_once(tmp_resources):
    """codex 2026-05-10 pass criterion #5: D_test evaluated only after the loop."""
    corpus = _v_loop_corpus()
    bench_split = split_benchmarks(corpus.benchmark_names, ratios=(0.6, 0.2, 0.2), seed=0)
    src = bench_split.train[0]

    overrides = _v_loop_overrides(
        tmp_resources, max_iter=2,
    )
    config = load_experiment_config(overrides)

    v0 = [{"id": "deductive_reasoning", "name": "D", "definition": "d"}]
    map_fn, _ = _make_mapper_fn()
    exec_fn, _ = _make_executer_fn([v0])

    run_part1(
        config, corpus=corpus,
        tag_fn=_uniform_tag_fn({}),
        improver_fn=_stepping_improver(),
        mapreduce_chat_fn=map_fn,
        executer_chat_fn=exec_fn,
    )

    run_dirs = os.listdir(tmp_resources["results_dir"])
    root = os.path.join(tmp_resources["results_dir"], run_dirs[0])
    split_metrics_path = os.path.join(root, "final", "split_metrics.json")
    assert os.path.isfile(split_metrics_path)
    with open(split_metrics_path, encoding="utf-8") as f:
        split_metrics = json.load(f)
    # Train + dev + test all present.
    for split_name in ("train", "dev", "test"):
        assert split_name in split_metrics, f"missing {split_name!r} in split_metrics"
    # Test split metrics finite (or nan-allowed) but the artifact must exist
    # exactly once — written from `final/` after the loop, not from any iter dir.
    iter_dirs = [d for d in os.listdir(root) if d.startswith("iter_")]
    for d in iter_dirs:
        assert not os.path.isfile(os.path.join(root, d, "split_metrics.json"))
