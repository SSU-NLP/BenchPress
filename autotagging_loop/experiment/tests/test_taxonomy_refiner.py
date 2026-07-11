"""Tests for experiment/taxonomy_refiner.py."""

from __future__ import annotations

import json

from autotagging_loop.experiment.taxonomy_refiner import (
    refine_taxonomy,
    validate_refined_taxonomy,
    vocab_quality_reasons,
)


SEED_VOCAB = [
    {"id": "analogical_reasoning", "name": "Analogical Reasoning", "definition": "Map relations."},
    {"id": "deductive_reasoning", "name": "Deductive Reasoning", "definition": "Apply explicit rules."},
]


def test_validate_refined_taxonomy_accepts_seed_plus_new_tag():
    vocab = [
        *SEED_VOCAB,
        {"id": "visual_pattern_recognition", "name": "Visual Pattern Recognition",
         "definition": "Identify non-textual visual patterns."},
    ]
    prompt = (
        "Use only analogical_reasoning, deductive_reasoning, and visual_pattern_recognition. "
        "Return JSON weights for every listed tag id and do not invent other tags."
    )

    ok, reasons = validate_refined_taxonomy(
        vocab,
        prompt,
        SEED_VOCAB,
        benchmark_names=["MMLU"],
        max_new_tags=2,
        base_prompt="Use analogical_reasoning and deductive_reasoning.",
    )

    assert ok, reasons


def test_validate_refined_taxonomy_rejects_missing_seed_tag():
    vocab = [
        {"id": "visual_pattern_recognition", "name": "Visual Pattern Recognition",
         "definition": "Identify non-textual visual patterns."},
    ]
    prompt = "Use only visual_pattern_recognition. Return JSON weights."

    ok, reasons = validate_refined_taxonomy(
        vocab,
        prompt,
        SEED_VOCAB,
        benchmark_names=[],
        retain_seed_tags=True,
    )

    assert not ok
    assert any("seed_tags_missing" in reason for reason in reasons)


def test_validate_refined_taxonomy_rejects_non_cognitive_vocab_axis():
    vocab = [
        *SEED_VOCAB,
        {"id": "hle_frontier_difficulty", "name": "HLE Frontier Difficulty",
         "definition": "Separates benchmarks by model leaderboard performance."},
    ]
    prompt = (
        "Use only analogical_reasoning, deductive_reasoning, and hle_frontier_difficulty. "
        "Return JSON weights for every listed tag id and no other tag ids."
    )

    ok, reasons = validate_refined_taxonomy(
        vocab,
        prompt,
        SEED_VOCAB,
        benchmark_names=["HLE"],
        max_new_tags=2,
        base_prompt="Use analogical_reasoning and deductive_reasoning.",
    )

    assert not ok
    assert any("vocab_leaks_non_cognitive_axis" in reason for reason in reasons)


def test_vocab_quality_allows_cognitive_mental_model_phrase():
    vocab = [
        {"id": "dynamic_state_tracking", "name": "Dynamic State Tracking",
         "definition": "Maintain a mental model of changing entities and values."},
    ]

    assert not vocab_quality_reasons(vocab, benchmark_names=[])


def test_vocab_quality_rejects_model_performance_axis():
    vocab = [
        {"id": "model_performance_cluster", "name": "Model Performance Cluster",
         "definition": "Separates benchmarks by model leaderboard performance."},
    ]

    reasons = vocab_quality_reasons(vocab, benchmark_names=[])

    assert any("vocab_leaks_non_cognitive_axis" in reason for reason in reasons)


def test_refine_taxonomy_accepts_clean_response():
    vocab = [
        *SEED_VOCAB,
        {"id": "visual_pattern_recognition", "name": "Visual Pattern Recognition",
         "definition": "Identify non-textual visual patterns."},
    ]
    prompt = (
        "Use only analogical_reasoning, deductive_reasoning, and visual_pattern_recognition. "
        "Return JSON weights for every listed tag id and no other tag ids."
    )

    def chat(_system, _user):
        return json.dumps({"vocab": vocab, "new_prompt": prompt, "rationale": "persistent residuals"})

    result = refine_taxonomy(
        seed_vocab=SEED_VOCAB,
        base_prompt="Use analogical_reasoning and deductive_reasoning.",
        best_prompt="Use analogical_reasoning and deductive_reasoning carefully.",
        residual_report=[],
        metrics={"residual_max": 0.8},
        benchmark_names=["FakeBench"],
        model="m",
        base_url=None,
        chat_fn=chat,
    )

    assert result.accepted, result.reasons
    assert [v["id"] for v in result.vocab] == [
        "analogical_reasoning",
        "deductive_reasoning",
        "visual_pattern_recognition",
    ]


def test_refine_taxonomy_payload_includes_protected_pairs():
    vocab = [
        *SEED_VOCAB,
        {"id": "visual_pattern_recognition", "name": "Visual Pattern Recognition",
         "definition": "Identify non-textual visual patterns."},
    ]
    prompt = (
        "Use only analogical_reasoning, deductive_reasoning, and visual_pattern_recognition. "
        "Return JSON weights for every listed tag id and no other tag ids."
    )
    protected_pairs = [{
        "benchmark_pair": ["BenchA", "BenchB"],
        "score_similarity": 0.92,
        "tag_similarity": 0.88,
        "residual_abs": 0.04,
    }]
    seen = {}

    def chat(_system, user):
        seen["payload"] = json.loads(user)
        return json.dumps({"vocab": vocab, "new_prompt": prompt, "rationale": "persistent residuals"})

    result = refine_taxonomy(
        seed_vocab=SEED_VOCAB,
        base_prompt="Use analogical_reasoning and deductive_reasoning.",
        best_prompt="Use analogical_reasoning and deductive_reasoning carefully.",
        residual_report=[],
        metrics={"residual_max": 0.8},
        benchmark_names=["FakeBench"],
        model="m",
        base_url=None,
        protected_pairs=protected_pairs,
        chat_fn=chat,
    )

    assert result.accepted, result.reasons
    assert seen["payload"]["protected_high_similarity_pairs"] == protected_pairs
    assert "protected" in seen["payload"]["instruction"]


def test_refine_taxonomy_prompt_keeps_tags_operation_grounded():
    vocab = [
        *SEED_VOCAB,
        {"id": "visual_pattern_recognition", "name": "Visual Pattern Recognition",
         "definition": "Identify non-textual visual patterns."},
    ]
    prompt = (
        "Use only analogical_reasoning, deductive_reasoning, and visual_pattern_recognition. "
        "Return JSON weights for every listed tag id and no other tag ids."
    )
    seen = {}

    def chat(system, user):
        seen["system"] = system
        seen["payload"] = json.loads(user)
        return json.dumps({"vocab": vocab, "new_prompt": prompt, "rationale": "persistent residuals"})

    result = refine_taxonomy(
        seed_vocab=SEED_VOCAB,
        base_prompt="Use analogical_reasoning and deductive_reasoning.",
        best_prompt="Use analogical_reasoning and deductive_reasoning carefully.",
        residual_report=[],
        metrics={"residual_max": 0.8},
        benchmark_names=["FakeBench"],
        model="m",
        base_url=None,
        protected_pairs=[],
        chat_fn=chat,
    )

    assert result.accepted, result.reasons
    assert "reusable cognitive operations visible in the evidence" in seen["system"]
    assert "not benchmark families" in seen["system"]
    assert "rate absent/weak/medium/strong/dominant" in seen["system"]
    assert "Do not break protected high-similarity pairs" in seen["system"]
    assert "Treat domain mismatch as weak evidence" in seen["payload"]["instruction"]
    assert "Avoid pure difficulty tags" in seen["payload"]["instruction"]
