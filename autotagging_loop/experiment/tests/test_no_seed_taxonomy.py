"""Tests for experiment/no_seed_taxonomy.py."""

from __future__ import annotations

import json

from autotagging_loop.experiment.corpus import Corpus
from autotagging_loop.experiment.no_seed_taxonomy import (
    induce_no_seed_taxonomy,
    validate_no_seed_taxonomy,
)


def test_validate_no_seed_taxonomy_accepts_clean_prompt():
    vocab = [
        {"id": "symbolic_problem_solving", "name": "Symbolic Problem Solving",
         "definition": "Solving abstract symbolic tasks."},
        {"id": "world_knowledge_use", "name": "World Knowledge Use",
         "definition": "Using external factual knowledge."},
    ]
    prompt = (
        "Use only symbolic_problem_solving and world_knowledge_use. "
        "Rate every listed tag id as absent, weak, medium, strong, or dominant."
    )

    ok, reasons = validate_no_seed_taxonomy(
        vocab,
        prompt,
        benchmark_names=["BenchA"],
        min_tags=2,
        max_tags=3,
    )

    assert ok, reasons


def test_validate_no_seed_taxonomy_rejects_answer_format_axis():
    vocab = [
        {"id": "multiple_choice_skill", "name": "Multiple Choice Skill",
         "definition": "Tracks the multiple choice answer format."},
        {"id": "world_knowledge_use", "name": "World Knowledge Use",
         "definition": "Using external factual knowledge."},
    ]
    prompt = (
        "Use only multiple_choice_skill and world_knowledge_use. "
        "Rate every listed tag id as absent, weak, medium, strong, or dominant."
    )

    ok, reasons = validate_no_seed_taxonomy(
        vocab,
        prompt,
        benchmark_names=["BenchA"],
        min_tags=2,
        max_tags=3,
    )

    assert not ok
    assert any("vocab_leaks_non_cognitive_axis" in reason for reason in reasons)


def test_validate_no_seed_taxonomy_rejects_weight_schema_prompt():
    vocab = [
        {"id": "symbolic_problem_solving", "name": "Symbolic Problem Solving",
         "definition": "Solving abstract symbolic tasks."},
        {"id": "world_knowledge_use", "name": "World Knowledge Use",
         "definition": "Using external factual knowledge."},
    ]
    prompt = (
        "Use only symbolic_problem_solving and world_knowledge_use. "
        "Assign confidence weight between 0.0 and 1.0 for every ability."
    )

    ok, reasons = validate_no_seed_taxonomy(
        vocab,
        prompt,
        benchmark_names=["BenchA"],
        min_tags=2,
        max_tags=3,
    )

    assert not ok
    assert any("prompt_conflicts_with_maker_schema" in reason for reason in reasons)


def test_induce_no_seed_taxonomy_uses_anonymized_benchmark_payload():
    vocab = [
        {"id": "symbolic_problem_solving", "name": "Symbolic Problem Solving",
         "definition": "Solving abstract symbolic tasks."},
        {"id": "world_knowledge_use", "name": "World Knowledge Use",
         "definition": "Using external factual knowledge."},
    ]
    prompt = (
        "Use only symbolic_problem_solving and world_knowledge_use. "
        "Rate every listed tag id as absent, weak, medium, strong, or dominant."
    )
    corpus = Corpus(
        benchmark_names=["SecretBench"],
        model_names=["m1", "m2"],
        Y={"SecretBench": {"m1": 1.0, "m2": 0.5}},
        descriptions={"SecretBench": "Benchmark: SecretBench\nA synthetic SecretBench symbolic task."},
        documents={
            "SecretBench": {
                "reviewed_rows": 1,
                "topic_counts": {"logic": 1},
                "reasoning_depth_counts": {"single_step": 1},
                "answer_format_counts": {"multiple_choice": 1},
                "examples": ["SecretBench Question: infer the rule\nAnswer: A"],
            }
        },
    )
    seen = {}

    def chat(_system, user):
        seen["payload"] = json.loads(user)
        return json.dumps({"vocab": vocab, "new_prompt": prompt, "rationale": "from evidence"})

    result = induce_no_seed_taxonomy(
        corpus=corpus,
        benchmark_names=corpus.benchmark_names,
        model="m",
        base_url=None,
        min_tags=2,
        max_tags=3,
        chat_fn=chat,
    )

    assert result.accepted, result.reasons
    brief = seen["payload"]["benchmark_evidence_anonymized"][0]
    assert brief["benchmark_ref"] == "benchmark_001"
    assert "benchmark" not in brief
    assert "SecretBench" not in json.dumps(seen["payload"])


def test_induce_no_seed_taxonomy_repairs_rejected_prompt():
    bad_vocab = [
        {"id": "semantic_disambiguation", "name": "Semantic Disambiguation",
         "definition": "Uses multiple-choice answer format cues."},
        {"id": "world_knowledge_use", "name": "World Knowledge Use",
         "definition": "Using external factual knowledge."},
    ]
    good_vocab = [
        {"id": "semantic_precision", "name": "Semantic Precision",
         "definition": "Resolving nuanced meanings and distractor distinctions."},
        {"id": "world_knowledge_use", "name": "World Knowledge Use",
         "definition": "Using external factual knowledge."},
    ]
    corpus = Corpus(
        benchmark_names=["SecretBench"],
        model_names=["m1", "m2"],
        Y={"SecretBench": {"m1": 1.0, "m2": 0.5}},
        descriptions={"SecretBench": "A synthetic symbolic task."},
        documents={"SecretBench": {"examples": ["Question: infer the rule"]}},
    )
    calls = []

    def chat(_system, user):
        calls.append(json.loads(user))
        if len(calls) == 1:
            return json.dumps({
                "vocab": bad_vocab,
                "new_prompt": (
                    "Use semantic_disambiguation and world_knowledge_use. "
                    "Assign confidence weight between 0.0 and 1.0."
                ),
                "rationale": "first attempt",
            })
        return json.dumps({
            "vocab": good_vocab,
            "new_prompt": (
                "Use only semantic_precision and world_knowledge_use. "
                "Rate every listed tag id as absent, weak, medium, strong, or dominant."
            ),
            "rationale": "repaired",
        })

    result = induce_no_seed_taxonomy(
        corpus=corpus,
        benchmark_names=corpus.benchmark_names,
        model="m",
        base_url=None,
        min_tags=2,
        max_tags=3,
        max_attempts=2,
        chat_fn=chat,
    )

    assert result.accepted, result.reasons
    assert len(calls) == 2
    assert calls[1]["previous_rejection"]["reasons"]


def test_induce_no_seed_taxonomy_prompt_keeps_axes_reusable():
    vocab = [
        {"id": "symbolic_problem_solving", "name": "Symbolic Problem Solving",
         "definition": "Solving abstract symbolic tasks."},
        {"id": "world_knowledge_use", "name": "World Knowledge Use",
         "definition": "Using external factual knowledge."},
    ]
    prompt = (
        "Use only symbolic_problem_solving and world_knowledge_use. "
        "Rate every listed tag id as absent, weak, medium, strong, or dominant."
    )
    corpus = Corpus(
        benchmark_names=["SecretBench"],
        model_names=["m1", "m2"],
        Y={"SecretBench": {"m1": 1.0, "m2": 0.5}},
        descriptions={"SecretBench": "A synthetic symbolic task."},
        documents={"SecretBench": {"examples": ["Question: infer the rule"]}},
    )
    seen = {}

    def chat(system, user):
        seen["system"] = system
        seen["payload"] = json.loads(user)
        return json.dumps({"vocab": vocab, "new_prompt": prompt, "rationale": "from evidence"})

    result = induce_no_seed_taxonomy(
        corpus=corpus,
        benchmark_names=corpus.benchmark_names,
        model="m",
        base_url=None,
        min_tags=2,
        max_tags=3,
        chat_fn=chat,
    )

    assert result.accepted, result.reasons
    assert "Separate surface task format from reusable reasoning operations" in seen["system"]
    assert "Prefer broad operations" in seen["system"]
    assert "observable evidence cues" in seen["system"]
    assert "Do not define axes by accuracy" in seen["system"]
    assert "Do not isolate simple, knowledge-heavy, math, coding, or reasoning" in seen["payload"]["instruction"]
    assert "Avoid pure difficulty tags" in seen["payload"]["instruction"]
    assert "Do not use accuracy, robustness, error resistance" in seen["payload"]["instruction"]


def test_induce_no_seed_taxonomy_adds_missing_vocab_ids_to_prompt():
    vocab = [
        {"id": "symbolic_problem_solving", "name": "Symbolic Problem Solving",
         "definition": (
             "Solving abstract symbolic tasks. Observable evidence includes "
             "code generation, chemistry, or benchmark-specific surface cues."
         )},
        {"id": "world_knowledge_use", "name": "World Knowledge Use",
         "definition": "Using external factual knowledge."},
    ]
    corpus = Corpus(
        benchmark_names=["SecretBench"],
        model_names=["m1", "m2"],
        Y={"SecretBench": {"m1": 1.0, "m2": 0.5}},
        descriptions={"SecretBench": "A synthetic symbolic task."},
        documents={"SecretBench": {"examples": ["Question: infer the rule"]}},
    )

    def chat(_system, _user):
        return json.dumps({
            "vocab": vocab,
            "new_prompt": "Use the returned cognitive ability vocabulary.",
            "rationale": "from evidence",
        })

    result = induce_no_seed_taxonomy(
        corpus=corpus,
        benchmark_names=corpus.benchmark_names,
        model="m",
        base_url=None,
        min_tags=2,
        max_tags=3,
        chat_fn=chat,
    )

    assert result.accepted, result.reasons
    assert "symbolic_problem_solving" in result.prompt
    assert "world_knowledge_use" in result.prompt
    assert "Use the returned cognitive ability vocabulary" not in result.prompt
    assert "Observable evidence includes" not in result.prompt
    assert "code generation" not in result.prompt
    assert len(result.prompt) < 1500


def test_induce_no_seed_taxonomy_passes_seed_to_chat_fn():
    vocab = [
        {"id": "symbolic_problem_solving", "name": "Symbolic Problem Solving",
         "definition": "Solving abstract symbolic tasks."},
        {"id": "world_knowledge_use", "name": "World Knowledge Use",
         "definition": "Using external factual knowledge."},
    ]
    corpus = Corpus(
        benchmark_names=["SecretBench"],
        model_names=["m1", "m2"],
        Y={"SecretBench": {"m1": 1.0, "m2": 0.5}},
        descriptions={"SecretBench": "A synthetic symbolic task."},
        documents={"SecretBench": {"examples": ["Question: infer the rule"]}},
    )
    seen = {}

    def chat(_system, _user, seed):
        seen["seed"] = seed
        return json.dumps({"vocab": vocab, "new_prompt": "Use the returned vocabulary."})

    result = induce_no_seed_taxonomy(
        corpus=corpus,
        benchmark_names=corpus.benchmark_names,
        model="m",
        base_url=None,
        min_tags=2,
        max_tags=3,
        chat_fn=chat,
        seed=42,
    )

    assert result.accepted, result.reasons
    assert seen["seed"] == 42
