"""Phase 0 gate tests: Mapper is vocab-free.

Two invariants enforced:
1. Cache key does NOT depend on vocab — vocab change must yield full cache hit.
2. Adversarial ability_* keys from a misbehaving small model fail the strict
   JSON contract instead of being silently stripped.
"""

from __future__ import annotations

import json

import pytest

from autotagging_loop.experiment.corpus import Corpus
from autotagging_loop.experiment.json_contract import JSONContractError
from autotagging_loop.experiment.mapreduce_evidence import build_mapreduce_descriptions


def _toy_corpus(n_examples: int = 2) -> Corpus:
    return Corpus(
        benchmark_names=["ToyBench"],
        model_names=["m1", "m2"],
        Y={"ToyBench": {"m1": 0.8, "m2": 0.6}},
        descriptions={"ToyBench": "raw fallback"},
        documents={
            "ToyBench": {
                "reviewed_rows": n_examples,
                "topic_counts": {"science": n_examples},
                "reasoning_depth_counts": {"single_step": n_examples},
                "answer_format_counts": {"multiple_choice": n_examples},
                "examples": [f"Question: q{i}\nAnswer: a" for i in range(n_examples)],
            }
        },
    )


def test_mapper_cache_invariant_under_vocab_change(tmp_path):
    """Two runs, same prompt, DIFFERENT vocab. Second run must hit cache fully."""
    corpus = _toy_corpus(2)
    config = {
        "mapreduce_model": {"name": "fake", "base_url": None},
        "mapreduce_chunk_examples": 1,
        "mapreduce_max_chunk_chars": 1000,
        "mapreduce_max_evidence_chars": 4000,
        "mapreduce_max_workers": 1,
        "mapreduce_cache_enabled": True,
        "mapreduce_cache_dir": str(tmp_path / "persistent_cache"),
        "mapreduce_write_run_cache_copy": True,
    }
    calls = {"n": 0}

    def chat(_system, _user):
        calls["n"] += 1
        return json.dumps({
            "chunk_summary": "summary",
            "task_patterns": [],
            "reasoning_patterns": [],
            "justifications": ["j"],
        })

    vocab_a = [
        {"id": "deductive_reasoning", "definition": "Apply explicit rules."},
        {"id": "long_term_knowledge_recall", "definition": "Recall stored facts."},
    ]
    _, agg_1 = build_mapreduce_descriptions(
        corpus,
        vocab_a,
        config,
        str(tmp_path / "run1"),
        chat_fn=chat,
        prompt="prompt v1",
    )
    assert calls["n"] == 2
    assert agg_1["ToyBench"]["cache_hits"] == 0

    vocab_b = [
        {"id": "deductive_reasoning", "definition": "Apply explicit rules."},
        {"id": "creative_reasoning", "definition": "Generate novel ideas."},
    ]

    def fail_chat(_system, _user):
        raise AssertionError("vocab change must not invalidate the mapper cache")

    _, agg_2 = build_mapreduce_descriptions(
        corpus,
        vocab_b,
        config,
        str(tmp_path / "run2"),
        chat_fn=fail_chat,
        prompt="prompt v1",
    )

    assert calls["n"] == 2
    assert agg_2["ToyBench"]["cache_hits"] == 2


def test_mapper_rejects_banned_ability_keys(tmp_path):
    """Misbehaving small model emits ability_* keys: fail the run."""
    corpus = _toy_corpus(2)

    def chat(_system, _user):
        return json.dumps({
            "chunk_summary": "adversarial",
            "task_patterns": ["pattern"],
            "reasoning_patterns": ["reasoning"],
            "justifications": ["just"],
            "ability_levels": {"deductive_reasoning": "strong"},
            "ability_scores": {"deductive_reasoning": 0.75},
            "ability_evidence": {"deductive_reasoning": "rule use"},
        })

    vocab = [{"id": "deductive_reasoning", "definition": "Apply explicit rules."}]
    with pytest.raises(JSONContractError, match="banned_keys"):
        build_mapreduce_descriptions(
            corpus,
            vocab,
            {
                "mapreduce_model": {"name": "fake", "base_url": None},
                "mapreduce_chunk_examples": 1,
                "mapreduce_max_chunk_chars": 1000,
                "mapreduce_max_evidence_chars": 4000,
                "mapreduce_max_workers": 1,
                "llm_json_contract_max_attempts": 1,
            },
            str(tmp_path),
            chat_fn=chat,
        )


def test_mapper_rejects_surface_format_as_reasoning_pattern(tmp_path):
    """Answer format belongs in task_patterns, not reusable reasoning_patterns."""
    corpus = _toy_corpus(2)

    def chat(_system, _user):
        return json.dumps({
            "chunk_summary": "Items require reading and answer selection.",
            "task_patterns": ["multiple choice"],
            "reasoning_patterns": ["multiple choice answer format handling"],
            "justifications": ["surface format is present"],
        })

    vocab = [{"id": "deductive_reasoning", "definition": "Apply explicit rules."}]
    with pytest.raises(JSONContractError, match="invalid_mapper_evidence"):
        build_mapreduce_descriptions(
            corpus,
            vocab,
            {
                "mapreduce_model": {"name": "fake", "base_url": None},
                "mapreduce_chunk_examples": 1,
                "mapreduce_max_chunk_chars": 1000,
                "mapreduce_max_evidence_chars": 4000,
                "mapreduce_max_workers": 1,
                "llm_json_contract_max_attempts": 1,
            },
            str(tmp_path),
            chat_fn=chat,
        )
