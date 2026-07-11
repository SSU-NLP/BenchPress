"""Tests for deterministic static tag-weight reduction."""

from __future__ import annotations

from autotagging_loop.experiment.static_tag_weights import build_static_tag_vectors_from_reducer_levels


VOCAB = [
    {"id": "deductive_reasoning"},
    {"id": "long_term_knowledge_recall"},
]


def test_reducer_level_weights_are_function_mapped_not_model_numeric_weights():
    T, meta = build_static_tag_vectors_from_reducer_levels(
        ["BenchA"],
        VOCAB,
        {
            "BenchA": {
                "ability_levels": {
                    "deductive_reasoning": "dominant",
                    "long_term_knowledge_recall": "weak",
                },
                "weights": {
                    "deductive_reasoning": 0.01,
                    "long_term_knowledge_recall": 0.99,
                },
                "mapped_examples": 10,
                "n_chunks": 3,
            }
        },
        {"weight_bounds": [0.0, 1.0], "static_tag_require_mapreduce": True},
    )

    assert T["BenchA"] == {
        "deductive_reasoning": 1.0,
        "long_term_knowledge_recall": 0.25,
    }
    assert meta["method"] == "mapreduce_llm_reducer_static_weights"
    assert meta["benchmarks"]["BenchA"]["source"] == "mapreduce_llm_reducer_levels"
