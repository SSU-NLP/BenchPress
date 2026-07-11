from __future__ import annotations

import json

from autotagging_loop.runner.mapreduce import build_tag_vectors


def test_build_tag_vectors_keeps_formula_weights_when_reducer_runs(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("map prompt", encoding="utf-8")
    vocab = [
        {"id": "reasoning", "name": "Reasoning", "definition": "Solve multi-step tasks."},
        {"id": "knowledge", "name": "Knowledge", "definition": "Recall facts."},
    ]
    documents = {
        "ToyBench": {
            "examples": ["Question: 1+1", "Question: 2+2"],
        }
    }
    config = {
        "prompt_path": str(prompt),
        "mapreduce_model": {"name": "mapper", "base_url": None},
        "mapreduce_reducer_model": {"name": "reducer", "base_url": None},
        "mapreduce_chunk_examples": 2,
        "mapreduce_max_chunk_chars": 1000,
    }
    calls = []

    def chat(system, user):
        calls.append((system, user))
        if "synthesizing chunk-level" in system:
            return json.dumps({
                "evidence_summary": "math only",
                "weight_cautions": {"reasoning": "", "knowledge": ""},
            })
        return json.dumps({
            "ability_levels": {"reasoning": "strong", "knowledge": "weak"},
            "ability_evidence": {"reasoning": "arithmetic", "knowledge": ""},
            "chunk_summary": "simple arithmetic",
        })

    T, metadata = build_tag_vectors(documents, vocab, config, chat_fn=chat)

    assert T["ToyBench"] == {"reasoning": 0.75, "knowledge": 0.25}
    assert metadata["benchmarks"]["ToyBench"]["reducer"]["source"] == "llm_reducer_evidence_synthesis"
    assert (
        metadata["benchmarks"]["ToyBench"]["reducer"]["final_weight_policy"]
        == "deterministic_chunk_level_weighted_average"
    )
    assert len(calls) == 2


def test_build_tag_vectors_accepts_part1_weight_prompt_output(tmp_path):
    prompt = tmp_path / "I_star.txt"
    prompt.write_text("part1 best prompt", encoding="utf-8")
    vocab = [
        {"id": "reasoning", "name": "Reasoning", "definition": "Solve multi-step tasks."},
        {"id": "knowledge", "name": "Knowledge", "definition": "Recall facts."},
    ]
    documents = {
        "ToyBench": {
            "examples": ["Question: 1+1", "Question: capital of France"],
        }
    }
    config = {
        "prompt_path": str(prompt),
        "mapreduce_model": {"name": "mapper", "base_url": None},
        "mapreduce_reducer_model": None,
        "mapreduce_chunk_examples": 2,
        "mapreduce_max_chunk_chars": 1000,
    }

    def chat(_system, _user):
        return json.dumps({
            "weights": {"reasoning": 0.8, "knowledge": 0.2},
            "rationale": "mixed arithmetic and factual recall",
        })

    T, metadata = build_tag_vectors(documents, vocab, config, chat_fn=chat)

    assert T["ToyBench"] == {"reasoning": 0.75, "knowledge": 0.25}
    assert metadata["prompt_path"] == str(prompt)
