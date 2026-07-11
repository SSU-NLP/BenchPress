"""Tests for experiment/mapreduce_evidence.py (v3 vocab-free Mapper)."""

from __future__ import annotations

import json
import threading
import time

from autotagging_loop.experiment.corpus import Corpus
from autotagging_loop.experiment.mapreduce_evidence import (
    _mapper_evidence_quality_reasons,
    build_mapreduce_descriptions,
)


VOCAB = [
    {"id": "deductive_reasoning", "definition": "Apply explicit rules."},
    {"id": "long_term_knowledge_recall", "definition": "Recall stored facts."},
]


def _toy_corpus(n_examples: int = 5) -> Corpus:
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


def test_mapreduce_descriptions_process_all_examples_in_chunks(tmp_path):
    corpus = _toy_corpus(5)
    calls = {"n": 0}

    def chat(_system, _user):
        calls["n"] += 1
        return json.dumps({
            "chunk_summary": f"summary {calls['n']}",
            "task_patterns": ["multiple choice"],
            "reasoning_patterns": ["rule use"],
            "justifications": ["benchmark requires explicit rule application"],
        })

    descriptions, aggregates = build_mapreduce_descriptions(
        corpus,
        VOCAB,
        {
            "mapreduce_model": {"name": "fake", "base_url": None},
            "mapreduce_chunk_examples": 2,
            "mapreduce_max_chunk_chars": 1000,
            "mapreduce_max_evidence_chars": 4000,
            "mapreduce_max_workers": 1,
        },
        str(tmp_path),
        chat_fn=chat,
    )

    assert calls["n"] == 3
    bench_agg = aggregates["ToyBench"]
    assert bench_agg["mapped_examples"] == 5
    assert bench_agg["n_chunks"] == 3
    assert "Full reviewed dataset was processed" in descriptions["ToyBench"]

    # New v3 schema: per-chunk chunk_evidence, flat justifications, no ability_* fields.
    assert "ability_score_means" not in bench_agg
    assert "ability_level_counts" not in bench_agg
    assert "top_ability_evidence" not in bench_agg
    assert "weight_formula" not in bench_agg
    assert "aggregate_chunk_ability_scores" not in descriptions["ToyBench"]

    chunk_evidence = bench_agg["chunk_evidence"]
    assert isinstance(chunk_evidence, list) and len(chunk_evidence) == 3
    expected_keys = {
        "chunk_index",
        "n_examples",
        "summary",
        "task_patterns",
        "reasoning_patterns",
        "justifications",
        "_banned_drift",
    }
    for item in chunk_evidence:
        assert set(item.keys()) == expected_keys
        assert item["_banned_drift"] == []
        # Mapper output must be vocab-free.
        for banned in ("ability_levels", "ability_scores", "ability_evidence"):
            assert banned not in item

    assert isinstance(bench_agg["justifications"], list)
    assert len(bench_agg["justifications"]) >= 1
    assert (tmp_path / "map_evidence" / "toybench" / "aggregate.json").is_file()


def test_mapreduce_reuses_persistent_cache_across_run_dirs(tmp_path):
    corpus = _toy_corpus(2)
    config = {
        "mapreduce_model": {"name": "fake", "base_url": None},
        "mapreduce_chunk_examples": 1,
        "mapreduce_max_chunk_chars": 1000,
        "mapreduce_max_evidence_chars": 4000,
        "mapreduce_max_workers": 1,
        "mapreduce_cache_enabled": True,
        "mapreduce_cache_dir": str(tmp_path / "persistent_cache"),
        "mapreduce_cache_schema_version": 2,
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

    _, aggregates_1 = build_mapreduce_descriptions(
        corpus,
        VOCAB,
        config,
        str(tmp_path / "run1"),
        chat_fn=chat,
        prompt="prompt v1",
    )

    assert calls["n"] == 2
    assert aggregates_1["ToyBench"]["cache_hits"] == 0

    def fail_chat(_system, _user):
        raise AssertionError("persistent cache should avoid a second API call")

    _, aggregates_2 = build_mapreduce_descriptions(
        corpus,
        VOCAB,
        config,
        str(tmp_path / "run2"),
        chat_fn=fail_chat,
        prompt="prompt v1",
    )

    assert aggregates_2["ToyBench"]["cache_hits"] == 2
    assert (tmp_path / "run2" / "map_evidence").is_dir()
    manifest_paths = list((tmp_path / "run2" / "map_evidence").glob("prompt_*/*/chunks_manifest.json"))
    assert manifest_paths


def _multi_corpus(bench_names: list[str], n_examples: int = 3) -> Corpus:
    return Corpus(
        benchmark_names=bench_names,
        model_names=["m1", "m2"],
        Y={b: {"m1": 0.5, "m2": 0.5} for b in bench_names},
        descriptions={b: "raw" for b in bench_names},
        documents={
            b: {
                "reviewed_rows": n_examples,
                "topic_counts": {"science": n_examples},
                "reasoning_depth_counts": {"single_step": n_examples},
                "answer_format_counts": {"multiple_choice": n_examples},
                "examples": [f"Q-{b}-{i}\nA-{b}-{i}" for i in range(n_examples)],
            }
            for b in bench_names
        },
    )


def test_mapreduce_outer_loop_parallelism_processes_all_benchmarks(tmp_path):
    """Phase G — every (benchmark, chunk) pair must be processed exactly once
    and aggregates dict must be ordered by Corpus benchmark_names."""
    bench_names = [f"B{i:02d}" for i in range(6)]
    corpus = _multi_corpus(bench_names, n_examples=2)
    call_lock = threading.Lock()
    seen: list[tuple[str, int]] = []

    def chat(_system, user):
        # Extract benchmark + chunk_index from user prompt for collision detection.
        # Prompt format: "Benchmark: <name>\nChunk: <1-indexed>/<total>\n..."
        benchmark = user.split("Benchmark: ", 1)[1].split("\n", 1)[0]
        chunk_one_indexed = user.split("Chunk: ", 1)[1].split("/", 1)[0]
        with call_lock:
            seen.append((benchmark, int(chunk_one_indexed) - 1))
        time.sleep(0.005)
        return json.dumps({
            "chunk_summary": f"sum-{benchmark}",
            "task_patterns": ["mc"],
            "reasoning_patterns": ["rule"],
            "justifications": ["needs explicit rule application"],
        })

    descriptions, aggregates = build_mapreduce_descriptions(
        corpus,
        VOCAB,
        {
            "mapreduce_model": {"name": "fake", "base_url": None},
            "mapreduce_chunk_examples": 1,  # 2 examples / 1 per chunk → 2 chunks per bench
            "mapreduce_max_chunk_chars": 1000,
            "mapreduce_max_evidence_chars": 4000,
            "mapreduce_max_workers": 4,
            "mapreduce_persistent_cache_dir": None,
        },
        str(tmp_path),
        chat_fn=chat,
    )

    # 6 benchmarks × 2 chunks = 12 distinct (bench, idx) pairs, each called once.
    assert sorted(seen) == [(b, i) for b in bench_names for i in range(2)]
    # All 6 benchmarks aggregated, in input order.
    assert list(aggregates.keys()) == bench_names
    assert list(descriptions.keys())[: len(bench_names)] == bench_names or set(
        bench_names
    ) <= set(descriptions)
    for b in bench_names:
        assert aggregates[b]["mapped_examples"] == 2
        assert aggregates[b]["n_chunks"] == 2


def test_mapper_quality_allows_task_ranking_language():
    parsed = {
        "chunk_summary": "Questions require ordering candidates from textual evidence.",
        "task_patterns": ["compare entities described in passages"],
        "reasoning_patterns": ["rank entities by evidence before selecting the answer"],
        "justifications": ["ranking candidates is an observed task operation"],
    }

    assert _mapper_evidence_quality_reasons(parsed, benchmark="Drop") == []


def test_mapper_quality_still_rejects_leaderboard_reputation():
    parsed = {
        "chunk_summary": "This benchmark is known from public reputation.",
        "task_patterns": [],
        "reasoning_patterns": ["leaderboard comparison"],
        "justifications": ["uses model performance expectations"],
    }

    reasons = _mapper_evidence_quality_reasons(parsed, benchmark="Drop")

    assert reasons
    assert any("leaderboard" in reason for reason in reasons)
