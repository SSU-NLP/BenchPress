"""Prompt guardrails for sparse benchmark-score alignment."""

from __future__ import annotations

import json
from pathlib import Path

from autotagging_loop.experiment.corpus import Corpus
from autotagging_loop.experiment.executer import EXECUTER_SCHEMA_VERSION, run_executer
from autotagging_loop.experiment.maker import MAKER_SCHEMA_VERSION, run_maker
from autotagging_loop.experiment.mapreduce_evidence import _DEFAULT_MAPPER_SCHEMA, build_mapreduce_descriptions


def _aggregate(benchmark: str = "BenchA") -> dict:
    return {
        "benchmark": benchmark,
        "reviewed_rows": 2,
        "n_chunks": 1,
        "mapped_examples": 2,
        "chunk_evidence": [
            {
                "chunk_index": 0,
                "n_examples": 2,
                "summary": "Tasks require reading, retrieval, and rule use.",
                "task_patterns": ["short answer"],
                "reasoning_patterns": ["retrieve facts", "apply a rule"],
                "justifications": ["Evidence shows shared cognitive operations."],
                "_banned_drift": [],
            }
        ],
        "justifications": ["Evidence shows shared cognitive operations."],
    }


def test_seed_prompt_contains_sparse_generalization_guardrails() -> None:
    text = (
        Path(__file__).resolve().parent.parent
        / "prompts"
        / "I_exec_seed.txt"
    ).read_text(encoding="utf-8")

    assert "Sparse-signal generalization criteria" in text
    assert "avoid splitting dimensions" in text
    assert "mostly absent capabilities" in text
    assert "Over-splitting failure mode to avoid" in text


def test_prompt_changes_bump_llm_cache_schema_versions() -> None:
    assert _DEFAULT_MAPPER_SCHEMA >= 5
    assert EXECUTER_SCHEMA_VERSION >= 7
    assert MAKER_SCHEMA_VERSION >= 7


def test_mapper_prompt_separates_surface_and_reasoning_patterns(tmp_path) -> None:
    captured: dict[str, str] = {}

    def chat(system: str, user: str) -> str:
        captured["system"] = system
        captured["user"] = user
        return json.dumps({
            "chunk_summary": "Tasks require retrieval and rule use.",
            "task_patterns": ["short answer"],
            "reasoning_patterns": ["retrieval", "rule application"],
            "justifications": ["Evidence shows reusable operations."],
        })

    corpus = Corpus(
        benchmark_names=["BenchA"],
        model_names=["m1", "m2"],
        Y={"BenchA": {"m1": 0.5, "m2": 0.6}},
        descriptions={"BenchA": "raw"},
        documents={
            "BenchA": {
                "reviewed_rows": 2,
                "topic_counts": {"knowledge": 2},
                "reasoning_depth_counts": {"single_step": 2},
                "answer_format_counts": {"free": 2},
                "examples": ["Question: q1\nAnswer: a1", "Question: q2\nAnswer: a2"],
            }
        },
        drop_log={},
    )

    build_mapreduce_descriptions(
        corpus,
        [],
        {
            "mapreduce_model": {"name": "fake", "base_url": None},
            "mapreduce_chunk_examples": 2,
            "mapreduce_max_chunk_chars": 2000,
            "mapreduce_max_evidence_chars": 4000,
            "mapreduce_max_workers": 1,
        },
        str(tmp_path),
        chat_fn=chat,
        prompt="Look for transferable operations.",
    )

    assert "Separate surface task patterns from reusable reasoning patterns" in captured["system"]
    assert "Use only the supplied examples" in captured["system"]
    assert "Make each reasoning_patterns entry comparable across domains" in captured["user"]


def test_executer_prompt_includes_sparse_generalization_guardrails(tmp_path) -> None:
    captured: dict[str, str] = {}

    def chat(system: str, user: str) -> str:
        captured["system"] = system
        captured["user"] = user
        return json.dumps({
            "vocab": [
                {
                    "id": "factual_retrieval",
                    "name": "Factual Retrieval",
                    "definition": "Retrieving supplied or stored facts.",
                }
            ],
            "rationale": "shared evidence",
        })

    vocab, metadata = run_executer(
        source_benchmarks=["BenchA"],
        source_aggregates={"BenchA": _aggregate("BenchA")},
        prompt_i_exec="Use broad transferable operations.",
        config={"executer_model": {"name": "fake", "base_url": None}},
        run_dir=str(tmp_path),
        version=1,
        label="iter_001",
        chat_fn=chat,
    )

    assert [item["id"] for item in vocab] == ["factual_retrieval"]
    assert metadata["executer_schema_version"] == EXECUTER_SCHEMA_VERSION
    assert "Prefer a compact vocabulary of broad reusable operations" in captured["system"]
    assert "Do not let a simple, knowledge-heavy" in captured["system"]
    assert "Treat domain distance as weak evidence" in captured["system"]
    assert "Avoid pure difficulty or frontier-status axes" in captured["system"]
    assert "Maker must be able to decide absent/weak/medium/strong/dominant" in captured["system"]
    assert "Design V for held-out benchmark evidence" in captured["user"]
    assert "can Maker rate it from evidence" in captured["user"]


def test_maker_prompt_includes_calibration_guardrails(tmp_path) -> None:
    captured: dict[str, str] = {}

    def chat(system: str, user: str) -> str:
        captured["system"] = system
        captured["user"] = user
        return json.dumps({
            "benchmark_summary": "The evidence requires retrieval and rule use.",
            "ability_levels": {"factual_retrieval": "strong"},
            "ability_rationale": {"factual_retrieval": "facts are directly required"},
        })

    _, metadata = run_maker(
        benchmark_names=["BenchA"],
        vocab=[
            {
                "id": "factual_retrieval",
                "name": "Factual Retrieval",
                "definition": "Retrieving supplied or stored facts.",
            }
        ],
        aggregates={"BenchA": _aggregate("BenchA")},
        config={"maker_model": {"name": "fake", "base_url": None}},
        run_dir=str(tmp_path),
        prompt="Use broad transferable operations.",
        version=1,
        label="iter_001",
        chat_fn=chat,
    )

    assert metadata["maker_schema_version"] == MAKER_SCHEMA_VERSION
    assert "Tag broad transferable operations" in captured["system"]
    assert "Domain mismatch is not evidence of absence" in captured["system"]
    assert "Calibration rules: assign similar levels" in captured["user"]
    assert "Rate each ability against its own definition" in captured["user"]
    assert "do not zero out common operations" in captured["user"]
    assert "Use dominant sparingly" in captured["user"]
    assert "Rationale strings must cite the observed operation patterns" in captured["user"]
