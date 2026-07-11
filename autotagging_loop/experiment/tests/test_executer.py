"""Tests for experiment/executer.py (v3 §2.2.4 Executer: produces V from Z_src + I_exec).

Multi-source contract: `source_benchmarks: list[str]` + `source_aggregates:
dict[str, dict]`. Cache keys are scoped by the order-insensitive source-set
signature so swapping bench order or membership picks the right cache slot.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotagging_loop.experiment.executer import (
    _executer_cache_key,
    _executer_cache_path,
    _format_chunk_evidence_multi,
    _source_set_sig,
    run_executer,
)
from autotagging_loop.experiment.json_contract import JSONContractError


def _aggregate(benchmark: str = "BenchA", n_chunks: int = 2, tag: str = "x") -> dict:
    return {
        "benchmark": benchmark,
        "reviewed_rows": n_chunks,
        "n_chunks": n_chunks,
        "mapped_examples": n_chunks,
        "chunk_evidence": [
            {
                "chunk_index": i,
                "n_examples": 1,
                "summary": f"chunk {i} of {benchmark} ({tag})",
                "task_patterns": ["mc"],
                "reasoning_patterns": ["rule"],
                "justifications": ["explicit rule use"],
                "_banned_drift": [],
            }
            for i in range(n_chunks)
        ],
        "justifications": [f"{benchmark} forces explicit rule use"],
        "text": f"Compact evidence summary for {benchmark} ({tag})",
    }


def _vocab_response(ids: list[str]) -> str:
    return json.dumps({
        "vocab": [
            {
                "id": tid,
                "name": tid.replace("_", " ").title(),
                "definition": f"definition for {tid}",
            }
            for tid in ids
        ],
        "rationale": "demo",
    })


def _config() -> dict:
    return {"experiment": {"executer_model": {"name": "fake-executer", "base_url": None}}}


def test_executer_returns_v_with_metadata(tmp_path):
    calls = {"n": 0}

    def chat(_system, _user):
        calls["n"] += 1
        return _vocab_response(["deductive_reasoning", "long_term_knowledge_recall"])

    vocab, metadata = run_executer(
        source_benchmarks=["BenchA"],
        source_aggregates={"BenchA": _aggregate("BenchA")},
        prompt_i_exec="I_exec v1",
        config=_config(),
        run_dir=str(tmp_path),
        version=1,
        label="iter_001",
        chat_fn=chat,
    )

    assert calls["n"] == 1
    assert [v["id"] for v in vocab] == ["deductive_reasoning", "long_term_knowledge_recall"]
    assert all("definition" in v and v["definition"] for v in vocab)
    assert metadata["source_benchmarks"] == ["BenchA"]
    assert metadata["vocab_size"] == 2
    assert metadata["cache_hit"] is False
    assert metadata["reasons"] == []


def test_executer_seed_partitions_cache(tmp_path):
    seen_seeds: list[int | None] = []

    def chat(_system, _user, seed=None):
        seen_seeds.append(seed)
        return _vocab_response(["deductive_reasoning"])

    for seed in (101, 202):
        run_executer(
            source_benchmarks=["BenchA"],
            source_aggregates={"BenchA": _aggregate("BenchA")},
            prompt_i_exec="I_exec v1",
            config=_config(),
            run_dir=str(tmp_path),
            version=1,
            label="iter_001",
            chat_fn=chat,
            seed=seed,
        )

    assert seen_seeds == [101, 202]


def test_executer_multi_source_evidence_concat(tmp_path):
    """Each source bench gets a `## Benchmark: <name>` section in the user prompt."""
    seen = {"user": ""}

    def chat(_system, user):
        seen["user"] = user
        return _vocab_response(["deductive_reasoning"])

    run_executer(
        source_benchmarks=["BenchA", "BenchB"],
        source_aggregates={
            "BenchA": _aggregate("BenchA"),
            "BenchB": _aggregate("BenchB"),
        },
        prompt_i_exec="prompt",
        config=_config(),
        run_dir=str(tmp_path),
        version=1,
        label="iter_001",
        chat_fn=chat,
    )
    assert "## Benchmark: BenchA" in seen["user"]
    assert "## Benchmark: BenchB" in seen["user"]


def test_executer_uses_compact_aggregate_text_not_raw_chunks(tmp_path):
    aggregate = _aggregate("BenchA", n_chunks=200)
    aggregate["text"] = "compact aggregate evidence"
    aggregate["chunk_evidence"][-1]["summary"] = "raw chunk that should stay out"
    seen = {"user": ""}

    def chat(_system, user):
        seen["user"] = user
        return _vocab_response(["deductive_reasoning"])

    run_executer(
        source_benchmarks=["BenchA"],
        source_aggregates={"BenchA": aggregate},
        prompt_i_exec="prompt",
        config=_config(),
        run_dir=str(tmp_path),
        version=1,
        label="iter_001",
        chat_fn=chat,
    )

    assert "compact aggregate evidence" in seen["user"]
    assert "raw chunk that should stay out" not in seen["user"]


def test_executer_cache_hit_skips_llm(tmp_path):
    calls = {"n": 0}

    def chat(_system, _user):
        calls["n"] += 1
        return _vocab_response(["deductive_reasoning"])

    args = dict(
        source_benchmarks=["BenchA"],
        source_aggregates={"BenchA": _aggregate("BenchA")},
        prompt_i_exec="prompt v1",
        config=_config(),
        run_dir=str(tmp_path),
        version=1,
        label="iter_001",
        chat_fn=chat,
    )

    v1, _ = run_executer(**args)

    def must_not_call(_system, _user):  # pragma: no cover - regression assertion
        raise AssertionError("Executer should hit cache instead of calling LLM")

    args["chat_fn"] = must_not_call
    args["label"] = "iter_002"
    v2, meta2 = run_executer(**args)

    assert calls["n"] == 1
    assert v1 == v2
    assert meta2["cache_hit"] is True


def test_executer_target_count_is_prompted_and_validated(tmp_path):
    seen = {"system": "", "user": ""}

    def chat(system, user):
        seen["system"] = system
        seen["user"] = user
        return _vocab_response(["reasoning_a", "reasoning_b", "reasoning_c"])

    vocab, metadata = run_executer(
        source_benchmarks=["BenchA"],
        source_aggregates={"BenchA": _aggregate("BenchA")},
        prompt_i_exec="prompt",
        config=_config(),
        run_dir=str(tmp_path),
        version=1,
        label="iter_001",
        chat_fn=chat,
        target_count=3,
    )

    assert len(vocab) == 3
    assert metadata["target_count"] == 3
    assert "exactly 3 vocab entries" in seen["system"]
    assert "Target vocab count for this candidate: exactly 3" in seen["user"]


def test_executer_rejects_target_count_mismatch(tmp_path):
    def chat(_system, _user):
        return _vocab_response(["reasoning_a", "reasoning_b"])

    with pytest.raises(JSONContractError, match="target_count_mismatch:2!=3"):
        run_executer(
            source_benchmarks=["BenchA"],
            source_aggregates={"BenchA": _aggregate("BenchA")},
            prompt_i_exec="prompt",
            config={**_config(), "llm_json_contract_max_attempts": 1},
            run_dir=str(tmp_path),
            version=1,
            label="iter_001",
            chat_fn=chat,
            target_count=3,
        )


def test_executer_target_count_misses_cache(tmp_path):
    calls = {"n": 0}

    def chat(_system, user):
        calls["n"] += 1
        if "exactly 2" in user:
            return _vocab_response(["reasoning_a", "reasoning_b"])
        return _vocab_response(["reasoning_a", "reasoning_b", "reasoning_c"])

    base = dict(
        source_benchmarks=["BenchA"],
        source_aggregates={"BenchA": _aggregate("BenchA")},
        prompt_i_exec="prompt",
        config=_config(),
        run_dir=str(tmp_path),
        version=1,
        label="iter_001",
        chat_fn=chat,
    )
    run_executer(target_count=2, **base)
    run_executer(target_count=3, **base)
    assert calls["n"] == 2


def test_executer_cache_hit_is_source_order_insensitive(tmp_path):
    """source_set_sig is order-insensitive — swapping bench order hits the cache."""
    calls = {"n": 0}

    def chat(_system, _user):
        calls["n"] += 1
        return _vocab_response(["deductive_reasoning"])

    common = dict(
        source_aggregates={
            "BenchA": _aggregate("BenchA"),
            "BenchB": _aggregate("BenchB"),
        },
        prompt_i_exec="p",
        config=_config(),
        run_dir=str(tmp_path),
        version=1,
        label="iter_001",
        chat_fn=chat,
    )
    run_executer(source_benchmarks=["BenchA", "BenchB"], **common)
    run_executer(source_benchmarks=["BenchB", "BenchA"], **common)
    assert calls["n"] == 1  # second call is a cache hit


def test_executer_different_prompt_misses_cache(tmp_path):
    seen_prompts: list[str] = []

    def chat(_system, user):
        seen_prompts.append(user)
        return _vocab_response(["deductive_reasoning"])

    base = dict(
        source_benchmarks=["BenchA"],
        source_aggregates={"BenchA": _aggregate("BenchA")},
        config=_config(),
        run_dir=str(tmp_path),
        version=1,
        label="iter_001",
        chat_fn=chat,
    )
    run_executer(prompt_i_exec="prompt A", **base)
    run_executer(prompt_i_exec="prompt B", **base)
    assert len(seen_prompts) == 2


def test_executer_different_source_set_misses_cache(tmp_path):
    """Different source sets → different cache keys (no stale V served)."""
    seen: list[str] = []

    def chat(_system, _user):
        seen.append("called")
        return _vocab_response(["deductive_reasoning"])

    common = dict(
        prompt_i_exec="prompt v1",
        config=_config(),
        run_dir=str(tmp_path),
        version=1,
        label="iter_001",
        chat_fn=chat,
    )
    run_executer(
        source_benchmarks=["BenchA"],
        source_aggregates={"BenchA": _aggregate("BenchA")},
        **common,
    )
    run_executer(
        source_benchmarks=["BenchB"],
        source_aggregates={"BenchB": _aggregate("BenchB")},
        **common,
    )
    assert len(seen) == 2

    sig_a = _source_set_sig(["BenchA"])
    sig_b = _source_set_sig(["BenchB"])
    k1 = _executer_cache_key(
        prompt_i_exec="x", z_src_sig="abc", source_set_sig=sig_a,
        executer_model="fake", schema_version=2,
    )
    k2 = _executer_cache_key(
        prompt_i_exec="x", z_src_sig="abc", source_set_sig=sig_b,
        executer_model="fake", schema_version=2,
    )
    assert k1 != k2


def test_executer_invalid_json_fails_contract(tmp_path):
    def chat(_system, _user):
        return "not json"

    with pytest.raises(JSONContractError, match="invalid_json"):
        run_executer(
            source_benchmarks=["BenchA"],
            source_aggregates={"BenchA": _aggregate("BenchA")},
            prompt_i_exec="prompt",
            config={
                "executer_model": {"name": "fake", "base_url": None},
                "llm_json_contract_max_attempts": 1,
            },
            run_dir=str(tmp_path),
            version=1,
            label="iter_001",
            chat_fn=chat,
        )


def test_executer_rejects_benchmark_and_difficulty_vocab_axes(tmp_path):
    bad_response = json.dumps({
        "vocab": [
            {
                "id": "mmlu_pro_difficulty",
                "name": "MMLU Pro Difficulty",
                "definition": "Groups benchmarks by leaderboard difficulty.",
            }
        ],
        "rationale": "bad shortcut",
    })

    def chat(_system, _user):
        return bad_response

    with pytest.raises(JSONContractError, match="invalid_vocab_quality"):
        run_executer(
            source_benchmarks=["MMLU-Pro"],
            source_aggregates={"MMLU-Pro": _aggregate("MMLU-Pro")},
            prompt_i_exec="prompt",
            config={
                "executer_model": {"name": "fake", "base_url": None},
                "llm_json_contract_max_attempts": 1,
            },
            run_dir=str(tmp_path),
            version=1,
            label="iter_001",
            chat_fn=chat,
        )


def test_executer_writes_cache_file_under_source_set_dir(tmp_path):
    def chat(_system, _user):
        return _vocab_response(["deductive_reasoning"])

    run_executer(
        source_benchmarks=["BenchA"],
        source_aggregates={"BenchA": _aggregate("BenchA")},
        prompt_i_exec="prompt",
        config=_config(),
        run_dir=str(tmp_path),
        version=1,
        label="iter_001",
        chat_fn=chat,
    )
    cache_root = Path(tmp_path) / "executer_cache"
    cached = list(cache_root.rglob("*.json"))
    assert len(cached) == 1
    rel = cached[0].relative_to(cache_root)
    assert rel.parts[0].startswith("sources_")
    payload = json.loads(cached[0].read_text())
    assert payload["source_benchmarks"] == ["BenchA"]
    assert payload["vocab"][0]["id"] == "deductive_reasoning"


def test_executer_requires_source_benchmarks_and_aggregates(tmp_path):
    with pytest.raises(ValueError):
        run_executer(
            source_benchmarks=[],
            source_aggregates={"BenchA": _aggregate("BenchA")},
            prompt_i_exec="p",
            config=_config(),
            run_dir=str(tmp_path),
            version=1,
            label="iter_001",
            chat_fn=lambda *_a: "{}",
        )

    with pytest.raises(ValueError):
        run_executer(
            source_benchmarks=["BenchA"],
            source_aggregates={},
            prompt_i_exec="p",
            config=_config(),
            run_dir=str(tmp_path),
            version=1,
            label="iter_001",
            chat_fn=lambda *_a: "{}",
        )


def test_executer_skips_missing_aggregates_with_warning(tmp_path, capsys):
    """A bench in source_benchmarks but missing from source_aggregates is skipped."""
    def chat(_system, _user):
        return _vocab_response(["deductive_reasoning"])

    vocab, metadata = run_executer(
        source_benchmarks=["BenchA", "BenchMissing"],
        source_aggregates={"BenchA": _aggregate("BenchA")},
        prompt_i_exec="p",
        config=_config(),
        run_dir=str(tmp_path),
        version=1,
        label="iter_001",
        chat_fn=chat,
    )
    captured = capsys.readouterr().out
    assert "BenchMissing" in captured
    assert "missing aggregate" in captured
    assert metadata["source_benchmarks"] == ["BenchA"]
    assert vocab[0]["id"] == "deductive_reasoning"


def test_executer_cache_path_uses_source_set_hash(tmp_path):
    sig = _source_set_sig(["BenchA", "BenchB"])
    path = _executer_cache_path(str(tmp_path), sig, "abc")
    assert path.endswith(f"executer_cache/sources_{sig}/abc.json")


def test_format_chunk_evidence_multi_orders_sections_by_input(tmp_path):
    sections = _format_chunk_evidence_multi([
        {"benchmark": "Z_Bench", "chunk_evidence": [{"chunk_index": 0, "summary": "z"}]},
        {"benchmark": "A_Bench", "chunk_evidence": [{"chunk_index": 0, "summary": "a"}]},
    ])
    # Sections appear in the input order (not alphabetized), so the loop's
    # canonical-sorted bench list is what controls prompt ordering.
    assert sections.index("## Benchmark: Z_Bench") < sections.index("## Benchmark: A_Bench")
