"""Tests for experiment/maker.py (v3 §2.2.5 Maker)."""

from __future__ import annotations

import json

import pytest

from autotagging_loop.experiment.json_contract import JSONContractError
from autotagging_loop.experiment.maker import run_maker


VOCAB = [
    {"id": "deductive_reasoning", "definition": "Apply explicit rules."},
    {"id": "long_term_knowledge_recall", "definition": "Recall stored facts."},
]


def _aggregate(benchmark: str, n_chunks: int = 2) -> dict:
    return {
        "benchmark": benchmark,
        "reviewed_rows": n_chunks,
        "n_chunks": n_chunks,
        "mapped_examples": n_chunks,
        "chunk_evidence": [
            {
                "chunk_index": i,
                "n_examples": 1,
                "summary": f"chunk {i} summary for {benchmark}",
                "task_patterns": ["multiple choice"],
                "reasoning_patterns": ["rule application"],
                "justifications": [f"chunk {i} requires explicit rule use"],
                "_banned_drift": [],
            }
            for i in range(n_chunks)
        ],
        "justifications": [f"{benchmark} requires explicit rule application"],
    }


def test_run_maker_assigns_levels_per_benchmark(tmp_path):
    aggregates = {"BenchA": _aggregate("BenchA"), "BenchB": _aggregate("BenchB")}
    calls = {"n": 0}

    def chat(_system, _user):
        calls["n"] += 1
        return json.dumps({
            "benchmark_summary": "synthesis",
            "ability_levels": {
                "deductive_reasoning": "strong",
                "long_term_knowledge_recall": "weak",
            },
            "ability_rationale": {
                "deductive_reasoning": "rule use",
                "long_term_knowledge_recall": "minor recall",
            },
        })

    config = {"maker_model": {"name": "fake", "base_url": None}}
    outputs, metadata = run_maker(
        benchmark_names=["BenchA", "BenchB"],
        vocab=VOCAB,
        aggregates=aggregates,
        config=config,
        run_dir=str(tmp_path),
        prompt="prompt v1",
        version=1,
        label="iter_001",
        chat_fn=chat,
    )

    assert calls["n"] == 2
    for bench in ("BenchA", "BenchB"):
        assert outputs[bench]["ability_levels"]["deductive_reasoning"] == "strong"
        assert outputs[bench]["ability_levels"]["long_term_knowledge_recall"] == "weak"
        assert outputs[bench]["ability_rationale"]["deductive_reasoning"] == "rule use"

    # Metadata exposes both new and legacy keys (legacy preserved for downstream consumers).
    assert metadata["maker_output_count"] == 2
    assert metadata["reducer_output_count"] == 2
    assert metadata["maker_cache_hits"] == 0


def test_run_maker_normalizes_numeric_levels_and_missing_summary(tmp_path):
    def chat(_system, _user):
        return json.dumps({
            "ability_levels": {
                "deductive_reasoning": 4,
                "long_term_knowledge_recall": 1,
            },
            "ability_rationale": {
                "deductive_reasoning": "explicit rule use",
                "long_term_knowledge_recall": "limited stored-fact cue",
            },
        })

    outputs, _metadata = run_maker(
        benchmark_names=["BenchA"],
        vocab=VOCAB,
        aggregates={"BenchA": _aggregate("BenchA")},
        config={"maker_model": {"name": "fake", "base_url": None}},
        run_dir=str(tmp_path),
        prompt="prompt v1",
        version=1,
        label="iter_001",
        chat_fn=chat,
    )

    assert outputs["BenchA"]["ability_levels"]["deductive_reasoning"] == "strong"
    assert outputs["BenchA"]["ability_levels"]["long_term_knowledge_recall"] == "absent"
    assert outputs["BenchA"]["benchmark_summary"]


def test_run_maker_caches_per_prompt_vocab_aggregate(tmp_path):
    aggregates = {"BenchA": _aggregate("BenchA")}
    calls = {"n": 0}

    def chat(_system, _user):
        calls["n"] += 1
        return json.dumps({
            "benchmark_summary": "synthesis",
            "ability_levels": {
                "deductive_reasoning": "medium",
                "long_term_knowledge_recall": "absent",
            },
            "ability_rationale": {
                "deductive_reasoning": "rule use",
                "long_term_knowledge_recall": "no recall cue",
            },
        })

    config = {"maker_model": {"name": "fake", "base_url": None}}
    outputs_1, _ = run_maker(
        benchmark_names=["BenchA"],
        vocab=VOCAB,
        aggregates=aggregates,
        config=config,
        run_dir=str(tmp_path),
        prompt="prompt v1",
        version=1,
        label="iter_001",
        chat_fn=chat,
    )
    assert calls["n"] == 1

    def fail_chat(_system, _user):
        raise AssertionError("cache hit expected")

    outputs_2, metadata_2 = run_maker(
        benchmark_names=["BenchA"],
        vocab=VOCAB,
        aggregates=aggregates,
        config=config,
        run_dir=str(tmp_path),
        prompt="prompt v1",
        version=1,
        label="iter_002",
        chat_fn=fail_chat,
    )
    assert outputs_2["BenchA"]["ability_levels"] == outputs_1["BenchA"]["ability_levels"]
    assert metadata_2["maker_cache_hits"] == 1


def test_run_maker_missing_rationale_fails_contract(tmp_path):
    def chat(_system, _user):
        return json.dumps({
            "benchmark_summary": "synthesis",
            "ability_levels": {
                "deductive_reasoning": "medium",
                "long_term_knowledge_recall": "absent",
            },
        })

    with pytest.raises(JSONContractError, match="missing_keys:ability_rationale"):
        run_maker(
            benchmark_names=["BenchA"],
            vocab=VOCAB,
            aggregates={"BenchA": _aggregate("BenchA")},
            config={
                "maker_model": {"name": "fake", "base_url": None},
                "llm_json_contract_max_attempts": 1,
            },
            run_dir=str(tmp_path),
            prompt="prompt v1",
            version=1,
            label="iter_001",
            chat_fn=chat,
        )


def test_run_maker_prompt_includes_complete_ability_skeleton(tmp_path):
    seen_user = {"text": ""}

    def chat(_system, user):
        seen_user["text"] = user
        return json.dumps({
            "benchmark_summary": "synthesis",
            "ability_levels": {
                "deductive_reasoning": "medium",
                "long_term_knowledge_recall": "absent",
            },
            "ability_rationale": {
                "deductive_reasoning": "rule use",
                "long_term_knowledge_recall": "no recall cue",
            },
        })

    run_maker(
        benchmark_names=["BenchA"],
        vocab=VOCAB,
        aggregates={"BenchA": _aggregate("BenchA")},
        config={"maker_model": {"name": "fake", "base_url": None}},
        run_dir=str(tmp_path),
        prompt="prompt v1",
        version=1,
        label="iter_001",
        chat_fn=chat,
    )

    assert '"deductive_reasoning": "absent|weak|medium|strong|dominant"' in seen_user["text"]
    assert '"long_term_knowledge_recall": "absent|weak|medium|strong|dominant"' in seen_user["text"]
    assert '"deductive_reasoning": "short justification; use no evidence if absent"' in seen_user["text"]
    assert '"long_term_knowledge_recall": "short justification; use no evidence if absent"' in seen_user["text"]
    assert "Forbidden rationale wording" in seen_user["text"]
    assert "grading/measurement" in seen_user["text"]
    assert "plausible-alternative filtering" in seen_user["text"]


def test_legacy_shim_re_exports_run_maker():
    from autotagging_loop.experiment.mapreduce_reducer import build_mapreduce_reducer_outputs
    from autotagging_loop.experiment.maker import run_maker as direct

    assert build_mapreduce_reducer_outputs is direct


def test_run_maker_parallel_outputs_all_benchmarks(tmp_path):
    """Phase B — parallel fan-out must produce one output per input benchmark, sorted."""
    import threading
    import time

    bench_names = [f"B{i:02d}" for i in range(8)]
    aggregates = {b: _aggregate(b) for b in bench_names}
    call_lock = threading.Lock()
    call_log: list[str] = []

    def chat(_system, user):
        # Extract benchmark id from the user msg (first "Benchmark: <id>" line).
        bench = user.split("Benchmark: ", 1)[1].split("\n", 1)[0]
        marker = f"rule-marker-{int(bench[1:]) + 100}"
        with call_lock:
            call_log.append(bench)
        time.sleep(0.01)  # encourage real overlap
        return json.dumps({
            "benchmark_summary": f"sum-{bench}",
            "ability_levels": {
                "deductive_reasoning": "strong",
                "long_term_knowledge_recall": "weak",
            },
            "ability_rationale": {
                "deductive_reasoning": marker,
                "long_term_knowledge_recall": "minor",
            },
        })

    config = {"maker_model": {"name": "fake", "base_url": None}, "maker_max_workers": 4}
    outputs, metadata = run_maker(
        benchmark_names=bench_names,
        vocab=VOCAB,
        aggregates=aggregates,
        config=config,
        run_dir=str(tmp_path),
        prompt="parallel prompt",
        version=1,
        label="iter_par",
        chat_fn=chat,
    )

    # All benchmarks tagged exactly once.
    assert sorted(call_log) == bench_names
    assert set(outputs) == set(bench_names)
    # Output dict is sorted regardless of completion order.
    assert list(outputs.keys()) == sorted(bench_names)
    # Per-benchmark rationale is correctly attributed (no race-y crossover).
    for bench in bench_names:
        marker = f"rule-marker-{int(bench[1:]) + 100}"
        assert outputs[bench]["ability_rationale"]["deductive_reasoning"] == marker
    assert metadata["maker_output_count"] == 8
    assert metadata["maker_cache_hits"] == 0


def test_run_maker_rejects_reputation_based_rationale(tmp_path):
    def chat(_system, _user):
        return json.dumps({
            "benchmark_summary": "synthesis",
            "ability_levels": {
                "deductive_reasoning": "strong",
                "long_term_knowledge_recall": "weak",
            },
            "ability_rationale": {
                "deductive_reasoning": (
                    "MMLU-Pro is a hard benchmark with leaderboard difficulty."
                ),
                "long_term_knowledge_recall": "minor recall from evidence",
            },
        })

    with pytest.raises(JSONContractError, match="invalid_maker_evidence"):
        run_maker(
            benchmark_names=["MMLU-Pro"],
            vocab=VOCAB,
            aggregates={"MMLU-Pro": _aggregate("MMLU-Pro")},
            config={
                "maker_model": {"name": "fake", "base_url": None},
                "llm_json_contract_max_attempts": 1,
            },
            run_dir=str(tmp_path),
            prompt="prompt v1",
            version=1,
            label="iter_001",
            chat_fn=chat,
        )


def test_run_maker_allows_operation_accuracy_wording(tmp_path):
    def chat(_system, _user):
        return json.dumps({
            "benchmark_summary": "synthesis",
            "ability_levels": {
                "deductive_reasoning": "strong",
                "long_term_knowledge_recall": "weak",
            },
            "ability_rationale": {
                "deductive_reasoning": (
                    "rule-chain verification improves answer accuracy on hard cases"
                ),
                "long_term_knowledge_recall": "minor recall from evidence",
            },
        })

    outputs, _metadata = run_maker(
        benchmark_names=["ARC Challenge"],
        vocab=VOCAB,
        aggregates={"ARC Challenge": _aggregate("ARC Challenge")},
        config={
            "maker_model": {"name": "fake", "base_url": None},
            "llm_json_contract_max_attempts": 1,
        },
        run_dir=str(tmp_path),
        prompt="prompt v1",
        version=1,
        label="iter_001",
        chat_fn=chat,
    )

    assert outputs["ARC Challenge"]["ability_levels"]["deductive_reasoning"] == "strong"


def test_run_maker_allows_task_content_scores(tmp_path):
    def chat(_system, _user):
        return json.dumps({
            "benchmark_summary": "numeric extraction over game narratives",
            "ability_levels": {
                "deductive_reasoning": "medium",
                "long_term_knowledge_recall": "absent",
            },
            "ability_rationale": {
                "deductive_reasoning": (
                    "Requires extracting team scores and point totals from game "
                    "narratives and computing margins between them."
                ),
                "long_term_knowledge_recall": "no recall cue",
            },
        })

    outputs, _metadata = run_maker(
        benchmark_names=["BenchA"],
        vocab=VOCAB,
        aggregates={"BenchA": _aggregate("BenchA")},
        config={
            "maker_model": {"name": "fake", "base_url": None},
            "llm_json_contract_max_attempts": 1,
        },
        run_dir=str(tmp_path),
        prompt="prompt v1",
        version=1,
        label="iter_001",
        chat_fn=chat,
    )

    assert outputs["BenchA"]["ability_levels"]["deductive_reasoning"] == "medium"


def test_run_maker_rejects_eval_context_scores(tmp_path):
    def chat(_system, _user):
        return json.dumps({
            "benchmark_summary": "eval leak",
            "ability_levels": {
                "deductive_reasoning": "medium",
                "long_term_knowledge_recall": "absent",
            },
            "ability_rationale": {
                "deductive_reasoning": (
                    "Strong model scores on these items justify a high level."
                ),
                "long_term_knowledge_recall": "no recall cue",
            },
        })

    with pytest.raises(JSONContractError, match="eval_score"):
        run_maker(
            benchmark_names=["BenchA"],
            vocab=VOCAB,
            aggregates={"BenchA": _aggregate("BenchA")},
            config={
                "maker_model": {"name": "fake", "base_url": None},
                "llm_json_contract_max_attempts": 1,
            },
            run_dir=str(tmp_path),
            prompt="prompt v1",
            version=1,
            label="iter_001",
            chat_fn=chat,
        )


def test_run_maker_rejects_level_rationale_contradiction(tmp_path):
    def chat(_system, _user):
        return json.dumps({
            "benchmark_summary": "synthesis",
            "ability_levels": {
                "deductive_reasoning": "strong",
                "long_term_knowledge_recall": "absent",
            },
            "ability_rationale": {
                "deductive_reasoning": "no direct evidence for rule use",
                "long_term_knowledge_recall": "no recall cue",
            },
        })

    with pytest.raises(JSONContractError, match="ability_level_rationale_contradiction"):
        run_maker(
            benchmark_names=["BenchA"],
            vocab=VOCAB,
            aggregates={"BenchA": _aggregate("BenchA")},
            config={
                "maker_model": {"name": "fake", "base_url": None},
                "llm_json_contract_max_attempts": 1,
            },
            run_dir=str(tmp_path),
            prompt="prompt v1",
            version=1,
            label="iter_001",
            chat_fn=chat,
        )


def test_run_maker_allows_negated_absent_and_surface_format_wording(tmp_path):
    def chat(_system, _user):
        return json.dumps({
            "benchmark_summary": "semantic comparison",
            "ability_levels": {
                "deductive_reasoning": "medium",
                "long_term_knowledge_recall": "absent",
            },
            "ability_rationale": {
                "deductive_reasoning": (
                    "The operation is not absent and goes beyond multiple choice "
                    "format because options require comparing subtle premises."
                ),
                "long_term_knowledge_recall": "no recall cue",
            },
        })

    outputs, _metadata = run_maker(
        benchmark_names=["BenchA"],
        vocab=VOCAB,
        aggregates={"BenchA": _aggregate("BenchA")},
        config={
            "maker_model": {"name": "fake", "base_url": None},
            "llm_json_contract_max_attempts": 1,
        },
        run_dir=str(tmp_path),
        prompt="prompt v1",
        version=1,
        label="iter_001",
        chat_fn=chat,
    )

    assert outputs["BenchA"]["ability_levels"]["deductive_reasoning"] == "medium"


def test_run_maker_allows_task_format_when_tied_to_operations(tmp_path):
    def chat(_system, _user):
        return json.dumps({
            "benchmark_summary": "schema mapping",
            "ability_levels": {
                "deductive_reasoning": "medium",
                "long_term_knowledge_recall": "absent",
            },
            "ability_rationale": {
                "deductive_reasoning": (
                    "task format requires schema mapping and value extraction before rule comparison"
                ),
                "long_term_knowledge_recall": "no recall cue",
            },
        })

    outputs, _metadata = run_maker(
        benchmark_names=["BenchA"],
        vocab=VOCAB,
        aggregates={"BenchA": _aggregate("BenchA")},
        config={
            "maker_model": {"name": "fake", "base_url": None},
            "llm_json_contract_max_attempts": 1,
        },
        run_dir=str(tmp_path),
        prompt="prompt v1",
        version=1,
        label="iter_001",
        chat_fn=chat,
    )

    assert outputs["BenchA"]["ability_levels"]["deductive_reasoning"] == "medium"


def test_run_maker_rejects_unqualified_surface_format_rationale(tmp_path):
    def chat(_system, _user):
        return json.dumps({
            "benchmark_summary": "format cue",
            "ability_levels": {
                "deductive_reasoning": "medium",
                "long_term_knowledge_recall": "absent",
            },
            "ability_rationale": {
                "deductive_reasoning": "multiple choice format drives the rating",
                "long_term_knowledge_recall": "no recall cue",
            },
        })

    with pytest.raises(JSONContractError, match="ability_rationale_leaks_non_evidence"):
        run_maker(
            benchmark_names=["BenchA"],
            vocab=VOCAB,
            aggregates={"BenchA": _aggregate("BenchA")},
            config={
                "maker_model": {"name": "fake", "base_url": None},
                "llm_json_contract_max_attempts": 1,
            },
            run_dir=str(tmp_path),
            prompt="prompt v1",
            version=1,
            label="iter_001",
            chat_fn=chat,
        )


def test_run_maker_retry_hint_repairs_non_evidence_rationale(tmp_path):
    seen_users: list[str] = []

    def chat(_system, user):
        seen_users.append(user)
        if len(seen_users) == 1:
            return json.dumps({
                "benchmark_summary": "synthesis",
                "ability_levels": {
                    "deductive_reasoning": "medium",
                    "long_term_knowledge_recall": "absent",
            },
            "ability_rationale": {
                "deductive_reasoning": "model score pattern drives the rating",
                "long_term_knowledge_recall": "no recall cue",
            },
        })
        return json.dumps({
            "benchmark_summary": "synthesis",
            "ability_levels": {
                "deductive_reasoning": "medium",
                "long_term_knowledge_recall": "absent",
            },
            "ability_rationale": {
                "deductive_reasoning": "rule comparison across chunk evidence",
                "long_term_knowledge_recall": "no recall cue",
            },
        })

    outputs, _metadata = run_maker(
        benchmark_names=["BenchA"],
        vocab=VOCAB,
        aggregates={"BenchA": _aggregate("BenchA")},
        config={
            "maker_model": {"name": "fake", "base_url": None},
            "llm_json_contract_max_attempts": 2,
        },
        run_dir=str(tmp_path),
        prompt="prompt v1",
        version=1,
        label="iter_001",
        chat_fn=chat,
    )

    assert outputs["BenchA"]["ability_levels"]["deductive_reasoning"] == "medium"
    assert len(seen_users) == 2
    assert "ability_rationale_leaks_non_evidence" in seen_users[1]
    assert "Remove every failed metadata/surface token" in seen_users[1]


def test_run_maker_seed_partitions_cache(tmp_path):
    seen_seeds: list[int | None] = []

    def chat(_system, _user, seed=None):
        seen_seeds.append(seed)
        marker = "alpha" if seed == 11 else "beta"
        return json.dumps({
            "benchmark_summary": "synthesis",
            "ability_levels": {
                "deductive_reasoning": "medium",
                "long_term_knowledge_recall": "absent",
            },
            "ability_rationale": {
                "deductive_reasoning": f"rule comparison {marker}",
                "long_term_knowledge_recall": "no recall cue",
            },
        })

    config = {"maker_model": {"name": "fake", "base_url": None}}
    run_maker(
        benchmark_names=["BenchA"],
        vocab=VOCAB,
        aggregates={"BenchA": _aggregate("BenchA")},
        config=config,
        run_dir=str(tmp_path),
        prompt="prompt v1",
        version=1,
        label="iter_001",
        chat_fn=chat,
        seed=11,
    )
    run_maker(
        benchmark_names=["BenchA"],
        vocab=VOCAB,
        aggregates={"BenchA": _aggregate("BenchA")},
        config=config,
        run_dir=str(tmp_path),
        prompt="prompt v1",
        version=1,
        label="iter_001",
        chat_fn=chat,
        seed=22,
    )

    assert seen_seeds == [11, 22]


def test_run_maker_parallel_cache_hits_tallied_correctly(tmp_path):
    """Cache-hit counter under threads must equal the number of pre-existing cache files."""
    bench_names = [f"B{i}" for i in range(5)]
    aggregates = {b: _aggregate(b) for b in bench_names}

    def first_chat(_system, _user):
        return json.dumps({
            "benchmark_summary": "s",
            "ability_levels": {
                "deductive_reasoning": "medium",
                "long_term_knowledge_recall": "weak",
            },
            "ability_rationale": {
                "deductive_reasoning": "rule use",
                "long_term_knowledge_recall": "minor recall",
            },
        })

    config = {"maker_model": {"name": "fake", "base_url": None}, "maker_max_workers": 4}
    run_maker(
        benchmark_names=bench_names,
        vocab=VOCAB,
        aggregates=aggregates,
        config=config,
        run_dir=str(tmp_path),
        prompt="cache prompt",
        version=1,
        label="seed",
        chat_fn=first_chat,
    )

    def fail_chat(_system, _user):
        raise AssertionError("should have hit cache")

    _outputs, metadata = run_maker(
        benchmark_names=bench_names,
        vocab=VOCAB,
        aggregates=aggregates,
        config=config,
        run_dir=str(tmp_path),
        prompt="cache prompt",
        version=1,
        label="rerun",
        chat_fn=fail_chat,
    )
    assert metadata["maker_cache_hits"] == 5
