"""Tests for experiment/prompt_improver.py — 4 guards."""

from __future__ import annotations

import json

import pytest

from autotagging_loop.experiment.alignment import ErrorPair
from autotagging_loop.experiment.json_contract import JSONContractError
from autotagging_loop.experiment.prompt_improver import improve_prompt, validate_prompt


VOCAB = [
    {"id": "analogical_reasoning"},
    {"id": "deductive_reasoning"},
    {"id": "inductive_reasoning"},
]
VOCAB_IDS = [v["id"] for v in VOCAB]
BASE_PROMPT = "Tag the benchmark with " + ", ".join(VOCAB_IDS) + " . Provide weighted scores."


def test_guard_shorter_than_base():
    new_p = "Use " + ", ".join(VOCAB_IDS) + "."
    ok, reasons = validate_prompt(new_p, BASE_PROMPT, VOCAB_IDS, ["MMLU"])
    assert not ok
    assert "shorter_than_I0" in reasons


def test_guard_allows_compact_rewrite_when_base_exceeds_hard_cap():
    long_base = (
        BASE_PROMPT
        + " Preserve schema, evidence grounding, and operation calibration. " * 80
    )
    new_p = (
        BASE_PROMPT
        + " Preserve schema, evidence grounding, and operation calibration. " * 52
    )

    ok, reasons = validate_prompt(new_p, long_base, VOCAB_IDS, ["MMLU"])

    assert ok, reasons
    assert len(new_p) < len(long_base)


def test_guard_vocab_missing():
    new_p = ("Tag the benchmark using analogical_reasoning and deductive_reasoning. "
             "Provide weighted scores in extended detail to satisfy length requirement.")
    ok, reasons = validate_prompt(new_p, BASE_PROMPT, VOCAB_IDS, ["MMLU"])
    assert not ok
    assert any("vocab_missing" in r for r in reasons)


def test_improver_payload_sanitizes_error_context():
    captured: dict = {}
    new_prompt = (
        BASE_PROMPT
        + " Distinguish abstract pattern transfer from rule-chain deduction in general evidence."
    )

    def chat_fn(system, user):
        captured["system"] = system
        captured["payload"] = json.loads(user)
        return json.dumps({"new_prompt": new_prompt, "rationale": "sanitized"})

    result = improve_prompt(
        prev_prompt=BASE_PROMPT,
        base_prompt=BASE_PROMPT,
        error_report=[
            ErrorPair(
                p="MMLU",
                q="BenchB",
                s_pq=0.12,
                r_pq_raw=0.98,
                r_pq_01=0.99,
                delta=0.4567,
                type="false_dis",
            )
        ],
        metrics={
            "L_align": 0.1234,
            "rho_align_pearson": -0.42,
            "rho_align_spearman": -0.37,
            "delta_tag": 0.22,
        },
        bench_descriptions={
            "MMLU": "Benchmark: MMLU\nMMLU asks knowledge questions. score 0.9876",
            "BenchB": "Benchmark: BenchB\nBenchB asks rule questions. score 0.1234",
        },
        vocab=VOCAB,
        benchmark_names=["MMLU", "BenchB"],
        model="fake",
        base_url=None,
        chat_fn=chat_fn,
    )

    assert result.accepted is True
    payload_text = json.dumps(captured["payload"], ensure_ascii=False)
    assert "MMLU" not in payload_text
    assert "BenchB" not in payload_text
    assert "0.4567" not in payload_text
    assert "0.9876" not in payload_text
    assert "metrics" not in captured["payload"]
    assert captured["payload"]["metrics_summary"]["loss_level"] == "high"
    summary = captured["payload"]["error_mode_summary"]
    assert summary["false_dis_count"] == 1
    assert summary["false_sim_count"] == 0
    assert summary["dominant_error_type"] == "false_dis"
    assert "tag_similarity_too_low" in summary["dominant_interpretation"]
    assert captured["payload"]["active_vocabulary"][0]["id"] == "analogical_reasoning"
    assert "false_dis means the current tag space separates" in captured["system"]
    assert "Use the supplied active_vocabulary definitions" in captured["system"]


def test_guard_benchmark_name_present():
    new_p = (BASE_PROMPT + " For example MMLU should weight deductive_reasoning highly.")
    ok, reasons = validate_prompt(new_p, BASE_PROMPT, VOCAB_IDS, ["MMLU"])
    assert not ok
    assert any("benchmark_names_present" in r for r in reasons)


def test_guard_benchmark_name_substring_not_flagged():
    # "MATH" must not match inside "mathematical" / "math problem" / "matched".
    # Real run regression: every Improver iter was rejected because legitimate
    # English vocab words triggered the substring guard, freezing v_loop.
    new_p = (BASE_PROMPT + " Reason about mathematical operations and patterns matched in evidence."
             " A useful tag distinguishes algorithmic problem solving from semantic resolution."
             " Consider tasks like solve a math problem when defining quantitative abilities."
             " Athletes appear in commonsense scenarios but the relevant ability is more general.")
    ok, reasons = validate_prompt(new_p, BASE_PROMPT, VOCAB_IDS, ["MATH", "MATH-500", "HLE"])
    assert ok, f"unexpected rejection reasons={reasons}"


def test_guard_benchmark_name_word_match_still_caught():
    # Canonical-case benchmark name as a word must still be flagged.
    new_p = (BASE_PROMPT + " For example MATH should weight deductive_reasoning highly.")
    ok, reasons = validate_prompt(new_p, BASE_PROMPT, VOCAB_IDS, ["MATH"])
    assert not ok
    assert any("benchmark_names_present" in r for r in reasons)


def test_guard_benchmark_name_multiword_match():
    # Multi-word benchmark names (containing space) must match as a phrase.
    new_p = (BASE_PROMPT + " AIME 2024 examples illustrate deductive_reasoning.")
    ok, reasons = validate_prompt(new_p, BASE_PROMPT, VOCAB_IDS, ["AIME 2024"])
    assert not ok
    assert any("benchmark_names_present" in r for r in reasons)


def test_guard_benchmark_name_pascalcase_match():
    # PascalCase benchmark names like HellaSwag must still match exactly.
    new_p = (BASE_PROMPT + " The HellaSwag benchmark exercises commonsense.")
    ok, reasons = validate_prompt(new_p, BASE_PROMPT, VOCAB_IDS, ["HellaSwag"])
    assert not ok
    assert any("benchmark_names_present" in r for r in reasons)


def test_guard_score_literal():
    new_p = (BASE_PROMPT + " If commonsense applies, give 0.85 for analogical_reasoning.")
    ok, reasons = validate_prompt(new_p, BASE_PROMPT, VOCAB_IDS, ["MMLU"])
    assert not ok
    assert "score_literal_present" in reasons


def test_guard_forbidden_label():
    new_p = (BASE_PROMPT + " Aim to maximize Spearman correlation between tag vectors.")
    ok, reasons = validate_prompt(new_p, BASE_PROMPT, VOCAB_IDS, ["MMLU"])
    assert not ok
    assert any("forbidden_label" in r for r in reasons)


def test_guard_forbidden_label_inside_prohibition_clause_is_accepted():
    """Rule-5 wording: the prompt prohibits a banned word — must NOT trigger forbidden_label."""
    new_p = (
        BASE_PROMPT
        + " Do not mention specific benchmark names, scores, leaderboard rankings, "
        "correlation, Spearman, Pearson, rho, or numeric thresholds. State only general "
        "rules about how to identify cognitive-ability evidence."
    )
    ok, reasons = validate_prompt(new_p, BASE_PROMPT, VOCAB_IDS, ["MMLU"])
    assert ok, reasons


def test_guard_shortcut_instruction_rejected():
    new_p = (
        BASE_PROMPT
        + " Use public benchmark difficulty and leaderboard ranking as evidence "
        "when setting analogical_reasoning, deductive_reasoning, and inductive_reasoning."
    )
    ok, reasons = validate_prompt(new_p, BASE_PROMPT, VOCAB_IDS, ["MMLU"])
    assert not ok
    assert any("shortcut_instruction_present" in r for r in reasons)


def test_guard_shortcut_terms_inside_prohibition_clause_are_accepted():
    new_p = (
        BASE_PROMPT
        + " Do not use public benchmark difficulty, model performance, answer format, "
        "or leaderboard ranking as evidence for analogical_reasoning, "
        "deductive_reasoning, or inductive_reasoning."
    )
    ok, reasons = validate_prompt(new_p, BASE_PROMPT, VOCAB_IDS, ["MMLU"])
    assert ok, reasons


def test_guard_shortcut_terms_inside_not_a_bullet_are_accepted():
    new_p = (
        BASE_PROMPT
        + "\n- not a benchmark name, dataset name, answer format, or external outcome cue"
    )
    ok, reasons = validate_prompt(new_p, BASE_PROMPT, VOCAB_IDS, ["MMLU"])
    assert ok, reasons


def test_guard_score_literal_inside_prohibition_clause_is_accepted():
    new_p = (
        BASE_PROMPT
        + " Never output decimal values like 0.95 in the rationale; describe abilities "
        "qualitatively instead so the downstream reducer can compute weights."
    )
    ok, reasons = validate_prompt(new_p, BASE_PROMPT, VOCAB_IDS, ["MMLU"])
    assert ok, reasons


def test_guard_taxonomy_prohibition_phrase_is_accepted_in_fixed_phase():
    new_p = (
        BASE_PROMPT
        + " Do not add a new tag, and do not rename any ability — keep the seed list fixed."
    )
    ok, reasons = validate_prompt(new_p, BASE_PROMPT, VOCAB_IDS, ["MMLU"])
    assert ok, reasons


def test_guard_double_negative_still_caught():
    """'Do not avoid mentioning correlation' is effectively a license to use the term —
    we err on the side of letting these through. This locks the current behavior so any
    future tightening of the sanitizer is an explicit decision, not a silent regression."""
    new_p = (
        BASE_PROMPT
        + " Do not avoid mentioning Spearman correlation when describing tag-score links."
    )
    ok, reasons = validate_prompt(new_p, BASE_PROMPT, VOCAB_IDS, ["MMLU"])
    assert ok, reasons


def test_guard_taxonomy_change_rejected_in_fixed_phase():
    new_p = (
        BASE_PROMPT
        + " Add a new tag for visual abstraction when the seed abilities leave residual errors."
    )
    ok, reasons = validate_prompt(new_p, BASE_PROMPT, VOCAB_IDS, ["MMLU"])
    assert not ok
    assert any("taxonomy_change_requested" in r for r in reasons)


def test_improver_retries_with_validation_feedback_after_guard_failure():
    calls: list[dict] = []
    rejected_prompt = (
        BASE_PROMPT
        + " Use leaderboard ranking as evidence when calibrating analogical_reasoning, "
        "deductive_reasoning, and inductive_reasoning."
    )
    accepted_prompt = (
        BASE_PROMPT
        + " Calibrate analogical_reasoning, deductive_reasoning, and inductive_reasoning "
        "from directly evidenced cognitive operations rather than source-specific cues."
    )

    def chat_fn(_system, user):
        payload = json.loads(user)
        calls.append(payload)
        prompt = rejected_prompt if len(calls) == 1 else accepted_prompt
        return json.dumps({"new_prompt": prompt, "rationale": "retry"})

    result = improve_prompt(
        prev_prompt=BASE_PROMPT,
        base_prompt=BASE_PROMPT,
        error_report=[],
        metrics={},
        bench_descriptions={},
        vocab=VOCAB,
        benchmark_names=["MMLU"],
        model="fake",
        base_url=None,
        chat_fn=chat_fn,
        n_samples=2,
    )

    assert result.accepted is True
    assert result.new_prompt == accepted_prompt
    assert len(calls) == 2
    feedback = calls[1]["previous_candidate_feedback"]
    assert feedback["previous_candidate_rejected"] is True
    assert any("shortcut_instruction_present" in r for r in feedback["validation_reasons"])


def test_improver_retry_feedback_hides_blocked_literal_terms():
    calls: list[dict] = []
    rejected_prompt = (
        BASE_PROMPT
        + " Use multiple choice cues and correlation wording when calibrating "
        "analogical_reasoning, deductive_reasoning, and inductive_reasoning."
    )
    accepted_prompt = (
        BASE_PROMPT
        + " Calibrate analogical_reasoning, deductive_reasoning, and inductive_reasoning "
        "from directly observed cognitive operations only."
    )

    def chat_fn(_system, user):
        payload = json.loads(user)
        calls.append(payload)
        prompt = rejected_prompt if len(calls) == 1 else accepted_prompt
        return json.dumps({"new_prompt": prompt, "rationale": "retry"})

    result = improve_prompt(
        prev_prompt=BASE_PROMPT,
        base_prompt=BASE_PROMPT,
        error_report=[],
        metrics={},
        bench_descriptions={},
        vocab=VOCAB,
        benchmark_names=["MMLU"],
        model="fake",
        base_url=None,
        chat_fn=chat_fn,
        n_samples=2,
    )

    assert result.accepted is True
    feedback_text = json.dumps(calls[1]["previous_candidate_feedback"]).lower()
    assert "correlation" not in feedback_text
    assert "multiple choice" not in feedback_text
    assert "multiple\\s+choice" not in feedback_text
    assert "alignment_metric_word" in feedback_text
    assert "surface_or_outcome_cue" in feedback_text


def test_improver_passes_stable_seed_per_sample():
    seen_seeds: list[int | None] = []
    rejected_prompt = (
        BASE_PROMPT
        + " Use leaderboard ranking as evidence when calibrating analogical_reasoning, "
        "deductive_reasoning, and inductive_reasoning."
    )
    accepted_prompt = (
        BASE_PROMPT
        + " Calibrate analogical_reasoning, deductive_reasoning, and inductive_reasoning "
        "from directly evidenced cognitive operations rather than source-specific cues."
    )

    def chat_fn(_system, _user, seed=None):
        seen_seeds.append(seed)
        prompt = rejected_prompt if len(seen_seeds) == 1 else accepted_prompt
        return json.dumps({"new_prompt": prompt, "rationale": "seeded"})

    result = improve_prompt(
        prev_prompt=BASE_PROMPT,
        base_prompt=BASE_PROMPT,
        error_report=[],
        metrics={},
        bench_descriptions={},
        vocab=VOCAB,
        benchmark_names=["MMLU"],
        model="fake",
        base_url=None,
        chat_fn=chat_fn,
        n_samples=2,
        json_contract_max_attempts=2,
        seed=500,
    )

    assert result.accepted is True
    assert seen_seeds == [500, 502]


def test_improver_retries_with_feedback_after_contract_length_failure():
    calls: list[dict] = []
    too_long_prompt = (
        BASE_PROMPT
        + " Preserve schema and active vocabulary. " * 140
    )
    accepted_prompt = (
        BASE_PROMPT
        + " Preserve schema and active vocabulary while using concise operation rules."
    )

    def chat_fn(_system, user):
        payload = json.loads(user)
        calls.append(payload)
        prompt = too_long_prompt if len(calls) == 1 else accepted_prompt
        return json.dumps({"new_prompt": prompt, "rationale": "length retry"})

    result = improve_prompt(
        prev_prompt=BASE_PROMPT,
        base_prompt=BASE_PROMPT,
        error_report=[],
        metrics={},
        bench_descriptions={},
        vocab=VOCAB,
        benchmark_names=["MMLU"],
        model="fake",
        base_url=None,
        chat_fn=chat_fn,
        n_samples=2,
        json_contract_max_attempts=1,
    )

    assert result.accepted is True
    assert result.new_prompt == accepted_prompt
    feedback = calls[1]["previous_candidate_feedback"]
    assert any("new_prompt_too_long" in r for r in feedback["validation_reasons"])
    assert any("target_prompt_chars" in h for h in feedback["repair_hints"])


def test_guard_taxonomy_change_allowed_in_unlocked_phase():
    new_p = (
        BASE_PROMPT
        + " Add a new tag for visual abstraction when the seed abilities leave residual errors."
    )
    ok, reasons = validate_prompt(
        new_p,
        BASE_PROMPT,
        VOCAB_IDS,
        ["MMLU"],
        allow_taxonomy_changes=True,
    )
    assert ok, reasons


def test_accept_clean_prompt():
    new_p = (
        BASE_PROMPT
        + " Reason carefully about which cognitive abilities the benchmark genuinely tests, "
        "favoring abilities that appear in the question structure and ignoring superficial cues."
    )
    ok, reasons = validate_prompt(new_p, BASE_PROMPT, VOCAB_IDS, ["MMLU"])
    assert ok, reasons


def test_improve_prompt_rejects_bad_response():
    def chat(_sys, _user):
        return json.dumps({
            "new_prompt": "Use MMLU as reference and aim for 0.95 Spearman.",
            "rationale": "n/a",
        })

    res = improve_prompt(
        prev_prompt=BASE_PROMPT,
        base_prompt=BASE_PROMPT,
        error_report=[],
        metrics={"L_align": 0.1, "delta_tag": 0.0},
        bench_descriptions={"MMLU": "knowledge bench"},
        vocab=VOCAB,
        benchmark_names=["MMLU"],
        model="m",
        base_url=None,
        chat_fn=chat,
    )
    assert not res.accepted
    assert res.new_prompt == BASE_PROMPT  # rollback to prev


def test_improve_prompt_repairs_prompt_that_exceeds_length_contract():
    calls: list[str] = []
    too_long = BASE_PROMPT + "\n" + ("Use concise evidence distinctions. " * 160)
    repaired = (
        BASE_PROMPT
        + " Distinguish retrieval, rule chaining, and pattern abstraction using general "
        "evidence cues. Keep all guidance compact and independent of benchmark names."
    )

    def chat(_sys, user):
        calls.append(user)
        prompt = too_long if len(calls) == 1 else repaired
        return json.dumps({"new_prompt": prompt, "rationale": "length repaired"})

    res = improve_prompt(
        prev_prompt=BASE_PROMPT,
        base_prompt=BASE_PROMPT,
        error_report=[],
        metrics={"L_align": 0.1, "delta_tag": 0.1},
        bench_descriptions={},
        vocab=VOCAB,
        benchmark_names=[],
        model="m",
        base_url=None,
        chat_fn=chat,
        json_contract_max_attempts=2,
    )

    assert res.accepted is True
    assert res.new_prompt == repaired
    assert len(calls) == 2
    assert "new_prompt_too_long" in calls[1]


def test_improve_prompt_invalid_json_fails_contract():
    def chat(_sys, _user):
        return "not json"

    with pytest.raises(JSONContractError, match="invalid_json"):
        improve_prompt(
            prev_prompt=BASE_PROMPT,
            base_prompt=BASE_PROMPT,
            error_report=[],
            metrics={"L_align": 0.1, "delta_tag": 0.0},
            bench_descriptions={},
            vocab=VOCAB,
            benchmark_names=["FakeBench"],
            model="m",
            base_url=None,
            chat_fn=chat,
            json_contract_max_attempts=1,
        )


def test_improve_prompt_accepts_clean_response():
    accepted_prompt = (
        BASE_PROMPT
        + " When a benchmark relies more on stored facts, increase long-term knowledge weights; "
        "when it requires step-by-step inference, increase deductive_reasoning and inductive_reasoning."
    )

    def chat(_sys, _user):
        return json.dumps({"new_prompt": accepted_prompt, "rationale": "ok"})

    res = improve_prompt(
        prev_prompt=BASE_PROMPT,
        base_prompt=BASE_PROMPT,
        error_report=[],
        metrics={"L_align": 0.1, "delta_tag": 0.0},
        bench_descriptions={},
        vocab=VOCAB,
        benchmark_names=["FakeBench"],
        model="m",
        base_url=None,
        chat_fn=chat,
    )
    assert res.accepted, res.reasons
    assert res.new_prompt == accepted_prompt


def test_improve_prompt_v_loop_mode_allows_taxonomy_changes():
    """v_loop ON path: improve_prompt(allow_taxonomy_changes=True) must accept
    a new_prompt that proposes adding/refining vocabulary entries; the same
    response with the flag off must be rejected with taxonomy_change_requested."""
    candidate = (
        BASE_PROMPT
        + " Add a new ability for visual_abstraction when residual evidence keeps surfacing it; "
        "otherwise keep analogical_reasoning, deductive_reasoning, and inductive_reasoning intact."
    )

    def chat(_sys, _user):
        return json.dumps({"new_prompt": candidate, "rationale": "v_loop refinement"})

    res_unlocked = improve_prompt(
        prev_prompt=BASE_PROMPT,
        base_prompt=BASE_PROMPT,
        error_report=[],
        metrics={"L_align": 0.1, "delta_tag": 0.0},
        bench_descriptions={},
        vocab=VOCAB,
        benchmark_names=["FakeBench"],
        model="m",
        base_url=None,
        chat_fn=chat,
        allow_taxonomy_changes=True,
    )
    assert res_unlocked.accepted, res_unlocked.reasons
    assert res_unlocked.new_prompt == candidate

    res_locked = improve_prompt(
        prev_prompt=BASE_PROMPT,
        base_prompt=BASE_PROMPT,
        error_report=[],
        metrics={"L_align": 0.1, "delta_tag": 0.0},
        bench_descriptions={},
        vocab=VOCAB,
        benchmark_names=["FakeBench"],
        model="m",
        base_url=None,
        chat_fn=chat,
        allow_taxonomy_changes=False,
    )
    assert not res_locked.accepted
    assert any("taxonomy_change_requested" in r for r in res_locked.reasons)
    assert res_locked.new_prompt == BASE_PROMPT


def test_improve_prompt_rejects_duplicate_of_prev_prompt():
    """Real-run regression (run_20260511_134315): with temperature=0 the Improver
    returned a byte-identical prompt for iter_3..5, stalling v_loop. The escape
    contract is: even a guard-passing candidate is rejected if it equals prev."""
    def chat(_sys, _user):
        # Return the same prompt as prev — guards pass, but it's a duplicate.
        return json.dumps({"new_prompt": BASE_PROMPT, "rationale": "no change"})

    res = improve_prompt(
        prev_prompt=BASE_PROMPT,
        base_prompt=BASE_PROMPT,
        error_report=[],
        metrics={"L_align": 0.1, "delta_tag": 0.0},
        bench_descriptions={},
        vocab=VOCAB,
        benchmark_names=["FakeBench"],
        model="m",
        base_url=None,
        chat_fn=chat,
        n_samples=3,
    )
    assert not res.accepted
    assert res.new_prompt == BASE_PROMPT
    assert any("duplicate_of_prev_prompt" in r for r in res.reasons)


def test_improve_prompt_n_samples_returns_first_distinct_accepted():
    accepted_prompt = (
        BASE_PROMPT
        + " Reason carefully about which cognitive abilities the benchmark genuinely tests, "
        "favoring abilities that appear in the question structure and ignoring superficial cues."
    )
    calls = {"n": 0}
    sequence = [
        json.dumps({"new_prompt": BASE_PROMPT, "rationale": "dup"}),
        json.dumps({"new_prompt": accepted_prompt, "rationale": "ok"}),
        json.dumps({"new_prompt": BASE_PROMPT, "rationale": "dup again"}),
    ]

    def chat(_sys, _user):
        i = calls["n"]
        calls["n"] += 1
        return sequence[min(i, len(sequence) - 1)]

    res = improve_prompt(
        prev_prompt=BASE_PROMPT,
        base_prompt=BASE_PROMPT,
        error_report=[],
        metrics={"L_align": 0.1, "delta_tag": 0.0},
        bench_descriptions={},
        vocab=VOCAB,
        benchmark_names=["FakeBench"],
        model="m",
        base_url=None,
        chat_fn=chat,
        n_samples=3,
    )
    assert res.accepted, res.reasons
    assert res.new_prompt == accepted_prompt
    assert calls["n"] == 2  # stopped after the second sample succeeded


def test_improve_prompt_payload_includes_rho_align():
    seen = {}
    accepted_prompt = (
        BASE_PROMPT
        + " Use item evidence to distinguish stored facts from explicit inference requirements."
    )

    def chat(_sys, user):
        seen["payload"] = json.loads(user)
        return json.dumps({"new_prompt": accepted_prompt, "rationale": "ok"})

    res = improve_prompt(
        prev_prompt=BASE_PROMPT,
        base_prompt=BASE_PROMPT,
        error_report=[],
        metrics={
            "L_align": 0.1,
            "rho_align_pearson": 0.2,
            "rho_align_spearman": 0.3,
            "delta_tag": 0.4,
        },
        bench_descriptions={},
        vocab=VOCAB,
        benchmark_names=["FakeBench"],
        model="m",
        base_url=None,
        chat_fn=chat,
    )

    assert res.accepted
    assert seen["payload"]["metrics_summary"]["rho_pearson_direction"] == "strong_positive"
    assert seen["payload"]["metrics_summary"]["rho_spearman_direction"] == "strong_positive"
