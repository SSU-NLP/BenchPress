"""Tests for experiment/tag_generator.py — JSON parse, vocab clamp, NaN reject, retry."""

from __future__ import annotations

import json
import math

from autotagging_loop.experiment.tag_generator import generate_tag_vector, random_tag_vectors


VOCAB = [
    {"id": "analogical_reasoning"},
    {"id": "deductive_reasoning"},
    {"id": "inductive_reasoning"},
]


def _chat(response: str):
    def fn(_sys, _user, _seed):
        return response
    return fn


def test_vocab_clamp_and_extras_dropped():
    payload = {
        "weights": {
            "analogical_reasoning": 1.5,    # clamp -> 1
            "deductive_reasoning": -0.3,    # clamp -> 0
            "inductive_reasoning": 0.4,
            "FAKE_KEY": 0.99,
        },
        "rationale": "x",
    }
    tv = generate_tag_vector(
        "B", "desc", None, VOCAB,
        prompt="P", model="m", base_url=None,
        chat_fn=_chat(json.dumps(payload)),
    )
    assert tv.weights["analogical_reasoning"] == 1.0
    assert tv.weights["deductive_reasoning"] == 0.0
    assert math.isclose(tv.weights["inductive_reasoning"], 0.4)
    assert "FAKE_KEY" not in tv.weights
    assert any("vocab_extras_dropped" in d for d in tv.drift_log)


def test_signed_weight_bounds_preserve_negative_values():
    payload = {
        "weights": {
            "analogical_reasoning": 1.5,
            "deductive_reasoning": -0.7,
            "inductive_reasoning": 0.4,
        },
        "rationale": "x",
    }
    tv = generate_tag_vector(
        "B", "desc", None, VOCAB,
        prompt="P", model="m", base_url=None,
        chat_fn=_chat(json.dumps(payload)),
        weight_bounds=(-1.0, 1.0),
    )
    assert tv.weights["analogical_reasoning"] == 1.0
    assert tv.weights["deductive_reasoning"] == -0.7
    assert math.isclose(tv.weights["inductive_reasoning"], 0.4)


def test_signed_zero_sum_nonzero_vector_does_not_retry():
    calls = {"n": 0}

    def fn(_sys, _user, _seed):
        calls["n"] += 1
        return json.dumps({
            "weights": {
                "analogical_reasoning": 0.5,
                "deductive_reasoning": -0.5,
                "inductive_reasoning": 0.0,
            }
        })

    tv = generate_tag_vector(
        "B", "desc", None, VOCAB,
        prompt="P", model="m", base_url=None,
        chat_fn=fn,
        weight_bounds=(-1.0, 1.0),
    )
    assert calls["n"] == 1
    assert tv.weights["analogical_reasoning"] == 0.5
    assert tv.weights["deductive_reasoning"] == -0.5


def test_missing_keys_filled_zero():
    payload = {"weights": {"analogical_reasoning": 0.5}, "rationale": "x"}
    tv = generate_tag_vector(
        "B", "desc", None, VOCAB,
        prompt="P", model="m", base_url=None,
        chat_fn=_chat(json.dumps(payload)),
    )
    assert tv.weights["deductive_reasoning"] == 0.0
    assert tv.weights["inductive_reasoning"] == 0.0
    assert any("vocab_missing_filled_zero" in d for d in tv.drift_log)


def test_nan_inf_rejected():
    raw = '{"weights": {"analogical_reasoning": "NaN", "deductive_reasoning": "Infinity", "inductive_reasoning": 0.7}, "rationale": "x"}'
    tv = generate_tag_vector(
        "B", "desc", None, VOCAB,
        prompt="P", model="m", base_url=None,
        chat_fn=_chat(raw),
    )
    assert tv.weights["analogical_reasoning"] == 0.0  # NaN as string -> non_numeric or non_finite, drift recorded
    # 0.7 should still be set
    assert math.isclose(tv.weights["inductive_reasoning"], 0.7)


def test_retry_on_zero_sum_then_fallback_uniform():
    calls = {"n": 0}

    def fn(_sys, _user, _seed):
        calls["n"] += 1
        # both calls return zero-sum
        return json.dumps({"weights": {tid["id"]: 0.0 for tid in VOCAB}})

    tv = generate_tag_vector(
        "B", "desc", None, VOCAB,
        prompt="P", model="m", base_url=None,
        chat_fn=fn,
    )
    assert calls["n"] == 2
    expected = 1.0 / len(VOCAB)
    for v in tv.weights.values():
        assert math.isclose(v, expected, rel_tol=1e-9)
    assert "fallback_uniform" in tv.drift_log


def test_zero_sum_fails_when_uniform_fallback_disabled():
    calls = {"n": 0}

    def fn(_sys, _user, _seed):
        calls["n"] += 1
        return json.dumps({"weights": {tid["id"]: 0.0 for tid in VOCAB}})

    try:
        generate_tag_vector(
            "B", "desc", None, VOCAB,
            prompt="P", model="m", base_url=None,
            chat_fn=fn,
            allow_uniform_fallback=False,
        )
    except RuntimeError as exc:
        assert "uniform fallback is disabled" in str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("expected zero-sum fallback to fail")
    assert calls["n"] == 2


def test_random_tag_vectors_deterministic():
    a = random_tag_vectors(["X", "Y"], VOCAB, seed=1)
    b = random_tag_vectors(["X", "Y"], VOCAB, seed=1)
    assert a == b
    for vec in a.values():
        assert all(0.0 <= v <= 1.0 for v in vec.values())
        assert set(vec.keys()) == {v["id"] for v in VOCAB}
