"""experiment/tag_generator.py — LLM 호출로 bounded weighted tag vector T_l 생성."""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TagVector:
    benchmark: str
    weights: dict[str, float]
    raw_response: str = ""
    prompt_version: int = 0
    drift_log: list[str] = field(default_factory=list)


# Type for an injectable chat function (for tests). Returns the raw response string.
ChatFn = Callable[[str, str, str | None], str]


def _default_chat_fn(
    model: str,
    base_url: str | None = None,
    *,
    base_url_env: str | None = None,
    api_key_env: str | None = None,
) -> ChatFn:
    from autotagging_loop.experiment.llm_client import shared_factory

    return shared_factory().chat_fn(
        model=model,
        base_url=base_url,
        base_url_env=base_url_env,
        api_key_env=api_key_env,
        response_format={"type": "json_object"},
        error_label="tag_generator",
    )


def _parse_json(raw: str) -> dict:
    if not raw:
        return {}
    cleaned = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", raw, flags=re.DOTALL).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                return {}
    return {}


def _coerce_weights(
    raw: dict,
    vocab_ids: list[str],
    weight_bounds: tuple[float, float] = (0.0, 1.0),
) -> tuple[dict[str, float], list[str]]:
    """Validate + clamp weights. Returns (weights, drift_log)."""
    lower, upper = float(weight_bounds[0]), float(weight_bounds[1])
    if lower > upper:
        raise ValueError(f"invalid weight bounds: {weight_bounds}")
    neutral = max(lower, min(upper, 0.0))
    drift: list[str] = []
    weights_raw = raw.get("weights")
    if not isinstance(weights_raw, dict):
        drift.append("missing_weights_dict")
        return {tid: neutral for tid in vocab_ids}, drift

    out: dict[str, float] = {tid: neutral for tid in vocab_ids}
    extras = []
    for k, v in weights_raw.items():
        if k not in out:
            extras.append(k)
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            drift.append(f"non_numeric:{k}")
            continue
        if not math.isfinite(f):
            drift.append(f"non_finite:{k}")
            continue
        out[k] = max(lower, min(upper, f))
    if extras:
        drift.append(f"vocab_extras_dropped:{len(extras)}")
    missing = [tid for tid in vocab_ids if weights_raw.get(tid) is None]
    if missing:
        drift.append(f"vocab_missing_filled_zero:{len(missing)}")
    return out, drift


def _nonzero_vector(weights: dict[str, float]) -> bool:
    return math.sqrt(sum(float(v) ** 2 for v in weights.values())) > 1e-9


def _fallback_weight(n_vocab: int, weight_bounds: tuple[float, float]) -> float:
    lower, upper = float(weight_bounds[0]), float(weight_bounds[1])
    candidate = 1.0 / max(n_vocab, 1)
    return max(lower, min(upper, candidate))


def generate_tag_vector(
    benchmark: str,
    description: str,
    samples: list[str] | None,
    vocab: list[dict],
    prompt: str,
    model: str,
    base_url: str | None,
    seed: int | None = None,
    prompt_version: int = 0,
    chat_fn: ChatFn | None = None,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
    base_url_env: str | None = None,
    api_key_env: str | None = None,
    allow_uniform_fallback: bool = True,
) -> TagVector:
    """Run a single-call LLM tag generation. `chat_fn` injectable for tests."""
    vocab_ids = [v["id"] for v in vocab]
    lower, upper = float(weight_bounds[0]), float(weight_bounds[1])
    sys_msg = prompt
    user_msg_parts = [f"Benchmark: {benchmark}"]
    if description:
        user_msg_parts.append(f"\nDescription:\n{description}")
    if samples:
        user_msg_parts.append("\nSamples:\n" + "\n".join(f"- {s}" for s in samples[:5]))
    user_msg_parts.append(
        f'\nReturn JSON: {{"weights": {{<tag_id>: <float in [{lower:g},{upper:g}]>, ...}}, "rationale": "..."}}'
    )
    user_msg_parts.append(f"\nValid tag ids ({len(vocab_ids)}): {', '.join(vocab_ids)}")
    user_msg = "\n".join(user_msg_parts)

    fn = chat_fn or _default_chat_fn(
        model,
        base_url,
        base_url_env=base_url_env,
        api_key_env=api_key_env,
    )
    seed_str = None if seed is None else str(seed)

    raw = fn(sys_msg, user_msg, seed_str)
    parsed = _parse_json(raw)
    weights, drift = _coerce_weights(parsed, vocab_ids, weight_bounds)

    if not _nonzero_vector(weights):
        # retry once
        raw2 = fn(sys_msg, user_msg, seed_str)
        parsed2 = _parse_json(raw2)
        w2, d2 = _coerce_weights(parsed2, vocab_ids, weight_bounds)
        drift.extend(["retried_zero_sum", *d2])
        if not _nonzero_vector(w2):
            if not allow_uniform_fallback:
                raise RuntimeError(
                    f"tag_generator produced a zero-sum vector for {benchmark} "
                    "after retry; uniform fallback is disabled"
                )
            uniform = _fallback_weight(len(vocab_ids), weight_bounds)
            weights = {tid: uniform for tid in vocab_ids}
            drift.append("fallback_uniform")
        else:
            weights = w2
        raw = raw2

    return TagVector(
        benchmark=benchmark,
        weights=weights,
        raw_response=raw,
        prompt_version=prompt_version,
        drift_log=drift,
    )


def random_tag_vectors(
    benchmark_names: list[str],
    vocab: list[dict],
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    vocab_ids = [v["id"] for v in vocab]
    return {
        b: {tid: rng.uniform(0.0, 1.0) for tid in vocab_ids}
        for b in benchmark_names
    }
