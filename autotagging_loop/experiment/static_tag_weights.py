"""Static benchmark tag-weight reducer.

The LLM may extract chunk-level evidence, but final benchmark weights are
computed here by deterministic aggregation. This keeps Part 1's tag vector
T_l interpretable as a statistic of the processed dataset rather than a
free-form numeric judgment from the model.
"""

from __future__ import annotations

import math
from typing import Any


ABILITY_LEVEL_SCORES: dict[str, float] = {
    "absent": 0.0,
    "weak": 0.25,
    "medium": 0.5,
    "strong": 0.75,
    "dominant": 1.0,
}

_LEVEL_ALIASES: dict[str, str] = {
    # ponytail: tolerate the common 1-5 ordinal leak from JSON-mode LLMs.
    "0": "absent",
    "0.0": "absent",
    "1": "absent",
    "1.0": "absent",
    "2": "weak",
    "2.0": "weak",
    "3": "medium",
    "3.0": "medium",
    "4": "strong",
    "4.0": "strong",
    "5": "dominant",
    "5.0": "dominant",
    "none": "absent",
    "missing": "absent",
    "low": "weak",
    "moderate": "medium",
    "mid": "medium",
    "high": "strong",
    "primary": "dominant",
    "central": "dominant",
}


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def coerce_ability_level(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    raw = _LEVEL_ALIASES.get(raw, raw)
    return raw if raw in ABILITY_LEVEL_SCORES else "absent"


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = float(bounds[0]), float(bounds[1])
    if lower > upper:
        raise ValueError(f"invalid weight bounds: {bounds}")
    return max(lower, min(upper, value))


def build_static_tag_vectors_from_reducer_levels(
    benchmark_names: list[str],
    vocab: list[dict],
    reducer_outputs: dict[str, dict],
    config: dict,
) -> tuple[dict[str, dict[str, float]], dict]:
    """Compute T_l from benchmark-level LLM reducer ordinal judgments.

    The reducer is allowed to synthesize qualitative levels only. Numeric
    weights are produced here by a deterministic level -> score mapping.
    """
    vocab_ids = [v["id"] for v in vocab]
    bounds = tuple(config.get("weight_bounds", [0.0, 1.0]))
    require_all = bool(config.get("static_tag_require_mapreduce", True))
    level_scores = dict(ABILITY_LEVEL_SCORES)

    T: dict[str, dict[str, float]] = {}
    per_benchmark: dict[str, dict] = {}
    missing: list[str] = []

    for benchmark in benchmark_names:
        reduced = reducer_outputs.get(benchmark)
        if not reduced:
            missing.append(benchmark)
            if require_all:
                continue
            T[benchmark] = {tag_id: _clamp(0.0, bounds) for tag_id in vocab_ids}
            per_benchmark[benchmark] = {
                "source": "missing_reducer_output_zero_fill",
                "weights": T[benchmark],
            }
            continue

        raw_levels = reduced.get("ability_levels")
        if not isinstance(raw_levels, dict):
            raw_levels = {}
        levels = {tag_id: coerce_ability_level(raw_levels.get(tag_id)) for tag_id in vocab_ids}
        weights = {
            tag_id: _clamp(level_scores[levels[tag_id]], bounds)
            for tag_id in vocab_ids
        }
        T[benchmark] = weights
        per_benchmark[benchmark] = {
            "source": "mapreduce_llm_reducer_levels",
            "mapped_examples": reduced.get("mapped_examples"),
            "n_chunks": reduced.get("n_chunks"),
            "levels": levels,
            "weights": weights,
        }

    if missing and require_all:
        missing_str = ", ".join(missing[:10])
        raise ValueError(
            "static_from_mapreduce requires LLM reducer outputs for every benchmark; "
            f"missing={missing_str}"
        )

    metadata = {
        "method": "mapreduce_llm_reducer_static_weights",
        "final_weight_policy": (
            "LLM reducer synthesizes benchmark-level ordinal ability levels only; "
            "final numeric weights are deterministic level-score mappings."
        ),
        "formula": "w_lk = level_score(final_level_lk)",
        "level_scores": level_scores,
        "weight_bounds": list(bounds),
        "require_all_reducer_outputs": require_all,
        "missing_reducer_outputs": missing,
        "benchmarks": per_benchmark,
    }
    return T, metadata
