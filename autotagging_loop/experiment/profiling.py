"""v3 §2.2.9 model profile construction with sign handling.

`relative` mode (default): P_r = Y_norm[r, :] @ T_star. Y_norm is centered on
zero (above/below the per-benchmark mean), so P entries can be negative — a
"strength relative to the cohort" signal.

`percentile` mode: Y_pct[r, b] is the rank-percentile of model r's score on
benchmark b in [0, 1]. Profile entries are guaranteed non-negative, useful
for absolute strength/weakness statements.

Both modes are returned per model so downstream consumers (e.g. the report
saver) can include both in `final/profiles.json`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def _model_score_vec(
    model: str,
    benchmark_names: list[str],
    Y_norm: dict[str, dict[str, float]],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for bench in benchmark_names:
        v = Y_norm.get(bench, {}).get(model)
        if v is None:
            continue
        try:
            out[bench] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _profile_from_score_vec(
    score_vec: dict[str, float],
    T: dict[str, dict[str, float]],
) -> dict[str, float]:
    profile: dict[str, float] = {}
    for bench, score in score_vec.items():
        weights = T.get(bench, {})
        if not weights:
            continue
        for tag_id, weight in weights.items():
            profile[tag_id] = profile.get(tag_id, 0.0) + float(score) * float(weight)
    return profile


def build_profile_relative(
    Y_norm: dict[str, dict[str, float]],
    T: dict[str, dict[str, float]],
    benchmark_names: list[str],
    model_names: Iterable[str],
) -> dict[str, dict[str, float]]:
    """Y_norm is the existing centered/normalized score table; entries may be negative."""
    out: dict[str, dict[str, float]] = {}
    for model in model_names:
        score_vec = _model_score_vec(model, benchmark_names, Y_norm)
        out[model] = _profile_from_score_vec(score_vec, T)
    return out


def y_to_percentile(
    Y_raw: dict[str, dict[str, float | None]],
    benchmark_names: list[str],
    model_names: Iterable[str],
) -> dict[str, dict[str, float]]:
    """Convert raw Y[bench][model] to per-benchmark rank percentiles in [0, 1].

    Within each benchmark, scores are ranked ascending. Ties get average rank.
    Missing entries propagate as missing.
    """
    models = list(model_names)
    out: dict[str, dict[str, float]] = {b: {} for b in benchmark_names}
    for bench in benchmark_names:
        row = Y_raw.get(bench, {})
        defined: list[tuple[float, str]] = []
        for model in models:
            v = row.get(model)
            if v is None:
                continue
            try:
                defined.append((float(v), model))
            except (TypeError, ValueError):
                continue
        if not defined:
            continue
        defined.sort(key=lambda x: x[0])
        n = len(defined)
        # Average-rank handling for ties.
        ranks: dict[str, float] = {}
        i = 0
        while i < n:
            j = i
            while j + 1 < n and defined[j + 1][0] == defined[i][0]:
                j += 1
            avg_rank = (i + j) / 2.0  # 0-indexed average rank
            for idx in range(i, j + 1):
                ranks[defined[idx][1]] = avg_rank
            i = j + 1
        denom = max(1, n - 1)
        for model, rank in ranks.items():
            out[bench][model] = float(rank) / float(denom)
    return out


def build_profile_percentile(
    Y_raw: dict[str, dict[str, float | None]],
    T: dict[str, dict[str, float]],
    benchmark_names: list[str],
    model_names: Iterable[str],
) -> dict[str, dict[str, float]]:
    """Profile in [0, +) by mapping Y to per-benchmark rank percentiles first."""
    Y_pct = y_to_percentile(Y_raw, benchmark_names, model_names)
    out: dict[str, dict[str, float]] = {}
    for model in model_names:
        score_vec: dict[str, float] = {}
        for bench in benchmark_names:
            v = Y_pct.get(bench, {}).get(model)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            score_vec[bench] = float(v)
        out[model] = _profile_from_score_vec(score_vec, T)
    return out


def build_profiles_both_modes(
    *,
    Y_norm: dict[str, dict[str, float]],
    Y_raw: dict[str, dict[str, float | None]],
    T: dict[str, dict[str, float]],
    benchmark_names: list[str],
    model_names: Iterable[str],
) -> dict:
    """Convenience wrapper used by the report saver.

    Returns:
        {"relative": {model: profile}, "percentile": {model: profile}}
    """
    models = list(model_names)
    return {
        "relative": build_profile_relative(Y_norm, T, benchmark_names, models),
        "percentile": build_profile_percentile(Y_raw, T, benchmark_names, models),
    }
