"""experiment/score_matrix.py — Y 정규화 + 페어 Spearman + R01."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from scipy.stats import spearmanr


def normalize_zscore(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    std = float(arr.std(ddof=0))
    if std <= 0.0:
        return [0.0 for _ in values]
    return ((arr - float(arr.mean())) / std).tolist()


def _rank_norm(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return []
    # average ranks for ties
    order = arr.argsort()
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    # tie handling: average rank over equal-value groups
    sorted_vals = arr[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j > i + 1:
            avg = (ranks[order[i]] + ranks[order[j - 1]]) / 2.0
            for k in range(i, j):
                ranks[order[k]] = avg
        i = j
    if n == 1:
        return [0.5]
    return (ranks / (n - 1)).tolist()


def normalize_matrix(
    Y: dict[str, dict[str, float]],
    method: Literal["zscore", "rank"] = "rank",
) -> dict[str, dict[str, float]]:
    """Per-benchmark normalization of model scores. Missing models stay missing."""
    out: dict[str, dict[str, float]] = {}
    for bench, scores in Y.items():
        if not scores:
            out[bench] = {}
            continue
        models = list(scores.keys())
        vals = [float(scores[m]) for m in models]
        if method == "rank":
            norm = _rank_norm(vals)
        elif method == "zscore":
            norm = normalize_zscore(vals)
        else:
            raise ValueError(f"unknown normalize method: {method}")
        out[bench] = {m: float(v) for m, v in zip(models, norm)}
    return out


def spearman_pair_matrix(
    Y_norm: dict[str, dict[str, float]],
    benchmark_names: list[str],
    min_common: int = 6,
    warn_below: int = 5,
) -> tuple[dict[tuple[str, str], float | None], dict[tuple[str, str], int]]:
    """Pairwise Spearman over common models. Pairs with too few common models -> None."""
    R: dict[tuple[str, str], float | None] = {}
    common_count: dict[tuple[str, str], int] = {}
    benchmark_names = sorted(set(benchmark_names))
    n = len(benchmark_names)
    for i in range(n):
        for j in range(i + 1, n):
            p, q = benchmark_names[i], benchmark_names[j]
            sp, sq = Y_norm.get(p, {}), Y_norm.get(q, {})
            common = sorted(set(sp) & set(sq))
            common_count[(p, q)] = len(common)
            if len(common) < min_common:
                R[(p, q)] = None
                if warn_below <= len(common) < min_common:
                    warnings.warn(
                        f"[score_matrix] pair ({p}, {q}) has {len(common)} common models "
                        f"(<min_common={min_common}); dropped.",
                        stacklevel=2,
                    )
                continue
            vec_p = [sp[m] for m in common]
            vec_q = [sq[m] for m in common]
            if len(set(vec_p)) < 2 or len(set(vec_q)) < 2:
                R[(p, q)] = None
                continue
            rho, _ = spearmanr(vec_p, vec_q)
            R[(p, q)] = float(rho) if not np.isnan(rho) else None
    return R, common_count


def to_R01(R_raw: dict[tuple[str, str], float | None]) -> dict[tuple[str, str], float | None]:
    """Map R_raw ∈ [-1, 1] to R01 ∈ [0, 1] via (R+1)/2."""
    out: dict[tuple[str, str], float | None] = {}
    for k, v in R_raw.items():
        out[k] = None if v is None else (v + 1.0) / 2.0
    return out


def valid_pairs(R: dict[tuple[str, str], float | None]) -> list[tuple[str, str]]:
    """Ω = {(p,q): p<q, R defined}. Tuple keys are already p<q from spearman_pair_matrix."""
    return [k for k, v in R.items() if v is not None]
