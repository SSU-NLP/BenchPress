"""Score normalization and benchmark-pair score-pattern similarity."""

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
    if std <= 0:
        return [0.0 for _ in values]
    return ((arr - float(arr.mean())) / std).tolist()


def rank_norm(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return []
    order = arr.argsort()
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    sorted_vals = arr[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j > i + 1:
            avg = (ranks[order[i]] + ranks[order[j - 1]]) / 2.0
            ranks[order[i:j]] = avg
        i = j
    if n == 1:
        return [0.5]
    return (ranks / (n - 1)).tolist()


def normalize_matrix(
    Y: dict[str, dict[str, float]],
    method: Literal["zscore", "rank"] = "rank",
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for benchmark, scores in Y.items():
        models = list(scores)
        values = [float(scores[m]) for m in models]
        if method == "rank":
            normalized = rank_norm(values)
        elif method == "zscore":
            normalized = normalize_zscore(values)
        else:
            raise ValueError(f"unknown normalize method: {method}")
        out[benchmark] = {model: float(value) for model, value in zip(models, normalized)}
    return out


def spearman_pair_matrix(
    Y_norm: dict[str, dict[str, float]],
    benchmark_names: list[str],
    *,
    min_common: int = 6,
    warn_below: int = 5,
) -> tuple[dict[tuple[str, str], float | None], dict[tuple[str, str], int]]:
    R: dict[tuple[str, str], float | None] = {}
    common_count: dict[tuple[str, str], int] = {}
    for i, p in enumerate(benchmark_names):
        for q in benchmark_names[i + 1:]:
            sp = Y_norm.get(p, {})
            sq = Y_norm.get(q, {})
            common = sorted(set(sp) & set(sq))
            common_count[(p, q)] = len(common)
            if len(common) < min_common:
                R[(p, q)] = None
                if warn_below <= len(common) < min_common:
                    warnings.warn(
                        f"pair ({p}, {q}) has {len(common)} common models",
                        stacklevel=2,
                    )
                continue
            vp = [sp[m] for m in common]
            vq = [sq[m] for m in common]
            if len(set(vp)) < 2 or len(set(vq)) < 2:
                R[(p, q)] = None
                continue
            rho, _ = spearmanr(vp, vq)
            R[(p, q)] = float(rho) if np.isfinite(rho) else None
    return R, common_count


def to_R01(R_raw: dict[tuple[str, str], float | None]) -> dict[tuple[str, str], float | None]:
    return {key: None if value is None else (float(value) + 1.0) / 2.0 for key, value in R_raw.items()}
