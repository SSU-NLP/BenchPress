"""v3 §2.2.11 run-stability and cross-model stability.

`run_stability` re-runs (Executer + Maker) at fixed `I*` with `n_runs`
different decoding seeds and reports orthogonal-Procrustes column-wise
correlation plus Frobenius residual against the reference T.

`cross_model_stability` swaps the maker / executer backbone (e.g. swap the
self-hosted model id) and reports the Pearson correlation between the
upper-triangular tag-similarity matrices.

Both functions take callable `_run_one(seed)` factories so they remain
decoupled from the loop. The loop wires them in Phase 7+.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Callable

import numpy as np


def _T_to_matrix(
    T: dict[str, dict[str, float]],
    benchmark_names: list[str],
    tag_ids: list[str],
) -> np.ndarray:
    n = len(benchmark_names)
    k = len(tag_ids)
    out = np.zeros((n, k), dtype=float)
    for i, bench in enumerate(benchmark_names):
        row = T.get(bench, {})
        for j, tag in enumerate(tag_ids):
            v = row.get(tag)
            if v is None:
                continue
            try:
                out[i, j] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def _column_correlations(A: np.ndarray, B: np.ndarray) -> list[float]:
    out: list[float] = []
    for j in range(A.shape[1]):
        a = A[:, j]
        b = B[:, j]
        if a.std() < 1e-12 or b.std() < 1e-12:
            out.append(float("nan"))
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        out.append(r)
    return out


def _orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return Q (k×k) minimizing ||A Q - B||_F. SVD on A^T B."""
    M = A.T @ B
    u, _s, vh = np.linalg.svd(M, full_matrices=False)
    return u @ vh


def _frobenius(M: np.ndarray) -> float:
    return float(np.linalg.norm(M, ord="fro"))


def _pair_similarity_upper(T: dict[str, dict[str, float]], names: list[str]) -> np.ndarray:
    n = len(names)
    if n < 2:
        return np.zeros((0,), dtype=float)
    vecs: list[dict[str, float]] = [T.get(b, {}) for b in names]
    norms = [math.sqrt(sum(v * v for v in vec.values())) or 1.0 for vec in vecs]
    out: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            keys = set(vecs[i]) | set(vecs[j])
            if not keys:
                out.append(0.0)
                continue
            dot = sum(float(vecs[i].get(k, 0.0)) * float(vecs[j].get(k, 0.0)) for k in keys)
            out.append(dot / (norms[i] * norms[j]))
    return np.asarray(out, dtype=float)


def run_stability(
    *,
    reference_T: dict[str, dict[str, float]],
    benchmark_names: list[str],
    tag_ids: list[str],
    seeds: Iterable[int],
    run_one: Callable[[int], dict[str, dict[str, float]]],
) -> dict:
    """Procrustes-aligned column correlations and Frobenius residual across seeds.

    `run_one(seed) -> T` re-executes the pinned (I*, V*) at the given seed.
    """
    seed_list = [int(s) for s in seeds]
    if not seed_list:
        return {"per_seed": [], "mean_column_correlation": float("nan"), "frobenius_residuals": []}
    A = _T_to_matrix(reference_T, benchmark_names, tag_ids)
    per_seed: list[dict] = []
    frobs: list[float] = []
    column_corrs_all: list[list[float]] = []
    for seed in seed_list:
        T_run = run_one(seed)
        B = _T_to_matrix(T_run, benchmark_names, tag_ids)
        Q = _orthogonal_procrustes(A, B)
        residual = _frobenius(A @ Q - B)
        col_corrs = _column_correlations(A @ Q, B)
        finite = [c for c in col_corrs if not math.isnan(c)]
        mean_corr = float(sum(finite) / len(finite)) if finite else float("nan")
        per_seed.append(
            {
                "seed": seed,
                "frobenius_residual": residual,
                "mean_column_correlation": mean_corr,
                "column_correlations": col_corrs,
            }
        )
        frobs.append(residual)
        column_corrs_all.append(col_corrs)
    flat = [c for col in column_corrs_all for c in col if not math.isnan(c)]
    return {
        "per_seed": per_seed,
        "mean_column_correlation": float(sum(flat) / len(flat)) if flat else float("nan"),
        "frobenius_residuals": frobs,
    }


def cross_model_stability(
    *,
    reference_T: dict[str, dict[str, float]],
    benchmark_names: list[str],
    backbone_runs: dict[str, dict[str, dict[str, float]]],
) -> dict:
    """Pearson correlation between upper-triangular pair-similarity vectors.

    `backbone_runs[name] = T_for_that_backbone`.
    """
    ref_vec = _pair_similarity_upper(reference_T, benchmark_names)
    if ref_vec.size == 0:
        return {"per_backbone": {}, "mean_correlation": float("nan")}
    per_backbone: dict[str, float] = {}
    corrs: list[float] = []
    for name, T_other in backbone_runs.items():
        other_vec = _pair_similarity_upper(T_other, benchmark_names)
        if other_vec.shape != ref_vec.shape:
            per_backbone[name] = float("nan")
            continue
        if ref_vec.std() < 1e-12 or other_vec.std() < 1e-12:
            per_backbone[name] = float("nan")
            continue
        r = float(np.corrcoef(ref_vec, other_vec)[0, 1])
        per_backbone[name] = r
        corrs.append(r)
    return {
        "per_backbone": per_backbone,
        "mean_correlation": float(sum(corrs) / len(corrs)) if corrs else float("nan"),
    }
