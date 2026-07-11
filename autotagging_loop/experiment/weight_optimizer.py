"""Pairwise tag-weight refinement for Part 1.

Given initial benchmark tag vectors T, tune bounded weights w so pairwise tag
cosine similarities approach empirical score-pattern similarities R_ij.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize


PairKey = tuple[str, str]
TargetScale = Literal["raw", "r01"]


@dataclass
class WeightOptimizationResult:
    T: dict[str, dict[str, float]]
    initial_loss: float
    optimized_loss: float
    n_pairs: int
    target_scale: TargetScale
    bounds: tuple[float, float]
    success: bool
    message: str
    iterations: int
    clipped_negative_targets: int


def _pairwise_cosines(X: np.ndarray, pair_indices: list[tuple[int, int]]) -> np.ndarray:
    values: list[float] = []
    for i, j in pair_indices:
        a = X[i]
        b = X[j]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        values.append(0.0 if denom <= 0.0 else float(np.dot(a, b) / denom))
    return np.asarray(values, dtype=float)


def _target_value(value: float, target_scale: TargetScale) -> float:
    if target_scale == "r01":
        return (value + 1.0) / 2.0
    return value


def optimize_tag_weights(
    T_initial: dict[str, dict[str, float]],
    R_raw: dict[PairKey, float | None],
    benchmark_names: list[str],
    vocab_ids: list[str],
    *,
    target_scale: TargetScale = "raw",
    bounds: tuple[float, float] = (0.0, 1.0),
    l2_lambda: float = 0.01,
    max_iter: int = 200,
) -> WeightOptimizationResult:
    """Optimize T so cos(T_i,T_j) matches R_ij as closely as possible.

    `target_scale="raw"` follows the v3 §1 formula directly. If bounds are
    non-negative, cosine similarity cannot be negative; negative R_ij values are
    then clipped to 0 for the constrained optimization target. With signed
    bounds such as (-1, 1), raw negative R_ij can be optimized directly.
    """
    lower, upper = float(bounds[0]), float(bounds[1])
    if lower > upper:
        raise ValueError(f"invalid weight bounds: {bounds}")
    nonnegative_bounds = lower >= 0.0
    names = [name for name in benchmark_names if name in T_initial]
    tag_ids = list(vocab_ids)
    if not names or not tag_ids:
        return WeightOptimizationResult(
            T=T_initial,
            initial_loss=float("nan"),
            optimized_loss=float("nan"),
            n_pairs=0,
            target_scale=target_scale,
            bounds=(lower, upper),
            success=False,
            message="empty_names_or_vocab",
            iterations=0,
            clipped_negative_targets=0,
        )

    name_to_idx = {name: idx for idx, name in enumerate(names)}
    pair_indices: list[tuple[int, int]] = []
    targets: list[float] = []
    clipped_negative = 0
    for (p, q), raw in R_raw.items():
        if raw is None or p not in name_to_idx or q not in name_to_idx:
            continue
        target = _target_value(float(raw), target_scale)
        if target < 0.0 and nonnegative_bounds:
            clipped_negative += 1
            target = 0.0
        target_lower = 0.0 if nonnegative_bounds else -1.0
        target = min(1.0, max(target_lower, target))
        pair_indices.append((name_to_idx[p], name_to_idx[q]))
        targets.append(target)

    if not pair_indices:
        return WeightOptimizationResult(
            T=T_initial,
            initial_loss=float("nan"),
            optimized_loss=float("nan"),
            n_pairs=0,
            target_scale=target_scale,
            bounds=(lower, upper),
            success=False,
            message="no_valid_pairs",
            iterations=0,
            clipped_negative_targets=clipped_negative,
        )

    X0 = np.asarray(
        [
            [float(T_initial.get(name, {}).get(tag_id, 0.0)) for tag_id in tag_ids]
            for name in names
        ],
        dtype=float,
    )
    X0 = np.clip(X0, lower, upper)
    target_arr = np.asarray(targets, dtype=float)

    def objective(flat: np.ndarray) -> float:
        X = flat.reshape(X0.shape)
        sims = _pairwise_cosines(X, pair_indices)
        mse = float(np.mean((sims - target_arr) ** 2))
        if l2_lambda > 0:
            mse += float(l2_lambda) * float(np.mean((X - X0) ** 2))
        return mse

    initial_loss = objective(X0.ravel())
    starts = [X0.ravel()]
    if lower < 0.0:
        starts.append(np.clip(-X0, lower, upper).ravel())
        rng = np.random.default_rng(0)
        for _ in range(4):
            starts.append(rng.uniform(lower, upper, size=X0.size))

    best_result = None
    best_loss = float("inf")
    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=[(lower, upper)] * X0.size,
            options={"maxiter": int(max_iter), "ftol": 1e-12},
        )
        loss = objective(result.x)
        if loss < best_loss:
            best_loss = loss
            best_result = result

    result = best_result
    X_best = np.clip(result.x.reshape(X0.shape), lower, upper)
    optimized_loss = objective(X_best.ravel())
    out: dict[str, dict[str, float]] = {}
    for row_idx, name in enumerate(names):
        out[name] = {
            tag_id: float(X_best[row_idx, col_idx])
            for col_idx, tag_id in enumerate(tag_ids)
        }

    return WeightOptimizationResult(
        T=out,
        initial_loss=float(initial_loss),
        optimized_loss=float(optimized_loss),
        n_pairs=len(pair_indices),
        target_scale=target_scale,
        bounds=(lower, upper),
        success=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit),
        clipped_negative_targets=clipped_negative,
    )
