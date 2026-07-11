"""v3 §2.2.7 offline power analysis.

Simulates score-pattern matrices `R` under a tunable correlation budget Δρ
between a planted "tag-similarity" vector `S` and the induced pair similarities
of `R`. For each (N, R, Δρ) cell on the grid, computes the power of:

  * a single-fold Spearman correlation test, and
  * the paired permutation test on a fold-pair of (random taxonomy vs planted
    taxonomy).

This module is **offline-only**: it does not call any LLM, does not touch
`run_dir`, and is never invoked from `experiment/loop.py`. Run it from a
notebook or script and write the JSON output under `analysis/`.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from scipy.stats import spearmanr

from autotagging_loop.experiment.alignment import paired_permutation_test


@dataclass
class PowerCell:
    n_benchmarks: int
    n_models: int
    delta_rho: float
    n_trials: int
    power_spearman: float
    power_paired_perm: float
    mean_rho: float
    std_rho: float


def _simulate_pair_matrix(
    n_benchmarks: int,
    n_models: int,
    delta_rho: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (S_pair, R_pair) where each is an n*(n-1)/2 vector.

    `S` is sampled uniformly in [0, 1]. `R` is generated so that its
    Spearman ρ with S is approximately `delta_rho`. Concretely, draw a
    latent benchmark vector for each side, compute pair similarities by
    cosine over `n_models` synthetic dimensions, and inject correlation
    via a shared component of weight `delta_rho`.
    """
    if n_benchmarks < 2:
        raise ValueError("n_benchmarks must be ≥ 2")
    if n_models < 2:
        raise ValueError("n_models must be ≥ 2")
    rho = max(0.0, min(1.0, float(delta_rho)))
    shared = rng.normal(0.0, 1.0, size=(n_benchmarks, n_models))
    s_noise = rng.normal(0.0, 1.0, size=(n_benchmarks, n_models))
    r_noise = rng.normal(0.0, 1.0, size=(n_benchmarks, n_models))
    a_s = math.sqrt(rho)
    a_n = math.sqrt(max(0.0, 1.0 - rho))
    s_emb = a_s * shared + a_n * s_noise
    r_emb = a_s * shared + a_n * r_noise

    def pair_cos(emb: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        unit = emb / norms
        out: list[float] = []
        for i in range(n_benchmarks):
            for j in range(i + 1, n_benchmarks):
                out.append(float(unit[i] @ unit[j]))
        return np.asarray(out, dtype=float)

    return pair_cos(s_emb), pair_cos(r_emb)


def power_cell(
    *,
    n_benchmarks: int,
    n_models: int,
    delta_rho: float,
    n_trials: int = 200,
    alpha: float = 0.05,
    perm_B: int = 1000,
    seed: int = 0,
) -> PowerCell:
    """Estimate detection power for a single (N, R, Δρ) grid cell."""
    rng = np.random.default_rng(seed)
    rhos: list[float] = []
    spearman_hits = 0
    perm_hits = 0
    for _ in range(int(n_trials)):
        s_pair, r_pair = _simulate_pair_matrix(n_benchmarks, n_models, delta_rho, rng)
        rho_obs, p_obs = spearmanr(s_pair, r_pair)
        if not np.isnan(rho_obs):
            rhos.append(float(rho_obs))
            if p_obs < alpha:
                spearman_hits += 1
        # Paired-permutation null: a "random" taxonomy gives ρ ≈ 0; the planted
        # taxonomy gives ρ ≈ Δρ. Per fold metric is the per-trial Spearman.
        s_pair_null, _ = _simulate_pair_matrix(n_benchmarks, n_models, 0.0, rng)
        rho_null, _ = spearmanr(s_pair_null, r_pair)
        if np.isnan(rho_null) or np.isnan(rho_obs):
            continue
        # Single-fold paired test is degenerate (n=1). Build a 5-fold pair by
        # repeated sampling so the permutation test has at least 2^5 = 32 sign
        # arrangements.
        a_folds: list[float] = []
        b_folds: list[float] = []
        for _k in range(5):
            sn, _ = _simulate_pair_matrix(n_benchmarks, n_models, 0.0, rng)
            sp, _ = _simulate_pair_matrix(n_benchmarks, n_models, delta_rho, rng)
            r_fold = r_pair  # fixed truth across folds
            rn, _ = spearmanr(sn, r_fold)
            rp, _ = spearmanr(sp, r_fold)
            if np.isnan(rn) or np.isnan(rp):
                continue
            a_folds.append(float(rn))
            b_folds.append(float(rp))
        if len(a_folds) >= 2:
            res = paired_permutation_test(a_folds, b_folds, B=perm_B, seed=seed)
            if res["p_value"] < alpha:
                perm_hits += 1

    rho_arr = np.asarray(rhos, dtype=float) if rhos else np.asarray([float("nan")])
    return PowerCell(
        n_benchmarks=n_benchmarks,
        n_models=n_models,
        delta_rho=delta_rho,
        n_trials=int(n_trials),
        power_spearman=spearman_hits / max(1, int(n_trials)),
        power_paired_perm=perm_hits / max(1, int(n_trials)),
        mean_rho=float(np.nanmean(rho_arr)),
        std_rho=float(np.nanstd(rho_arr, ddof=0)),
    )


def _power_cell_worker(kwargs: dict) -> PowerCell:
    """Top-level adapter so multiprocessing.Pool can pickle the call."""
    return power_cell(**kwargs)


def run_grid(
    *,
    n_benchmarks_grid: list[int],
    n_models_grid: list[int],
    delta_rho_grid: list[float],
    n_trials: int = 200,
    alpha: float = 0.05,
    perm_B: int = 1000,
    seed: int = 0,
    workers: int = 1,
) -> list[PowerCell]:
    """Run a (N, R, Δρ) grid and return power estimates per cell.

    `workers > 1` runs cells in parallel via `multiprocessing.Pool`. Each cell
    has a fixed seed derived from its grid position, so output is independent
    of completion order.
    """
    base_seed = int(seed)
    cell_seed = base_seed
    cell_args: list[dict] = []
    for n in n_benchmarks_grid:
        for r in n_models_grid:
            for dr in delta_rho_grid:
                cell_args.append({
                    "n_benchmarks": int(n),
                    "n_models": int(r),
                    "delta_rho": float(dr),
                    "n_trials": int(n_trials),
                    "alpha": float(alpha),
                    "perm_B": int(perm_B),
                    "seed": cell_seed,
                })
                cell_seed += 1

    nworkers = max(1, int(workers))
    if nworkers <= 1 or len(cell_args) <= 1:
        return [_power_cell_worker(a) for a in cell_args]
    with multiprocessing.Pool(processes=min(nworkers, len(cell_args))) as pool:
        return list(pool.imap(_power_cell_worker, cell_args))


def save_grid(
    cells: list[PowerCell],
    out_dir: str = "analysis",
    *,
    tag: str | None = None,
) -> str:
    """Write the grid to `analysis/power_<UTC date>[_tag].json`."""
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    path = os.path.join(out_dir, f"power_{stamp}{suffix}.json")
    payload = {
        "generated_utc": stamp,
        "n_cells": len(cells),
        "cells": [cell.__dict__ for cell in cells],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path
