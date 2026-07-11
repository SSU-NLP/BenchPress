"""Tests for experiment/power_analysis.py (offline §2.2.7 power simulator)."""

from __future__ import annotations

import json

from autotagging_loop.experiment.power_analysis import (
    PowerCell,
    _simulate_pair_matrix,
    power_cell,
    run_grid,
    save_grid,
)
import numpy as np


def test_simulate_pair_matrix_shape_and_correlation():
    rng = np.random.default_rng(0)
    s, r = _simulate_pair_matrix(n_benchmarks=20, n_models=15, delta_rho=0.7, rng=rng)
    n_pairs = 20 * 19 // 2
    assert s.shape == (n_pairs,)
    assert r.shape == (n_pairs,)
    # With Δρ=0.7 the planted shared component should yield positive Spearman
    rho = np.corrcoef(s, r)[0, 1]
    assert rho > 0.3


def test_simulate_pair_matrix_uncorrelated_for_zero_delta():
    rng = np.random.default_rng(1)
    rhos = []
    for _ in range(20):
        s, r = _simulate_pair_matrix(n_benchmarks=30, n_models=15, delta_rho=0.0, rng=rng)
        rhos.append(np.corrcoef(s, r)[0, 1])
    assert abs(float(np.mean(rhos))) < 0.15


def test_power_cell_higher_for_stronger_signal():
    weak = power_cell(
        n_benchmarks=20, n_models=15, delta_rho=0.05,
        n_trials=20, perm_B=200, seed=0,
    )
    strong = power_cell(
        n_benchmarks=20, n_models=15, delta_rho=0.7,
        n_trials=20, perm_B=200, seed=0,
    )
    assert isinstance(weak, PowerCell)
    assert strong.power_spearman > weak.power_spearman


def test_run_grid_writes_one_cell_per_combo(tmp_path):
    cells = run_grid(
        n_benchmarks_grid=[10, 20],
        n_models_grid=[10],
        delta_rho_grid=[0.0, 0.5],
        n_trials=10,
        perm_B=100,
        seed=0,
    )
    assert len(cells) == 4

    path = save_grid(cells, out_dir=str(tmp_path), tag="unit")
    with open(path) as fh:
        payload = json.load(fh)
    assert payload["n_cells"] == 4
    assert {c["n_benchmarks"] for c in payload["cells"]} == {10, 20}
    assert {c["delta_rho"] for c in payload["cells"]} == {0.0, 0.5}


def test_run_grid_parallel_matches_sequential():
    """workers>1 must return the same cells (same order, same metadata) as workers=1."""
    grid_kwargs = dict(
        n_benchmarks_grid=[10, 15],
        n_models_grid=[8],
        delta_rho_grid=[0.0, 0.3, 0.6],
        n_trials=8,
        perm_B=64,
        seed=7,
    )
    seq = run_grid(**grid_kwargs, workers=1)
    par = run_grid(**grid_kwargs, workers=2)
    assert len(seq) == len(par) == 6
    for a, b in zip(seq, par):
        assert (a.n_benchmarks, a.n_models, a.delta_rho) == (
            b.n_benchmarks,
            b.n_models,
            b.delta_rho,
        )
        # Each cell is seeded by its grid position, so values are deterministic.
        assert a.power_spearman == b.power_spearman
        assert a.power_paired_perm == b.power_paired_perm
