from __future__ import annotations

from autotagging_loop.runner.alignment import compute_metrics
from autotagging_loop.runner.score_matrix import normalize_matrix, spearman_pair_matrix


def test_score_matrix_and_alignment_metrics_are_independent():
    Y = {
        "BenchA": {"m1": 0.1, "m2": 0.2, "m3": 0.3},
        "BenchB": {"m1": 0.2, "m2": 0.3, "m3": 0.4},
        "BenchC": {"m1": 0.9, "m2": 0.2, "m3": 0.1},
    }
    Y_norm = normalize_matrix(Y, method="rank")
    R, common = spearman_pair_matrix(Y_norm, ["BenchA", "BenchB", "BenchC"], min_common=3)
    T = {
        "BenchA": {"reasoning": 1.0, "knowledge": 0.0},
        "BenchB": {"reasoning": 1.0, "knowledge": 0.0},
        "BenchC": {"reasoning": 0.0, "knowledge": 1.0},
    }

    metrics, S, residuals = compute_metrics(T, ["BenchA", "BenchB", "BenchC"], R, bootstrap_B=5, seed=0)

    assert common[("BenchA", "BenchB")] == 3
    assert S[("BenchA", "BenchB")] == 1.0
    assert metrics["n_pairs"] == 3
    assert residuals
