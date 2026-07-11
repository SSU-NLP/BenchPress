"""Tests for experiment/score_matrix.py."""

from __future__ import annotations

import math

from scipy.stats import spearmanr

from autotagging_loop.experiment.score_matrix import (
    normalize_matrix,
    spearman_pair_matrix,
    to_R01,
    valid_pairs,
)


def test_rank_normalization_known_values():
    Y = {"A": {"m1": 0.1, "m2": 0.5, "m3": 0.9}}
    out = normalize_matrix(Y, method="rank")
    vals = sorted(out["A"].values())
    assert vals == [0.0, 0.5, 1.0]


def test_zscore_normalization_zero_std():
    Y = {"A": {"m1": 0.5, "m2": 0.5, "m3": 0.5}}
    out = normalize_matrix(Y, method="zscore")
    assert all(abs(v) < 1e-9 for v in out["A"].values())


def test_spearman_min_common(recwarn):
    Y_norm = {
        "A": {"m1": 0.1, "m2": 0.2, "m3": 0.3},
        "B": {"m1": 0.3, "m2": 0.2, "m3": 0.1, "m4": 0.0},
        "C": {"m1": 0.1, "m2": 0.2},
    }
    R, common = spearman_pair_matrix(Y_norm, ["A", "B", "C"], min_common=3, warn_below=2)
    assert common[("A", "B")] == 3
    assert R[("A", "B")] is not None
    assert R[("A", "C")] is None
    assert common[("A", "C")] == 2


def test_spearman_pair_matrix_canonicalizes_unsorted_input():
    Y_norm = {
        "A": {"m1": 0.1, "m2": 0.2, "m3": 0.3},
        "B": {"m1": 0.3, "m2": 0.2, "m3": 0.1},
        "C": {"m1": 0.2, "m2": 0.3, "m3": 0.1},
    }
    R, common = spearman_pair_matrix(
        Y_norm,
        ["C", "A", "B"],
        min_common=3,
        warn_below=2,
    )
    assert list(R) == [("A", "B"), ("A", "C"), ("B", "C")]
    assert list(common) == [("A", "B"), ("A", "C"), ("B", "C")]


def test_spearman_matches_scipy():
    Y_norm = {
        "A": {"m1": 0.1, "m2": 0.5, "m3": 0.9, "m4": 0.7, "m5": 0.3, "m6": 0.4},
        "B": {"m1": 0.2, "m2": 0.4, "m3": 0.8, "m4": 0.6, "m5": 0.5, "m6": 0.1},
    }
    R, _ = spearman_pair_matrix(Y_norm, ["A", "B"], min_common=3, warn_below=2)
    expected, _ = spearmanr(
        [Y_norm["A"][f"m{i}"] for i in range(1, 7)],
        [Y_norm["B"][f"m{i}"] for i in range(1, 7)],
    )
    assert math.isclose(R[("A", "B")], float(expected), rel_tol=1e-9)


def test_to_R01_in_range():
    R = {("A", "B"): -1.0, ("A", "C"): 0.0, ("B", "C"): 1.0, ("X", "Y"): None}
    R01 = to_R01(R)
    assert R01[("A", "B")] == 0.0
    assert R01[("A", "C")] == 0.5
    assert R01[("B", "C")] == 1.0
    assert R01[("X", "Y")] is None


def test_valid_pairs_p_lt_q():
    R = {("A", "B"): 0.5, ("A", "C"): None, ("B", "C"): 0.3}
    pairs = valid_pairs(R)
    for p, q in pairs:
        assert p < q
    assert ("A", "C") not in pairs
    assert ("A", "B") in pairs
