"""Tests for experiment/weight_optimizer.py."""

from __future__ import annotations

from autotagging_loop.experiment.alignment import alignment_loss, cosine_pair_matrix
from autotagging_loop.experiment.weight_optimizer import optimize_tag_weights


def test_optimizer_reduces_pairwise_loss_for_two_benchmarks():
    T0 = {
        "A": {"x": 1.0, "y": 0.2},
        "B": {"x": 0.9, "y": 0.1},
    }
    R = {("A", "B"): 0.0}
    S0 = cosine_pair_matrix(T0, ["A", "B"])
    before = alignment_loss(S0, R)

    result = optimize_tag_weights(
        T0,
        R,
        ["A", "B"],
        ["x", "y"],
        target_scale="raw",
        l2_lambda=0.0,
        max_iter=100,
    )

    S1 = cosine_pair_matrix(result.T, ["A", "B"])
    after = alignment_loss(S1, R)
    assert result.n_pairs == 1
    assert after < before
    assert result.optimized_loss < result.initial_loss
    for vec in result.T.values():
        assert all(0.0 <= v <= 1.0 for v in vec.values())


def test_negative_raw_targets_are_clipped_under_nonnegative_cosine_constraint():
    T0 = {
        "A": {"x": 1.0, "y": 0.0},
        "B": {"x": 0.0, "y": 1.0},
    }
    result = optimize_tag_weights(
        T0,
        {("A", "B"): -0.8},
        ["A", "B"],
        ["x", "y"],
        target_scale="raw",
        l2_lambda=0.0,
        max_iter=10,
    )

    assert result.clipped_negative_targets == 1
    assert result.n_pairs == 1


def test_signed_bounds_can_fit_negative_raw_target():
    T0 = {
        "A": {"x": 1.0, "y": 0.0},
        "B": {"x": 0.5, "y": 0.5},
    }
    R = {("A", "B"): -0.8}
    result = optimize_tag_weights(
        T0,
        R,
        ["A", "B"],
        ["x", "y"],
        target_scale="raw",
        bounds=(-1.0, 1.0),
        l2_lambda=0.0,
        max_iter=200,
    )
    S = cosine_pair_matrix(result.T, ["A", "B"])

    assert result.clipped_negative_targets == 0
    assert alignment_loss(S, R) < 1e-6
    assert any(v < 0.0 for vec in result.T.values() for v in vec.values())
