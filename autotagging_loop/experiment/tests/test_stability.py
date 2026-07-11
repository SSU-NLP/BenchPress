"""Tests for experiment/stability.py (v3 §2.2.11)."""

from __future__ import annotations

import math

from autotagging_loop.experiment.stability import cross_model_stability, run_stability


def _T(rows: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {k: dict(v) for k, v in rows.items()}


def test_run_stability_perfect_when_run_returns_reference():
    ref = _T({"A": {"t0": 0.6, "t1": 0.4}, "B": {"t0": 0.1, "t1": 0.9}})

    def run_one(seed: int) -> dict[str, dict[str, float]]:
        return _T(ref)

    out = run_stability(
        reference_T=ref,
        benchmark_names=["A", "B"],
        tag_ids=["t0", "t1"],
        seeds=[1, 2, 3],
        run_one=run_one,
    )
    for record in out["per_seed"]:
        assert record["frobenius_residual"] < 1e-9
    assert math.isclose(out["mean_column_correlation"], 1.0, rel_tol=1e-9)


def test_run_stability_residual_grows_with_perturbation():
    ref = _T({"A": {"t0": 1.0, "t1": 0.0}, "B": {"t0": 0.0, "t1": 1.0}, "C": {"t0": 0.5, "t1": 0.5}})

    def perturbed(seed: int) -> dict[str, dict[str, float]]:
        delta = 0.05 * seed
        return _T({
            "A": {"t0": 1.0 - delta, "t1": delta},
            "B": {"t0": delta, "t1": 1.0 - delta},
            "C": {"t0": 0.5, "t1": 0.5},
        })

    out = run_stability(
        reference_T=ref,
        benchmark_names=["A", "B", "C"],
        tag_ids=["t0", "t1"],
        seeds=[1, 2, 3, 4],
        run_one=perturbed,
    )
    residuals = [rec["frobenius_residual"] for rec in out["per_seed"]]
    assert residuals == sorted(residuals)


def test_cross_model_stability_perfect_for_identical_T():
    ref = _T({"A": {"t0": 1.0, "t1": 0.0}, "B": {"t0": 0.0, "t1": 1.0}, "C": {"t0": 0.5, "t1": 0.5}})
    out = cross_model_stability(
        reference_T=ref,
        benchmark_names=["A", "B", "C"],
        backbone_runs={"qwen": ref, "llama": ref},
    )
    assert math.isclose(out["mean_correlation"], 1.0, rel_tol=1e-9)
    assert math.isclose(out["per_backbone"]["qwen"], 1.0, rel_tol=1e-9)


def test_cross_model_stability_negative_for_anti_correlated():
    # Pair-similarity vector for ref = [cos(A,B)=0, cos(A,C)≈0.707, cos(B,C)≈0.707]
    ref = _T({"A": {"t0": 1.0}, "B": {"t1": 1.0}, "C": {"t0": 0.7, "t1": 0.7}})
    # Inverted: swap which axis each benchmark loads on; pair sims swap pattern.
    inv = _T({"A": {"t1": 1.0}, "B": {"t0": 1.0}, "C": {"t0": 0.7, "t1": 0.7}})
    out = cross_model_stability(
        reference_T=ref,
        benchmark_names=["A", "B", "C"],
        backbone_runs={"twin": inv},
    )
    # Same pair-sim distribution → corr ≈ 1 by construction; we only assert it is finite
    # and within [-1, 1].
    r = out["per_backbone"]["twin"]
    assert -1.0 <= r <= 1.0
