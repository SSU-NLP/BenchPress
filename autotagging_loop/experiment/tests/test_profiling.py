"""Tests for experiment/profiling.py (v3 §2.2.9 sign handling)."""

from __future__ import annotations

import math

from autotagging_loop.experiment.profiling import (
    build_profile_percentile,
    build_profile_relative,
    build_profiles_both_modes,
    y_to_percentile,
)


def test_relative_profile_can_be_negative():
    Y_norm = {"BenchA": {"m1": -0.5}, "BenchB": {"m1": 0.4}}
    T = {"BenchA": {"reasoning": 1.0}, "BenchB": {"recall": 1.0}}
    profiles = build_profile_relative(Y_norm, T, ["BenchA", "BenchB"], ["m1"])
    p1 = profiles["m1"]
    assert math.isclose(p1["reasoning"], -0.5, rel_tol=1e-9)
    assert math.isclose(p1["recall"], 0.4, rel_tol=1e-9)


def test_y_to_percentile_yields_zero_to_one():
    Y_raw = {"B1": {"m1": 0.1, "m2": 0.5, "m3": 0.9}}
    pct = y_to_percentile(Y_raw, ["B1"], ["m1", "m2", "m3"])
    assert pct["B1"]["m1"] == 0.0
    assert math.isclose(pct["B1"]["m2"], 0.5, rel_tol=1e-9)
    assert pct["B1"]["m3"] == 1.0


def test_y_to_percentile_ties_get_average_rank():
    Y_raw = {"B1": {"m1": 0.5, "m2": 0.5, "m3": 0.9}}
    pct = y_to_percentile(Y_raw, ["B1"], ["m1", "m2", "m3"])
    # Two tied at rank 0,1 → average 0.5; denominator n-1 = 2 → 0.25.
    assert math.isclose(pct["B1"]["m1"], 0.25, rel_tol=1e-9)
    assert math.isclose(pct["B1"]["m2"], 0.25, rel_tol=1e-9)
    assert pct["B1"]["m3"] == 1.0


def test_percentile_profile_is_nonnegative():
    Y_raw = {"B1": {"m1": 0.1, "m2": 0.9}, "B2": {"m1": 0.5, "m2": 0.5}}
    T = {"B1": {"reasoning": 1.0}, "B2": {"recall": 1.0}}
    profiles = build_profile_percentile(Y_raw, T, ["B1", "B2"], ["m1", "m2"])
    for model_profile in profiles.values():
        for v in model_profile.values():
            assert v >= 0.0


def test_build_profiles_both_modes_returns_both():
    Y_norm = {"B1": {"m1": -0.2}}
    Y_raw = {"B1": {"m1": 0.3, "m2": 0.7}}
    T = {"B1": {"reasoning": 1.0}}
    out = build_profiles_both_modes(
        Y_norm=Y_norm, Y_raw=Y_raw, T=T,
        benchmark_names=["B1"], model_names=["m1", "m2"],
    )
    assert set(out) == {"relative", "percentile"}
    assert "m1" in out["relative"] and "m1" in out["percentile"]
