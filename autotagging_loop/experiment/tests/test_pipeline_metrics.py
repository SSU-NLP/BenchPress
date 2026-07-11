"""Tests for pipeline metric contracts."""

from __future__ import annotations

import math

from autotagging_loop.experiment.pipeline.run import _compute_metrics, _selection_delta_tag
from autotagging_loop.experiment.split_metrics import compute_split_metrics
from autotagging_loop.experiment.splits import BenchmarkSplit


def test_compute_metrics_delta_uses_only_comparable_pairs():
    T = {
        "A": {"x": 1.0, "y": 0.0},
        "B": {"x": 0.0, "y": 1.0},
        "C": {"x": 1.0, "y": 1.0},
        "D": {"x": 1.0, "y": 0.0},
    }
    R_raw = {
        ("A", "B"): -0.8,
        ("A", "C"): 0.2,
        ("A", "D"): None,
        ("B", "C"): 0.4,
        ("B", "D"): None,
        ("C", "D"): None,
    }
    R01 = {k: (None if v is None else (v + 1.0) / 2.0) for k, v in R_raw.items()}
    names = ["D", "C", "A", "B"]

    _S, metrics, _boot = _compute_metrics(
        T,
        names,
        R_raw,
        R01,
        q_p=0.80,
        q_n=0.20,
        bootstrap_B=0,
        seed=0,
    )
    blocks = compute_split_metrics(
        S=_S,
        R_raw=R_raw,
        R01=R01,
        benchmark_split=BenchmarkSplit(
            train=[],
            dev=names,
            test=[],
            seed=0,
            ratios=(0.0, 1.0, 0.0),
        ),
        q_p=0.80,
        q_n=0.20,
        bootstrap_B=0,
        seed=0,
    )

    assert metrics["n_pairs"] == 3
    for key in ("L_align", "rho_align_spearman", "delta_tag"):
        assert math.isclose(metrics[key], blocks["dev"][key], rel_tol=1e-12)


def test_selection_delta_tag_does_not_repair_non_finite_dev_signal():
    assert math.isnan(_selection_delta_tag({"delta_tag": None}))
    assert math.isnan(_selection_delta_tag({"delta_tag": float("nan")}))
    assert _selection_delta_tag({"delta_tag": -0.05}) == -0.05
