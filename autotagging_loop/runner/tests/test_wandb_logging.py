from __future__ import annotations

from autotagging_loop.runner.run import flatten_metrics


def test_flatten_metrics_handles_bootstrap_metrics():
    metrics = {
        "L_align": 0.1,
        "bootstrap": {
            "L_align": {"mean": 0.11, "std": 0.02},
            "rho_pearson": {"mean": 0.5, "std": 0.1},
        },
    }

    flat = flatten_metrics(metrics)

    assert flat["L_align"] == 0.1
    assert flat["bootstrap/L_align/mean"] == 0.11
    assert flat["bootstrap/rho_pearson/std"] == 0.1
