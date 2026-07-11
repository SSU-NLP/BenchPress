"""Tests for v3 §2.2.6 Δ_tag>0 hard reject gate (Phase 7)."""

from __future__ import annotations

import json
import os

import pytest

from autotagging_loop.experiment.loop import (
    _candidate_improvement_status,
    _is_better,
    _passes_delta_tag_gate,
    _write_stop_reason,
)
from autotagging_loop.experiment.pipeline.run import (
    _improver_selection_guard_skip_reason,
    _selection_stability_components,
)


def test_gate_rejects_zero_delta_tag():
    cand = {"L_align": 0.01, "rho_align_pearson": 0.9, "delta_tag": 0.0}
    assert _passes_delta_tag_gate(cand) is False


def test_gate_rejects_negative_delta_tag():
    cand = {"L_align": 0.01, "rho_align_pearson": 0.9, "delta_tag": -0.05}
    assert _passes_delta_tag_gate(cand) is False


def test_gate_rejects_nan_delta_tag():
    cand = {"L_align": 0.01, "rho_align_pearson": 0.9, "delta_tag": float("nan")}
    assert _passes_delta_tag_gate(cand) is False


def test_gate_accepts_positive_delta_tag():
    cand = {"L_align": 0.01, "rho_align_pearson": 0.9, "delta_tag": 0.001}
    assert _passes_delta_tag_gate(cand) is True


def test_gate_threshold_admits_negative_when_relaxed():
    """Run C ablation: threshold=-0.10 should admit fold1's Δ_tag=-0.0857."""
    cand = {"L_align": 0.01, "rho_align_pearson": 0.9, "delta_tag": -0.0857}
    assert _passes_delta_tag_gate(cand, threshold=-0.10) is True
    assert _passes_delta_tag_gate(cand, threshold=0.0) is False


def test_gate_threshold_tolerates_relaxed_boundary_roundoff():
    cand = {
        "L_align": 0.01,
        "rho_align_pearson": 0.9,
        "delta_tag": -0.10000000000000009,
    }
    assert _passes_delta_tag_gate(cand, threshold=-0.10) is True
    assert _passes_delta_tag_gate({"delta_tag": 0.0}, threshold=0.0) is False


def test_gate_threshold_still_rejects_nan():
    cand = {"L_align": 0.01, "rho_align_pearson": 0.9, "delta_tag": float("nan")}
    assert _passes_delta_tag_gate(cand, threshold=-0.5) is False


def test_is_better_threshold_forwarded():
    """_is_better must forward delta_tag_threshold to the gate."""
    cand = {"L_align": 0.05, "rho_align_pearson": 0.8, "delta_tag": -0.02}
    assert _is_better(cand, None) is False
    assert _is_better(cand, None, delta_tag_threshold=-0.10) is True


def test_is_better_rejects_l_align_improver_with_negative_delta():
    """A candidate that improves L_align but has Δ_tag ≤ 0 must lose to the prior best."""
    prior = {"L_align": 0.05, "rho_align_pearson": 0.8, "delta_tag": 0.02}
    candidate = {"L_align": 0.01, "rho_align_pearson": 0.9, "delta_tag": -0.01}
    assert _is_better(candidate, prior) is False


def test_is_better_seed_candidate_must_pass_gate_too():
    """When best is None, the candidate still must pass the gate."""
    candidate = {"L_align": 0.05, "rho_align_pearson": 0.8, "delta_tag": -0.01}
    assert _is_better(candidate, None) is False


def test_is_better_picks_best_among_gate_passers():
    a = {"L_align": 0.05, "rho_align_pearson": 0.8, "delta_tag": 0.01}
    b = {"L_align": 0.03, "rho_align_pearson": 0.9, "delta_tag": 0.02}
    assert _is_better(b, a) is True


def test_dev_l_align_selection_rejects_missing_dev_metric():
    candidate = {
        "L_align": 0.01,
        "rho_align_pearson": 0.9,
        "delta_tag": 0.02,
    }
    assert _is_better(
        candidate,
        None,
        selection_cfg={"mode": "dev_l_align"},
    ) is False


def test_dev_l_align_selection_does_not_fall_back_to_train_metric():
    prior = {
        "L_align": 0.5,
        "dev_L_align": 0.2,
        "rho_align_pearson": 0.0,
        "dev_rho_pearson": 0.0,
        "delta_tag": 0.02,
    }
    candidate = {
        "L_align": 0.01,
        "rho_align_pearson": 0.9,
        "delta_tag": 0.03,
    }
    assert _is_better(
        candidate,
        prior,
        selection_cfg={"mode": "dev_l_align"},
    ) is False


def test_dev_l_align_selection_rejects_dev_spearman_drop_when_configured():
    prior = {
        "dev_L_align": 0.20,
        "dev_rho_spearman": 0.35,
        "dev_rho_pearson": 0.30,
        "delta_tag": 0.20,
    }
    candidate = {
        "dev_L_align": 0.10,
        "dev_rho_spearman": 0.34,
        "dev_rho_pearson": 0.50,
        "delta_tag": 0.30,
    }

    assert _is_better(
        candidate,
        prior,
        selection_cfg={
            "mode": "dev_l_align",
            "dev_rho_drop_tolerance": 0.0,
        },
    ) is False


def test_dev_l_align_selection_rejects_seed_below_dev_spearman_floor():
    candidate = {
        "dev_L_align": 0.10,
        "dev_rho_spearman": 0.19,
        "dev_rho_pearson": 0.50,
        "delta_tag": 0.30,
    }

    assert _is_better(
        candidate,
        None,
        selection_cfg={
            "mode": "dev_l_align",
            "dev_rho_floor": 0.20,
        },
    ) is False


def test_dev_l_align_selection_allows_dev_spearman_drop_within_tolerance():
    prior = {
        "dev_L_align": 0.20,
        "dev_rho_spearman": 0.35,
        "dev_rho_pearson": 0.30,
        "delta_tag": 0.20,
    }
    candidate = {
        "dev_L_align": 0.10,
        "dev_rho_spearman": 0.34,
        "dev_rho_pearson": 0.50,
        "delta_tag": 0.30,
    }

    assert _is_better(
        candidate,
        prior,
        selection_cfg={
            "mode": "dev_l_align",
            "dev_rho_drop_tolerance": 0.02,
        },
    ) is True


def test_dev_l_align_selection_rejects_train_l_instability_when_configured():
    prior = {
        "dev_L_align": 0.20,
        "dev_rho_spearman": 0.50,
        "train_L_align": 0.05,
        "train_rho_spearman": 0.50,
        "delta_tag": 0.20,
    }
    candidate = {
        "dev_L_align": 0.10,
        "dev_rho_spearman": 0.50,
        "train_L_align": 0.07,
        "train_rho_spearman": 0.50,
        "delta_tag": 0.30,
    }

    assert _is_better(
        candidate,
        prior,
        selection_cfg={
            "mode": "dev_l_align",
            "train_l_increase_tolerance": 0.01,
        },
    ) is False


def test_dev_l_align_selection_rejects_train_spearman_drop_when_configured():
    prior = {
        "dev_L_align": 0.20,
        "dev_rho_spearman": 0.50,
        "train_L_align": 0.05,
        "train_rho_spearman": 0.50,
        "delta_tag": 0.20,
    }
    candidate = {
        "dev_L_align": 0.10,
        "dev_rho_spearman": 0.50,
        "train_L_align": 0.05,
        "train_rho_spearman": 0.39,
        "delta_tag": 0.30,
    }

    assert _is_better(
        candidate,
        prior,
        selection_cfg={
            "mode": "dev_l_align",
            "train_rho_drop_tolerance": 0.10,
        },
    ) is False


def test_dev_l_align_selection_allows_train_instability_within_tolerance():
    prior = {
        "dev_L_align": 0.20,
        "dev_rho_spearman": 0.50,
        "train_L_align": 0.05,
        "train_rho_spearman": 0.50,
        "delta_tag": 0.20,
    }
    candidate = {
        "dev_L_align": 0.10,
        "dev_rho_spearman": 0.50,
        "train_L_align": 0.055,
        "train_rho_spearman": 0.45,
        "delta_tag": 0.30,
    }

    assert _is_better(
        candidate,
        prior,
        selection_cfg={
            "mode": "dev_l_align",
            "train_l_increase_tolerance": 0.01,
            "train_rho_drop_tolerance": 0.10,
            "train_rho_floor": 0.20,
        },
    ) is True


def test_dev_l_align_selection_rejects_model_probe_spearman_floor():
    candidate = {
        "dev_L_align": 0.10,
        "dev_rho_spearman": 0.50,
        "train_L_align": 0.05,
        "train_rho_spearman": 0.50,
        "model_probe_dev_rho_spearman_min": -0.05,
        "delta_tag": 0.30,
    }

    assert _is_better(
        candidate,
        None,
        selection_cfg={
            "mode": "dev_l_align",
            "model_probe_dev_rho_floor": 0.0,
        },
    ) is False


def test_dev_l_align_selection_rejects_model_probe_spearman_drop():
    prior = {
        "dev_L_align": 0.20,
        "dev_rho_spearman": 0.50,
        "model_probe_dev_rho_spearman_min": 0.45,
        "delta_tag": 0.20,
    }
    candidate = {
        "dev_L_align": 0.10,
        "dev_rho_spearman": 0.50,
        "model_probe_dev_rho_spearman_min": 0.10,
        "delta_tag": 0.30,
    }

    assert _is_better(
        candidate,
        prior,
        selection_cfg={
            "mode": "dev_l_align",
            "model_probe_dev_rho_drop_tolerance": 0.30,
        },
    ) is False


def test_dev_l_align_selection_rejects_model_probe_l_instability():
    prior = {
        "dev_L_align": 0.20,
        "dev_rho_spearman": 0.50,
        "model_probe_dev_L_align_mean": 0.25,
        "delta_tag": 0.20,
    }
    candidate = {
        "dev_L_align": 0.10,
        "dev_rho_spearman": 0.50,
        "model_probe_dev_L_align_mean": 0.40,
        "delta_tag": 0.30,
    }

    assert _is_better(
        candidate,
        prior,
        selection_cfg={
            "mode": "dev_l_align",
            "model_probe_dev_l_increase_tolerance": 0.10,
        },
    ) is False


def test_dev_l_align_selection_uses_model_probe_l_max_for_instability():
    prior = {
        "dev_L_align": 0.20,
        "dev_rho_spearman": 0.50,
        "model_probe_dev_L_align_mean": 0.20,
        "model_probe_dev_L_align_max": 0.30,
        "delta_tag": 0.20,
    }
    candidate = {
        "dev_L_align": 0.10,
        "dev_rho_spearman": 0.50,
        "model_probe_dev_L_align_mean": 0.19,
        "model_probe_dev_L_align_max": 0.45,
        "delta_tag": 0.30,
    }

    assert _is_better(
        candidate,
        prior,
        selection_cfg={
            "mode": "dev_l_align",
            "model_probe_dev_l_increase_tolerance": 0.10,
        },
    ) is False


def test_selection_rejects_candidate_outside_tag_count_range():
    candidate = {
        "dev_L_align": 0.01,
        "dev_selection_score": 0.01,
        "dev_rho_spearman": 0.50,
        "dev_rho_pearson": 0.50,
        "delta_tag": 0.30,
        "tag_count": 2,
    }

    assert _is_better(
        candidate,
        None,
        selection_cfg={
            "mode": "dev_l_align",
            "objective_key": "dev_selection_score",
            "tag_count_min": 4,
            "tag_count_max": 14,
        },
    ) is False


def test_selection_uses_penalized_objective_key_before_raw_l_align():
    prior = {
        "dev_L_align": 0.10,
        "dev_selection_score": 0.10,
        "dev_rho_spearman": 0.50,
        "dev_rho_pearson": 0.50,
        "delta_tag": 0.30,
        "tag_count": 8,
    }
    candidate = {
        "dev_L_align": 0.09,
        "dev_selection_score": 0.12,
        "dev_rho_spearman": 0.60,
        "dev_rho_pearson": 0.60,
        "delta_tag": 0.40,
        "tag_count": 14,
    }

    assert _is_better(
        candidate,
        prior,
        selection_cfg={
            "mode": "dev_l_align",
            "objective_key": "dev_selection_score",
            "tag_count_min": 4,
            "tag_count_max": 14,
        },
    ) is False


def test_dev_stability_selection_uses_stability_objective_before_raw_dev_l():
    prior = {
        "dev_L_align": 0.20,
        "stability_selection_score": 0.20,
        "dev_rho_spearman": 0.50,
        "delta_tag": 0.30,
    }
    candidate = {
        "dev_L_align": 0.10,
        "stability_selection_score": 0.25,
        "dev_rho_spearman": 0.60,
        "delta_tag": 0.40,
    }

    assert _is_better(
        candidate,
        prior,
        selection_cfg={
            "mode": "dev_stability_l_align",
            "objective_key": "stability_selection_score",
        },
    ) is False


def test_dev_stability_selection_prefers_weakest_rho_before_l_align():
    prior = {
        "dev_L_align": 0.20,
        "stability_selection_rho_min": 0.30,
        "stability_selection_l_max": 0.20,
        "delta_tag": 0.30,
    }
    candidate = {
        "dev_L_align": 0.10,
        "stability_selection_rho_min": 0.45,
        "stability_selection_l_max": 0.30,
        "delta_tag": 0.40,
    }

    assert _is_better(
        candidate,
        prior,
        selection_cfg={"mode": "dev_stability_l_align"},
    ) is True


def test_dev_stability_selection_rejects_train_negative_when_floor_configured():
    candidate = {
        "dev_L_align": 0.05,
        "dev_rho_spearman": 0.80,
        "train_L_align": 0.70,
        "train_rho_spearman": -0.30,
        "stability_selection_rho_min": -0.30,
        "stability_selection_l_max": 0.70,
        "delta_tag": 0.80,
    }

    assert _is_better(
        candidate,
        None,
        selection_cfg={
            "mode": "dev_stability_l_align",
            "train_rho_floor": 0.0,
        },
    ) is False


def test_improver_selection_guard_skips_train_negative_candidate():
    reason = _improver_selection_guard_skip_reason(
        m_dev={"delta_tag": 0.80, "rho_align_spearman": 0.70},
        m_train={"rho_align_spearman": -0.30},
        model_probe_dev=None,
        config={
            "delta_tag_threshold": -0.10,
            "best_iter_dev_rho_floor": 0.0,
            "best_iter_train_rho_floor": 0.0,
        },
    )

    assert reason == "train_rho_floor_failed"


def test_dev_stability_selection_uses_l_align_as_tie_breaker():
    prior = {
        "dev_L_align": 0.20,
        "stability_selection_rho_min": 0.30,
        "stability_selection_l_max": 0.20,
        "delta_tag": 0.30,
    }
    candidate = {
        "dev_L_align": 0.10,
        "stability_selection_rho_min": 0.30,
        "stability_selection_l_max": 0.30,
        "delta_tag": 0.40,
    }

    assert _is_better(
        candidate,
        prior,
        selection_cfg={"mode": "dev_stability_l_align"},
    ) is False


def test_stability_selection_score_uses_worst_l_and_weakest_rho():
    components = _selection_stability_components(
        dev_metrics={"L_align": 0.10, "rho_align_spearman": 0.70},
        train_metrics={"L_align": 0.30, "rho_align_spearman": 0.40},
        model_probe_dev_metrics={
            "L_align_mean": 0.25,
            "L_align_max": 0.40,
            "rho_align_spearman_min": 0.20,
        },
        tag_count_penalty=0.02,
        config={"best_iter_stability_rho_weight": 0.50},
    )

    assert components["l_max"] == pytest.approx(0.42)
    assert components["rho_min"] == pytest.approx(0.20)
    assert components["score"] == pytest.approx(0.32)


def test_write_stop_reason_creates_json(tmp_path):
    run_dir = str(tmp_path)
    _write_stop_reason(
        run_dir,
        stalled_delta_tag=True,
        consecutive_no_improve=2,
        threshold=2,
    )
    out = os.path.join(run_dir, "final", "stop_reason.json")
    assert os.path.exists(out)
    with open(out) as fh:
        payload = json.load(fh)
    assert payload["status"] == "stalled_delta_tag"
    assert payload["stalled_delta_tag"] is True
    assert payload["consecutive_no_improve"] == 2


def test_is_better_static_seed_beats_worse_v_loop_iter():
    """2026-05-12 selector fix: with static iter seeded as `best`, a v_loop
    iter that has higher train L_align must NOT replace it, even if it passes
    the gate. This is the bug Run C fold1 hit before the fix.
    """
    static_seed = {
        "L_align": 0.2588,  # train-split L for static iter
        "rho_align_pearson": 0.3168,
        "delta_tag": 0.3020,
    }
    v_loop_candidate = {
        "L_align": 0.4001,  # higher train L
        "rho_align_pearson": 0.1809,
        "delta_tag": -0.0857,
    }
    # v_loop candidate passes the relaxed gate but has worse train L -> reject.
    assert _is_better(v_loop_candidate, static_seed, delta_tag_threshold=-0.10) is False


def test_gate_passing_but_worse_candidate_counts_as_no_improvement():
    static_seed = {
        "L_align": 0.05,
        "rho_align_pearson": 0.4,
        "delta_tag": 0.10,
    }
    worse_v_loop_candidate = {
        "L_align": 0.12,
        "rho_align_pearson": 0.2,
        "delta_tag": 0.20,
    }

    is_better, gate_pass, reason = _candidate_improvement_status(
        worse_v_loop_candidate,
        static_seed,
    )

    assert is_better is False
    assert gate_pass is True
    assert reason == "not_better_than_current_best"


def test_gate_passing_seed_that_fails_selection_guard_is_not_best():
    candidate = {
        "dev_L_align": 0.05,
        "dev_rho_spearman": 0.10,
        "dev_rho_pearson": 0.40,
        "delta_tag": 0.20,
    }

    is_better, gate_pass, reason = _candidate_improvement_status(
        candidate,
        None,
        selection_cfg={
            "mode": "dev_l_align",
            "dev_rho_floor": 0.20,
        },
    )

    assert is_better is False
    assert gate_pass is True
    assert reason == "selection_gate_failed"


def test_is_better_static_seed_yields_to_better_v_loop_iter():
    """Selector fix counterpart: when a v_loop iter genuinely improves train
    L_align over the seeded static, it should win.
    """
    static_seed = {
        "L_align": 0.2588,
        "rho_align_pearson": 0.3168,
        "delta_tag": 0.3020,
    }
    v_loop_candidate = {
        "L_align": 0.1500,
        "rho_align_pearson": 0.5000,
        "delta_tag": 0.2000,
    }
    assert _is_better(v_loop_candidate, static_seed) is True


def test_write_stop_reason_ok_when_not_stalled(tmp_path):
    run_dir = str(tmp_path)
    _write_stop_reason(
        run_dir,
        stalled_delta_tag=False,
        consecutive_no_improve=0,
        threshold=2,
    )
    with open(os.path.join(run_dir, "final", "stop_reason.json")) as fh:
        payload = json.load(fh)
    assert payload["status"] == "ok"
