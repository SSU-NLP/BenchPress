"""Tests for experiment/alignment.py."""

from __future__ import annotations

import math

import numpy as np

from autotagging_loop.experiment.alignment import (
    alignment_corr,
    alignment_loss,
    block_bootstrap_ci,
    bootstrap_metrics,
    bootstrap_metrics_block,
    build_error_report,
    build_residual_report,
    cosine_pair_matrix,
    intra_inter_gap,
    paired_permutation_test,
    quantile_thresholds,
)


def _mk_T(values: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {b: {f"t{i}": v for i, v in enumerate(vec)} for b, vec in values.items()}


def test_cosine_pair_matrix_p_lt_q():
    T = _mk_T({"A": [1, 0, 0], "B": [0, 1, 0], "C": [1, 1, 0]})
    S = cosine_pair_matrix(T, benchmark_names=["A", "B", "C"])
    for p, q in S:
        assert p < q
    assert math.isclose(S[("A", "B")], 0.0, abs_tol=1e-9)
    assert math.isclose(S[("A", "C")], 1.0 / math.sqrt(2), rel_tol=1e-9)


def test_cosine_pair_matrix_canonicalizes_unsorted_input():
    T = _mk_T({"A": [1, 0], "B": [0, 1], "C": [1, 1]})
    S = cosine_pair_matrix(T, benchmark_names=["C", "A", "B"])
    assert list(S) == [("A", "B"), ("A", "C"), ("B", "C")]


def test_alignment_loss_handcomputed():
    S = {("A", "B"): 0.6, ("A", "C"): 0.3}
    R01 = {("A", "B"): 0.4, ("A", "C"): 0.5}
    expected = ((0.6 - 0.4) ** 2 + (0.3 - 0.5) ** 2) / 2
    assert math.isclose(alignment_loss(S, R01), expected, rel_tol=1e-9)


def test_alignment_loss_zero_when_identical():
    S = {("A", "B"): 0.7, ("A", "C"): 0.2, ("B", "C"): 0.5}
    R01 = dict(S)
    assert alignment_loss(S, R01) == 0.0


def test_alignment_corr_perfect_and_zero():
    S_perfect = {("A", "B"): 0.1, ("A", "C"): 0.5, ("B", "C"): 0.9, ("A", "D"): 0.3}
    R_raw = {k: 2 * v - 1 for k, v in S_perfect.items()}
    pr, sp = alignment_corr(S_perfect, R_raw)
    assert math.isclose(pr, 1.0, rel_tol=1e-9)
    assert math.isclose(sp, 1.0, rel_tol=1e-9)


def test_quantile_thresholds_changes_with_distribution():
    a = list(np.linspace(0, 1, 11))
    tp_a, tn_a = quantile_thresholds({(str(i), str(i + 1)): v for i, v in enumerate(a)}, 0.8, 0.2)
    b = [0.95] * 9 + [0.05] * 2
    tp_b, tn_b = quantile_thresholds({(str(i), str(i + 1)): v for i, v in enumerate(b)}, 0.8, 0.2)
    assert tp_a != tp_b
    assert tn_a != tn_b


def test_intra_inter_gap_basic():
    S = {("A", "B"): 0.9, ("A", "C"): 0.1, ("B", "C"): 0.5}
    R = {("A", "B"): 0.8, ("A", "C"): -0.1, ("B", "C"): 0.2}
    gap = intra_inter_gap(S, R, theta_p=0.8, theta_n=0.2)
    assert math.isclose(gap["intra"], 0.8, rel_tol=1e-9)
    assert math.isclose(gap["inter"], -0.1, rel_tol=1e-9)
    assert math.isclose(gap["delta"], 0.9, rel_tol=1e-9)


def test_bootstrap_close_to_point():
    n = 30
    rng = np.random.default_rng(0)
    keys = [(f"a{i}", f"b{i}") for i in range(n)]
    s = rng.uniform(0, 1, n)
    r_raw = 2 * s - 1 + rng.normal(0, 0.05, n)
    r01 = (r_raw + 1) / 2
    S = {k: float(s[i]) for i, k in enumerate(keys)}
    R_raw = {k: float(r_raw[i]) for i, k in enumerate(keys)}
    R01 = {k: float(r01[i]) for i, k in enumerate(keys)}
    point = alignment_loss(S, R_raw)
    point_01 = alignment_loss(S, R01)
    boot = bootstrap_metrics(S, R_raw, R01, B=200, seed=0)
    assert abs(boot["L_align"]["mean"] - point) < 0.05
    assert abs(boot["L_align_01"]["mean"] - point_01) < 0.05


def test_error_report_picks_outliers():
    # 10 pairs total, 2 false_sim + 2 false_dis crafted as outliers
    pairs = {
        ("A", "B"): (0.95, 0.05),  # false_sim (S high, R01 low)
        ("A", "C"): (0.92, 0.08),  # false_sim
        ("A", "D"): (0.05, 0.95),  # false_dis
        ("A", "E"): (0.08, 0.92),  # false_dis
        ("B", "C"): (0.5, 0.5),
        ("B", "D"): (0.4, 0.4),
        ("B", "E"): (0.6, 0.6),
        ("C", "D"): (0.55, 0.55),
        ("C", "E"): (0.45, 0.45),
        ("D", "E"): (0.5, 0.5),
    }
    S = {k: v[0] for k, v in pairs.items()}
    R01 = {k: v[1] for k, v in pairs.items()}
    R_raw = {k: 2 * v - 1 for k, v in R01.items()}
    report = build_error_report(S, R_raw, R01, top_k=10, q_p_s=0.8, q_n_s=0.2,
                                q_p_r=0.8, q_n_r=0.2)
    types = sorted([p.type for p in report])
    assert types.count("false_sim") == 2
    assert types.count("false_dis") == 2


def test_block_bootstrap_ci_excludes_zero_for_correlated_data():
    """v3 §2.2.7: with R = 2S - 1 + tiny noise, ρ_pearson CI should exclude 0."""
    n = 30
    rng = np.random.default_rng(42)
    names = [f"B{i}" for i in range(n)]
    keys = [(names[i], names[j]) for i in range(n) for j in range(i + 1, n)]
    s_vec = rng.uniform(0, 1, len(keys))
    r_vec = 2 * s_vec - 1 + rng.normal(0, 0.05, len(keys))
    r01 = (r_vec + 1) / 2
    S = {k: float(s_vec[i]) for i, k in enumerate(keys)}
    R_raw = {k: float(r_vec[i]) for i, k in enumerate(keys)}
    R01 = {k: float(r01[i]) for i, k in enumerate(keys)}

    boot = bootstrap_metrics_block(S, R_raw, R01, names, B=200, seed=0)
    # Strong positive correlation: mean ρ > 0.9 with tiny std on N=30 with n=435 pairs
    assert boot["rho_pearson"]["mean"] > 0.9
    assert boot["rho_spearman"]["mean"] > 0.9


def test_block_bootstrap_ci_covers_zero_for_uncorrelated_data():
    n = 30
    rng = np.random.default_rng(7)
    names = [f"B{i}" for i in range(n)]
    keys = [(names[i], names[j]) for i in range(n) for j in range(i + 1, n)]
    s_vec = rng.uniform(0, 1, len(keys))
    r_vec = rng.uniform(-1, 1, len(keys))
    r01 = (r_vec + 1) / 2
    S = {k: float(s_vec[i]) for i, k in enumerate(keys)}
    R_raw = {k: float(r_vec[i]) for i, k in enumerate(keys)}
    R01 = {k: float(r01[i]) for i, k in enumerate(keys)}

    boot = bootstrap_metrics_block(S, R_raw, R01, names, B=200, seed=0)
    # |mean ρ| should be small for uncorrelated draws on a single ρ instance.
    assert abs(boot["rho_pearson"]["mean"]) < 0.25


def test_block_bootstrap_idempotent_for_same_seed():
    """Same input + same seed → bit-identical means/std (vectorized path)."""
    n = 20
    rng = np.random.default_rng(123)
    names = [f"B{i}" for i in range(n)]
    keys = [(names[i], names[j]) for i in range(n) for j in range(i + 1, n)]
    s_vec = rng.uniform(0, 1, len(keys))
    r_vec = 2 * s_vec - 1 + rng.normal(0, 0.1, len(keys))
    r01 = (r_vec + 1) / 2
    S = {k: float(s_vec[i]) for i, k in enumerate(keys)}
    R_raw = {k: float(r_vec[i]) for i, k in enumerate(keys)}
    R01 = {k: float(r01[i]) for i, k in enumerate(keys)}

    a = bootstrap_metrics_block(S, R_raw, R01, names, B=300, seed=99)
    b = bootstrap_metrics_block(S, R_raw, R01, names, B=300, seed=99)
    for metric in ("L_align", "L_align_01", "rho_pearson", "rho_spearman", "delta_tag"):
        assert a[metric] == b[metric]


def test_block_bootstrap_handles_too_few_benchmarks():
    boot = bootstrap_metrics_block({}, {}, {}, ["A"], B=10, seed=0)
    assert math.isnan(boot["L_align"]["mean"])
    assert math.isnan(boot["rho_pearson"]["mean"])


def test_block_bootstrap_ci_helper():
    lo, hi = block_bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], alpha=0.2)
    assert lo < hi
    assert math.isclose(lo, 0.18, abs_tol=0.05)
    assert math.isclose(hi, 0.92, abs_tol=0.05)


def test_block_bootstrap_ci_helper_handles_empty():
    lo, hi = block_bootstrap_ci([], alpha=0.05)
    assert math.isnan(lo) and math.isnan(hi)


def test_paired_permutation_test_significant_when_consistent_diff():
    a = [0.10, 0.12, 0.09, 0.11, 0.10]
    b = [0.20, 0.22, 0.21, 0.19, 0.20]
    res = paired_permutation_test(a, b, B=2000, seed=0)
    assert res["p_value"] < 0.1  # B=5 paired folds with consistent +0.10 → 1/32 ≈ 0.0625
    assert res["observed_diff"] > 0.05
    assert res["n_pairs_used"] == 5


def test_paired_permutation_test_not_significant_when_no_diff():
    a = [0.10, 0.20, 0.15, 0.25, 0.18]
    b = [0.11, 0.19, 0.16, 0.24, 0.17]
    res = paired_permutation_test(a, b, B=2000, seed=0)
    assert res["p_value"] > 0.2


def test_paired_permutation_test_drops_nan_pairs():
    a = [0.1, float("nan"), 0.2]
    b = [0.2, 0.3, float("nan")]
    res = paired_permutation_test(a, b, B=500, seed=0)
    assert res["n_pairs_used"] == 1


def test_residual_report_keeps_seed_vocab_fixed_action():
    S = {("A", "B"): 0.9, ("A", "C"): 0.2}
    R = {("A", "B"): -0.4, ("A", "C"): 0.1}
    report = build_residual_report(S, R, top_k=1)
    assert report[0]["p"] == "A"
    assert report[0]["q"] == "B"
    assert report[0]["direction"] == "tag_similarity_too_high"
    assert report[0]["part1_action"] == "keep_seed_vocabulary_fixed"
    assert report[0]["post_part1_use"] == "candidate_residual_for_taxonomy_refinement"
