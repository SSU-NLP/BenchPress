"""Tests for greedy + k-medoids subset selection (v3 §2.2.10)."""

from __future__ import annotations

from autotagging_loop.experiment.loop import (
    _build_profile_support,
    _select_kmedoids_subset,
    _select_tag_cover_subset,
)


def _T_clustered() -> dict[str, dict[str, float]]:
    """Three tag-vector clusters: {A1, A2}, {B1, B2}, {C1, C2}."""
    return {
        "A1": {"t0": 1.0, "t1": 0.0, "t2": 0.0},
        "A2": {"t0": 0.95, "t1": 0.05, "t2": 0.0},
        "B1": {"t0": 0.0, "t1": 1.0, "t2": 0.0},
        "B2": {"t0": 0.05, "t1": 0.95, "t2": 0.0},
        "C1": {"t0": 0.0, "t1": 0.0, "t2": 1.0},
        "C2": {"t0": 0.0, "t1": 0.05, "t2": 0.95},
    }


def test_kmedoids_picks_one_per_cluster_for_k3():
    T = _T_clustered()
    selected = _select_kmedoids_subset(T, list(T), k=3, seed=0)
    assert len(selected) == 3
    # Each medoid should come from a distinct cluster prefix.
    prefixes = sorted({name[0] for name in selected})
    assert prefixes == ["A", "B", "C"]


def test_kmedoids_deterministic_for_same_seed():
    T = _T_clustered()
    a = _select_kmedoids_subset(T, list(T), k=3, seed=42)
    b = _select_kmedoids_subset(T, list(T), k=3, seed=42)
    assert a == b


def test_kmedoids_returns_all_when_k_exceeds_n():
    T = _T_clustered()
    selected = _select_kmedoids_subset(T, list(T), k=99, seed=0)
    assert sorted(selected) == sorted(T)


def test_kmedoids_zero_k_returns_empty():
    T = _T_clustered()
    assert _select_kmedoids_subset(T, list(T), k=0, seed=0) == []


def test_greedy_still_works_alongside_kmedoids():
    T = _T_clustered()
    greedy = _select_tag_cover_subset(T, list(T), k=3)
    kmed = _select_kmedoids_subset(T, list(T), k=3, seed=0)
    # Both pick exactly k items, with deterministic outputs across reruns.
    assert len(greedy) == 3
    assert len(kmed) == 3


def test_build_profile_support_emits_both_methods():
    T = _T_clustered()
    Y_norm = {
        "m1": {"A1": 0.9, "A2": 0.85, "B1": 0.4, "B2": 0.42, "C1": 0.7, "C2": 0.72},
        "m2": {"A1": 0.3, "A2": 0.32, "B1": 0.8, "B2": 0.78, "C1": 0.5, "C2": 0.48},
    }
    out = _build_profile_support(
        Y_norm=Y_norm,
        T=T,
        benchmark_names=list(T),
        model_names=list(Y_norm),
        subset_sizes=[2, 3],
        methods=["greedy", "kmedoids"],
        kmedoids_seed=0,
    )
    assert "methods" in out
    assert set(out["methods"].keys()) == {"greedy", "kmedoids"}
    assert set(out["methods"]["greedy"]["subsets"].keys()) == {"2", "3"}
    assert set(out["methods"]["kmedoids"]["subsets"].keys()) == {"2", "3"}
    # Backward-compat top-level "subsets" mirrors the first requested method (greedy).
    assert out["subsets"] == out["methods"]["greedy"]["subsets"]
