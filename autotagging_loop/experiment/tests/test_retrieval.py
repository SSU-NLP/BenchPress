"""Tests for experiment/retrieval.py (v3 §2.2.10 Recall@K)."""

from __future__ import annotations

import math

from autotagging_loop.experiment.retrieval import recall_at_k


def test_recall_at_k_perfect_when_S_matches_R():
    names = ["A", "B", "C", "D"]
    R = {("A", "B"): 0.9, ("A", "C"): 0.5, ("A", "D"): 0.1,
         ("B", "C"): 0.8, ("B", "D"): 0.2, ("C", "D"): 0.7}
    S = dict(R)
    out = recall_at_k(benchmark_names=names, S=S, R=R, k_values=(1, 2))
    # Identical similarity rankings → recall = 1 at every k.
    assert out["per_k"]["1"]["mean"] == 1.0
    assert out["per_k"]["2"]["mean"] == 1.0


def test_recall_at_k_zero_when_S_inverts_R():
    names = ["A", "B", "C", "D"]
    R = {("A", "B"): 0.9, ("A", "C"): 0.6, ("A", "D"): 0.1,
         ("B", "C"): 0.8, ("B", "D"): 0.2, ("C", "D"): 0.7}
    # S inverts: highest R becomes lowest S
    S = {k: 1.0 - v for k, v in R.items()}
    out = recall_at_k(benchmark_names=names, S=S, R=R, k_values=(1,))
    # Top-1 by S is the bottom-1 by R; no overlap.
    assert out["per_k"]["1"]["mean"] == 0.0


def test_recall_at_k_skips_benchmarks_with_too_few_neighbors():
    names = ["A", "B", "C"]
    # Only one defined R neighbor for A → recall@2 undefined for A.
    R = {("A", "B"): 0.9}
    S = {("A", "B"): 0.9, ("A", "C"): 0.5, ("B", "C"): 0.4}
    out = recall_at_k(benchmark_names=names, S=S, R=R, k_values=(2,))
    # A drops out (only 1 R neighbor); B drops out (only 1 R neighbor); C drops out (0).
    assert out["per_k"]["2"]["n_benchmarks"] == 0
    assert math.isnan(out["per_k"]["2"]["mean"])


def test_recall_at_k_ties_broken_by_lex_order():
    names = ["A", "B", "C", "D"]
    R = {("A", "B"): 0.5, ("A", "C"): 0.5, ("A", "D"): 0.5,
         ("B", "C"): 0.5, ("B", "D"): 0.5, ("C", "D"): 0.5}
    S = dict(R)
    out = recall_at_k(benchmark_names=names, S=S, R=R, k_values=(2,))
    # All ties → S top-2 == R top-2 by lex order, so recall = 1.0 for every benchmark.
    assert out["per_k"]["2"]["mean"] == 1.0
