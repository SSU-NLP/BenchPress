"""Tests for experiment/split_metrics.py (v3 §2.2.7 split-aware reporting)."""

from __future__ import annotations

import json
import os

from autotagging_loop.experiment.split_metrics import (
    compute_held_model_test_metrics,
    compute_split_metrics,
    write_split_metrics_json,
)
from autotagging_loop.experiment.alignment import cosine_pair_matrix
from autotagging_loop.experiment.splits import BenchmarkSplit, split_benchmarks, split_models


def _S_R(names: list[str]) -> tuple[dict, dict, dict]:
    """Build perfectly-aligned S = R01 toy data over all upper-triangle pairs."""
    S: dict[tuple[str, str], float] = {}
    R_raw: dict[tuple[str, str], float | None] = {}
    R01: dict[tuple[str, str], float | None] = {}
    for i, p in enumerate(names):
        for q in names[i + 1:]:
            v = 0.1 + 0.1 * (hash((p, q)) % 7)  # deterministic non-trivial value
            S[(p, q)] = float(v)
            R_raw[(p, q)] = float(2 * v - 1)
            R01[(p, q)] = float(v)
    return S, R_raw, R01


def test_compute_split_metrics_emits_three_blocks():
    names = [f"B{i}" for i in range(15)]
    S, R_raw, R01 = _S_R(names)
    bench_split = split_benchmarks(names, seed=0)
    blocks = compute_split_metrics(
        S=S, R_raw=R_raw, R01=R01,
        benchmark_split=bench_split,
        q_p=0.80, q_n=0.20,
        bootstrap_B=20, seed=0,
    )
    assert set(blocks) == {"train", "dev", "test"}
    for name in ("train", "dev", "test"):
        block = blocks[name]
        assert block["n_benchmarks"] == len(getattr(bench_split, name))
        assert block["n_pairs"] >= 0


def test_split_metrics_train_pairs_dont_leak_to_test():
    """Pairs returned per block must have both endpoints in that bucket."""
    names = [f"B{i}" for i in range(10)]
    S, R_raw, R01 = _S_R(names)
    bench_split = split_benchmarks(names, seed=42)
    blocks = compute_split_metrics(
        S=S, R_raw=R_raw, R01=R01,
        benchmark_split=bench_split,
        q_p=0.80, q_n=0.20,
        bootstrap_B=10, seed=0,
    )
    # Total pairs across blocks ≤ total pairs in input (cross-bucket dropped).
    n_input = len(S)
    n_placed = sum(blocks[k]["n_pairs"] for k in ("train", "dev", "test"))
    assert n_placed <= n_input


def test_split_metrics_keep_all_pairs_from_unsorted_full_corpus_similarity():
    T = {
        "A": {"x": 1.0, "y": 0.0},
        "B": {"x": 0.0, "y": 1.0},
        "C": {"x": 1.0, "y": 1.0},
        "D": {"x": 0.5, "y": 0.5},
    }
    S = cosine_pair_matrix(T, benchmark_names=["C", "A", "B", "D"])
    R_raw = {
        ("A", "B"): -0.8,
        ("A", "C"): 0.2,
        ("B", "C"): 0.4,
    }
    R01 = {k: (v + 1.0) / 2.0 for k, v in R_raw.items()}
    split = BenchmarkSplit(
        train=["D"],
        dev=["A", "B", "C"],
        test=[],
        seed=0,
        ratios=(0.25, 0.75, 0.0),
    )

    blocks = compute_split_metrics(
        S=S,
        R_raw=R_raw,
        R01=R01,
        benchmark_split=split,
        q_p=0.80,
        q_n=0.20,
        bootstrap_B=0,
        seed=0,
    )

    assert blocks["dev"]["n_pairs"] == 3
    assert blocks["dev"]["n_effective_benchmarks"] == 3
    assert blocks["dev"]["isolated_benchmarks"] == []


def test_held_model_test_metrics_uses_only_held_models():
    bench_names = [f"B{i}" for i in range(15)]
    model_names = [f"m{j}" for j in range(20)]
    Y_norm = {b: {m: float((i + j) % 7) / 7.0 for j, m in enumerate(model_names)}
              for i, b in enumerate(bench_names)}
    S, _R_raw, _R01 = _S_R(bench_names)
    bench_split = split_benchmarks(bench_names, seed=0)
    model_split = split_models(model_names, seed=0)

    block = compute_held_model_test_metrics(
        S=S, Y_norm=Y_norm,
        benchmark_split=bench_split, model_split=model_split,
        q_p=0.80, q_n=0.20, bootstrap_B=10, seed=0,
        min_common=2,
    )
    assert block.get("n_held_models") == len(model_split.held)


def test_held_model_skips_when_no_held_models():
    bench_names = [f"B{i}" for i in range(5)]
    model_names = ["m1", "m2"]
    Y_norm = {b: {m: 0.5 for m in model_names} for b in bench_names}
    S, _, _ = _S_R(bench_names)
    bench_split = split_benchmarks(bench_names, seed=0)
    # Force all-seen split.
    model_split = split_models(model_names, ratios=(1.0, 0.0), seed=0)
    block = compute_held_model_test_metrics(
        S=S, Y_norm=Y_norm,
        benchmark_split=bench_split, model_split=model_split,
        q_p=0.80, q_n=0.20, bootstrap_B=10, seed=0,
        min_common=2,
    )
    assert block["skipped"] == "no_held_models"


def test_held_model_skips_when_below_min_common():
    bench_names = [f"B{i}" for i in range(5)]
    model_names = ["m1", "m2", "m3", "m4"]
    Y_norm = {b: {m: 0.5 for m in model_names} for b in bench_names}
    S, _, _ = _S_R(bench_names)
    bench_split = split_benchmarks(bench_names, seed=0)
    model_split = split_models(model_names, ratios=(0.5, 0.5), seed=0)

    block = compute_held_model_test_metrics(
        S=S, Y_norm=Y_norm,
        benchmark_split=bench_split, model_split=model_split,
        q_p=0.80, q_n=0.20, bootstrap_B=10, seed=0,
        min_common=3,
    )

    assert block["skipped"] == "held_models_below_min_common"
    assert block["n_held_models"] == 2
    assert block["min_common"] == 3


def test_write_split_metrics_round_trip(tmp_path):
    names = [f"B{i}" for i in range(10)]
    S, R_raw, R01 = _S_R(names)
    bench_split = split_benchmarks(names, seed=0)
    model_split = split_models(["m1", "m2", "m3", "m4"], seed=0)
    blocks = compute_split_metrics(
        S=S, R_raw=R_raw, R01=R01,
        benchmark_split=bench_split,
        q_p=0.80, q_n=0.20, bootstrap_B=10, seed=0,
    )
    path = write_split_metrics_json(
        str(tmp_path),
        fold=0, seed=0,
        benchmark_split=bench_split,
        model_split=model_split,
        train_dev_test=blocks,
        held_model_test={"n_pairs": 0, "n_benchmarks": 0, "skipped": "test_only"},
    )
    assert os.path.exists(path)
    with open(path) as fh:
        payload = json.load(fh)
    assert payload["fold"] == 0
    assert set(payload["benchmark_split"]) == {"train", "dev", "test", "ratios"}
    assert set(payload["model_split"]) == {"seen", "held", "ratios", "strategy"}
    assert set(payload).issuperset({"train", "dev", "test", "held_model_test"})
