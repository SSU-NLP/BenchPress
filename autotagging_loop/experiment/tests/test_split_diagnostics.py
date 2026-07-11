"""Tests for split diagnostics used by v-loop preflight."""

from __future__ import annotations

from autotagging_loop.experiment.split_diagnostics import (
    benchmark_split_from_config,
    coverage_pair_count,
    effective_benchmark_count,
    split_effective_benchmark_counts,
    split_pair_count_failures,
    split_coverage_pair_counts,
    split_valid_pair_counts,
    valid_pair_count,
)
from autotagging_loop.experiment.splits import BenchmarkSplit, induced_pair_set


def test_benchmark_split_from_config_honors_dev_train_split():
    names = [f"Bench{i}" for i in range(8)]

    default_split = benchmark_split_from_config(
        names,
        {"cv_folds": 2, "fold": 0, "benchmark_seed": 0},
    )
    wider_dev_split = benchmark_split_from_config(
        names,
        {
            "cv_folds": 2,
            "fold": 0,
            "benchmark_seed": 0,
            "dev_train_split": [1.0, 1.0],
        },
    )

    assert (len(default_split.dev), len(default_split.train)) == (1, 3)
    assert (len(wider_dev_split.dev), len(wider_dev_split.train)) == (2, 2)


def test_valid_pair_count_ignores_none_values():
    pair_dict = {
        ("A", "B"): 1.0,
        ("A", "C"): None,
        ("B", "C"): -0.5,
    }

    assert valid_pair_count(pair_dict, ["A", "B", "C"]) == 2


def test_effective_benchmark_count_ignores_isolated_benchmarks():
    pair_dict = {
        ("A", "B"): 1.0,
        ("A", "C"): None,
        ("B", "C"): -0.5,
        ("D", "E"): None,
    }

    assert effective_benchmark_count(pair_dict, ["A", "B", "C", "D", "E"]) == 3


def test_coverage_pair_count_uses_common_count_not_score_value():
    common_count = {
        ("A", "B"): 6,
        ("A", "C"): 5,
        ("B", "C"): 3,
    }

    assert coverage_pair_count(common_count, ["A", "B", "C"], min_common=5) == 2


def test_split_valid_pair_counts_stays_inside_buckets():
    split = BenchmarkSplit(
        train=["A", "B", "C"],
        dev=["D", "E"],
        test=["F"],
        seed=0,
        ratios=(0.5, 0.33, 0.17),
    )
    pair_dict = {pair: 1.0 for pair in induced_pair_set(["A", "B", "C", "D", "E", "F"])}

    assert split_valid_pair_counts(pair_dict, split) == {
        "train": 3,
        "dev": 1,
        "test": 0,
    }


def test_split_effective_benchmark_counts_stays_inside_buckets():
    split = BenchmarkSplit(
        train=["A", "B", "C"],
        dev=["D", "E"],
        test=["F"],
        seed=0,
        ratios=(0.5, 0.33, 0.17),
    )
    pair_dict = {
        ("A", "B"): 1.0,
        ("A", "C"): None,
        ("B", "C"): None,
        ("D", "E"): 0.5,
    }

    assert split_effective_benchmark_counts(pair_dict, split) == {
        "train": 2,
        "dev": 2,
        "test": 0,
    }


def test_split_coverage_pair_counts_stays_inside_buckets():
    split = BenchmarkSplit(
        train=["A", "B", "C"],
        dev=["D", "E"],
        test=["F"],
        seed=0,
        ratios=(0.5, 0.33, 0.17),
    )
    common_count = {
        pair: 6
        for pair in induced_pair_set(["A", "B", "C", "D", "E", "F"])
    }
    common_count[("A", "C")] = 4

    assert split_coverage_pair_counts(common_count, split, min_common=5) == {
        "train": 2,
        "dev": 1,
        "test": 0,
    }


def test_split_pair_count_failures_are_stable():
    failures = split_pair_count_failures(
        {"train": 3, "dev": 0, "test": 1},
        {"train": 1, "dev": 1, "test": 2},
    )

    assert failures == ["dev:0<1", "test:1<2"]
