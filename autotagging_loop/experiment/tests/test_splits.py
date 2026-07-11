"""Tests for experiment/splits.py (v3 §2.2.7 validation splits)."""

from __future__ import annotations

from autotagging_loop.experiment.splits import (
    _default_benchmark_stratum,
    _default_model_family,
    induced_pair_set,
    restrict_pair_dict,
    split_benchmarks,
    split_benchmarks_kfold,
    split_benchmarks_kfold_score_balanced,
    split_benchmarks_kfold_stratified,
    split_models,
    split_pair_dict_by_benchmark,
)


def test_split_benchmarks_disjoint_and_union_equals_input():
    names = [f"B{i}" for i in range(30)]
    split = split_benchmarks(names, ratios=(0.6, 0.2, 0.2), seed=42)

    train_set = set(split.train)
    dev_set = set(split.dev)
    test_set = set(split.test)

    assert train_set.isdisjoint(dev_set)
    assert train_set.isdisjoint(test_set)
    assert dev_set.isdisjoint(test_set)
    assert train_set | dev_set | test_set == set(names)
    # Ratios on N=30 → train=18, dev=6, test=6
    assert len(split.train) == 18
    assert len(split.dev) == 6
    assert len(split.test) == 6


def test_split_benchmarks_deterministic_for_same_seed():
    names = [f"B{i}" for i in range(20)]
    a = split_benchmarks(names, seed=7)
    b = split_benchmarks(names, seed=7)
    assert a.train == b.train and a.dev == b.dev and a.test == b.test


def test_split_benchmarks_differs_across_seeds():
    names = [f"B{i}" for i in range(20)]
    a = split_benchmarks(names, seed=1)
    b = split_benchmarks(names, seed=2)
    # Highly unlikely identical at N=20 with different seeds.
    assert (a.train, a.dev, a.test) != (b.train, b.dev, b.test)


def test_split_models_disjoint_and_union_equals_input():
    names = [f"m{i}" for i in range(50)]
    split = split_models(names, ratios=(0.8, 0.2), seed=0)
    seen = set(split.seen)
    held = set(split.held)
    assert seen.isdisjoint(held)
    assert seen | held == set(names)
    assert len(split.seen) == 40
    assert len(split.held) == 10


def test_split_models_family_stratified_balances_repeated_families():
    names = [
        "Claude-3.5-Sonnet",
        "Claude-Sonnet-4",
        "GPT-4o",
        "GPT-oss-120b",
        "GPT-oss-20b",
        "Qwen2.5-72B",
        "Qwen3-235B",
        "DeepSeek-v3",
        "Gemma-3-4B",
        "KIMI-K2",
        "Llama-3.1-8B",
        "Phi-4-Mini",
        "GLM-4.7",
    ]
    split = split_models(
        names,
        ratios=(0.5, 0.5),
        seed=157,
        strategy="family_stratified",
    )

    assert set(split.seen).isdisjoint(split.held)
    assert set(split.seen) | set(split.held) == set(names)
    assert len(split.seen) == 6
    assert len(split.held) == 7
    for family in ("claude", "gpt", "qwen"):
        assert any(_default_model_family(model) == family for model in split.seen)
        assert any(_default_model_family(model) == family for model in split.held)


def test_induced_pair_set_is_sorted_unique_upper_triangle():
    pairs = induced_pair_set(["B", "A", "C", "A"])
    assert pairs == [("A", "B"), ("A", "C"), ("B", "C")]
    # No (A, A) self-pairs even though "A" appears twice.
    assert all(p < q for p, q in pairs)


def test_restrict_pair_dict_keeps_only_listed_pairs():
    pair_dict = {("A", "B"): 1.0, ("A", "C"): 2.0, ("B", "C"): 3.0}
    restricted = restrict_pair_dict(pair_dict, [("A", "B"), ("B", "C")])
    assert restricted == {("A", "B"): 1.0, ("B", "C"): 3.0}


def test_split_pair_dict_drops_cross_bucket_pairs():
    names = ["A", "B", "C", "D"]
    # Force a known split: train=A,B; dev=C; test=D
    split = split_benchmarks(names, ratios=(0.5, 0.25, 0.25), seed=0)
    # Build all 6 pairs.
    pair_dict = {pq: 1.0 for pq in induced_pair_set(names)}
    buckets = split_pair_dict_by_benchmark(pair_dict, split)

    # Every bucket-pair must have both endpoints in that bucket.
    train_names = set(split.train)
    dev_names = set(split.dev)
    test_names = set(split.test)
    for (p, q) in buckets["train"]:
        assert p in train_names and q in train_names
    for (p, q) in buckets["dev"]:
        assert p in dev_names and q in dev_names
    for (p, q) in buckets["test"]:
        assert p in test_names and q in test_names

    # Cross-bucket pairs are dropped by design.
    placed = sum(len(b) for b in buckets.values())
    cross = len(pair_dict) - placed
    assert cross >= 0
    assert placed == sum(
        len(induced_pair_set(getattr(split, b))) for b in ("train", "dev", "test")
    )


def test_split_benchmarks_rejects_bad_ratios():
    import pytest
    with pytest.raises(ValueError):
        split_benchmarks(["A", "B", "C"], ratios=(0.5, 0.3, 0.3))


# --- K-fold (stratified) ---------------------------------------------------

_PART2_LIKE = [
    "AIME", "MATH-500", "GSM8K",
    "HumanEval", "MBPP", "CodeContests",
    "MMLU", "GPQA", "HLE", "SimpleQA", "SuperGPQA",
    "BBH", "ARC", "HellaSwag", "WinoGrande", "TruthfulQA", "AGIEval", "DROP",
]


def test_default_stratum_covers_part2_families():
    strata = {b: _default_benchmark_stratum(b) for b in _PART2_LIKE}
    assert strata["AIME"] == "math"
    assert strata["HumanEval"] == "code"
    assert strata["MMLU"] == "knowledge"
    assert strata["BBH"] == "reasoning"
    assert _default_benchmark_stratum("Unknown-Bench") == "reasoning"  # fallback


def test_stratified_kfold_test_folds_disjoint_and_cover_input():
    K = 4
    seen: list[str] = []
    for k in range(K):
        s = split_benchmarks_kfold_stratified(_PART2_LIKE, n_folds=K, fold=k, seed=0)
        seen.extend(s.test)
    assert sorted(seen) == sorted(set(_PART2_LIKE))
    assert len(seen) == len(set(seen))  # disjoint


def test_stratified_kfold_balances_families():
    """Per-fold test sets must not be dominated by a single family.

    With K=4 over an 18-bench corpus (4 families), every fold's test set
    should contain >=2 different strata so the loss isn't computed on a
    monoculture (which causes n_pos=0 dev splits — the fold0/fold1 bug).
    """
    K = 4
    for k in range(K):
        s = split_benchmarks_kfold_stratified(_PART2_LIKE, n_folds=K, fold=k, seed=0)
        strata_in_test = {_default_benchmark_stratum(b) for b in s.test}
        assert len(strata_in_test) >= 2, (
            f"fold{k} test={s.test} has only one stratum {strata_in_test}"
        )


def test_stratified_kfold_deterministic_for_same_seed():
    a = split_benchmarks_kfold_stratified(_PART2_LIKE, n_folds=4, fold=1, seed=7)
    b = split_benchmarks_kfold_stratified(_PART2_LIKE, n_folds=4, fold=1, seed=7)
    assert a.train == b.train and a.dev == b.dev and a.test == b.test


def test_stratified_kfold_differs_from_plain_kfold_in_general():
    """Stratification should produce a different test-fold composition than
    plain K-fold for a non-trivial corpus (otherwise the flag does nothing).
    """
    s_strat = split_benchmarks_kfold_stratified(_PART2_LIKE, n_folds=4, fold=0, seed=0)
    s_plain = split_benchmarks_kfold(_PART2_LIKE, n_folds=4, fold=0, seed=0)
    assert s_strat.test != s_plain.test


def test_stratified_kfold_dev_train_ratio_applied():
    K = 4
    s = split_benchmarks_kfold_stratified(
        _PART2_LIKE, n_folds=K, fold=0, seed=0, dev_train_split=(1.0, 2.0),
    )
    n_non_test = len(s.train) + len(s.dev)
    # 1:2 within non-test → dev ≈ 1/3 of non-test
    assert abs(len(s.dev) - round(n_non_test / 3)) <= 1


def test_stratified_kfold_rejects_bad_args():
    import pytest
    with pytest.raises(ValueError):
        split_benchmarks_kfold_stratified(_PART2_LIKE, n_folds=1, fold=0)
    with pytest.raises(ValueError):
        split_benchmarks_kfold_stratified(_PART2_LIKE, n_folds=4, fold=4)
    with pytest.raises(ValueError):
        split_benchmarks_kfold_stratified(["A", "B"], n_folds=4, fold=0)


def test_score_balanced_kfold_spreads_anti_correlated_pairs():
    names = ["HLE", "A", "B", "C", "D", "E", "F"]
    R = {}
    for bad in ("A", "B", "C"):
        R[tuple(sorted(("HLE", bad)))] = -1.0
    for good in ("D", "E", "F"):
        R[tuple(sorted(("HLE", good)))] = 0.5

    folds = [
        split_benchmarks_kfold_score_balanced(
            names, n_folds=2, fold=k, seed=0, R_raw=R,
        )
        for k in range(2)
    ]
    hle_fold = next(split for split in folds if "HLE" in split.test)

    assert not ({"A", "B", "C"} & set(hle_fold.test))
    assert set().union(*(set(split.test) for split in folds)) == set(names)
    assert set(folds[0].test).isdisjoint(folds[1].test)
