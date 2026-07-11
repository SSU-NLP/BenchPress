"""Split diagnostics for v-loop benchmark partitions."""

from __future__ import annotations

from autotagging_loop.experiment.splits import (
    BenchmarkSplit,
    induced_pair_set,
    parse_dev_train_split,
    restrict_pair_dict,
    split_benchmarks,
    split_benchmarks_kfold,
    split_benchmarks_kfold_score_balanced,
    split_benchmarks_kfold_stratified,
)


def benchmark_split_from_config(
    benchmark_names: list[str],
    splits_cfg: dict | None,
    *,
    score_pair_dict: dict | None = None,
    required_pair_dicts: list[dict] | None = None,
    min_test_valid_pairs: int = 0,
    min_test_effective_benchmarks: int = 0,
) -> BenchmarkSplit:
    """Build the benchmark split exactly as the v-loop should read config."""
    cfg = splits_cfg or {}
    bench_ratios = tuple(cfg.get("benchmark_ratios", (0.6, 0.2, 0.2)))
    bench_seed = int(cfg.get("benchmark_seed", 0))
    cv_folds = int(cfg.get("cv_folds", 1))
    fold = int(cfg.get("fold", 0))
    if cv_folds > 1:
        dev_train_split = parse_dev_train_split(cfg.get("dev_train_split"))
        strategy = str(
            cfg.get(
                "benchmark_split_strategy",
                "family_stratified" if bool(cfg.get("stratified", False)) else "random",
            )
        ).strip().lower()
        if strategy in {"score_balanced", "score"} and score_pair_dict is not None:
            return split_benchmarks_kfold_score_balanced(
                benchmark_names,
                n_folds=cv_folds,
                fold=fold,
                seed=bench_seed,
                dev_train_split=dev_train_split,
                R_raw=score_pair_dict,
                required_pair_dicts=required_pair_dicts,
                min_test_valid_pairs=min_test_valid_pairs,
                min_test_effective_benchmarks=min_test_effective_benchmarks,
                search_iters=int(cfg.get("score_balance_search_iters", 5000)),
            )
        if strategy in {"family_stratified", "family"}:
            return split_benchmarks_kfold_stratified(
                benchmark_names,
                n_folds=cv_folds,
                fold=fold,
                seed=bench_seed,
                dev_train_split=dev_train_split,
            )
        if strategy not in {"random", "score_balanced", "score"}:
            raise ValueError(
                "benchmark split strategy must be one of "
                "{'random', 'family_stratified', 'score_balanced'}, "
                f"got {strategy!r}"
            )
        return split_benchmarks_kfold(
            benchmark_names,
            n_folds=cv_folds,
            fold=fold,
            seed=bench_seed,
            dev_train_split=dev_train_split,
        )
    return split_benchmarks(
        benchmark_names,
        ratios=bench_ratios,
        seed=bench_seed,
    )


def valid_pair_count(pair_dict: dict, benchmarks: list[str]) -> int:
    """Count finite score-comparable pairs induced wholly inside benchmarks."""
    restricted = restrict_pair_dict(pair_dict, induced_pair_set(benchmarks))
    return sum(1 for value in restricted.values() if value is not None)


def effective_benchmark_count(pair_dict: dict, benchmarks: list[str]) -> int:
    """Count benchmarks that appear in at least one finite within-split pair."""
    endpoints: set[str] = set()
    restricted = restrict_pair_dict(pair_dict, induced_pair_set(benchmarks))
    for pair, value in restricted.items():
        if value is None:
            continue
        endpoints.update(pair)
    return len(endpoints)


def split_effective_benchmark_counts(pair_dict: dict, split: BenchmarkSplit) -> dict[str, int]:
    """Count non-isolated benchmarks inside each benchmark split bucket."""
    return {
        "train": effective_benchmark_count(pair_dict, split.train),
        "dev": effective_benchmark_count(pair_dict, split.dev),
        "test": effective_benchmark_count(pair_dict, split.test),
    }


def coverage_pair_count(
    common_count: dict,
    benchmarks: list[str],
    *,
    min_common: int,
) -> int:
    """Count pairs with enough overlapping score cells, independent of score values."""
    restricted = restrict_pair_dict(common_count, induced_pair_set(benchmarks))
    return sum(1 for value in restricted.values() if int(value or 0) >= min_common)


def split_coverage_pair_counts(
    common_count: dict,
    split: BenchmarkSplit,
    *,
    min_common: int,
) -> dict[str, int]:
    """Count coverage-feasible pairs inside each benchmark split bucket."""
    return {
        "train": coverage_pair_count(common_count, split.train, min_common=min_common),
        "dev": coverage_pair_count(common_count, split.dev, min_common=min_common),
        "test": coverage_pair_count(common_count, split.test, min_common=min_common),
    }


def split_valid_pair_counts(pair_dict: dict, split: BenchmarkSplit) -> dict[str, int]:
    """Count score-comparable pairs inside each benchmark split bucket."""
    return {
        "train": valid_pair_count(pair_dict, split.train),
        "dev": valid_pair_count(pair_dict, split.dev),
        "test": valid_pair_count(pair_dict, split.test),
    }


def split_pair_count_failures(
    counts: dict[str, int],
    thresholds: dict[str, int],
) -> list[str]:
    """Return stable failure strings for split pair-count thresholds."""
    return [
        f"{name}:{int(counts.get(name, 0))}<{int(thresholds.get(name, 0))}"
        for name in ("train", "dev", "test")
        if int(counts.get(name, 0)) < int(thresholds.get(name, 0))
    ]
