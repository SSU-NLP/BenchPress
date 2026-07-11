"""v3 §2.2.7 validation splits.

The Improver and selection use D_dev; D_train remains a generalization guard
and diagnostic view; D_test is untouched until end-of-run reporting. Held-model
testing partitions the score-pattern model set F into F_seen / F_held so v3
§2.2.11 can evaluate generalization to unseen models without leaking them
into the loop.

All splits are deterministic given a seed (`random.Random` based shuffle on
the canonical sort order). K_fold reporting runs `run_part1` once per seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable


PairKey = tuple[str, str]


@dataclass
class BenchmarkSplit:
    train: list[str]
    dev: list[str]
    test: list[str]
    seed: int
    ratios: tuple[float, float, float]


@dataclass
class ModelSplit:
    seen: list[str]
    held: list[str]
    seed: int
    ratios: tuple[float, float]
    strategy: str = "random"


def parse_dev_train_split(
    value,
    *,
    default: tuple[float, float] = (1.0, 3.0),
) -> tuple[float, float]:
    """Normalize a config value into the K-fold dev:train ratio tuple."""
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            f"dev_train_split must be a 2-item list/tuple, got {value!r}"
        )
    try:
        dev_r = float(value[0])
        train_r = float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"dev_train_split values must be numeric, got {value!r}"
        ) from exc
    if dev_r < 0 or train_r < 0:
        raise ValueError(
            f"dev_train_split must be non-negative, got {(dev_r, train_r)!r}"
        )
    return (dev_r, train_r)


def split_benchmarks(
    benchmark_names: Iterable[str],
    *,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
) -> BenchmarkSplit:
    """Disjoint random partition of benchmark names.

    The split is deterministic for a given (seed, sorted-input). Ratios use
    `round` for the train and dev counts; the test bucket absorbs any rounding
    residue so the union exactly equals the input.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    names = sorted(set(benchmark_names))
    rng = random.Random(seed)
    rng.shuffle(names)
    n = len(names)
    n_train = round(n * ratios[0])
    n_dev = round(n * ratios[1])
    if n_train + n_dev > n:
        n_dev = max(0, n - n_train)
    train = sorted(names[:n_train])
    dev = sorted(names[n_train : n_train + n_dev])
    test = sorted(names[n_train + n_dev :])
    return BenchmarkSplit(train=train, dev=dev, test=test, seed=seed, ratios=ratios)


def _default_benchmark_stratum(name: str) -> str:
    """Heuristic benchmark-family classifier for default stratification.

    Returns one of {"math", "code", "knowledge", "reasoning"}. Fallback is
    "reasoning" so unknown benches still get a stratum and are distributed
    evenly. The map covers the `labels_part2` corpus used in v3.
    """
    up = name.upper()
    if any(k in up for k in ("AIME", "MATH", "GSM")):
        return "math"
    if any(k in up for k in ("HUMANEVAL", "MBPP", "HUMAN EVAL", "CODE")):
        return "code"
    if any(k in up for k in ("MMLU", "GPQA", "HLE", "SIMPLEQA", "SUPERGPQA")):
        return "knowledge"
    return "reasoning"


def split_benchmarks_kfold_stratified(
    benchmark_names: Iterable[str],
    *,
    n_folds: int,
    fold: int,
    seed: int = 0,
    dev_train_split: tuple[float, float] = (1.0, 3.0),
    stratum_fn: "Callable[[str], str] | None" = None,
) -> BenchmarkSplit:
    """Stratified K-fold partition. Each fold's test set contains a balanced
    sample of benchmark families so no fold is dominated by one family.

    Algorithm: per-stratum shuffle (seeded) then deterministic round-robin
    assignment of stratum members to test folds. Non-test → dev/train via
    `dev_train_split` ratio applied per fold globally.

    Properties:
      - Test folds are disjoint; their union = input.
      - Each test fold has at most ⌈|stratum|/K⌉ members from any stratum.
      - Deterministic for fixed (sorted-input, seed, n_folds, fold).
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be ≥ 2 (got {n_folds})")
    if not (0 <= fold < n_folds):
        raise ValueError(f"fold must be in [0, {n_folds}); got {fold}")
    if any(r < 0 for r in dev_train_split):
        raise ValueError(f"dev_train_split must be non-negative; got {dev_train_split}")

    sfn = stratum_fn or _default_benchmark_stratum
    names = sorted(set(benchmark_names))
    if len(names) < n_folds:
        raise ValueError(f"n_folds={n_folds} exceeds benchmark count {len(names)}")

    # Group by stratum (sorted name within each stratum), then per-stratum shuffle
    # with deterministic seed so different folds see a consistent assignment.
    strata: dict[str, list[str]] = {}
    for nm in names:
        strata.setdefault(sfn(nm), []).append(nm)
    rng = random.Random(seed)
    fold_assignment: dict[str, int] = {}
    for stratum in sorted(strata.keys()):
        members = sorted(strata[stratum])
        rng.shuffle(members)
        offset = rng.randrange(n_folds)
        for idx, nm in enumerate(members):
            fold_assignment[nm] = (idx + offset) % n_folds

    test = sorted(nm for nm in names if fold_assignment[nm] == fold)
    non_test = [nm for nm in names if fold_assignment[nm] != fold]

    # Dev/train within non-test: deterministic shuffle then prefix-split.
    rng2 = random.Random(seed * 31 + fold)
    rng2.shuffle(non_test)
    dev_r, train_r = dev_train_split
    denom = dev_r + train_r
    n_non_test = len(non_test)
    n_dev = 0 if denom <= 0 else round(n_non_test * (dev_r / denom))
    if n_dev > n_non_test:
        n_dev = n_non_test
    dev = sorted(non_test[:n_dev])
    train = sorted(non_test[n_dev:])

    n = len(names)
    ratios = (
        len(train) / n if n else 0.0,
        len(dev) / n if n else 0.0,
        len(test) / n if n else 0.0,
    )
    return BenchmarkSplit(
        train=train, dev=dev, test=test, seed=seed, ratios=ratios,
    )


def _finite_score_pair(
    R_raw: dict[PairKey, float | None],
    left: str,
    right: str,
) -> float | None:
    if left == right:
        return None
    pair = (left, right) if left < right else (right, left)
    value = R_raw.get(pair)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def split_benchmarks_kfold_score_balanced(
    benchmark_names: Iterable[str],
    *,
    n_folds: int,
    fold: int,
    R_raw: dict[PairKey, float | None],
    seed: int = 0,
    dev_train_split: tuple[float, float] = (1.0, 3.0),
    required_pair_dicts: list[dict[PairKey, float | None]] | None = None,
    min_test_valid_pairs: int = 0,
    min_test_effective_benchmarks: int = 0,
    search_iters: int = 5000,
) -> BenchmarkSplit:
    """K-fold split that spreads anti-correlated score-pattern pairs.

    Family stratification can still put HLE-like anti-correlated benchmarks in
    one held-out test fold. This splitter minimizes within-fold negative-R
    cost while preserving optional seen/held test coverage constraints.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be ≥ 2 (got {n_folds})")
    if not (0 <= fold < n_folds):
        raise ValueError(f"fold must be in [0, {n_folds}); got {fold}")
    if any(r < 0 for r in dev_train_split):
        raise ValueError(f"dev_train_split must be non-negative; got {dev_train_split}")

    names = sorted(set(benchmark_names))
    n = len(names)
    if n < n_folds:
        raise ValueError(f"n_folds={n_folds} exceeds benchmark count {n}")

    base = n // n_folds
    rem = n % n_folds
    targets = [base + (1 if k < rem else 0) for k in range(n_folds)]

    required = list(required_pair_dicts or [R_raw])

    def valid_pair_count(pair_dict: dict[PairKey, float | None], members: list[str]) -> int:
        count = 0
        for i, left in enumerate(members):
            for right in members[i + 1:]:
                if _finite_score_pair(pair_dict, left, right) is not None:
                    count += 1
        return count

    def effective_count(pair_dict: dict[PairKey, float | None], members: list[str]) -> int:
        endpoints: set[str] = set()
        for i, left in enumerate(members):
            for right in members[i + 1:]:
                if _finite_score_pair(pair_dict, left, right) is not None:
                    endpoints.update((left, right))
        return len(endpoints)

    def negative_cost(members: list[str]) -> float:
        cost = 0.0
        for i, left in enumerate(members):
            for right in members[i + 1:]:
                rv = _finite_score_pair(R_raw, left, right)
                if rv is not None and rv < 0.0:
                    cost += float((-rv) ** 2)
        return cost

    def assignment_score(candidate: list[list[str]]) -> tuple[int, float, float]:
        missing = 0
        neg_costs: list[float] = []
        for members in candidate:
            for pair_dict in required:
                missing += max(0, int(min_test_valid_pairs) - valid_pair_count(pair_dict, members))
                missing += max(
                    0,
                    int(min_test_effective_benchmarks) - effective_count(pair_dict, members),
                )
            neg_costs.append(negative_cost(members))
        return (missing, max(neg_costs, default=0.0), sum(neg_costs))

    anti_degree: dict[str, float] = {}
    for name in names:
        total = 0.0
        for other in names:
            rv = _finite_score_pair(R_raw, name, other)
            if rv is not None and rv < 0.0:
                total += float((-rv) ** 2)
        anti_degree[name] = total

    rng = random.Random(seed)
    tie_break = {name: rng.random() for name in names}
    order = sorted(names, key=lambda nm: (-anti_degree[nm], tie_break[nm], nm))

    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for name in order:
        choices = [idx for idx in range(n_folds) if len(folds[idx]) < targets[idx]]

        def added_negative_cost(idx: int) -> tuple[float, int, float]:
            cost = 0.0
            for other in folds[idx]:
                rv = _finite_score_pair(R_raw, name, other)
                if rv is not None and rv < 0.0:
                    cost += float((-rv) ** 2)
            return (cost, len(folds[idx]), tie_break[name] + idx * 1e-9)

        best_fold = min(choices, key=added_negative_cost)
        folds[best_fold].append(name)

    best_folds = [sorted(members) for members in folds]
    best_score = assignment_score(best_folds)

    rng_search = random.Random(seed + 991_027)
    for _ in range(max(0, int(search_iters))):
        shuffled = names[:]
        rng_search.shuffle(shuffled)
        candidate: list[list[str]] = []
        cursor = 0
        for size in targets:
            candidate.append(sorted(shuffled[cursor : cursor + size]))
            cursor += size
        score = assignment_score(candidate)
        if score < best_score:
            best_folds = candidate
            best_score = score
            if score[0] == 0 and score[1] <= 0.0:
                break

    test = sorted(best_folds[fold])
    non_test = [nm for nm in names if nm not in set(test)]

    rng2 = random.Random(seed * 31 + fold)
    rng2.shuffle(non_test)
    dev_r, train_r = dev_train_split
    denom = dev_r + train_r
    n_non_test = len(non_test)
    n_dev = 0 if denom <= 0 else round(n_non_test * (dev_r / denom))
    if n_dev > n_non_test:
        n_dev = n_non_test
    dev = sorted(non_test[:n_dev])
    train = sorted(non_test[n_dev:])

    ratios = (
        len(train) / n if n else 0.0,
        len(dev) / n if n else 0.0,
        len(test) / n if n else 0.0,
    )
    return BenchmarkSplit(train=train, dev=dev, test=test, seed=seed, ratios=ratios)


def split_benchmarks_kfold(
    benchmark_names: Iterable[str],
    *,
    n_folds: int,
    fold: int,
    seed: int = 0,
    dev_train_split: tuple[float, float] = (1.0, 3.0),
) -> BenchmarkSplit:
    """Deterministic K-fold partition for benchmark splits.

    The full benchmark set is sorted then shuffled once with `seed`. The
    shuffled list is sliced into `n_folds` contiguous chunks; fold `fold`'s
    chunk becomes the test set. The remaining (1 − 1/K) benchmarks are
    further partitioned into dev/train using the `dev_train_split` ratio
    (default 1:3 — with source = train per v3 §2.2.4, train carries the
    Executer's evidence base so a fatter train and a leaner dev are
    preferred over the older 1:2 split).

    Properties (verified by `tests/test_splits_kfold.py`):
      - Every benchmark appears in test for exactly one fold ∈ [0, n_folds).
      - Test sets across folds are disjoint; their union equals the input.
      - Deterministic for fixed (sorted-input, seed, n_folds, fold).
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be ≥ 2 (got {n_folds}); use split_benchmarks for single split")
    if not (0 <= fold < n_folds):
        raise ValueError(f"fold must be in [0, {n_folds}); got {fold}")
    if any(r < 0 for r in dev_train_split):
        raise ValueError(f"dev_train_split must be non-negative; got {dev_train_split}")

    names = sorted(set(benchmark_names))
    rng = random.Random(seed)
    rng.shuffle(names)
    n = len(names)
    if n < n_folds:
        raise ValueError(f"n_folds={n_folds} exceeds benchmark count {n}")

    base = n // n_folds
    rem = n % n_folds
    starts = []
    cursor = 0
    for k in range(n_folds):
        size_k = base + (1 if k < rem else 0)
        starts.append((cursor, cursor + size_k))
        cursor += size_k
    test_start, test_end = starts[fold]
    test = names[test_start:test_end]
    non_test = names[:test_start] + names[test_end:]

    n_non_test = len(non_test)
    dev_r, train_r = dev_train_split
    denom = dev_r + train_r
    if denom <= 0:
        n_dev = 0
    else:
        n_dev = round(n_non_test * (dev_r / denom))
    if n_dev > n_non_test:
        n_dev = n_non_test
    dev = non_test[:n_dev]
    train = non_test[n_dev:]

    train_sorted = sorted(train)
    dev_sorted = sorted(dev)
    test_sorted = sorted(test)
    ratios = (
        len(train_sorted) / n if n else 0.0,
        len(dev_sorted) / n if n else 0.0,
        len(test_sorted) / n if n else 0.0,
    )
    return BenchmarkSplit(
        train=train_sorted,
        dev=dev_sorted,
        test=test_sorted,
        seed=seed,
        ratios=ratios,
    )


def _default_model_family(name: str) -> str:
    lower = str(name).strip().lower()
    for prefix, family in (
        ("claude", "claude"),
        ("gpt-", "gpt"),
        ("gpt_", "gpt"),
        ("qwen", "qwen"),
        ("llama", "llama"),
        ("gemma", "gemma"),
        ("deepseek", "deepseek"),
        ("kimi", "kimi"),
        ("glm", "glm"),
        ("phi", "phi"),
    ):
        if lower.startswith(prefix):
            return family
    return lower.split("-", 1)[0] or lower


def _rebalance_model_split(
    seen: list[str],
    held: list[str],
    *,
    target_seen: int,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    if len(seen) > target_seen:
        rng.shuffle(seen)
        held.extend(seen[target_seen:])
        seen = seen[:target_seen]
    elif len(seen) < target_seen:
        rng.shuffle(held)
        need = target_seen - len(seen)
        seen.extend(held[:need])
        held = held[need:]
    return seen, held


def split_models(
    model_names: Iterable[str],
    *,
    ratios: tuple[float, float] = (0.8, 0.2),
    seed: int = 0,
    strategy: str = "random",
) -> ModelSplit:
    """Disjoint random partition of the score-pattern model set F."""
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    strategy = str(strategy or "random").strip().lower()
    names = sorted(set(model_names))
    rng = random.Random(seed)
    n = len(names)
    n_seen = round(n * ratios[0])
    if n_seen > n:
        n_seen = n
    if strategy == "random":
        rng.shuffle(names)
        seen = sorted(names[:n_seen])
        held = sorted(names[n_seen:])
        return ModelSplit(seen=seen, held=held, seed=seed, ratios=ratios, strategy=strategy)
    if strategy not in {"family_stratified", "family"}:
        raise ValueError(
            "model split strategy must be one of {'random', 'family_stratified'}, "
            f"got {strategy!r}"
        )

    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(_default_model_family(name), []).append(name)
    seen: list[str] = []
    held: list[str] = []
    families = sorted(groups)
    rng.shuffle(families)
    singletons: list[str] = []
    for family in families:
        group = sorted(groups[family])
        rng.shuffle(group)
        if len(group) == 1:
            singletons.extend(group)
            continue
        k = round(len(group) * ratios[0])
        if 0.0 < ratios[0] < 1.0:
            k = min(max(1, k), len(group) - 1)
        seen.extend(group[:k])
        held.extend(group[k:])
    rng.shuffle(singletons)
    for name in singletons:
        if len(seen) < n_seen:
            seen.append(name)
        else:
            held.append(name)
    seen, held = _rebalance_model_split(seen, held, target_seen=n_seen, rng=rng)
    return ModelSplit(
        seen=sorted(seen),
        held=sorted(held),
        seed=seed,
        ratios=ratios,
        strategy="family_stratified",
    )


def induced_pair_set(benchmarks: Iterable[str]) -> list[PairKey]:
    """All (p, q) with p < q lex-sorted over the unique names in `benchmarks`."""
    names = sorted(set(benchmarks))
    pairs: list[PairKey] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j]))
    return pairs


def restrict_pair_dict(
    pair_dict: dict[PairKey, float | None],
    pair_set: Iterable[PairKey],
) -> dict[PairKey, float | None]:
    """Filter a pair-keyed dict to the given pair set."""
    keep = set(pair_set)
    return {k: v for k, v in pair_dict.items() if k in keep}


def split_pair_dict_by_benchmark(
    pair_dict: dict[PairKey, float | None],
    split: BenchmarkSplit,
) -> dict[str, dict[PairKey, float | None]]:
    """Partition a pair-keyed dict into train/dev/test using the benchmark split.

    A pair (p, q) is assigned to a bucket only if BOTH p and q are in that
    bucket. Cross-bucket pairs are dropped — they would leak between splits.
    """
    buckets = {
        "train": set(split.train),
        "dev": set(split.dev),
        "test": set(split.test),
    }
    out: dict[str, dict[PairKey, float | None]] = {k: {} for k in buckets}
    for (p, q), v in pair_dict.items():
        for name, members in buckets.items():
            if p in members and q in members:
                out[name][(p, q)] = v
                break
    return out
