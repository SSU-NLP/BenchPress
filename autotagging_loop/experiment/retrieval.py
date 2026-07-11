"""v3 §2.2.10 Recall@K retrieval metric.

For each benchmark in the test split, rank all other benchmarks by tag
similarity (top-k_S) and by score-pattern similarity (top-k_R). Recall@K
is the mean fraction of the top-k_R set recovered inside the top-k_S set.

Inputs are pair-keyed dicts produced upstream by `experiment/alignment.py`
and `experiment/score_matrix.py` so this module stays decoupled from the
loop.
"""

from __future__ import annotations


PairKey = tuple[str, str]


def _pair_value(d: dict[PairKey, float | None], a: str, b: str) -> float | None:
    if a == b:
        return None
    key = (a, b) if a < b else (b, a)
    return d.get(key)


def _top_k(query: str, others: list[str], pair_d: dict[PairKey, float | None], k: int) -> list[str]:
    """Top-k neighbors of `query` ranked by descending pair_d value. None values dropped.

    Ties broken by lex order on the neighbor name so the ranking is reproducible.
    """
    scored: list[tuple[float, str]] = []
    for other in others:
        if other == query:
            continue
        v = _pair_value(pair_d, query, other)
        if v is None:
            continue
        scored.append((float(v), other))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [name for _, name in scored[: max(0, int(k))]]


def recall_at_k(
    *,
    benchmark_names: list[str],
    S: dict[PairKey, float],
    R: dict[PairKey, float | None],
    k_values: list[int] = (1, 3, 5),
) -> dict:
    """Mean recall@k across benchmarks.

    Returns:
        {
          "k_values": [...],
          "per_k": {str(k): {"mean": float, "n_benchmarks": int}},
          "per_benchmark": {b: {str(k): float}},
        }

    Benchmarks for which R has fewer than `k` defined neighbors are excluded
    from the per-k mean (their recall denominator would be 0).
    """
    benches = list(benchmark_names)
    per_benchmark: dict[str, dict[str, float]] = {}
    per_k_sum: dict[int, list[float]] = {int(k): [] for k in k_values}
    for query in benches:
        per_benchmark[query] = {}
        for k_raw in k_values:
            k = int(k_raw)
            r_top = _top_k(query, benches, R, k)
            if len(r_top) < k:
                continue
            s_top = _top_k(query, benches, S, k)
            hit = len(set(r_top) & set(s_top))
            recall = hit / float(k)
            per_benchmark[query][str(k)] = recall
            per_k_sum[k].append(recall)
    per_k_summary = {
        str(k): {
            "mean": float(sum(vals) / len(vals)) if vals else float("nan"),
            "n_benchmarks": len(vals),
        }
        for k, vals in per_k_sum.items()
    }
    return {
        "k_values": [int(k) for k in k_values],
        "per_k": per_k_summary,
        "per_benchmark": per_benchmark,
    }
