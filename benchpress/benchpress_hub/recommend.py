"""Tag-driven recommendation: benchmarks by tag relevance, models by leaderboard scores."""

from __future__ import annotations

import math


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity over the union of keys; zero-norm input yields 0.0."""
    keys = set(a) | set(b)
    dot = sum(float(a.get(k, 0.0)) * float(b.get(k, 0.0)) for k in keys)
    norm_a = math.sqrt(sum(float(v) ** 2 for v in a.values()))
    norm_b = math.sqrt(sum(float(v) ** 2 for v in b.values()))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def relevance_ranking(
    tag_scores: dict[str, dict[str, float]],
    target_tags: list[str] | dict[str, float],
) -> list[tuple[str, float]]:
    """Rank every benchmark by cosine similarity to the target tag-weight vector."""
    weights = dict(target_tags) if isinstance(target_tags, dict) else {t: 1.0 for t in target_tags}
    if not weights:
        raise ValueError("target_tags is empty")
    known = {tag for vector in tag_scores.values() for tag in vector}
    if not set(weights) & known:
        raise ValueError(f"no target tag appears in any benchmark vector: {sorted(weights)}")
    ranking = [(bench, _cosine(vector, weights)) for bench, vector in tag_scores.items()]
    return sorted(ranking, key=lambda item: (-item[1], item[0]))


def rank_models(leaderboard: dict[str, dict], benchmarks: list[str]) -> list[tuple[str, float]]:
    """Mean min-max-normalized score per model over the given benchmarks.

    Only models scored on every given benchmark that exists in the leaderboard
    are ranked; returns [] when no benchmark overlaps the leaderboard.
    """
    normalized: list[dict[str, float]] = []
    for bench in benchmarks:
        scores = leaderboard.get(bench)
        if bench.startswith("_") or not isinstance(scores, dict):
            continue
        numeric = {
            m: float(s)
            for m, s in scores.items()
            if isinstance(s, (int, float)) and not isinstance(s, bool)
        }
        lo, hi = (min(numeric.values()), max(numeric.values())) if numeric else (0.0, 0.0)
        if hi == lo:
            normalized.append({m: 0.5 for m in numeric})
        else:
            normalized.append({m: (s - lo) / (hi - lo) for m, s in numeric.items()})
    if not normalized:
        return []
    covered = set.intersection(*(set(n) for n in normalized))
    ranked = [(m, sum(n[m] for n in normalized) / len(normalized)) for m in covered]
    return sorted(ranked, key=lambda item: (-item[1], item[0]))
