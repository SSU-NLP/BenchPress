"""Alignment metrics for Part 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr

PairKey = tuple[str, str]


@dataclass
class ResidualPair:
    p: str
    q: str
    s_pq: float
    r_pq_raw: float
    residual_abs: float
    direction: str

    def as_dict(self) -> dict:
        return {
            "p": self.p,
            "q": self.q,
            "s_pq": self.s_pq,
            "r_pq_raw": self.r_pq_raw,
            "residual_abs": self.residual_abs,
            "direction": self.direction,
        }


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) | set(b))
    if not keys:
        return 0.0
    va = np.asarray([float(a.get(k, 0.0)) for k in keys], dtype=float)
    vb = np.asarray([float(b.get(k, 0.0)) for k in keys], dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def cosine_pair_matrix(
    T: dict[str, dict[str, float]],
    benchmark_names: list[str],
) -> dict[PairKey, float]:
    S: dict[PairKey, float] = {}
    for i, p in enumerate(benchmark_names):
        for q in benchmark_names[i + 1:]:
            if p in T and q in T:
                S[(p, q)] = cosine_similarity(T[p], T[q])
    return S


def aligned_arrays(
    S: dict[PairKey, float],
    R: dict[PairKey, float | None],
) -> tuple[np.ndarray, np.ndarray, list[PairKey]]:
    keys: list[PairKey] = []
    s_vals: list[float] = []
    r_vals: list[float] = []
    for key, s_value in S.items():
        r_value = R.get(key)
        if r_value is None:
            continue
        keys.append(key)
        s_vals.append(float(s_value))
        r_vals.append(float(r_value))
    return np.asarray(s_vals, dtype=float), np.asarray(r_vals, dtype=float), keys


def alignment_loss(S: dict[PairKey, float], R: dict[PairKey, float | None]) -> float:
    s, r, _ = aligned_arrays(S, R)
    if s.size == 0:
        return float("nan")
    return float(np.mean((s - r) ** 2))


def alignment_corr(S: dict[PairKey, float], R: dict[PairKey, float | None]) -> tuple[float, float]:
    s, r, _ = aligned_arrays(S, R)
    if s.size < 3 or len(set(s.tolist())) < 2 or len(set(r.tolist())) < 2:
        return float("nan"), float("nan")
    pr, _ = pearsonr(s, r)
    sp, _ = spearmanr(s, r)
    return float(pr) if np.isfinite(pr) else float("nan"), float(sp) if np.isfinite(sp) else float("nan")


def quantile_thresholds(values: dict[PairKey, float], q_p: float = 0.80, q_n: float = 0.20) -> tuple[float, float]:
    arr = np.asarray(list(values.values()), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(arr, q_p)), float(np.quantile(arr, q_n))


def intra_inter_gap(
    S: dict[PairKey, float],
    R_raw: dict[PairKey, float | None],
    theta_p: float,
    theta_n: float,
) -> dict[str, float]:
    intra: list[float] = []
    inter: list[float] = []
    for key, s_value in S.items():
        r_value = R_raw.get(key)
        if r_value is None:
            continue
        if s_value >= theta_p:
            intra.append(float(r_value))
        if s_value <= theta_n:
            inter.append(float(r_value))
    intra_mean = float(np.mean(intra)) if intra else float("nan")
    inter_mean = float(np.mean(inter)) if inter else float("nan")
    delta = float("nan") if np.isnan(intra_mean) or np.isnan(inter_mean) else intra_mean - inter_mean
    return {"intra": intra_mean, "inter": inter_mean, "delta": delta, "n_pos": len(intra), "n_neg": len(inter)}


def bootstrap_metrics(
    S: dict[PairKey, float],
    R_raw: dict[PairKey, float | None],
    B: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    s, r, _ = aligned_arrays(S, R_raw)
    if s.size == 0 or B <= 0:
        return {}
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    pearsons: list[float] = []
    spearmans: list[float] = []
    deltas: list[float] = []
    n = s.size
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        ss = s[idx]
        rr = r[idx]
        losses.append(float(np.mean((ss - rr) ** 2)))
        if len(set(ss.tolist())) >= 2 and len(set(rr.tolist())) >= 2:
            pr, _ = pearsonr(ss, rr)
            sp, _ = spearmanr(ss, rr)
            if np.isfinite(pr):
                pearsons.append(float(pr))
            if np.isfinite(sp):
                spearmans.append(float(sp))
        hi = float(np.quantile(ss, 0.80))
        lo = float(np.quantile(ss, 0.20))
        intra = rr[ss >= hi]
        inter = rr[ss <= lo]
        if intra.size and inter.size:
            deltas.append(float(intra.mean() - inter.mean()))

    def summary(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}

    return {
        "L_align": summary(losses),
        "rho_pearson": summary(pearsons),
        "rho_spearman": summary(spearmans),
        "delta_tag": summary(deltas),
    }


def residual_report(
    S: dict[PairKey, float],
    R_raw: dict[PairKey, float | None],
    *,
    top_k: int = 20,
) -> list[dict]:
    rows: list[ResidualPair] = []
    for (p, q), s_value in S.items():
        r_value = R_raw.get((p, q))
        if r_value is None:
            continue
        residual = abs(float(s_value) - float(r_value))
        direction = "tag_similarity_too_high" if s_value > r_value else "tag_similarity_too_low"
        rows.append(ResidualPair(p, q, float(s_value), float(r_value), residual, direction))
    rows.sort(key=lambda item: item.residual_abs, reverse=True)
    return [item.as_dict() for item in rows[:top_k]]


def compute_metrics(
    T: dict[str, dict[str, float]],
    benchmark_names: list[str],
    R_raw: dict[PairKey, float | None],
    *,
    bootstrap_B: int,
    seed: int,
) -> tuple[dict, dict[PairKey, float], list[dict]]:
    S = cosine_pair_matrix(T, benchmark_names)
    L = alignment_loss(S, R_raw)
    rho_p, rho_s = alignment_corr(S, R_raw)
    theta_p, theta_n = quantile_thresholds(S)
    gap = intra_inter_gap(S, R_raw, theta_p, theta_n)
    report = residual_report(S, R_raw)
    all_residuals = [
        abs(float(s_value) - float(r_value))
        for key, s_value in S.items()
        for r_value in [R_raw.get(key)]
        if r_value is not None
    ]
    metrics = {
        "L_align": L,
        "rho_align_pearson": rho_p,
        "rho_align_spearman": rho_s,
        "delta_tag": gap["delta"],
        "intra_tag_score_similarity": gap["intra"],
        "inter_tag_score_similarity": gap["inter"],
        "n_pairs": sum(1 for value in R_raw.values() if value is not None),
        "residual_mean": float(np.mean(all_residuals)) if all_residuals else float("nan"),
        "residual_max": float(np.max(all_residuals)) if all_residuals else float("nan"),
        "bootstrap": bootstrap_metrics(S, R_raw, bootstrap_B, seed),
    }
    return metrics, S, report
