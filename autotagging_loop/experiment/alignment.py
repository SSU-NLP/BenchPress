"""experiment/alignment.py — S, L_align, ρ_align, Δ_tag, bootstrap, error report.

Ω = {(p,q): p<q, R 정의됨}. 문서 v3 §1의 L_align 은 R_raw 위에서 MSE.
R01 기반 MSE는 range-normalized 보조 진단으로만 저장한다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import pearsonr, spearmanr


PairKey = tuple[str, str]


@dataclass
class ErrorPair:
    p: str
    q: str
    s_pq: float
    r_pq_raw: float
    r_pq_01: float
    delta: float
    type: Literal["false_sim", "false_dis"]


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
    benchmark_names: list[str] | None = None,
) -> dict[PairKey, float]:
    """S_{p,q} = cos(T_p, T_q) for p < q (lexicographic)."""
    names = sorted(set(benchmark_names if benchmark_names is not None else T.keys()))
    S: dict[PairKey, float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p, q = names[i], names[j]
            if p not in T or q not in T:
                continue
            S[(p, q)] = float(cosine_similarity(T[p], T[q]))
    return S


def _aligned_arrays(
    S: dict[PairKey, float],
    R: dict[PairKey, float | None],
) -> tuple[np.ndarray, np.ndarray, list[PairKey]]:
    keys: list[PairKey] = []
    s_vals: list[float] = []
    r_vals: list[float] = []
    for k, sv in S.items():
        rv = R.get(k)
        if rv is None:
            continue
        keys.append(k)
        s_vals.append(float(sv))
        r_vals.append(float(rv))
    return np.asarray(s_vals, dtype=float), np.asarray(r_vals, dtype=float), keys


def alignment_loss(S: dict[PairKey, float], R01: dict[PairKey, float | None]) -> float:
    """MSE(S, R) over Ω."""
    s, r, _ = _aligned_arrays(S, R01)
    if s.size == 0:
        return float("nan")
    return float(np.mean((s - r) ** 2))


def alignment_corr(
    S: dict[PairKey, float],
    R_raw: dict[PairKey, float | None],
) -> tuple[float, float]:
    """(pearson, spearman) of S vs R_raw over Ω."""
    s, r, _ = _aligned_arrays(S, R_raw)
    if s.size < 3 or len(set(s.tolist())) < 2 or len(set(r.tolist())) < 2:
        return (float("nan"), float("nan"))
    pr, _ = pearsonr(s, r)
    sp, _ = spearmanr(s, r)
    pr = float(pr) if not np.isnan(pr) else float("nan")
    sp = float(sp) if not np.isnan(sp) else float("nan")
    return (pr, sp)


def quantile_thresholds(
    values: dict[PairKey, float] | list[float] | np.ndarray,
    q_p: float = 0.80,
    q_n: float = 0.20,
) -> tuple[float, float]:
    if isinstance(values, dict):
        arr = np.asarray(list(values.values()), dtype=float)
    else:
        arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    return float(np.quantile(arr, q_p)), float(np.quantile(arr, q_n))


def intra_inter_gap(
    S: dict[PairKey, float],
    R_raw: dict[PairKey, float | None],
    theta_p: float,
    theta_n: float,
) -> dict[str, float]:
    """IntraTagSim = E[R_raw | S>=θ_p],  InterTagSim = E[R_raw | S<=θ_n]."""
    intra: list[float] = []
    inter: list[float] = []
    for k, sv in S.items():
        rv = R_raw.get(k)
        if rv is None:
            continue
        if sv >= theta_p:
            intra.append(float(rv))
        if sv <= theta_n:
            inter.append(float(rv))
    intra_mean = float(np.mean(intra)) if intra else float("nan")
    inter_mean = float(np.mean(inter)) if inter else float("nan")
    delta = (
        float("nan")
        if (np.isnan(intra_mean) or np.isnan(inter_mean))
        else intra_mean - inter_mean
    )
    return {
        "intra": intra_mean,
        "inter": inter_mean,
        "delta": delta,
        "n_pos": len(intra),
        "n_neg": len(inter),
    }


def bootstrap_metrics(
    S: dict[PairKey, float],
    R_raw: dict[PairKey, float | None],
    R01: dict[PairKey, float | None],
    B: int = 200,
    seed: int = 0,
    q_p: float = 0.80,
    q_n: float = 0.20,
) -> dict[str, dict[str, float]]:
    """Resample Ω with replacement; report mean/std for each metric."""
    s, r_raw, keys = _aligned_arrays(S, R_raw)
    _, r_01, _ = _aligned_arrays(S, R01)
    n = s.size
    if n == 0 or B <= 0:
        return {
            "L_align": {"mean": float("nan"), "std": float("nan")},
            "rho_pearson": {"mean": float("nan"), "std": float("nan")},
            "rho_spearman": {"mean": float("nan"), "std": float("nan")},
            "delta_tag": {"mean": float("nan"), "std": float("nan")},
        }
    rng = np.random.default_rng(seed)
    L_b: list[float] = []
    L01_b: list[float] = []
    rp_b: list[float] = []
    rs_b: list[float] = []
    dt_b: list[float] = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        ss = s[idx]
        rr = r_raw[idx]
        r01s = r_01[idx]
        L_b.append(float(np.mean((ss - rr) ** 2)))
        L01_b.append(float(np.mean((ss - r01s) ** 2)))
        if len(set(ss.tolist())) >= 2 and len(set(rr.tolist())) >= 2:
            pr, _ = pearsonr(ss, rr)
            sp, _ = spearmanr(ss, rr)
            rp_b.append(float(pr) if not np.isnan(pr) else float("nan"))
            rs_b.append(float(sp) if not np.isnan(sp) else float("nan"))
        tp = float(np.quantile(ss, q_p))
        tn = float(np.quantile(ss, q_n))
        intra = rr[ss >= tp]
        inter = rr[ss <= tn]
        if intra.size and inter.size:
            dt_b.append(float(intra.mean() - inter.mean()))

    def summary(vals: list[float]) -> dict[str, float]:
        arr = np.asarray([v for v in vals if not np.isnan(v)], dtype=float)
        if arr.size == 0:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}

    return {
        "L_align": summary(L_b),
        "L_align_01": summary(L01_b),
        "rho_pearson": summary(rp_b),
        "rho_spearman": summary(rs_b),
        "delta_tag": summary(dt_b),
    }


def bootstrap_metrics_block(
    S: dict[PairKey, float],
    R_raw: dict[PairKey, float | None],
    R01: dict[PairKey, float | None],
    benchmark_names: list[str],
    B: int = 1000,
    seed: int = 0,
    q_p: float = 0.80,
    q_n: float = 0.20,
) -> dict[str, dict[str, float]]:
    """v3 §2.2.7 block bootstrap. Resample benchmarks with replacement and
    induce all upper-triangle pairs from the sample on every iteration.

    Pair-level resampling (`bootstrap_metrics`) treats each (p, q) as i.i.d.,
    which understates uncertainty when pairs share a benchmark. The block
    variant resamples the benchmark set itself, then derives the pair set,
    so the dependence structure of the score-pattern matrix is preserved.
    """
    names = list(benchmark_names)
    n = len(names)
    if n < 2 or B <= 0:
        nan = {"mean": float("nan"), "std": float("nan")}
        return {
            "L_align": dict(nan),
            "L_align_01": dict(nan),
            "rho_pearson": dict(nan),
            "rho_spearman": dict(nan),
            "delta_tag": dict(nan),
        }
    # Pre-materialize dense pair matrices indexed by name position. Pairs not
    # present in S/R_raw/R01 become NaN and are masked per iteration.
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    S_mat = np.full((n, n), np.nan, dtype=float)
    R_mat = np.full((n, n), np.nan, dtype=float)
    R01_mat = np.full((n, n), np.nan, dtype=float)
    for (p, q), sv in S.items():
        if p in name_to_idx and q in name_to_idx and p < q:
            i, j = name_to_idx[p], name_to_idx[q]
            S_mat[i, j] = float(sv)
    for (p, q), rv in R_raw.items():
        if p in name_to_idx and q in name_to_idx and p < q and rv is not None:
            i, j = name_to_idx[p], name_to_idx[q]
            R_mat[i, j] = float(rv)
    for (p, q), rv in R01.items():
        if p in name_to_idx and q in name_to_idx and p < q and rv is not None:
            i, j = name_to_idx[p], name_to_idx[q]
            R01_mat[i, j] = float(rv)

    ti, tj = np.triu_indices(n, k=1)  # position pairs (i, j) with i < j

    rng = np.random.default_rng(seed)
    L_b: list[float] = []
    L01_b: list[float] = []
    rp_b: list[float] = []
    rs_b: list[float] = []
    dt_b: list[float] = []
    for _ in range(int(B)):
        idx = rng.integers(0, n, size=n)
        si = idx[ti]
        sj = idx[tj]
        lo = np.minimum(si, sj)
        hi = np.maximum(si, sj)
        s_gather = S_mat[lo, hi]
        r_gather = R_mat[lo, hi]
        r01_gather = R01_mat[lo, hi]
        valid = (
            (si != sj)
            & ~np.isnan(s_gather)
            & ~np.isnan(r_gather)
            & ~np.isnan(r01_gather)
        )
        if not np.any(valid):
            continue
        ss = s_gather[valid]
        rr = r_gather[valid]
        r01s = r01_gather[valid]
        L_b.append(float(np.mean((ss - rr) ** 2)))
        L01_b.append(float(np.mean((ss - r01s) ** 2)))
        if len(set(ss.tolist())) >= 2 and len(set(rr.tolist())) >= 2:
            pr, _ = pearsonr(ss, rr)
            sp, _ = spearmanr(ss, rr)
            if not np.isnan(pr):
                rp_b.append(float(pr))
            if not np.isnan(sp):
                rs_b.append(float(sp))
        tp = float(np.quantile(ss, q_p))
        tn = float(np.quantile(ss, q_n))
        intra = rr[ss >= tp]
        inter = rr[ss <= tn]
        if intra.size and inter.size:
            dt_b.append(float(intra.mean() - inter.mean()))

    def summary(vals: list[float]) -> dict[str, float]:
        arr = np.asarray([v for v in vals if not np.isnan(v)], dtype=float)
        if arr.size == 0:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}

    return {
        "L_align": summary(L_b),
        "L_align_01": summary(L01_b),
        "rho_pearson": summary(rp_b),
        "rho_spearman": summary(rs_b),
        "delta_tag": summary(dt_b),
    }


def block_bootstrap_ci(
    values: list[float],
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile (1−α) confidence interval. NaNs are dropped."""
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    lo = float(np.quantile(arr, alpha / 2))
    hi = float(np.quantile(arr, 1 - alpha / 2))
    return lo, hi


def build_error_report(
    S: dict[PairKey, float],
    R_raw: dict[PairKey, float | None],
    R01: dict[PairKey, float | None],
    top_k: int = 20,
    q_p_s: float = 0.80,
    q_n_s: float = 0.20,
    q_p_r: float = 0.80,
    q_n_r: float = 0.20,
) -> list[ErrorPair]:
    """false_sim/false_dis pairs by S quantile vs R01 quantile, top-k by |s - r01|."""
    s_vals: list[float] = []
    r01_vals: list[float] = []
    keys: list[PairKey] = []
    for k, sv in S.items():
        rv01 = R01.get(k)
        if rv01 is None:
            continue
        keys.append(k)
        s_vals.append(float(sv))
        r01_vals.append(float(rv01))

    if not keys:
        return []

    s_arr = np.asarray(s_vals)
    r01_arr = np.asarray(r01_vals)
    s_hi = float(np.quantile(s_arr, q_p_s))
    s_lo = float(np.quantile(s_arr, q_n_s))
    r_hi = float(np.quantile(r01_arr, q_p_r))
    r_lo = float(np.quantile(r01_arr, q_n_r))

    pairs: list[ErrorPair] = []
    for (p, q), sv, rv01 in zip(keys, s_vals, r01_vals):
        rv_raw = R_raw.get((p, q))
        if rv_raw is None:
            continue
        delta = abs(sv - rv01)
        if sv >= s_hi and rv01 <= r_lo:
            pairs.append(ErrorPair(p, q, sv, float(rv_raw), rv01, delta, "false_sim"))
        elif sv <= s_lo and rv01 >= r_hi:
            pairs.append(ErrorPair(p, q, sv, float(rv_raw), rv01, delta, "false_dis"))

    pairs.sort(key=lambda x: x.delta, reverse=True)
    return pairs[:top_k]


def build_residual_report(
    S: dict[PairKey, float],
    R_raw: dict[PairKey, float | None],
    top_k: int = 20,
) -> list[dict]:
    """Largest raw residuals |S_pq - R_pq| for post-Part-1 diagnosis only."""
    rows: list[dict] = []
    for (p, q), sv in S.items():
        rv = R_raw.get((p, q))
        if rv is None:
            continue
        residual = abs(float(sv) - float(rv))
        rows.append(
            {
                "p": p,
                "q": q,
                "s_pq": float(sv),
                "r_pq_raw": float(rv),
                "residual_abs": float(residual),
                "direction": "tag_similarity_too_high" if float(sv) > float(rv) else "tag_similarity_too_low",
                "part1_action": "keep_seed_vocabulary_fixed",
                "post_part1_use": "candidate_residual_for_taxonomy_refinement",
            }
        )
    rows.sort(key=lambda row: row["residual_abs"], reverse=True)
    return rows[:top_k]


def paired_permutation_test(
    a_per_fold: list[float],
    b_per_fold: list[float],
    B: int = 10000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Two-tailed paired permutation test for paired-fold metrics.

    For each iteration the sign of every (b - a) pair is flipped uniformly at random.
    Reported `p_value` is the two-tailed proportion of permuted mean-diffs whose
    magnitude is at least the observed |mean(b) - mean(a)|. Empty / nan-only inputs
    yield `p_value=nan`.
    """
    a_arr = np.asarray([float(x) for x in a_per_fold], dtype=float)
    b_arr = np.asarray([float(x) for x in b_per_fold], dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"paired_permutation_test: shape mismatch a={a_arr.shape} b={b_arr.shape}"
        )
    finite = np.isfinite(a_arr) & np.isfinite(b_arr)
    a = a_arr[finite]
    b = b_arr[finite]
    n = a.size
    if n == 0 or B <= 0:
        return {
            "p_value": float("nan"),
            "B_iters": int(B),
            "observed_diff": float("nan"),
            "n_pairs_used": int(n),
        }
    diff = b - a
    observed = float(np.mean(diff))
    abs_obs = abs(observed)
    rng = np.random.default_rng(seed)
    # Vectorized sign-flip permutation: draw all (B, n) signs in one shot,
    # compute B permuted means with a single matrix-vector mean. RNG stream
    # differs from the per-iteration version, but the test is distributional.
    signs = rng.choice(np.array([-1.0, 1.0], dtype=float), size=(int(B), n))
    permuted_means = (signs * diff[None, :]).mean(axis=1)
    extreme = int(np.sum(np.abs(permuted_means) >= abs_obs))
    # Add 1 to numerator and denominator for stability (standard permutation-test trick)
    p_value = (extreme + 1) / (int(B) + 1)
    return {
        "p_value": float(p_value),
        "B_iters": int(B),
        "observed_diff": observed,
        "n_pairs_used": int(n),
    }


def error_pairs_to_dicts(pairs: list[ErrorPair]) -> list[dict]:
    return [
        {
            "p": p.p,
            "q": p.q,
            "s_pq": p.s_pq,
            "r_pq_raw": p.r_pq_raw,
            "r_pq_01": p.r_pq_01,
            "delta": p.delta,
            "type": p.type,
        }
        for p in pairs
    ]
