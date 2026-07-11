"""scripts/permutation_test_run.py — Null test for an existing v3 run.

Answers "is the test ρ_s observed at the end of a run different from chance?"
The CI on a single run's test ρ_s is too wide to decide (bootstrap σ ≈ 0.5 on
n_pairs=10), so we shuffle the truth vector against the predicted similarity
B times and report the two-tailed (and one-tailed lower) p-value for ρ_s, ρ_p,
and L_align on each of train / dev / test.

Usage:
    python scripts/permutation_test_run.py --run-dir results/part2_experiment/run_20260511_140200
        [--B 10000] [--seed 0]

Outputs `<run-dir>/analysis/permutation_test.json` and prints a markdown
summary + ASCII histogram of the null distribution per split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Iterable

import numpy as np
from scipy.stats import pearsonr, spearmanr

# Make `experiment.*` importable when this script is run as `python scripts/...`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autotagging_loop.experiment.alignment import alignment_loss, cosine_pair_matrix
from autotagging_loop.experiment.splits import induced_pair_set, restrict_pair_dict


PairKey = tuple[str, str]


def _parse_pair_key(s: str) -> PairKey:
    """JSON-serialized 'p||q' → (p, q)."""
    p, _, q = s.partition("||")
    return (p, q)


def _load_pair_dict(d: dict, *, drop_none: bool = False) -> dict[PairKey, float | None]:
    out: dict[PairKey, float | None] = {}
    for k, v in d.items():
        key = _parse_pair_key(k)
        if v is None:
            if drop_none:
                continue
            out[key] = None
        else:
            out[key] = float(v)
    return out


def _block_arrays(
    S: dict[PairKey, float],
    R_raw: dict[PairKey, float | None],
    R01: dict[PairKey, float | None],
    benches: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[PairKey]]:
    """Mirror `_block_metrics` filtering: keep pairs where R_raw is not None."""
    pair_set = induced_pair_set(benches)
    S_b = restrict_pair_dict(S, pair_set)
    R_b = restrict_pair_dict(R_raw, pair_set)
    R01_b = restrict_pair_dict(R01, pair_set)
    keys: list[PairKey] = []
    s_vals: list[float] = []
    r_vals: list[float] = []
    r01_vals: list[float] = []
    for k, rv in R_b.items():
        if rv is None:
            continue
        sv = S_b.get(k)
        if sv is None:
            continue
        rv01 = R01_b.get(k)
        if rv01 is None:
            continue
        sv_f, rv_f, rv01_f = float(sv), float(rv), float(rv01)
        if not (math.isfinite(sv_f) and math.isfinite(rv_f) and math.isfinite(rv01_f)):
            continue
        keys.append(k)
        s_vals.append(sv_f)
        r_vals.append(rv_f)
        r01_vals.append(rv01_f)
    return (
        np.asarray(s_vals, dtype=float),
        np.asarray(r_vals, dtype=float),
        np.asarray(r01_vals, dtype=float),
        keys,
    )


def _permutation_test(
    s: np.ndarray,
    r_raw: np.ndarray,
    r01: np.ndarray,
    *,
    B: int,
    seed: int,
) -> dict:
    """Permutation null for ρ_p, ρ_s, and L_align.

    Reasoning for what gets shuffled:
        - ρ uses R_raw to match `alignment_corr` in split_metrics.
        - L_align uses R_raw to match `alignment_loss` in split_metrics.
        - L_align_01 is kept as a range-normalized auxiliary diagnostic.
        - In each case we shuffle the *truth* vector against the fixed S; this
          tests the null "predicted and truth are independent" without altering
          the marginal distributions of either vector.
    """
    n = s.size
    if n < 3 or len(set(s.tolist())) < 2 or len(set(r_raw.tolist())) < 2:
        return {
            "n_pairs": int(n),
            "skipped": "insufficient_variation",
        }

    obs_pr = float(pearsonr(s, r_raw).statistic)
    obs_sp = float(spearmanr(s, r_raw).statistic)
    obs_L = float(np.mean((s - r_raw) ** 2))
    obs_L01 = float(np.mean((s - r01) ** 2))

    rng = np.random.default_rng(seed)
    perm_pr = np.empty(B, dtype=float)
    perm_sp = np.empty(B, dtype=float)
    perm_L = np.empty(B, dtype=float)
    perm_L01 = np.empty(B, dtype=float)
    for i in range(B):
        idx = rng.permutation(n)
        r_shuf_raw = r_raw[idx]
        r_shuf_01 = r01[idx]
        perm_pr[i] = pearsonr(s, r_shuf_raw).statistic
        perm_sp[i] = spearmanr(s, r_shuf_raw).statistic
        perm_L[i] = float(np.mean((s - r_shuf_raw) ** 2))
        perm_L01[i] = float(np.mean((s - r_shuf_01) ** 2))

    def _p_two(perm: np.ndarray, obs: float) -> float:
        return float((np.sum(np.abs(perm) >= abs(obs)) + 1) / (B + 1))

    def _p_one_neg(perm: np.ndarray, obs: float) -> float:
        return float((np.sum(perm <= obs) + 1) / (B + 1))

    def _p_one_low(perm: np.ndarray, obs: float) -> float:
        # L_align: lower is better, so "low tail" = p-value for our gain
        return float((np.sum(perm <= obs) + 1) / (B + 1))

    def _summary(perm: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(perm)),
            "std": float(np.std(perm, ddof=1)) if perm.size > 1 else 0.0,
            "pct_2_5": float(np.percentile(perm, 2.5)),
            "pct_97_5": float(np.percentile(perm, 97.5)),
        }

    return {
        "n_pairs": int(n),
        "rho_pearson": {
            "observed": obs_pr,
            "p_two_sided": _p_two(perm_pr, obs_pr),
            "p_one_sided_neg": _p_one_neg(perm_pr, obs_pr),
            "null": _summary(perm_pr),
        },
        "rho_spearman": {
            "observed": obs_sp,
            "p_two_sided": _p_two(perm_sp, obs_sp),
            "p_one_sided_neg": _p_one_neg(perm_sp, obs_sp),
            "null": _summary(perm_sp),
        },
        "L_align": {
            "observed": obs_L,
            "p_one_sided_low": _p_one_low(perm_L, obs_L),
            "null": _summary(perm_L),
        },
        "L_align_01": {
            "observed": obs_L01,
            "p_one_sided_low": _p_one_low(perm_L01, obs_L01),
            "null": _summary(perm_L01),
        },
        "_perm_rho_spearman": perm_sp.tolist(),  # for histogram
    }


def _ascii_hist(values: Iterable[float], *, obs: float, bins: int = 30, width: int = 40) -> str:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return "(empty)"
    lo = min(float(np.min(arr)), obs)
    hi = max(float(np.max(arr)), obs)
    if hi <= lo:
        hi = lo + 1e-9
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    cmax = int(np.max(counts)) if counts.size else 0
    if cmax == 0:
        return "(flat)"
    lines: list[str] = []
    obs_bin = int(np.clip(np.searchsorted(edges, obs) - 1, 0, bins - 1))
    for i, c in enumerate(counts):
        bar_len = int(round(width * c / cmax))
        marker = "<" if i == obs_bin else " "
        lines.append(
            f"  [{edges[i]:+.3f}, {edges[i+1]:+.3f}) | {'#' * bar_len:<{width}} {c:>5d} {marker}"
        )
    return "\n".join(lines)


def _selected_final_dir(run_dir: str) -> tuple[str, dict]:
    selection_path = os.path.join(run_dir, "selection.json")
    final_dir = os.path.join(run_dir, "final")
    selection: dict = {}
    if os.path.exists(selection_path):
        selection = json.load(open(selection_path))
        selected_source = selection.get("selected_source")
        mode = selection.get("mode")
        if selected_source == "taxonomy_refinement" or mode == "taxonomy":
            rel_path = selection.get("taxonomy_path") or "taxonomy_refinement/final"
            tax_dir = os.path.join(run_dir, rel_path)
            if os.path.exists(os.path.join(tax_dir, "T_star.json")):
                return tax_dir, selection
    return final_dir, selection


def _load_run_artifacts(run_dir: str):
    """Returns selected T plus split metadata for a single v3 run dir."""
    sm_path = os.path.join(run_dir, "score_matrix.json")
    selected_dir, selection = _selected_final_dir(run_dir)
    selected_splits_path = os.path.join(selected_dir, "split_metrics.json")
    fixed_splits_path = os.path.join(run_dir, "final", "split_metrics.json")
    splits_path = selected_splits_path if os.path.exists(selected_splits_path) else fixed_splits_path
    t_path = os.path.join(selected_dir, "T_star.json")
    for p in (sm_path, splits_path, t_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing artifact: {p}")
    sm = json.load(open(sm_path))
    splits_data = json.load(open(splits_path))
    T_star = json.load(open(t_path))
    R_raw = _load_pair_dict(sm["R_raw"])
    R01 = _load_pair_dict(sm["R01"])
    artifact_info = {
        "selected_final_dir": selected_dir,
        "split_metrics_path": splits_path,
        "selection": selection,
        "strict_reproduction_check": (
            os.path.commonpath([os.path.abspath(splits_path), os.path.abspath(selected_dir)])
            == os.path.abspath(selected_dir)
        ),
    }
    return R_raw, R01, T_star, splits_data, artifact_info


def _gather_test_block(run_dir: str) -> dict:
    """Return arrays + benchmarks for the test split of a run, for pooling across folds."""
    R_raw, R01, T_star, splits_data, artifact_info = _load_run_artifacts(run_dir)
    bench_names = sorted(T_star.keys())
    S = cosine_pair_matrix(T_star, benchmark_names=bench_names)
    test_benches = list(splits_data["benchmark_split"]["test"])
    s, r, r01v, keys = _block_arrays(S, R_raw, R01, test_benches)
    raw_sp = splits_data["test"].get("rho_align_spearman")
    stored_sp = float(raw_sp) if raw_sp is not None else float("nan")
    return {
        "run_dir": run_dir,
        "selected_final_dir": artifact_info["selected_final_dir"],
        "selected_source": (artifact_info["selection"] or {}).get("selected_source", "fixed"),
        "test_benches": test_benches,
        "s": s, "r_raw": r, "r01": r01v, "keys": keys,
        "stored_test_rho_spearman": stored_sp,
    }


def run_single(run_dir: str, *, B: int = 10_000, seed: int = 0) -> dict:
    """Per-run permutation test on train/dev/test. Writes <run_dir>/analysis/permutation_test.json."""
    run_dir = os.path.abspath(run_dir)
    R_raw, R01, T_star, splits_data, artifact_info = _load_run_artifacts(run_dir)
    bench_names = sorted(T_star.keys())
    S = cosine_pair_matrix(T_star, benchmark_names=bench_names)

    test_benches = list(splits_data["benchmark_split"]["test"])
    s_t, r_t, r01_t, _ = _block_arrays(S, R_raw, R01, test_benches)
    if s_t.size >= 3 and len(set(s_t.tolist())) >= 2 and len(set(r_t.tolist())) >= 2:
        sp_repro = float(spearmanr(s_t, r_t).statistic)
    else:
        sp_repro = float("nan")
    stored_sp = float(splits_data["test"]["rho_align_spearman"])
    diff = abs(sp_repro - stored_sp) if not math.isnan(sp_repro) else float("inf")
    if artifact_info["strict_reproduction_check"] and diff > 1e-9:
        raise RuntimeError(
            f"reproduction mismatch in {run_dir}: stored={stored_sp:.9f} "
            f"reproduced={sp_repro:.9f} diff={diff:.2e}"
        )

    out_splits: dict[str, dict] = {}
    for split_name in ("train", "dev", "test"):
        benches = list(splits_data["benchmark_split"][split_name])
        s, r, r01v, keys = _block_arrays(S, R_raw, R01, benches)
        result = _permutation_test(s, r, r01v, B=B, seed=seed)
        result["n_benchmarks"] = len(benches)
        result["pair_keys"] = ["||".join(k) for k in keys]
        out_splits[split_name] = result

    persisted = {
        "run_dir": run_dir,
        "B": B,
        "rng_seed": seed,
        "best_iter": _read_best_iter(run_dir),
        "selected_source": (artifact_info["selection"] or {}).get("selected_source", "fixed"),
        "selected_final_dir": artifact_info["selected_final_dir"],
        "split_metrics_path": artifact_info["split_metrics_path"],
        "reproduction_check": {
            "stored_test_rho_s": stored_sp,
            "reproduced_test_rho_s": sp_repro,
            "abs_diff": diff,
            "strict": artifact_info["strict_reproduction_check"],
        },
        "splits": {
            name: {k: v for k, v in payload.items() if not k.startswith("_perm_")}
            for name, payload in out_splits.items()
        },
    }
    out_dir = os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "permutation_test.json")
    with open(out_path, "w") as fh:
        json.dump(persisted, fh, indent=2, sort_keys=True)
    return {
        "run_dir": run_dir,
        "out_path": out_path,
        "reproduction": {
            "stored": stored_sp,
            "reproduced": sp_repro,
            "diff": diff,
            "strict": artifact_info["strict_reproduction_check"],
        },
        "splits": out_splits,
    }


def run_pooled(
    fold_dirs: list[str],
    *,
    out_path: str,
    B: int = 10_000,
    seed: int = 0,
) -> dict:
    """K-fold pooled permutation test: concatenate test pairs across folds.

    Each fold contributes its own test split (disjoint by K-fold construction).
    The pooled (S, R_raw, R01) arrays go through one permutation test on the
    full ~K·C(test_size,2) pair set. Per-fold permutation results are also
    computed and included for breakdown.
    """
    if not fold_dirs:
        raise ValueError("run_pooled requires at least one fold directory")

    per_fold = []
    pooled_s: list[np.ndarray] = []
    pooled_r: list[np.ndarray] = []
    pooled_r01: list[np.ndarray] = []
    pooled_keys: list[PairKey] = []
    for fd in fold_dirs:
        block = _gather_test_block(fd)
        pf_perm = _permutation_test(
            block["s"], block["r_raw"], block["r01"], B=B, seed=seed,
        )
        pf_perm["n_benchmarks"] = len(block["test_benches"])
        pf_perm["pair_keys"] = ["||".join(k) for k in block["keys"]]
        pf_perm["stored_test_rho_spearman"] = block["stored_test_rho_spearman"]
        per_fold.append({
            "fold_dir": fd,
            "test_benches": block["test_benches"],
            "permutation": {k: v for k, v in pf_perm.items() if not k.startswith("_perm_")},
        })
        pooled_s.append(block["s"])
        pooled_r.append(block["r_raw"])
        pooled_r01.append(block["r01"])
        pooled_keys.extend(block["keys"])

    s_all = np.concatenate(pooled_s)
    r_all = np.concatenate(pooled_r)
    r01_all = np.concatenate(pooled_r01)

    seen_keys: set[PairKey] = set()
    dups: list[PairKey] = []
    for k in pooled_keys:
        if k in seen_keys:
            dups.append(k)
        else:
            seen_keys.add(k)
    if dups:
        raise RuntimeError(
            f"pooled test pairs contain duplicates across folds: {dups[:5]}... "
            f"({len(dups)} total). K-fold test sets must be disjoint."
        )

    pooled_perm = _permutation_test(s_all, r_all, r01_all, B=B, seed=seed)
    pooled_perm["n_benchmarks"] = sum(
        len(pf["test_benches"]) for pf in per_fold
    )
    pooled_perm["pair_keys"] = ["||".join(k) for k in pooled_keys]

    persisted = {
        "fold_dirs": [os.path.abspath(fd) for fd in fold_dirs],
        "B": B,
        "rng_seed": seed,
        "per_fold": per_fold,
        "pooled": {k: v for k, v in pooled_perm.items() if not k.startswith("_perm_")},
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(persisted, fh, indent=2, sort_keys=True)

    print()
    print(f"  pooled K-fold permutation test (K={len(fold_dirs)}, B={B}):")
    print()
    print("| scope | n_pairs | ρ_s (obs) | p_two | p_neg | L_align (obs) | p_low |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for pf in per_fold:
        r = pf["permutation"]
        if "skipped" in r:
            print(f"| fold {os.path.basename(pf['fold_dir'])} | {r['n_pairs']} | (skipped) ||||||")
            continue
        print(
            f"| fold {os.path.basename(pf['fold_dir'])} | {r['n_pairs']} | "
            f"{r['rho_spearman']['observed']:+.4f} | "
            f"{r['rho_spearman']['p_two_sided']:.4f} | "
            f"{r['rho_spearman']['p_one_sided_neg']:.4f} | "
            f"{r['L_align']['observed']:.4f} | "
            f"{r['L_align']['p_one_sided_low']:.4f} |"
        )
    r = pooled_perm
    if "skipped" not in r:
        print(
            f"| **pooled** | {r['n_pairs']} | "
            f"{r['rho_spearman']['observed']:+.4f} | "
            f"{r['rho_spearman']['p_two_sided']:.4f} | "
            f"{r['rho_spearman']['p_one_sided_neg']:.4f} | "
            f"{r['L_align']['observed']:.4f} | "
            f"{r['L_align']['p_one_sided_low']:.4f} |"
        )
        print()
        print(f"  null distribution of pooled ρ_s (n_pairs={r['n_pairs']}, obs marked '<'):")
        print(_ascii_hist(r["_perm_rho_spearman"], obs=r["rho_spearman"]["observed"]))
    else:
        print(f"| **pooled** | {r['n_pairs']} | (skipped: {r['skipped']}) ||||||")
    print()
    print(f"  wrote {out_path}")
    return persisted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--run-dir", help="Path to single results/.../run_<ts> directory")
    grp.add_argument(
        "--parent-dir",
        help="Path to K-fold parent dir containing fold0/ fold1/ ... subdirs",
    )
    parser.add_argument("--B", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.parent_dir:
        parent = os.path.abspath(args.parent_dir)
        fold_dirs = sorted(
            os.path.join(parent, d) for d in os.listdir(parent)
            if d.startswith("fold") and os.path.isdir(os.path.join(parent, d))
        )
        if not fold_dirs:
            print(f"no fold*/ subdirs found under {parent}", file=sys.stderr)
            return 2
        out_path = os.path.join(parent, "agg", "permutation_test.json")
        try:
            run_pooled(fold_dirs, out_path=out_path, B=int(args.B), seed=int(args.seed))
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3
        return 0

    run_dir = os.path.abspath(args.run_dir)
    try:
        result = run_single(run_dir, B=int(args.B), seed=int(args.seed))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    out_splits = result["splits"]
    repro = result["reproduction"]
    repro_status = "ok" if repro["strict"] else "not strict; selected T used fallback split_metrics"
    print()
    print(
        f"  reproduction: stored={repro['stored']:.6f}, "
        f"reproduced={repro['reproduced']:.6f}, {repro_status}"
    )
    print(f"  B={args.B}, seed={args.seed}")
    print()
    print("| split | n_pairs | ρ_s (obs) | p_two | p_neg | ρ_p (obs) | p_two | L_align (obs) | p_low |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in ("train", "dev", "test"):
        r = out_splits[name]
        if "skipped" in r:
            print(f"| {name} | {r['n_pairs']} | (skipped: {r['skipped']}) |||||||")
            continue
        print(
            f"| {name} | {r['n_pairs']} | "
            f"{r['rho_spearman']['observed']:+.4f} | "
            f"{r['rho_spearman']['p_two_sided']:.4f} | "
            f"{r['rho_spearman']['p_one_sided_neg']:.4f} | "
            f"{r['rho_pearson']['observed']:+.4f} | "
            f"{r['rho_pearson']['p_two_sided']:.4f} | "
            f"{r['L_align']['observed']:.4f} | "
            f"{r['L_align']['p_one_sided_low']:.4f} |"
        )
    print()
    for name in ("train", "dev", "test"):
        r = out_splits[name]
        if "skipped" in r:
            continue
        print(f"  null distribution of ρ_s on {name} (n_pairs={r['n_pairs']}, obs marked '<'):")
        print(_ascii_hist(r["_perm_rho_spearman"], obs=r["rho_spearman"]["observed"]))
        print()
    print(f"  wrote {result['out_path']}")
    return 0


def _read_best_iter(run_dir: str) -> str:
    final_dir, _selection = _selected_final_dir(run_dir)
    p = os.path.join(final_dir, "best_iter.txt")
    if os.path.exists(p):
        return open(p).read().strip()
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
