"""scripts/posthoc_selection_simulation.py — replay best-iter selection under
different criteria on an existing run.

Reads per-iter `tag_vectors.json` and reconstructs train/dev/test metrics, then
applies several candidate selection rules:

    * `train_l_align` (current v3 default) — argmin train L_align s.t. Δ_tag > 0
    * `dev_l_align`                         — argmin dev L_align s.t. Δ_tag > 0
    * `dev_rho_spearman`                    — argmax dev ρ_s, tie-break dev L_align
    * `dev_l_with_rho_floor`                — argmin dev L s.t. dev ρ_s ≥ 0
                                              (codex recommendation: dev L primary,
                                               dev ρ guards catastrophic collapse)

For each rule reports the selected iter and the test metrics (and permutation
two-sided p-value for ρ_s) that *would have been* the final result.

Usage:
    python scripts/posthoc_selection_simulation.py --run-dir <path>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autotagging_loop.experiment.alignment import alignment_corr, alignment_loss, cosine_pair_matrix
from autotagging_loop.experiment.splits import induced_pair_set, restrict_pair_dict


def _parse_pair_dict(d: dict) -> dict[tuple[str, str], float | None]:
    out: dict[tuple[str, str], float | None] = {}
    for k, v in d.items():
        p, _, q = k.partition("||")
        out[(p, q)] = None if v is None else float(v)
    return out


def _block(S, R_raw, R01, benches):
    pset = induced_pair_set(benches)
    Sb = restrict_pair_dict(S, pset)
    Rb = restrict_pair_dict(R_raw, pset)
    R01b = restrict_pair_dict(R01, pset)
    Sd = {k: v for k, v in Sb.items() if Rb.get(k) is not None and v is not None}
    Rd = {k: float(v) for k, v in Rb.items() if v is not None and k in Sd}
    R01d = {k: float(R01b[k]) for k in Rd if R01b.get(k) is not None}
    if len(Rd) < 2:
        return None
    pr, sp = alignment_corr(Sd, Rd)
    L = alignment_loss(Sd, Rd)        # match split_metrics: MSE vs R_raw
    L01 = alignment_loss(Sd, R01d)
    return {
        "n_pairs": len(Rd),
        "L_align": L,
        "L_align_01": L01,
        "rho_pearson": pr,
        "rho_spearman": sp,
    }


def _permutation_p(S_test: dict, R_test: dict, *, B: int = 10_000, seed: int = 0) -> float:
    if len(R_test) < 3:
        return float("nan")
    keys = list(S_test.keys())
    s = np.asarray([S_test[k] for k in keys], dtype=float)
    r = np.asarray([R_test[k] for k in keys], dtype=float)
    if len(set(s.tolist())) < 2 or len(set(r.tolist())) < 2:
        return float("nan")
    obs = float(spearmanr(s, r).statistic)
    rng = np.random.default_rng(seed)
    perms = np.empty(B, dtype=float)
    for i in range(B):
        perms[i] = spearmanr(s, r[rng.permutation(len(r))]).statistic
    return float((np.sum(np.abs(perms) >= abs(obs)) + 1) / (B + 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--B", type=int, default=10_000)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    sm = json.load(open(os.path.join(run_dir, "score_matrix.json")))
    splits = json.load(open(os.path.join(run_dir, "final", "split_metrics.json")))
    R_raw = _parse_pair_dict(sm["R_raw"])
    R01 = _parse_pair_dict(sm["R01"])
    train_b = list(splits["benchmark_split"]["train"])
    dev_b = list(splits["benchmark_split"]["dev"])
    test_b = list(splits["benchmark_split"]["test"])

    iter_data: list[dict] = []
    for name in sorted(os.listdir(run_dir)):
        if not name.startswith("iter_"):
            continue
        tv = os.path.join(run_dir, name, "tag_vectors.json")
        if not os.path.exists(tv):
            continue
        T = json.load(open(tv))
        S = cosine_pair_matrix(T, benchmark_names=sorted(T.keys()))
        tr = _block(S, R_raw, R01, train_b)
        de = _block(S, R_raw, R01, dev_b)
        te = _block(S, R_raw, R01, test_b)
        # delta_tag from stored metrics so we use the same gate definition
        try:
            stored = json.load(open(os.path.join(run_dir, name, "metrics.json")))
            delta_tag = float(stored.get("delta_tag", float("nan")))
        except Exception:
            delta_tag = float("nan")
        # Test pair dicts for permutation test
        pset_test = induced_pair_set(test_b)
        S_test = {k: v for k, v in restrict_pair_dict(S, pset_test).items() if v is not None}
        R_test = {k: float(v) for k, v in restrict_pair_dict(R_raw, pset_test).items() if v is not None}
        S_test = {k: v for k, v in S_test.items() if k in R_test}
        iter_data.append({
            "label": name,
            "delta_tag": delta_tag,
            "train": tr, "dev": de, "test": te,
            "S_test": S_test, "R_test": R_test,
        })

    # ---- selection rules ----
    def passes_gate(it):
        # baseline_random has no delta_tag in some setups; treat NaN as 0 (fail).
        # Static baseline always passes (it's the reference).
        if "baseline_random" in it["label"]:
            return True
        if "baseline_static" in it["label"]:
            return True
        d = it["delta_tag"]
        return not math.isnan(d) and d > 0

    def pick_train_l(its):
        cs = [it for it in its if passes_gate(it) and it["train"] is not None]
        if not cs: return None
        return min(cs, key=lambda it: it["train"]["L_align"])

    def pick_dev_l(its):
        cs = [it for it in its if passes_gate(it) and it["dev"] is not None]
        if not cs: return None
        return min(cs, key=lambda it: it["dev"]["L_align"])

    def pick_dev_rho(its):
        cs = [it for it in its if passes_gate(it) and it["dev"] is not None]
        if not cs: return None
        # primary: max dev rho_s; tie-break: min dev L_align
        return min(cs, key=lambda it: (-it["dev"]["rho_spearman"], it["dev"]["L_align"]))

    def pick_dev_l_with_rho_floor(its):
        cs = [
            it for it in its
            if passes_gate(it) and it["dev"] is not None and it["dev"]["rho_spearman"] >= 0.0
        ]
        if not cs:
            cs = [it for it in its if passes_gate(it) and it["dev"] is not None]
        if not cs:
            return None
        return min(cs, key=lambda it: it["dev"]["L_align"])

    rules = {
        "train_l_align (v3 default)": pick_train_l,
        "dev_l_align": pick_dev_l,
        "dev_rho_spearman": pick_dev_rho,
        "dev_l_with_rho_floor": pick_dev_l_with_rho_floor,
    }

    def _n_pairs(block):
        return "na" if block is None else str(block["n_pairs"])

    def _fmt(block, key):
        if block is None:
            return "    na"
        value = block.get(key)
        return f"{value:>+7.3f}" if key.startswith("rho") else f"{value:>6.3f}"

    print()
    print(
        "per-iter (n_pairs train/dev/test = "
        f"{_n_pairs(iter_data[0]['train'])}/"
        f"{_n_pairs(iter_data[0]['dev'])}/"
        f"{_n_pairs(iter_data[0]['test'])})"
    )
    if any(it["test"] is None for it in iter_data):
        print(
            "note: test metrics are unavailable for iterations whose tag_vectors "
            "exclude held-out test benchmarks; v-loop only applies the selected "
            "candidate to test at finalization."
        )
    print(f"{'label':<28} {'Δtag':>7} | {'tr_L':>6} {'tr_ρ':>7} | {'dev_L':>6} {'dev_ρ':>7} | {'te_L':>6} {'te_ρ':>7}")
    for it in iter_data:
        d = it["delta_tag"]
        tr = it["train"]; de = it["dev"]; te = it["test"]
        print(
            f"{it['label']:<28} {d:>+7.3f} | "
            f"{_fmt(tr, 'L_align')} {_fmt(tr, 'rho_spearman')} | "
            f"{_fmt(de, 'L_align')} {_fmt(de, 'rho_spearman')} | "
            f"{_fmt(te, 'L_align')} {_fmt(te, 'rho_spearman')}"
        )
    print()

    print("selection simulation:")
    print(f"{'rule':<34} {'chosen iter':<18} {'test ρ_s':>10} {'test L':>10} {'perm p_two':>12}")
    out: dict = {"run_dir": run_dir, "rules": {}}
    for rule_name, rule_fn in rules.items():
        chosen = rule_fn(iter_data)
        if chosen is None:
            print(f"{rule_name:<34} (no candidate passed gate)")
            continue
        if chosen["test"] is None:
            test_rho = None
            test_L = None
            p = None
            print(
                f"{rule_name:<34} {chosen['label']:<18} "
                f"{'unavailable':>10} {'unavailable':>10} {'unavailable':>12}"
            )
        else:
            test_rho = chosen["test"]["rho_spearman"]
            test_L = chosen["test"]["L_align"]
            p = _permutation_p(chosen["S_test"], chosen["R_test"], B=args.B, seed=0)
            print(f"{rule_name:<34} {chosen['label']:<18} {test_rho:>+10.4f} {test_L:>10.4f} {p:>12.4f}")
        out["rules"][rule_name] = {
            "chosen_iter": chosen["label"],
            "test_rho_spearman": test_rho,
            "test_L_align": test_L,
            "permutation_p_two_sided": p,
        }

    out_path = os.path.join(run_dir, "analysis", "posthoc_selection.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print()
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
