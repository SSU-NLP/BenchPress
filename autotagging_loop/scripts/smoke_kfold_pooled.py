"""Offline smoke for K-fold `run_pooled` aggregator.

Synthesizes 4 fold directories from a single existing v3 run by partitioning
its 20 benchmarks into disjoint test slices via `split_benchmarks_kfold`. Each
fake fold reuses the source run's T_star + score_matrix unchanged and writes a
minimal `final/split_metrics.json` whose `benchmark_split.test` is that fold's
slice and whose `test.rho_align_spearman` is recomputed from the cosine pair
matrix against R_raw.

Then calls `run_pooled(...)` and asserts:
  - per-fold n_pairs == C(test_size, 2) after R_raw NaN filtering
  - pooled n_pairs == sum of per-fold n_pairs (disjoint guard)
  - agg/permutation_test.json exists and parses

No LLM calls. ~1-2s wall.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autotagging_loop.experiment.alignment import cosine_pair_matrix
from autotagging_loop.experiment.splits import (
    induced_pair_set,
    restrict_pair_dict,
    split_benchmarks_kfold,
)
from autotagging_loop.scripts.permutation_test_run import _load_pair_dict, run_pooled


SOURCE_RUN = "results/part2_experiment/run_20260511_140200"


def _compute_test_rho_s(T_star, R_raw, test_benches):
    bench_names = sorted(T_star.keys())
    S = cosine_pair_matrix(T_star, benchmark_names=bench_names)
    pair_set = induced_pair_set(test_benches)
    S_b = restrict_pair_dict(S, pair_set)
    R_b = restrict_pair_dict(R_raw, pair_set)
    s_vals, r_vals = [], []
    for k, rv in R_b.items():
        if rv is None:
            continue
        sv = S_b.get(k)
        if sv is None:
            continue
        s_vals.append(float(sv))
        r_vals.append(float(rv))
    if len(s_vals) < 3:
        return float("nan"), len(s_vals)
    return float(spearmanr(np.asarray(s_vals), np.asarray(r_vals)).statistic), len(s_vals)


def _make_fold_dir(parent: str, fold_idx: int, source: str, test_benches, train_benches, dev_benches, T_star, R_raw):
    fold_dir = os.path.join(parent, f"fold{fold_idx}")
    final_dir = os.path.join(fold_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    shutil.copy(os.path.join(source, "score_matrix.json"), os.path.join(fold_dir, "score_matrix.json"))
    shutil.copy(os.path.join(source, "final", "T_star.json"), os.path.join(final_dir, "T_star.json"))

    rho_s, n_pairs = _compute_test_rho_s(T_star, R_raw, test_benches)
    split_metrics = {
        "fold": fold_idx,
        "seed": 44,
        "benchmark_split": {
            "train": sorted(train_benches),
            "dev": sorted(dev_benches),
            "test": sorted(test_benches),
            "ratios": [0.5, 0.25, 0.25],
        },
        "test": {"rho_align_spearman": rho_s, "n_pairs": n_pairs},
        "dev": {},
        "train": {},
        "model_split": {},
        "held_model_test": {},
    }
    with open(os.path.join(final_dir, "split_metrics.json"), "w") as fh:
        json.dump(split_metrics, fh, indent=2, sort_keys=True)
    return fold_dir, n_pairs


def main() -> int:
    source = os.path.abspath(SOURCE_RUN)
    if not os.path.isdir(source):
        print(f"source run not found: {source}", file=sys.stderr)
        return 2
    T_star = json.load(open(os.path.join(source, "final", "T_star.json")))
    R_raw = _load_pair_dict(json.load(open(os.path.join(source, "score_matrix.json")))["R_raw"])
    bench_names = sorted(T_star.keys())
    print(f"  [smoke] source run has {len(bench_names)} benchmarks")
    if len(bench_names) < 4:
        print("  [smoke] too few benchmarks for 4-fold smoke", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="kfold_smoke_") as parent:
        per_fold_n = []
        fold_dirs = []
        for k in range(4):
            split = split_benchmarks_kfold(bench_names, n_folds=4, fold=k, seed=44)
            fd, n = _make_fold_dir(
                parent, k, source,
                test_benches=split.test,
                train_benches=split.train,
                dev_benches=split.dev,
                T_star=T_star, R_raw=R_raw,
            )
            fold_dirs.append(fd)
            per_fold_n.append(n)
            print(f"  [smoke] fold{k}: test={split.test}, n_pairs(filtered)={n}")

        out_path = os.path.join(parent, "agg", "permutation_test.json")
        result = run_pooled(fold_dirs, out_path=out_path, B=200, seed=0)

        pooled = result["pooled"]
        per_fold = result["per_fold"]
        assert pooled["n_pairs"] == sum(per_fold_n), (
            f"pooled n_pairs={pooled['n_pairs']} != sum per-fold {sum(per_fold_n)}"
        )
        assert len(per_fold) == 4
        assert os.path.exists(out_path)
        print()
        print(f"  [smoke] PASS — pooled n_pairs={pooled['n_pairs']} matches sum(per_fold)={sum(per_fold_n)}")
        print(f"  [smoke] pooled ρ_s={pooled['rho_spearman']['observed']:+.4f}, p_two={pooled['rho_spearman']['p_two_sided']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
