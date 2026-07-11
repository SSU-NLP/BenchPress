"""v3 §2.2.7 split-aware reporting.

Given a final tag-similarity dict `S`, the score-pattern dict `R_raw` / `R01`,
the benchmark/model splits, and the normalized score table `Y_norm`, produce
the four blocks the v3 contract requires in `final/metrics.json`:

    {"train": {...}, "dev": {...}, "test": {...}, "held_model_test": {...}}

The "held_model_test" block recomputes pair similarities R using **only**
F_held (held-out models) over the test benchmarks, so the metric measures
generalization to a model fold that the loop never saw.
"""

from __future__ import annotations

import math

from autotagging_loop.experiment.alignment import (
    alignment_corr,
    alignment_loss,
    bootstrap_metrics_block,
    intra_inter_gap,
    quantile_thresholds,
)
from autotagging_loop.experiment.score_matrix import spearman_pair_matrix, to_R01
from autotagging_loop.experiment.splits import (
    BenchmarkSplit,
    ModelSplit,
    induced_pair_set,
    restrict_pair_dict,
)


PairKey = tuple[str, str]


def _block_metrics(
    S_block: dict[PairKey, float],
    R_raw_block: dict[PairKey, float | None],
    R01_block: dict[PairKey, float | None],
    benchmark_block: list[str],
    *,
    q_p: float,
    q_n: float,
    bootstrap_B: int,
    seed: int,
) -> dict:
    """Compute one split's metric dict. Drops pair entries with R_raw is None."""
    S_def = {k: v for k, v in S_block.items() if R_raw_block.get(k) is not None}
    R_def = {k: float(v) for k, v in R_raw_block.items() if v is not None}
    R01_def = {k: float(R01_block[k]) for k in R_def}
    n_pairs = len(R_def)
    effective = sorted({name for pair in R_def for name in pair})
    isolated = sorted(set(benchmark_block) - set(effective))
    if n_pairs == 0:
        return {
            "n_pairs": 0,
            "n_benchmarks": len(benchmark_block),
            "n_effective_benchmarks": 0,
            "isolated_benchmarks": sorted(benchmark_block),
        }
    L = alignment_loss(S_def, R_def)
    L01 = alignment_loss(S_def, R01_def)
    pr, sp = alignment_corr(S_def, R_def)
    theta_p, theta_n = quantile_thresholds(S_def, q_p=q_p, q_n=q_n)
    gap = intra_inter_gap(S_def, R_def, theta_p, theta_n)
    boot = bootstrap_metrics_block(
        S_def, R_def, R01_def, benchmark_block, B=bootstrap_B, seed=seed,
        q_p=q_p, q_n=q_n,
    )
    return {
        "L_align": L,
        "L_align_01": L01,
        "rho_align_pearson": pr,
        "rho_align_spearman": sp,
        "theta_p": theta_p,
        "theta_n": theta_n,
        "intra_tag_sim": gap["intra"],
        "inter_tag_sim": gap["inter"],
        "delta_tag": gap["delta"],
        "n_pairs": n_pairs,
        "n_benchmarks": len(benchmark_block),
        "n_effective_benchmarks": len(effective),
        "isolated_benchmarks": isolated,
        "bootstrap": boot,
    }


def compute_split_metrics(
    *,
    S: dict[PairKey, float],
    R_raw: dict[PairKey, float | None],
    R01: dict[PairKey, float | None],
    benchmark_split: BenchmarkSplit,
    q_p: float,
    q_n: float,
    bootstrap_B: int,
    seed: int,
) -> dict:
    """Returns {"train": {...}, "dev": {...}, "test": {...}}.

    Pairs whose endpoints span buckets are dropped per `induced_pair_set`.
    """
    out: dict = {}
    for name in ("train", "dev", "test"):
        bench_block: list[str] = list(getattr(benchmark_split, name))
        pair_set = induced_pair_set(bench_block)
        S_block = restrict_pair_dict(S, pair_set)
        R_block = restrict_pair_dict(R_raw, pair_set)
        R01_block = restrict_pair_dict(R01, pair_set)
        out[name] = _block_metrics(
            S_block, R_block, R01_block, bench_block,
            q_p=q_p, q_n=q_n, bootstrap_B=bootstrap_B, seed=seed,
        )
    return out


def compute_held_model_test_metrics(
    *,
    S: dict[PairKey, float],
    Y_norm: dict[str, dict[str, float]],
    benchmark_split: BenchmarkSplit,
    model_split: ModelSplit,
    q_p: float,
    q_n: float,
    bootstrap_B: int,
    seed: int,
    min_common: int = 8,
) -> dict:
    """Held-model evaluation: recompute R using only F_held over D_test.

    Tag similarities S are unchanged (taxonomy is what we are evaluating). R
    on the held subset measures how the same taxonomy generalizes to a model
    fold that never participated in score-pattern grounding during training.
    """
    held = list(model_split.held)
    if not held:
        return {"n_pairs": 0, "n_benchmarks": len(benchmark_split.test), "skipped": "no_held_models"}
    if len(held) < int(min_common):
        return {
            "n_pairs": 0,
            "n_benchmarks": len(benchmark_split.test),
            "n_held_models": len(held),
            "min_common": int(min_common),
            "skipped": "held_models_below_min_common",
        }
    test_benches = list(benchmark_split.test)
    if len(test_benches) < 2:
        return {
            "n_pairs": 0,
            "n_benchmarks": len(test_benches),
            "n_held_models": len(held),
            "min_common": int(min_common),
            "skipped": "too_few_test_benchmarks",
        }
    Y_held = {
        bench: {m: float(Y_norm[bench][m]) for m in Y_norm.get(bench, {}) if m in held}
        for bench in test_benches
    }
    R_held, _ = spearman_pair_matrix(
        Y_held, test_benches, min_common=min_common,
    )
    R01_held = to_R01(R_held)
    pair_set = induced_pair_set(test_benches)
    S_block = restrict_pair_dict(S, pair_set)
    R_block = restrict_pair_dict(R_held, pair_set)
    R01_block = restrict_pair_dict(R01_held, pair_set)
    block = _block_metrics(
        S_block, R_block, R01_block, test_benches,
        q_p=q_p, q_n=q_n, bootstrap_B=bootstrap_B, seed=seed,
    )
    block["n_held_models"] = len(held)
    block["min_common"] = int(min_common)
    if block.get("n_pairs") == 0:
        block["skipped"] = "no_score_comparable_pairs"
    return block


def write_split_metrics_json(
    run_dir: str,
    *,
    fold: int,
    seed: int,
    benchmark_split: BenchmarkSplit,
    model_split: ModelSplit | None,
    train_dev_test: dict,
    held_model_test: dict | None,
) -> str:
    """Persist `final/split_metrics.json` per the v3 contract."""
    import json
    import os

    final_dir = os.path.join(run_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    payload = {
        "fold": int(fold),
        "seed": int(seed),
        "benchmark_split": {
            "train": list(benchmark_split.train),
            "dev": list(benchmark_split.dev),
            "test": list(benchmark_split.test),
            "ratios": list(benchmark_split.ratios),
        },
        "model_split": (
            {
                "seen": list(model_split.seen),
                "held": list(model_split.held),
                "ratios": list(model_split.ratios),
                "strategy": getattr(model_split, "strategy", "random"),
            }
            if model_split is not None
            else None
        ),
        "train": train_dev_test["train"],
        "dev": train_dev_test["dev"],
        "test": train_dev_test["test"],
        "held_model_test": held_model_test,
    }
    out_path = os.path.join(final_dir, "split_metrics.json")
    with open(out_path, "w") as fh:
        json.dump(_jsonable(payload), fh, indent=2, sort_keys=True)
    return out_path


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        return obj
    return obj
