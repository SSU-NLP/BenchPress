"""§4.3 Surrogate ranking — do tag-similar benchmarks give an interpretable
reference for a held-out benchmark's model ranking?

Task: hold out benchmark X. Using X's ability-tag profile, pick its top-k
nearest benchmarks (tag-cosine) among the rest; predict X's per-model score as
the tag-cos-weighted mean of those neighbours; rank the models by the prediction
and compare to X's true model ranking (Spearman). Averaged over all held-out X.

Baselines at the same k: domain-label neighbours (same coarse family), random
neighbours, and global mean (no neighbour selection). If ability tags beat
domain labels, "math/code/knowledge" is too coarse and the ability profile
carries real selection signal.

Circularity: the tag profile is built from item text only (never sees model
ranks), and X is held out, so this is a genuine out-of-sample prediction. The
relative margin vs domain/random is the headline (baselines share any residual
circular advantage). Offline — no LLM calls.

Usage: python -m experiment.surrogate_ranking [--profiles seed|best] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from itertools import combinations
from pathlib import Path

import numpy as np

from autotagging_loop.experiment import vloop, vloop_pilot, splits
import autotagging_loop.experiment.calibration as cal

REPO = Path(__file__).resolve().parents[2]
V2_PATH = REPO / "results/target_v2/score_matrix.json"
# seed-vocab census profiles (19 benches, never optimized on rank -> no circularity)
PROFILE_DIRS = {
    "seed": (REPO / "results/vloop_main/main2_fold0/final/seed/tags",
             REPO / "results/vloop_main/main2_fold0/iter_0"),
    "best": (REPO / "results/vloop_pilot/pilot_005/iter_1/tags",
             REPO / "results/vloop_pilot/pilot_005/iter_1"),
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_profiles(tag_dir: Path, state_dir: Path) -> tuple[dict, list[str]]:
    state = vloop_pilot.load_state(state_dir)
    axes = [a.id for a in state.vocab]
    P = {}
    for f in tag_dir.glob("*.jsonl"):
        cached = cal._load_cached(f)
        rows = [[vloop.LEVEL_VALUE[lv[a]] for a in axes]
                for lv in (r.get("levels") for r in cached.values())
                if lv and set(lv) >= set(axes)]
        if rows:
            P[f.stem] = np.mean(rows, axis=0)
    return P, axes


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def spearman_models(pred: dict, true: dict) -> float | None:
    common = sorted(set(pred) & set(true))
    if len(common) < 4:
        return None
    return vloop.spearman([pred[m] for m in common], [true[m] for m in common])


def weighted_pred(neighbors: list[str], weights: list[float], Y: dict) -> dict:
    """tag-cos-weighted mean of neighbour model scores."""
    out: dict[str, float] = {}
    models = set().union(*(set(Y[b]) for b in neighbors))
    for m in models:
        num = den = 0.0
        for b, w in zip(neighbors, weights):
            if m in Y[b]:
                num += w * Y[b][m]
                den += w
        if den > 0:
            out[m] = num / den
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", choices=["seed", "best"], default="seed")
    ap.add_argument("--out", type=Path, default=REPO / "results/vloop_main/surrogate_ranking.json")
    ap.add_argument("--n-random", type=int, default=200)
    args = ap.parse_args(argv)

    V2 = json.loads(V2_PATH.read_text())["Y_norm"]
    disp = {_norm(k): k for k in V2}
    tag_dir, state_dir = PROFILE_DIRS[args.profiles]
    P, axes = load_profiles(tag_dir, state_dir)
    benches = sorted(b for b in P if _norm(b) in disp)
    Y = {b: {m: v for m, v in V2[disp[_norm(b)]].items()} for b in benches}
    strat = {b: splits._default_benchmark_stratum(b) for b in benches}
    print(f"profiles={args.profiles}, {len(benches)} benches, {len(axes)} axes")

    def eval_strategy(pick_fn, ks, rounds=1):
        """pick_fn(X, others, k, rng) -> (neighbors, weights). Returns {k: mean spearman}."""
        out = {}
        for k in ks:
            accs = []
            for r in range(rounds):
                rng = random.Random(1000 + r)
                per_x = []
                for X in benches:
                    others = [b for b in benches if b != X]
                    nb, w = pick_fn(X, others, k, rng)
                    if not nb:
                        continue
                    s = spearman_models(weighted_pred(nb, w, Y), Y[X])
                    if s is not None:
                        per_x.append(s)
                accs.append(np.mean(per_x))
            out[k] = (float(np.mean(accs)), float(np.std(accs)))
        return out

    def pick_tag(X, others, k, rng):
        ranked = sorted(others, key=lambda b: -cos(P[X], P[b]))[:k]
        return ranked, [cos(P[X], P[b]) for b in ranked]

    def pick_domain(X, others, k, rng):
        same = [b for b in others if strat[b] == strat[X]]
        pool = same if len(same) >= 1 else others
        rng.shuffle(pool)
        nb = pool[:k]
        return nb, [1.0] * len(nb)

    def pick_random(X, others, k, rng):
        pool = others[:]; rng.shuffle(pool); nb = pool[:k]
        return nb, [1.0] * len(nb)

    def pick_global(X, others, k, rng):
        return others, [1.0] * len(others)

    ks = [1, 3, 5]
    res = {
        "ability_tag": eval_strategy(pick_tag, ks),
        "domain_label": eval_strategy(pick_domain, ks, rounds=args.n_random),
        "random": eval_strategy(pick_random, ks, rounds=args.n_random),
        "global_mean": eval_strategy(pick_global, [1])[1],  # k-independent
    }
    print(f"\n{'strategy':<16}" + "".join(f"{'k='+str(k):>14}" for k in ks))
    for name in ("ability_tag", "domain_label", "random"):
        row = f"{name:<16}"
        for k in ks:
            m, sd = res[name][k]
            row += f"{m:>8.3f}±{sd:.3f}"
        print(row)
    gm, gsd = res["global_mean"]
    print(f"{'global_mean':<16}{gm:>8.3f}±{gsd:.3f}  (k-independent)")

    args.out.write_text(json.dumps({
        "profiles": args.profiles, "n_benches": len(benches), "ks": ks,
        "results": {k: (v if k == "global_mean" else {str(kk): vv for kk, vv in v.items()})
                    for k, v in res.items()},
    }, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
