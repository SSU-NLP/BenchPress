"""§4.2 Tag validity — ability tags vs surface domain labels.

Emits, for all benchmark pairs, (tag-cos, rank-sim, same-domain?) and:
  - Pearson(tag-cos, rank-sim) vs Pearson(domain-match, rank-sim) + partial
  - case study: same-domain-but-far-tag / cross-domain-but-near-tag pairs
  - same- vs cross-domain rank-sim means (domain barely separates ranking)
Offline: target v2 + seed-vocab census profiles. No LLM calls.
"""
from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np

from autotagging_loop.experiment import vloop, vloop_pilot, splits
import autotagging_loop.experiment.calibration as cal

REPO = Path(__file__).resolve().parents[2]
V2_PATH = REPO / "results/target_v2/score_matrix.json"
PROF_DIR = REPO / "results/vloop_main/main2_fold0/final/seed/tags"
STATE_DIR = REPO / "results/vloop_main/main2_fold0/iter_0"
OUT = REPO / "results/vloop_main/tag_validity.json"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def collect() -> dict:
    Y = json.loads(V2_PATH.read_text())["Y_norm"]
    disp = {_norm(k): k for k in Y}
    state = vloop_pilot.load_state(STATE_DIR)
    axes = [a.id for a in state.vocab]
    P = {}
    for f in PROF_DIR.glob("*.jsonl"):
        cached = cal._load_cached(f)
        rows = [[vloop.LEVEL_VALUE[lv[a]] for a in axes]
                for lv in (r.get("levels") for r in cached.values())
                if lv and set(lv) >= set(axes)]
        if rows:
            P[f.stem] = np.mean(rows, axis=0)
    benches = sorted(b for b in P if _norm(b) in disp)
    strat = {b: splits._default_benchmark_stratum(b) for b in benches}

    pairs = []
    for x, y in combinations(benches, 2):
        rs, _ = vloop.rank_similarity(Y[disp[_norm(x)]], Y[disp[_norm(y)]], 4)
        if rs is None:
            continue
        pairs.append({"a": x, "b": y, "same_domain": strat[x] == strat[y],
                      "domain_a": strat[x], "domain_b": strat[y],
                      "tag_cos": _cos(P[x], P[y]), "rank_sim": rs})

    tc = [p["tag_cos"] for p in pairs]
    dm = [1.0 if p["same_domain"] else 0.0 for p in pairs]
    rk = [p["rank_sim"] for p in pairs]

    def resid(y, x):
        b = vloop.pearson(x, y) * (np.std(y) / (np.std(x) or 1))
        return list(np.array(y) - (np.mean(y) + b * (np.array(x) - np.mean(x))))

    same = [p for p in pairs if p["same_domain"]]
    diff = [p for p in pairs if not p["same_domain"]]
    return {
        "pairs": pairs,
        "pearson_tag": vloop.pearson(tc, rk),
        "pearson_domain": vloop.pearson(dm, rk),
        "partial_tag_given_domain": vloop.pearson(resid(tc, dm), resid(rk, dm)),
        "same_domain_ranksim_mean": float(np.mean([p["rank_sim"] for p in same])),
        "cross_domain_ranksim_mean": float(np.mean([p["rank_sim"] for p in diff])),
        "case_same_far": sorted([p for p in same], key=lambda p: p["tag_cos"])[:6],
        "case_cross_near": sorted(diff, key=lambda p: -p["tag_cos"])[:6],
    }


def main() -> None:
    r = collect()
    print(f"Pearson(tag-cos, rank-sim)       = {r['pearson_tag']:+.3f}")
    print(f"Pearson(domain-match, rank-sim)  = {r['pearson_domain']:+.3f}")
    print(f"partial(tag | domain)            = {r['partial_tag_given_domain']:+.3f}")
    print(f"same-domain rank-sim mean = {r['same_domain_ranksim_mean']:+.3f}  "
          f"cross-domain = {r['cross_domain_ranksim_mean']:+.3f}")
    print("\nsame-domain but far-tag (domain label would wrongly group):")
    for p in r["case_same_far"][:4]:
        print(f"  [{p['domain_a']}] {p['a']}~{p['b']}: tag {p['tag_cos']:.3f}, rank {p['rank_sim']:+.3f}")
    print("\ncross-domain but near-tag (domain label would wrongly split):")
    for p in r["case_cross_near"][:4]:
        print(f"  {p['a']}({p['domain_a']})~{p['b']}({p['domain_b']}): "
              f"tag {p['tag_cos']:.3f}, rank {p['rank_sim']:+.3f}")
    OUT.write_text(json.dumps(r, indent=2))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
