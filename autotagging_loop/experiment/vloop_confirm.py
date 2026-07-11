"""Fresh-draw confirmation pass — out-of-sample re-test of gate-rejected v_loop candidates.

pilot_004 left two candidates at smoothed +0.015~0.017 over seed with p_gt 0.83/0.80:
consistent positive signal below the 0.9 bar. This harness re-runs the paired comparison
on a DISJOINT item draw (the calibration plan's 승자-확인 methodology): tag fresh items
under the seed state and each candidate state, then compute smoothed scores and the
paired-bootstrap p_gt on the fresh sample only. High p_gt out-of-sample confirms the
improver lift; ~0.5 means the in-sample delta was selection luck.

Benches whose original 100-item draw already covered the full pool (aime*, hmmt, mmlu,
mmluredux, scicode) have no fresh items and are excluded — the objective is recomputed
on the surviving bench set for BOTH states, so the comparison stays paired and fair.

Usage:
  python -m experiment.vloop_confirm --run-id confirm_001 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

import autotagging_loop.experiment.calibration as cal
from autotagging_loop.experiment import vloop
from autotagging_loop.experiment.vloop_pilot import (
    GatewayJsonLLM, TAGGER_MODEL, MAX_WORKERS, SCORES_PATH, OUT_DIR,
    _bench_matrices, _norm, _pair_ranksims, _score_from_Tb,
    gate_p_better, load_state, smoothed_score, tag_profiles,
)

DEFAULT_SEED_DIR = "pilot_004/iter_0"
DEFAULT_CAND_DIRS = "pilot_004/iter_1,pilot_004/iter_2"
MIN_FRESH = 30  # exclude benches with fewer unseen items (livecodebench has 21)


def fresh_benchmarks(
    exclude_dir: Path, n_items: int, sample_seed: int,
    drop: frozenset[str] = frozenset(),
) -> tuple[list[vloop.Benchmark], dict[str, str]]:
    """Benchmarks rebuilt from items DISJOINT from `exclude_dir`'s tag cache."""
    sm = json.loads(SCORES_PATH.read_text())
    y_norm = sm["Y_norm"]
    disp = {_norm(k): k for k in y_norm}
    rng = random.Random(sample_seed)
    drop_n = {_norm(d) for d in drop}

    benches, skipped = [], {}
    for slug in sorted(p.name for p in cal.DATA_DIR.iterdir() if p.is_dir()):
        if _norm(slug) in drop_n:
            skipped[slug] = "excluded (dropped benchmark)"
            continue
        key = disp.get(_norm(slug))
        if key is None:
            continue  # no model scores — same exclusion as the pilots
        used = set(cal._load_cached(exclude_dir / "tags" / f"{slug}.jsonl"))
        pool = [
            json.loads(line)
            for line in (cal.DATA_DIR / slug / "tasks.jsonl").read_text().splitlines()
            if line.strip()
        ]
        fresh = [it for it in pool if it["item_id"] not in used]
        if len(fresh) < MIN_FRESH:
            skipped[slug] = f"fresh={len(fresh)} < {MIN_FRESH}"
            continue
        rng.shuffle(fresh)
        items = [vloop.Item(id=it["item_id"], prompt=cal._item_block(it)) for it in fresh[:n_items]]
        benches.append(vloop.Benchmark(name=slug, items=items, scores=dict(y_norm[key])))
    return benches, skipped


def hard_score(state_dir: Path, state: vloop.PromptState, benches, ranksims) -> float:
    mats = _bench_matrices(state_dir, benches, state)
    Tb = {n: (m.mean(axis=0) if m.shape[0] else np.zeros(m.shape[1])) for n, (_i, m) in mats.items()}
    return _score_from_Tb(Tb, ranksims)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="v_loop fresh-draw confirmation pass")
    parser.add_argument("--run-id", default="confirm_001")
    parser.add_argument("--n-items", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=777)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--seed-dir", default=DEFAULT_SEED_DIR, help="seed state dir under vloop_pilot/")
    parser.add_argument("--cand-dirs", default=DEFAULT_CAND_DIRS, help="comma-separated candidate state dirs")
    parser.add_argument("--exclude", default="", help="comma-separated benchmark slugs to drop")
    parser.add_argument("--dry-run", action="store_true", help="print fresh counts, no LLM")
    args = parser.parse_args(argv)

    seed_dir = OUT_DIR / args.seed_dir
    cand_dirs = [OUT_DIR / c.strip() for c in args.cand_dirs.split(",") if c.strip()]
    drop = frozenset(s.strip() for s in args.exclude.split(",") if s.strip())

    benches, skipped = fresh_benchmarks(seed_dir, args.n_items, args.sample_seed, drop)
    print(f"confirm: seed={args.seed_dir} cand={[c.name for c in cand_dirs]} "
          f"{len(benches)} benches with fresh items "
          f"({len(list(_pair_ranksims(benches)))} pairs); skipped: {skipped}")
    for b in benches:
        print(f"  {b.name}: {len(b.items)} fresh items")
    if args.dry_run:
        return

    run_dir = OUT_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "sample_manifest.json").write_text(json.dumps({
        "sample_seed": args.sample_seed, "excluded": skipped,
        "benches": {b.name: [it.id for it in b.items] for b in benches},
    }, indent=2))

    tagger = vloop.ItemTagger(GatewayJsonLLM(TAGGER_MODEL, reasoning=False))
    ranksims = _pair_ranksims(benches)
    states = [("seed", seed_dir)] + [
        (f"cand_{d.parent.name}_{d.name}", d) for d in cand_dirs
    ]

    results = {}
    for label, src_dir in states:
        state = load_state(src_dir)
        out_dir = run_dir / label
        _profiles, invalid = tag_profiles(benches, state, out_dir, tagger, args.workers)
        results[label] = {
            "state_from": str(src_dir), "n_axes": len(state.vocab), "n_invalid": invalid,
            "smooth": smoothed_score(out_dir, state, benches, ranksims),
            "hard": hard_score(out_dir, state, benches, ranksims),
        }
        print(f"  {label}: smooth={results[label]['smooth']:.4f} "
              f"hard={results[label]['hard']:.4f} invalid={invalid}")

    rng = np.random.default_rng(42)
    seed_state = load_state(seed_dir)
    lines = ["# V-Loop fresh-draw confirmation", "",
             f"{len(benches)} benches, {len(ranksims)} pairs, seed smooth "
             f"{results['seed']['smooth']:.4f} (hard {results['seed']['hard']:.4f})", "",
             "| candidate | smooth | Δ smooth | hard | p_gt (fresh) |", "|---|---|---|---|---|"]
    for label, src_dir in states[1:]:
        p_gt = gate_p_better(run_dir / label, run_dir / "seed", load_state(src_dir),
                             seed_state, benches, ranksims, rng)
        results[label]["p_gt_fresh"] = p_gt
        d = results[label]["smooth"] - results["seed"]["smooth"]
        print(f"  {label}: Δsmooth={d:+.4f} p_gt(fresh)={p_gt:.2f}")
        lines.append(f"| {label} | {results[label]['smooth']:.4f} | {d:+.4f} | "
                     f"{results[label]['hard']:.4f} | {p_gt:.2f} |")

    (run_dir / "report.json").write_text(json.dumps(results, indent=2))
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"done -> {run_dir}/report.md")


if __name__ == "__main__":
    main()
