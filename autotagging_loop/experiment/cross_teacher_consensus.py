"""Cross-teacher consensus analysis for tagger calibration (§4.2 Tagger selection).

calibration.py's report ranks candidates against a SINGLE teacher (Qwen-397b) —
which favours same-family candidates (cand35b looks best there). This recomputes
each candidate's worst-bench cosine against EACH of several teachers, then takes
the consensus (mean) and worst-teacher (min). A candidate that only matches the
Qwen teacher is a family-mimicry artifact, not a good tagger.

Offline: reads a calibration run's tags/. No LLM calls. Confirms cal_003's
finding — qwen27b is the consensus-robust winner (high across all teachers),
cand35b is Qwen-only.

Usage: python -m experiment.cross_teacher_consensus [--run-id cal_003] [--min-consensus 0.80]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autotagging_loop.experiment.calibration import RESULTS_DIR
from autotagging_loop.experiment.vloop import LEVEL_VALUE

AXES = [
    "analogical_reasoning", "commonsense_causal_reasoning", "contextual_recall",
    "deductive_reasoning", "inductive_reasoning", "long_term_knowledge_recall",
    "quantitative_reasoning", "semantic_relationship_comprehension",
    "spatial_geometrical_reasoning", "temporal_reasoning",
]
# tag-dir names in the cal_003 run (teacher = qwen3.5-397b)
DEFAULT_TEACHERS = ["teacher", "gpt5", "deepseekv4pro", "gemini31pro"]
DEFAULT_CANDS = ["qwen27b", "cand35b", "cand122b", "cand9b",
                 "llama33_70b", "deepseekv31", "gemini25flashlite"]


def load_items(tags_root: Path, model: str) -> dict[str, np.ndarray]:
    """bench -> (n_items x n_axes) matrix of valid tags (all passes pooled)."""
    out: dict[str, list] = {}
    for f in (tags_root / model).glob("*.jsonl"):
        bench = f.stem.split("_pass")[0]
        for line in f.read_text().splitlines():
            t = json.loads(line)
            if t.get("valid") and t.get("tags") and set(t["tags"]) >= set(AXES):
                out.setdefault(bench, []).append([LEVEL_VALUE[t["tags"][a]] for a in AXES])
    return {b: np.array(v) for b, v in out.items()}


def profile(tags_root: Path, model: str) -> dict[str, np.ndarray]:
    """bench -> mean tag vector over all valid items (all passes)."""
    return {b: m.mean(axis=0) for b, m in load_items(tags_root, model).items()}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def worst_bench_cos(cand: dict, teacher: dict) -> float:
    benches = sorted(set(cand) & set(teacher))
    return min(_cos(cand[b], teacher[b]) for b in benches) if benches else float("nan")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="cal_003")
    ap.add_argument("--teachers", nargs="+", default=DEFAULT_TEACHERS)
    ap.add_argument("--candidates", nargs="+", default=DEFAULT_CANDS)
    ap.add_argument("--min-consensus", type=float, default=0.80,
                    help="pass iff worst-teacher (min) cosine >= this")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="item-bootstrap B rounds -> confirmation-pass CI (0 = point only)")
    args = ap.parse_args(argv)

    tags_root = RESULTS_DIR / args.run_id / "tags"
    prof = {m: profile(tags_root, m) for m in args.teachers + args.candidates}

    print(f"{'candidate':<16}" + "".join(f"{t:>14}" for t in args.teachers)
          + f"{'CONS':>8}{'MIN':>8}  gate")
    results = {}
    for c in args.candidates:
        per_t = {t: worst_bench_cos(prof[c], prof[t]) for t in args.teachers}
        cons = float(np.mean(list(per_t.values())))
        mn = float(min(per_t.values()))
        results[c] = {"per_teacher": per_t, "consensus": cons, "min": mn,
                      "pass": mn >= args.min_consensus}
        gate = "PASS" if mn >= args.min_consensus else ""
        print(f"{c:<16}" + "".join(f"{per_t[t]:>14.3f}" for t in args.teachers)
              + f"{cons:>8.3f}{mn:>8.3f}  {gate}")

    winners = [c for c, r in results.items() if r["pass"]]
    print(f"\nconsensus winner(s) [min>={args.min_consensus}]: {winners}")

    if args.bootstrap > 0:
        items = {m: load_items(tags_root, m) for m in args.teachers + args.candidates}
        benches = sorted(items[args.teachers[0]])
        rng = np.random.default_rng(0)

        def boot_min(c: str) -> float:
            per_t = []
            for T in args.teachers:
                worst = 1.0
                for b in benches:
                    cm, tm = items[c].get(b), items[T].get(b)
                    if cm is None or tm is None or not len(cm) or not len(tm):
                        continue
                    pc = cm[rng.integers(0, len(cm), len(cm))].mean(0)
                    pt = tm[rng.integers(0, len(tm), len(tm))].mean(0)
                    worst = min(worst, _cos(pc, pt))
                per_t.append(worst)
            return min(per_t)

        draws = {c: [] for c in args.candidates}
        sole = 0
        for _ in range(args.bootstrap):
            mins = {c: boot_min(c) for c in args.candidates}
            for c in args.candidates:
                draws[c].append(mins[c])
            if [c for c in args.candidates if mins[c] >= args.min_consensus] == winners:
                sole += 1
        print(f"\nitem-bootstrap B={args.bootstrap} (confirmation pass):")
        print(f"{'candidate':<16}{'MIN 95% CI':>22}{'P(pass)':>9}")
        for c in args.candidates:
            arr = np.array(draws[c]); lo, hi = np.percentile(arr, [2.5, 97.5])
            results[c]["ci95"] = [float(lo), float(hi)]
            results[c]["p_pass"] = float((arr >= args.min_consensus).mean())
            print(f"{c:<16}   [{lo:.3f}, {hi:.3f}]{results[c]['p_pass']:>9.2f}")
        print(f"\nP(winner set stable) = {sole/args.bootstrap:.2f}")

    (RESULTS_DIR / args.run_id / "cross_teacher_consensus.json").write_text(
        json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
