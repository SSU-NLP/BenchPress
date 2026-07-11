"""Stage 2 main experiment — held-out bench-split v_loop (generalization claim).

Design (agreed 2026-07-09):
  - 19 benches (full-MATH excluded — corrupted target vector, see pilot_005).
  - Bench split: family-stratified K-fold (splits.py), test fold held out entirely.
    Test benches are NEVER tagged during the loop; the improver never sees them.
  - Items: uniform random, fixed seed, manifest via tag cache. n_items=200 means
    pool<=200 benches use their FULL item set (census) automatically.
  - Loop: improver sees the TRAIN-pair report only; the gate accepts on
    DEV-involving pairs (dev x train + dev x dev) — smoothed delta AND
    paired-bootstrap p_gt >= GATE_CONFIDENCE. Patience early-stop as in pilots.
  - Final: seed and best states tag the TEST benches on their FULL item sets
    (census — zero item-sampling variance in the reported metrics). Report
    positive-pair score over test-involving pairs + per-test-bench retrieval.

Usage:
  python -m experiment.vloop_main --run-id main_001 [--dry-run] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import autotagging_loop.experiment.config  # noqa: F401 — loads .env
from autotagging_loop.experiment import splits, vloop
from autotagging_loop.experiment.vloop_pilot import (
    GATE_CONFIDENCE, IMPROVER_MODEL, IMPROVER_TEMP, MAX_WORKERS, PATIENCE,
    TAGGER_MODEL, FeedbackImprover, GatewayJsonLLM, _bench_matrices,
    _pair_ranksims, _score_from_Tb, gate_p_better, initial_state,
    load_benchmarks, load_state, save_state, smoothed_score, tag_profiles,
)

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results/vloop_main"
CENSUS_N = 10 ** 9  # sample_items caps at pool size -> full item set
RETRIEVAL_K = 3


def _stratum(name: str) -> str:
    if "HMMT" in name.upper():
        return "math"  # default classifier files HMMT under reasoning
    return splits._default_benchmark_stratum(name)


def involving(ranksims: dict, group: set[str]) -> dict:
    """Pairs with at least one endpoint in `group`."""
    return {p: v for p, v in ranksims.items() if p[0] in group or p[1] in group}


def profiles_from(tag_dir: Path, state: vloop.PromptState, benches) -> dict[str, np.ndarray]:
    mats = _bench_matrices(tag_dir, benches, state)
    return {n: (m.mean(axis=0) if m.shape[0] else np.zeros(m.shape[1]))
            for n, (_ids, m) in mats.items()}


def retrieval_report(P: dict, ranksims: dict, test_names, k: int = RETRIEVAL_K) -> dict:
    """Per test bench: mean rank-sim of its top-k tag-cos neighbours vs all-neighbour mean."""
    out = {}
    for t in test_names:
        cands = []
        for o in sorted(P):
            key = tuple(sorted((t, o)))
            if o == t or key not in ranksims:
                continue
            na, nb = float(np.linalg.norm(P[t])), float(np.linalg.norm(P[o]))
            cos = float(P[t] @ P[o]) / (na * nb) if na > 0 and nb > 0 else 0.0
            cands.append((cos, o, ranksims[key]))
        cands.sort(reverse=True)
        top = cands[:k]
        out[t] = {
            "top_neighbors": [(o, round(c, 4), round(r, 4)) for c, o, r in top],
            f"top{k}_mean_ranksim": sum(r for _, _, r in top) / len(top),
            "all_mean_ranksim": sum(r for _, _, r in cands) / len(cands),
            "n_candidates": len(cands),
        }
    return out


def run_main(
    run_dir: Path, bench_split: splits.BenchmarkSplit,
    loop_benches, test_benches, tagger_llm, improver_llm,
    max_iter: int, workers: int, seed: int = 42,
) -> dict:
    tagger = vloop.ItemTagger(tagger_llm)
    objective = vloop.PositivePairObjective()
    improver = FeedbackImprover(improver_llm)
    rng = np.random.default_rng(seed)

    train_set, dev_set = set(bench_split.train), set(bench_split.dev)
    dev_rs = involving(_pair_ranksims(loop_benches), dev_set)

    history: list[dict] = []
    best = None
    consec_rej = 0

    for k in range(max_iter):
        iter_dir = run_dir / f"iter_{k}"
        cand = load_state(iter_dir)
        if cand is None:
            if k == 0:
                cand = initial_state()
            else:
                try:
                    cand = improver.improve(best["state"], best["train_report"])
                except Exception as exc:
                    print(f"  iter {k}: improver failed ({exc}); skipping")
                    history.append({"iteration": k, "dev_smooth": None, "accepted": False,
                                    "error": str(exc)[:200]})
                    continue
            save_state(iter_dir, cand)

        profiles, invalid = tag_profiles(loop_benches, cand, iter_dir, tagger, workers)
        train_report = objective.evaluate([p for p in profiles if p.benchmark.name in train_set])
        s_dev = smoothed_score(iter_dir, cand, loop_benches, dev_rs)

        if best is None:
            p_gt, accepted = None, True
        else:
            p_gt = gate_p_better(iter_dir, best["dir"], cand, best["state"],
                                 loop_benches, dev_rs, rng)
            accepted = s_dev > best["dev_smooth"] and p_gt >= GATE_CONFIDENCE

        row = {
            "iteration": k, "dev_smooth": s_dev, "train_score": train_report.score,
            "accepted": accepted, "p_better_than_best": p_gt,
            "n_axes": len(cand.vocab), "n_invalid": invalid,
            "from_iter": None if best is None else best["iter"],
        }
        history.append(row)
        (iter_dir / "report.json").write_text(json.dumps(row, indent=2))
        p_txt = "-" if p_gt is None else f"{p_gt:.2f}"
        print(f"  iter {k}: dev_smooth={s_dev:.4f} train={train_report.score:.4f} "
              f"axes={len(cand.vocab)} invalid={invalid} p_gt={p_txt} "
              f"[{'ACCEPT' if accepted else 'reject'}]")

        if accepted:
            best = {"iter": k, "dev_smooth": s_dev, "state": cand,
                    "train_report": train_report, "dir": iter_dir}
            consec_rej = 0
        else:
            improver.note_rejection(cand, s_dev, best["dev_smooth"], p_gt)
            consec_rej += 1
            if consec_rej >= PATIENCE:
                print(f"  early stop: {PATIENCE} consecutive rejections (best=iter {best['iter']})")
                break

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    final = final_eval(run_dir, best, loop_benches, test_benches, bench_split,
                       tagger, workers, rng)
    return {"history": history, "best_iter": best["iter"], "final": final}


def final_eval(
    run_dir: Path, best: dict, loop_benches, test_benches,
    bench_split: splits.BenchmarkSplit, tagger: vloop.ItemTagger,
    workers: int, rng: np.random.Generator,
) -> dict:
    """Census-tag the held-out test benches under seed and best states; report
    test-involving positive-pair score + per-test-bench retrieval."""
    all_benches = list(loop_benches) + list(test_benches)
    rs_test = involving(_pair_ranksims(all_benches), set(bench_split.test))
    seed_state = load_state(run_dir / "iter_0")

    results: dict = {"split": {"train": bench_split.train, "dev": bench_split.dev,
                               "test": bench_split.test}, "best_iter": best["iter"]}
    for label, src_dir, st in [("seed", run_dir / "iter_0", seed_state),
                               ("best", best["dir"], best["state"])]:
        fdir = run_dir / "final" / label
        (fdir / "tags").mkdir(parents=True, exist_ok=True)
        for b in loop_benches:  # reuse loop tags — cache hit, zero LLM calls
            src = src_dir / "tags" / f"{b.name}.jsonl"
            dst = fdir / "tags" / f"{b.name}.jsonl"
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)
        _profs, invalid = tag_profiles(all_benches, st, fdir, tagger, workers)
        P = profiles_from(fdir, st, all_benches)
        results[label] = {
            "n_axes": len(st.vocab), "n_invalid": invalid,
            "test_hard": _score_from_Tb(P, rs_test),
            "test_smooth": smoothed_score(fdir, st, all_benches, rs_test),
            "retrieval": retrieval_report(P, _pair_ranksims(all_benches), bench_split.test),
        }
        print(f"  final[{label}]: test_smooth={results[label]['test_smooth']:.4f} "
              f"test_hard={results[label]['test_hard']:.4f}")

    results["p_gt_test"] = gate_p_better(
        run_dir / "final" / "best", run_dir / "final" / "seed",
        best["state"], seed_state, all_benches, rs_test, rng)
    d = results["best"]["test_smooth"] - results["seed"]["test_smooth"]
    print(f"  final: Δtest_smooth={d:+.4f} p_gt(test)={results['p_gt_test']:.2f}")

    (run_dir / "final" / "report.json").write_text(json.dumps(results, indent=2))
    lines = ["# V-Loop main experiment — held-out test report", "",
             f"split: train={bench_split.train}", f"dev={bench_split.dev}",
             f"test={bench_split.test}", "",
             "| state | test smooth | test hard | axes |", "|---|---|---|---|"]
    for label in ("seed", "best"):
        r = results[label]
        lines.append(f"| {label} | {r['test_smooth']:.4f} | {r['test_hard']:.4f} | {r['n_axes']} |")
    lines += ["", f"Δ test_smooth = {d:+.4f}, p_gt(test) = {results['p_gt_test']:.2f}", "",
              "## Retrieval (per held-out test bench, top-3 tag-cos neighbours)", ""]
    for label in ("seed", "best"):
        lines.append(f"### {label}")
        for t, r in results[label]["retrieval"].items():
            lines.append(f"- {t}: top3 mean rank-sim {r['top3_mean_ranksim']:+.3f} "
                         f"(all-neighbour mean {r['all_mean_ranksim']:+.3f}) "
                         f"neighbours={r['top_neighbors']}")
    (run_dir / "final" / "report.md").write_text("\n".join(lines) + "\n")
    return results


# ---------- offline smoke ----------

def _smoke() -> None:
    import random
    import tempfile

    ids = ["a1", "a2", "a3"]

    class FakeTagger:
        def __init__(self): self.rng = random.Random(0)
        def complete_json(self, prompt):
            return {"tags": {i: self.rng.choice(list(vloop.LEVELS)) for i in ids}}

    class FakeImprover:
        def complete_json(self, prompt):
            return {"tagger_instructions": "instr",
                    "vocab": [{"id": i, "name": i, "definition": "d"} for i in ids]}

    names = ["math500", "gsm8k", "humaneval", "mbpp", "mmlu", "gpqa", "bbh", "drop"]
    benches = []
    for bi, nm in enumerate(names):
        items = [vloop.Item(id=f"{nm}_i{j}", prompt=f"item {j}") for j in range(3)]
        scores = {f"m{m}": float((m * 7 + bi * 3) % 11) for m in range(5)}
        benches.append(vloop.Benchmark(name=nm, items=items, scores=scores))

    sp = splits.split_benchmarks_kfold_stratified(
        names, n_folds=4, fold=0, seed=0, stratum_fn=_stratum)
    loop_b = [b for b in benches if b.name not in set(sp.test)]
    test_b = [b for b in benches if b.name in set(sp.test)]
    assert test_b and loop_b

    import autotagging_loop.experiment.vloop_main as vm
    real_init = vm.initial_state
    vm.initial_state = lambda: vloop.PromptState(0, "instr", [vloop.TagAxis(i, i, "d") for i in ids])
    try:
        with tempfile.TemporaryDirectory() as td:
            out = run_main(Path(td), sp, loop_b, test_b, FakeTagger(), FakeImprover(),
                           max_iter=4, workers=4)
    finally:
        vm.initial_state = real_init

    hist = out["history"]
    assert hist[0]["accepted"] is True
    assert all(h["p_better_than_best"] >= GATE_CONFIDENCE for h in hist[1:] if h.get("accepted"))
    fin = out["final"]
    assert set(fin["split"]["test"]) == set(sp.test)
    assert 0.0 <= fin["p_gt_test"] <= 1.0
    for label in ("seed", "best"):
        assert len(fin[label]["retrieval"]) == len(sp.test)
    print("smoke OK:", [(h["iteration"], h["accepted"]) for h in hist],
          f"p_gt_test={fin['p_gt_test']:.2f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="held-out bench-split v_loop main experiment")
    parser.add_argument("--run-id")
    parser.add_argument("--n-items", type=int, default=200, help="pool<=n benches become census")
    parser.add_argument("--seed", type=int, default=42, help="item draw seed")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--exclude", default="math", help="comma-separated benchmarks to drop")
    parser.add_argument("--scores-path", type=Path, default=None,
                        help="pinned target matrix (default: vloop_pilot.SCORES_PATH)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        _smoke()
        return
    if not args.run_id:
        parser.error("--run-id required (or use --smoke)")

    exclude = frozenset(s.strip() for s in args.exclude.split(",") if s.strip())
    loop_pool = load_benchmarks(args.seed, args.n_items, exclude, scores_path=args.scores_path)
    names = [b.name for b in loop_pool]
    sp = splits.split_benchmarks_kfold_stratified(
        names, n_folds=args.n_folds, fold=args.fold, seed=args.split_seed, stratum_fn=_stratum)
    census_pool = load_benchmarks(args.seed, CENSUS_N, exclude, scores_path=args.scores_path)
    loop_benches = [b for b in loop_pool if b.name not in set(sp.test)]
    test_benches = [b for b in census_pool if b.name in set(sp.test)]

    n_loop = sum(len(b.items) for b in loop_benches)
    n_test = sum(len(b.items) for b in test_benches)
    print(f"split (fold {args.fold}/{args.n_folds}, seed {args.split_seed}):")
    print(f"  train ({len(sp.train)}): {sp.train}")
    print(f"  dev   ({len(sp.dev)}): {sp.dev}")
    print(f"  test  ({len(sp.test)}): {sp.test}")
    print(f"items/iter (train+dev, n<={args.n_items}): {n_loop}")
    print(f"test census items (x2 states at final): {n_test}")
    if args.dry_run:
        return

    run_dir = OUT_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        "n_items": args.n_items, "item_seed": args.seed, "split_seed": args.split_seed,
        "n_folds": args.n_folds, "fold": args.fold, "exclude": sorted(exclude),
        "split": {"train": sp.train, "dev": sp.dev, "test": sp.test},
        "tagger": TAGGER_MODEL, "improver": IMPROVER_MODEL,
        "gate_confidence": GATE_CONFIDENCE, "patience": PATIENCE,
        "scores_path": str(args.scores_path) if args.scores_path else "default(SCORES_PATH)",
        "scores_provenance": json.loads(args.scores_path.read_text()).get("provenance")
        if args.scores_path else None,
    }, indent=2))
    run_main(
        run_dir, sp, loop_benches, test_benches,
        GatewayJsonLLM(TAGGER_MODEL, reasoning=False),
        GatewayJsonLLM(IMPROVER_MODEL, reasoning=True, temperature=IMPROVER_TEMP),
        args.max_iter, args.workers, seed=args.seed,
    )


if __name__ == "__main__":
    main()
