"""Stage 1.5 pilot — run the new positive-pair v_loop (experiment/vloop.py) with
the calibration winner (qwen27b) as item-tagger and qwen3.5-397b as improver.

Wires the pure vloop skeleton to real inputs:
  - tagger/improver JsonLLM -> SELFHOST gateway (empty/invalid => retry then RAISE;
    this is the guard against the Run-D silent error_fallback='{}' cascade)
  - benchmarks -> data/labels_part2_full items + Y_norm model-score matrix (20x13)
  - initial PromptState -> tagger_calibration.txt + cognitive_abilities.json (10 axes)

Each iteration re-tags all items under the current vocab (vocab is the thing being
improved), so tags are cached per iter dir and the run is resume-safe.

Usage:
  python -m experiment.vloop_pilot --run-id pilot_001 [--n-items 100] [--max-iter 3]
  python -m experiment.vloop_pilot --smoke   # offline wiring check, no gateway
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
from itertools import combinations
from pathlib import Path

import numpy as np
from tqdm import tqdm

import autotagging_loop.experiment.config  # noqa: F401 — import loads .env (SELFHOST_BASE_URL / SELFHOST_API_KEY)
import autotagging_loop.experiment.calibration as cal  # sample_items, _item_block, _append_row, _load_cached
from autotagging_loop.experiment import vloop
from autotagging_loop.experiment.llm_client import shared_factory

REPO = Path(__file__).resolve().parents[2]
SCORES_PATH = REPO / "results/part2_experiment/run_cv_20260624_141839/fold0/score_matrix.json"
ABILITIES_PATH = REPO / "data/cognitive_abilities.json"
INSTRUCTIONS_PATH = REPO / "experiment/prompts/tagger_calibration.txt"
OUT_DIR = REPO / "results/vloop_pilot"

TAGGER_MODEL = "openrouter/qwen/qwen3.5-27b"
IMPROVER_MODEL = "openrouter/qwen/qwen3.5-397b-a17b"
MAX_WORKERS = 100
LLM_TRIES = 3
IMPROVER_TEMP = 0.7  # >0 so repeated improve-from-best explores (gate always improves from best)
GATE_BOOTSTRAP = 500  # paired item-bootstrap resamples for the acceptance gate (zero LLM calls)
GATE_CONFIDENCE = 0.9  # accept iff P_boot(score_cand > score_best) >= this (same-state noise ~ +-0.01)
PATIENCE = 3  # early-stop after this many consecutive gate rejections
FEEDBACK_MAX = 5  # most recent rejected attempts shown back to the improver


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _chat(model_id: str, prompt: str, reasoning: bool, temperature: float = 0.0) -> str:
    """One JSON chat completion; returns stripped content ('' on empty)."""
    client = shared_factory().get(base_url_env="SELFHOST_BASE_URL", api_key_env="SELFHOST_API_KEY")
    extra = {} if reasoning else {"extra_body": {"reasoning": {"enabled": False}}}
    resp = client.chat.completions.create(
        model=model_id,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        **extra,
    )
    return (resp.choices[0].message.content or "").strip()


class GatewayJsonLLM:
    """vloop.JsonLLM over the SELFHOST gateway. Empty/invalid content RETRIES then RAISES
    — never returns a silent '{}' (the Run-D failure mode)."""

    def __init__(self, model_id: str, reasoning: bool, tries: int = LLM_TRIES, temperature: float = 0.0):
        self.model_id = model_id
        self.reasoning = reasoning
        self.tries = tries
        self.temperature = temperature

    def complete_json(self, prompt: str) -> dict:
        last = "unset"
        for _ in range(self.tries):
            try:
                content = _chat(self.model_id, prompt, self.reasoning, self.temperature)
            except Exception as exc:  # gateway/network — retry
                last = f"exc:{exc}"
                continue
            if not content:
                last = "empty-response"
                continue
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                last = f"bad-json:{content[:120]}"
        raise RuntimeError(f"{self.model_id}: no valid JSON after {self.tries} tries ({last})")


class FeedbackImprover(vloop.PromptImprover):
    """PromptImprover that also shows the improver its gate-rejected attempts, so it stops
    re-proposing edits that already failed (pilot_002's dominant failure: axis over-splitting)."""

    def __init__(self, llm):
        super().__init__(llm)
        self.rejections: list[dict] = []

    def note_rejection(
        self, state: vloop.PromptState, score: float, best_score: float, p_gt: float | None
    ) -> None:
        self.rejections.append({
            "n_axes": len(state.vocab),
            "axis_ids": [a.id for a in state.vocab],
            "score": round(score, 4),
            "best_score": round(best_score, 4),
            "p_better_than_best": None if p_gt is None else round(p_gt, 2),
        })

    def improve(self, state: vloop.PromptState, report: vloop.ObjectiveReport) -> vloop.PromptState:
        prompt = vloop.render_improver_prompt(state, report) + self._feedback_block()
        data = self.llm.complete_json(prompt)
        axes = [
            vloop.TagAxis(id=str(x["id"]), name=str(x["name"]), definition=str(x["definition"]))
            for x in data["vocab"]
        ]
        return vloop.PromptState(state.iteration + 1, str(data["tagger_instructions"]), axes)

    def _feedback_block(self) -> str:
        if not self.rejections:
            return ""
        recent = self.rejections[-FEEDBACK_MAX:]
        return (
            "\n\nGate feedback — these earlier vocab proposals were REJECTED (they did not beat "
            "the current best with statistical confidence). Do not repeat similar edits; try a "
            "qualitatively different change:\n"
            + json.dumps(recent, ensure_ascii=False, indent=2)
        )


# ---------- inputs ----------

def load_benchmarks(
    seed: int, n_items: int, exclude: frozenset[str] = frozenset(),
    scores_path: Path | None = None,
) -> list[vloop.Benchmark]:
    sm = json.loads((scores_path or SCORES_PATH).read_text())
    y_norm = sm["Y_norm"]  # display -> {model -> score}
    disp = {_norm(k): k for k in y_norm}
    slugs = sorted(p.name for p in cal.DATA_DIR.iterdir() if p.is_dir())
    excl = {_norm(e) for e in exclude}

    benches = []
    for slug in slugs:
        if _norm(slug) in excl:
            continue  # duplicate/corrupted target dropped (e.g. full-MATH ⊃ MATH-500)
        key = disp.get(_norm(slug))
        if key is None:
            continue  # no model scores -> can't contribute to the objective
        items = [
            vloop.Item(id=it["item_id"], prompt=cal._item_block(it))
            for it in cal.sample_items(slug, seed, n_items)
        ]
        benches.append(vloop.Benchmark(name=slug, items=items, scores=dict(y_norm[key])))
    return benches


def initial_state() -> vloop.PromptState:
    axes = json.loads(ABILITIES_PATH.read_text())
    vocab = [vloop.TagAxis(a["id"], a["name"], a["definition"]) for a in axes]
    return vloop.PromptState(0, INSTRUCTIONS_PATH.read_text(), vocab)


# ---------- per-iter state persistence (resume) ----------

def _state_path(iter_dir: Path) -> Path:
    return iter_dir / "state.json"


def save_state(iter_dir: Path, state: vloop.PromptState) -> None:
    iter_dir.mkdir(parents=True, exist_ok=True)
    _state_path(iter_dir).write_text(json.dumps({
        "iteration": state.iteration,
        "tagger_instructions": state.tagger_instructions,
        "vocab": [ax.__dict__ for ax in state.vocab],
    }, ensure_ascii=False, indent=2))


def load_state(iter_dir: Path) -> vloop.PromptState | None:
    p = _state_path(iter_dir)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    vocab = [vloop.TagAxis(a["id"], a["name"], a["definition"]) for a in d["vocab"]]
    return vloop.PromptState(d["iteration"], d["tagger_instructions"], vocab)


# ---------- tagging (parallel + cached) ----------

def tag_profiles(
    benches: list[vloop.Benchmark], state: vloop.PromptState, iter_dir: Path,
    tagger: vloop.ItemTagger, workers: int,
) -> tuple[list[vloop.BenchmarkProfile], int]:
    """Tag every item under state.vocab (cached per bench jsonl); returns (profiles, n_invalid)."""
    def path_for(b: vloop.Benchmark) -> Path:
        p = iter_dir / "tags" / f"{b.name}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    axis_set = {a.id for a in state.vocab}
    tasks = []
    for b in benches:
        cached = cal._load_cached(path_for(b))
        for it in b.items:
            row = cached.get(it.id)
            levels = row.get("levels") if row else None
            # re-tag if: missing, error row (levels=None), OR cached under a DIFFERENT
            # vocab (axis mismatch) — the latter guards resume when best's state changed
            # (e.g. seed->iter_1) but a stale-vocab tag cache was left behind.
            if not levels or set(levels) != axis_set:
                tasks.append((b, it))

    def _run(task) -> None:
        b, it = task
        try:
            tag = tagger.tag(it, state)
            row = {"item_id": it.id, "levels": tag.levels}
        except Exception as exc:  # invalid/unrecoverable for this item -> record, drop from profile
            row = {"item_id": it.id, "levels": None, "error": str(exc)[:200]}
        cal._append_row(path_for(b), row)

    if tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_run, t) for t in tasks]
            for f in tqdm(concurrent.futures.as_completed(futs), total=len(futs),
                          desc=f"iter{state.iteration} tag"):
                f.result()

    profiles, invalid = [], 0
    for b in benches:
        cached = cal._load_cached(path_for(b))
        item_tags = []
        for iid, row in cached.items():
            if row.get("levels"):
                item_tags.append(vloop.ItemTag(iid, row["levels"]))
            else:
                invalid += 1
        vec = vloop.aggregate_item_tags(item_tags, state.vocab)
        profiles.append(vloop.BenchmarkProfile(b, vec))
    return profiles, invalid


# ---------- acceptance gate (paired item bootstrap over cached tags; zero LLM calls) ----------

def _bench_matrices(
    iter_dir: Path, benches: list[vloop.Benchmark], state: vloop.PromptState
) -> dict[str, tuple[list[str], np.ndarray]]:
    """Per bench: (item_ids, n_items x n_axes matrix) of valid cached tags under state.vocab."""
    axis_ids = [a.id for a in state.vocab]
    out = {}
    for b in benches:
        cached = cal._load_cached(iter_dir / "tags" / f"{b.name}.jsonl")
        ids, rows = [], []
        for iid, row in cached.items():
            lv = row.get("levels")
            if lv:
                ids.append(iid)
                rows.append([vloop.LEVEL_VALUE[lv[a]] for a in axis_ids])
        mat = np.array(rows, dtype=float) if rows else np.zeros((0, len(axis_ids)))
        out[b.name] = (ids, mat)
    return out


def _pair_ranksims(
    benches: list[vloop.Benchmark], min_common: int = 4
) -> dict[tuple[str, str], float]:
    """Model-rank similarity per benchmark pair — fixed, independent of tags."""
    out = {}
    for x, y in combinations(sorted(benches, key=lambda b: b.name), 2):
        val, _ = vloop.rank_similarity(x.scores, y.scores, min_common)
        if val is not None:
            out[(x.name, y.name)] = val
    return out


def _score_from_Tb(Tb: dict[str, np.ndarray], ranksims: dict, top_q: float = 0.2) -> float:
    """Same objective as vloop.PositivePairObjective, computed from T_b vectors directly."""
    ranked = []
    for (a, b), rsim in ranksims.items():
        va, vb = Tb[a], Tb[b]
        na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        cos = float(va @ vb) / (na * nb) if na > 0 and nb > 0 else 0.0
        ranked.append((-cos, a, b, rsim))
    ranked.sort()
    k = max(1, math.ceil(len(ranked) * top_q))
    return sum(r[3] for r in ranked[:k]) / k


def smoothed_score(
    iter_dir: Path, state: vloop.PromptState, benches: list[vloop.Benchmark],
    ranksims: dict, n_boot: int = GATE_BOOTSTRAP, seed: int = 7,
) -> float:
    """Bagged objective — mean top-k score over item resamples (zero LLM calls).
    The hard top-k cut is discontinuous exactly where pairs are densest (neighbor
    gaps ~0.001 vs same-state rerun noise ~0.004), so the point estimate jumps by
    whole-pair swaps; averaging over resamples turns membership into selection
    frequency and cut same-state noise 0.0355 -> 0.0076 (measured, p2it0 vs p3it0)."""
    mats = _bench_matrices(iter_dir, benches, state)
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_boot):
        Tb = {}
        for name, (_ids, m) in mats.items():
            n = m.shape[0]
            Tb[name] = m[rng.integers(0, n, n)].mean(axis=0) if n else np.zeros(m.shape[1])
        total += _score_from_Tb(Tb, ranksims)
    return total / n_boot


def gate_p_better(
    cand_dir: Path, best_dir: Path, cand_state: vloop.PromptState, best_state: vloop.PromptState,
    benches: list[vloop.Benchmark], ranksims: dict, rng: np.random.Generator,
) -> float:
    """P(score(cand) > score(best)) under paired item resampling — the same item draw is
    applied to both states so decode/sampling noise cancels (calibration-style paired design)."""
    cand_m = _bench_matrices(cand_dir, benches, cand_state)
    best_m = _bench_matrices(best_dir, benches, best_state)
    shared = {}
    for name in cand_m:
        c_ids, cm = cand_m[name]
        b_ids, bm = best_m[name]
        common = sorted(set(c_ids) & set(b_ids))
        ci = {i: k for k, i in enumerate(c_ids)}
        bi = {i: k for k, i in enumerate(b_ids)}
        shared[name] = (cm[[ci[i] for i in common]], bm[[bi[i] for i in common]])
    wins = 0
    for _ in range(GATE_BOOTSTRAP):
        Tc, Tb = {}, {}
        for name, (cm, bm) in shared.items():
            n = cm.shape[0]
            if n == 0:
                Tc[name] = np.zeros(cm.shape[1])
                Tb[name] = np.zeros(bm.shape[1])
                continue
            idx = rng.integers(0, n, n)
            Tc[name] = cm[idx].mean(axis=0)
            Tb[name] = bm[idx].mean(axis=0)
        if _score_from_Tb(Tc, ranksims) > _score_from_Tb(Tb, ranksims):
            wins += 1
    return wins / GATE_BOOTSTRAP


# ---------- loop ----------

def run_loop(
    run_dir: Path, benches: list[vloop.Benchmark], state0: vloop.PromptState,
    tagger_llm, improver_llm, max_iter: int, workers: int, seed: int = 42,
) -> dict:
    """Gated hill-climb. Every improver step starts from the BEST state so far; a candidate
    is adopted only when its SMOOTHED (bagged) score beats best AND the paired item-bootstrap
    says so with >= GATE_CONFIDENCE (kills noise-acceptances like pilot_002's +0.0002).
    Rejections are fed back into the improver prompt; PATIENCE consecutive rejections end
    the run early."""
    tagger = vloop.ItemTagger(tagger_llm)
    objective = vloop.PositivePairObjective()
    improver = FeedbackImprover(improver_llm)
    ranksims = _pair_ranksims(benches)
    rng = np.random.default_rng(seed)

    history: list[dict] = []
    best = None
    consec_rej = 0

    for k in range(max_iter):
        iter_dir = run_dir / f"iter_{k}"
        cand = load_state(iter_dir)  # resume: reuse the saved candidate (non-deterministic improver)
        if cand is None:
            if k == 0:
                cand = state0
            else:
                try:
                    cand = improver.improve(best["state"], best["report"])  # always from best
                except Exception as exc:
                    print(f"  iter {k}: improver failed ({exc}); skipping")
                    history.append({"iteration": k, "score": None, "accepted": False,
                                    "n_axes": None, "n_invalid": None, "error": str(exc)[:200]})
                    continue
            save_state(iter_dir, cand)

        profiles, invalid = tag_profiles(benches, cand, iter_dir, tagger, workers)
        report = objective.evaluate(profiles)
        s_smooth = smoothed_score(iter_dir, cand, benches, ranksims)

        if best is None:
            p_gt, accepted = None, True  # seed is always adopted
        else:
            p_gt = gate_p_better(iter_dir, best["dir"], cand, best["state"], benches, ranksims, rng)
            # smoothed improvement AND bootstrap confidence — the smoothed point estimate
            # replaces the hard top-k one (noise 0.0355 -> 0.0076); p_gt still required
            accepted = s_smooth > best["smooth"] and p_gt >= GATE_CONFIDENCE

        row = {
            "iteration": k, "score": report.score, "score_smooth": s_smooth,
            "loss": report.loss, "accepted": accepted,
            "p_better_than_best": p_gt,
            "n_selected": len(report.selected_pairs), "n_pairs": len(report.all_pairs),
            "n_axes": len(cand.vocab), "n_invalid": invalid,
            "from_iter": None if best is None else best["iter"],
        }
        history.append(row)
        (iter_dir / "report.json").write_text(json.dumps({
            **row, "selected_pairs": [pr.__dict__ for pr in report.selected_pairs],
        }, ensure_ascii=False, indent=2))
        base = "seed" if best is None else f"from best(iter{best['iter']})"
        p_txt = "-" if p_gt is None else f"{p_gt:.2f}"
        print(f"  iter {k}: smooth={s_smooth:.4f} (hard={report.score:.4f}) axes={len(cand.vocab)} "
              f"invalid={invalid} p_gt={p_txt} [{'ACCEPT' if accepted else 'reject'}] {base}")

        if accepted:
            best = {"iter": k, "score": report.score, "smooth": s_smooth, "state": cand,
                    "report": report, "dir": iter_dir}
            consec_rej = 0
        else:
            improver.note_rejection(cand, s_smooth, best["smooth"], p_gt)
            consec_rej += 1
            if consec_rej >= PATIENCE:
                print(f"  early stop: {PATIENCE} consecutive rejections (converged at iter {best['iter']})")
                break

    _write_summary(run_dir, history, best)
    return {"history": history, "best": best}


def _write_summary(run_dir: Path, history: list[dict], best: dict | None) -> None:
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    lines = ["# V-Loop Pilot (gated, smoothed objective)", "",
             "| iter | smooth | hard | p_gt | n_axes | selected/pairs | invalid | gate |",
             "|---|---|---|---|---|---|---|---|"]
    for h in history:
        sm = f"{h['score_smooth']:.4f}" if h.get("score_smooth") is not None else "— (improver fail)"
        sc = f"{h['score']:.4f}" if h.get("score") is not None else "—"
        pg = h.get("p_better_than_best")
        pgs = f"{pg:.2f}" if pg is not None else "—"
        ax = h.get("n_axes"); axs = str(ax) if ax is not None else "—"
        sp = f"{h.get('n_selected', '—')}/{h.get('n_pairs', '—')}"
        inv = h.get("n_invalid"); invs = str(inv) if inv is not None else "—"
        gate = "✓ best" if h.get("accepted") else "reject"
        lines.append(f"| {h['iteration']} | {sm} | {sc} | {pgs} | {axs} | {sp} | {invs} | {gate} |")
    if best is not None:
        lines += ["", f"**Best: iter {best['iter']}, smooth {best['smooth']:.4f} "
                      f"(hard {best['score']:.4f})**"]
        (run_dir / "best_state.json").write_text(json.dumps({
            "iter": best["iter"], "score": best["score"], "score_smooth": best["smooth"],
            "tagger_instructions": best["state"].tagger_instructions,
            "vocab": [ax.__dict__ for ax in best["state"].vocab],
        }, ensure_ascii=False, indent=2))
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


# ---------- smoke (offline wiring check) ----------

def _smoke() -> None:
    import random
    import tempfile

    ids = ["a1", "a2", "a3"]
    instr = "test instructions"

    class FakeTagger:
        def __init__(self): self.rng = random.Random(0)
        def complete_json(self, prompt): return {"tags": {i: self.rng.choice(list(vloop.LEVELS)) for i in ids}}

    class FakeImprover:
        def complete_json(self, prompt):
            return {"tagger_instructions": instr, "vocab": [{"id": i, "name": i, "definition": "d"} for i in ids]}

    # 4 fake benches, 3 items each, 5 shared models with varied scores
    benches = []
    for bi in range(4):
        items = [vloop.Item(id=f"b{bi}_i{j}", prompt=f"item {j}") for j in range(3)]
        scores = {f"m{m}": float((m * 7 + bi * 3) % 11) for m in range(5)}
        benches.append(vloop.Benchmark(name=f"bench{bi}", items=items, scores=scores))

    state0 = vloop.PromptState(0, instr, [vloop.TagAxis(i, i, "d") for i in ids])
    with tempfile.TemporaryDirectory() as td:
        out = run_loop(Path(td), benches, state0, FakeTagger(), FakeImprover(), max_iter=3, workers=4)
    hist = out["history"]
    assert out["best"] is not None
    assert hist[0]["accepted"] is True  # seed always adopted
    acc_scores = [h["score_smooth"] for h in hist if h.get("accepted")]
    assert abs(out["best"]["smooth"] - max(acc_scores)) < 1e-12, (out["best"]["smooth"], acc_scores)
    # bootstrap gate: every non-seed acceptance carried the required confidence
    assert all(
        h["p_better_than_best"] >= GATE_CONFIDENCE for h in hist[1:] if h.get("accepted")
    ), hist
    print("smoke OK:", [(round(h["score"], 3), h["accepted"], h.get("p_better_than_best"))
                        for h in hist])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 1.5 positive-pair v_loop pilot")
    parser.add_argument("--run-id")
    parser.add_argument("--n-items", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--exclude", default="", help="comma-separated benchmark slugs to drop")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        _smoke()
        return
    if not args.run_id:
        parser.error("--run-id required (or use --smoke)")

    run_dir = OUT_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    exclude = frozenset(s.strip() for s in args.exclude.split(",") if s.strip())
    benches = load_benchmarks(args.seed, args.n_items, exclude)
    print(f"pilot: {len(benches)} benches, tagger={TAGGER_MODEL}, improver={IMPROVER_MODEL}, "
          f"n_items<={args.n_items}, max_iter={args.max_iter}")
    run_loop(
        run_dir, benches, initial_state(),
        GatewayJsonLLM(TAGGER_MODEL, reasoning=False),
        GatewayJsonLLM(IMPROVER_MODEL, reasoning=True, temperature=IMPROVER_TEMP),
        args.max_iter, args.workers, seed=args.seed,
    )


if __name__ == "__main__":
    main()
