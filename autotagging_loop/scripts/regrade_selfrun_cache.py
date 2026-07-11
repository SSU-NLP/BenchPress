"""Re-grade cached self-run responses without model calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autotagging_loop.scripts.run_selfrun_scores import LOADERS, MODEL_SLUGS, VALIDATION, grade

DEFAULT_OUT = ROOT / "results" / "item_grades.jsonl"
DEFAULT_SUMMARY = ROOT / "results" / "item_grade_summary.json"

MODEL_BY_SLUG = {slug: model for model, slug in MODEL_SLUGS.items()}


def default_cache_paths() -> list[Path]:
    return sorted((ROOT / "results").glob("selfrun_cache*.jsonl"))


def load_cache(paths: list[Path]) -> dict[tuple[str, str, str], str]:
    rows: dict[tuple[str, str, str], str] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rows[(rec["slug"], rec["bench"], str(rec["id"]))] = rec.get("text", "")
    return rows


def load_items(benches: set[str]) -> dict[str, dict[str, dict]]:
    loaded: dict[str, dict[str, dict]] = {}
    for bench in sorted(benches):
        if bench not in LOADERS:
            continue
        try:
            items = LOADERS[bench]()
        except Exception as exc:  # noqa: BLE001 - external HF/cache load
            print(f"skip {bench}: loader failed ({exc.__class__.__name__})")
            continue
        loaded[bench] = {str(item["id"]): item for item in items}
    return loaded


def regrade(cache: dict[tuple[str, str, str], str]) -> tuple[list[dict], dict]:
    items_by_bench = load_items({bench for _, bench, _ in cache})
    rows: list[dict] = []
    missing = 0
    for (slug, bench, item_id), text in sorted(cache.items()):
        item = items_by_bench.get(bench, {}).get(item_id)
        if item is None:
            missing += 1
            continue
        score = grade(bench, text, item)
        rows.append({
            "slug": slug,
            "model": MODEL_BY_SLUG.get(slug, slug),
            "benchmark": bench,
            "item_id": item_id,
            "score": score,
        })
    summary = summarize(rows)
    summary["_meta"] = {
        "cache_records": len(cache),
        "graded_records": len(rows),
        "missing_items": missing,
    }
    return rows, summary


def summarize(rows: list[dict]) -> dict:
    buckets: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        buckets.setdefault((row["benchmark"], row["model"]), []).append(float(row["score"]))
    cells = []
    for (bench, model), scores in sorted(buckets.items()):
        cells.append({
            "benchmark": bench,
            "model": model,
            "n": len(scores),
            "score": sum(scores) / len(scores) if scores else 0.0,
        })
    return {"cells": cells}


def write_jsonl(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_validation(summary: dict) -> None:
    by_cell = {(c["benchmark"], c["model"]): c for c in summary["cells"]}
    print(f"{'cell':<32}{'cache':>8}{'ref':>8}{'delta':>9}{'n':>8}")
    print("-" * 65)
    for cell, ref in sorted(VALIDATION.items()):
        got = by_cell.get(cell)
        if not got:
            continue
        delta = got["score"] - ref
        print(f"{cell[0]+'/'+cell[1]:<32}{got['score']:>8.3f}{ref:>8.3f}{delta:>+9.3f}{got['n']:>8}")


def _leaderboard_value(path: Path, bench: str, model: str) -> float | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = (data.get(bench) or {}).get(model)
    if isinstance(raw, dict):
        raw = raw.get("score")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def print_leaderboard_validation(summary: dict, path: Path, *, tolerance: float = 0.002) -> dict:
    print(f"{'cell':<32}{'cache':>8}{'leader':>8}{'delta':>9}{'n':>8}{'status':>9}")
    print("-" * 74)
    rows = []
    for cell in summary["cells"]:
        ref = _leaderboard_value(path, cell["benchmark"], cell["model"])
        if ref is None:
            continue
        delta = cell["score"] - ref
        status = "ok" if abs(delta) <= tolerance else "check"
        rows.append({**cell, "leaderboard": ref, "delta": delta, "status": status})
        print(f"{cell['benchmark']+'/'+cell['model']:<32}{cell['score']:>8.4f}{ref:>8.4f}"
              f"{delta:>+9.4f}{cell['n']:>8}{status:>9}")
    return {"tolerance": tolerance, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="*", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--summary-out", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--leaderboard", default=None,
                    help="optional leaderboard_scores.json for cached-cell validation")
    args = ap.parse_args()

    paths = [Path(p) for p in args.cache] if args.cache else default_cache_paths()
    cache = load_cache(paths)
    if not cache:
        print("cache absent; skipped re-grade")
        return

    rows, summary = regrade(cache)
    write_jsonl(rows, Path(args.out))
    Path(args.summary_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(rows)} graded records)")
    print(f"wrote {args.summary_out}")
    if args.validate:
        print()
        print_validation(summary)
    if args.leaderboard:
        print()
        summary["leaderboard_validation"] = print_leaderboard_validation(summary, Path(args.leaderboard))
        Path(args.summary_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")


if __name__ == "__main__":
    main()
