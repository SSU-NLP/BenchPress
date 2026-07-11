"""
scripts/inspect_gt_vocab.py
============================
data/labels/*/tasks.jsonl의 reviewed row를 훑어
gt_reasoning_depth와 gt_answer_format의 실제 등장값을
벤치마크별 + 전체 합계로 출력하고, canonical vocab summary를
data/labels/vocab_summary.json에 저장.

목적: evaluator의 canonical vocab을 추측이 아닌 실측값으로 확정.

Usage:
    python scripts/inspect_gt_vocab.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = ROOT / "data" / "labels"
AXES = ["gt_reasoning_depth", "gt_answer_format"]
OUT_PATH = LABELS_DIR / "vocab_summary.json"


def load_reviewed(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("reviewer_status") == "reviewed":
                rows.append(row)
    return rows


def _print_axis(ctr: Counter, axis: str, indent: str = "  ") -> None:
    total = sum(ctr.values())
    print(f"{indent}{axis} ({total} rows):")
    for val, cnt in ctr.most_common():
        pct = 100 * cnt / total if total else 0.0
        label = val if val != "" else "<empty>"
        print(f"{indent}  {label:<30} {cnt:>7} ({pct:5.1f}%)")


def main() -> None:
    per_bench: dict[str, dict[str, Counter]] = defaultdict(
        lambda: {axis: Counter() for axis in AXES}
    )
    totals: dict[str, Counter] = {axis: Counter() for axis in AXES}

    bench_dirs = sorted(p for p in LABELS_DIR.iterdir() if p.is_dir())
    if not bench_dirs:
        print(f"no benchmark dirs under {LABELS_DIR}")
        return

    for bench_dir in bench_dirs:
        tasks_path = bench_dir / "tasks.jsonl"
        if not tasks_path.exists():
            continue
        bench = bench_dir.name
        rows = load_reviewed(tasks_path)
        for row in rows:
            for axis in AXES:
                val = row.get(axis)
                key = val if isinstance(val, str) else ""
                per_bench[bench][axis][key] += 1
                totals[axis][key] += 1

    for bench in sorted(per_bench):
        print(f"\n=== {bench} ===")
        for axis in AXES:
            _print_axis(per_bench[bench][axis], axis)

    print("\n=== TOTAL (all benchmarks) ===")
    for axis in AXES:
        _print_axis(totals[axis], axis)

    canonical = {axis: sorted(totals[axis].keys()) for axis in AXES}
    print("\n=== CANONICAL VOCAB (for evaluator constants) ===")
    for axis, vals in canonical.items():
        print(f"  {axis}: {vals}")

    summary = {
        "per_benchmark": {
            bench: {axis: dict(per_bench[bench][axis]) for axis in AXES}
            for bench in per_bench
        },
        "totals": {axis: dict(totals[axis]) for axis in AXES},
        "canonical": canonical,
    }
    OUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote summary → {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
