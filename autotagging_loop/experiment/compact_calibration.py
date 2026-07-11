"""Compact calibration tag caches: keep only the last row per item_id.

Tag JSONL files are append-only — a retried item leaves its stale (invalid) row
in place plus a fresh row. `_load_cached` already dedups last-wins at read time,
so metrics are correct, but the raw files look padded with dead rows. This
rewrites each file to its deduped final state, in place and atomically.
Idempotent; safe to resume the run afterwards.

Usage: python -m experiment.compact_calibration [run_dir ...]
       (default: every run dir under results/calibration/)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "calibration"


def compact_file(path: Path) -> tuple[int, int]:
    """Rewrite path keeping the last row per item_id; returns (before, after) row counts."""
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    last: dict[str, str] = {}
    for ln in lines:
        last[json.loads(ln)["item_id"]] = ln  # last-wins; dict keeps first-seen key order
    kept = list(last.values())
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(kept) + ("\n" if kept else ""))
    os.replace(tmp, path)  # atomic; no partial file on crash
    reloaded = {json.loads(ln)["item_id"] for ln in path.read_text().splitlines() if ln.strip()}
    assert reloaded == set(last), f"item_id set changed after compaction: {path}"
    return len(lines), len(kept)


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    roots = [Path(a) for a in args] or sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir())
    before_tot = after_tot = 0
    for root in roots:
        for f in sorted(root.rglob("*.jsonl")):
            before, after = compact_file(f)
            before_tot += before
            after_tot += after
            if before != after:
                print(f"{f}: {before} -> {after}")
    print(f"done: {before_tot} -> {after_tot} rows ({before_tot - after_tot} stale rows dropped)")


if __name__ == "__main__":
    main()
