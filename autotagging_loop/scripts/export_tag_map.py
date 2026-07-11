"""Export a CV fold's tag data (T_star.json + vocab_star.json) to space/tag_map.json.

Usage:
    uv run python scripts/export_tag_map.py --run results/part2_experiment/<run>/fold2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = _REPO_ROOT / "space" / "tag_map.json"


def build_tag_map(run_dir: Path) -> dict[str, Any]:
    """Read and validate a fold's T_star/vocab_star; return the tag_map payload."""
    if (run_dir / "final").is_dir():
        run_dir = run_dir / "final"
    try:
        with open(run_dir / "T_star.json", encoding="utf-8") as fh:
            tag_scores = json.load(fh)
        with open(run_dir / "vocab_star.json", encoding="utf-8") as fh:
            vocab = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read run files in {run_dir}: {exc}") from exc
    if not isinstance(vocab, list) or not vocab:
        raise ValueError("vocab_star.json must be a non-empty list")
    for entry in vocab:
        if not isinstance(entry, dict) or not all(entry.get(k) for k in ("id", "name", "definition")):
            raise ValueError(f"vocab entry missing id/name/definition: {entry!r}")
    vocab_ids = {entry["id"] for entry in vocab}
    if not isinstance(tag_scores, dict) or not tag_scores:
        raise ValueError("T_star.json must be a non-empty object")
    for bench, vector in tag_scores.items():
        if not isinstance(vector, dict):
            raise ValueError(f"tag scores for '{bench}' must be an object")
        unknown = sorted(set(vector) - vocab_ids)
        if unknown:
            raise ValueError(f"'{bench}' uses tag ids missing from vocab: {unknown}")
        for tag_id, score in vector.items():
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise ValueError(f"non-numeric score for '{bench}'/{tag_id}: {score!r}")
    try:
        run_label = str(run_dir.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        run_label = str(run_dir)
    return {
        "meta": {"run": run_label, "n_benchmarks": len(tag_scores), "n_tags": len(vocab)},
        "vocab": vocab,
        "tag_scores": tag_scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export T_star/vocab_star to space/tag_map.json")
    parser.add_argument("--run", required=True, help="fold dir or its final/ dir")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output path (default: space/tag_map.json)")
    args = parser.parse_args()
    try:
        tag_map = build_tag_map(Path(args.run))
    except ValueError as exc:
        print(f"export_tag_map: {exc}", file=sys.stderr)
        return 1
    out_path = Path(args.out)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(tag_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, out_path)
    meta = tag_map["meta"]
    print(f"wrote {out_path} ({meta['n_benchmarks']} benchmarks, {meta['n_tags']} tags)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
