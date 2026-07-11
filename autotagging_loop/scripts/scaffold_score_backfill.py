"""Create a TODO skeleton for curated score backfill records.

The output is intentionally not valid backfill data: scores are null and
sources are placeholders. It exists to prevent omitted cells while a human
fills exact, source-grounded benchmark/model scores.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autotagging_loop.runner.config import load_config
from autotagging_loop.scripts.validate_score_backfill import DEFAULT_MISSING_CSV, load_missing_cell_plan


def build_score_backfill_skeleton(
    missing_cells: dict[tuple[str, str], dict[str, str]],
    *,
    missing_csv: str | Path,
) -> dict[str, Any]:
    rows = list(missing_cells.values())
    return {
        "_meta": {
            "description": "TODO skeleton for manually curated missing score cells.",
            "generated_from": str(Path(missing_csv)),
            "record_count": len(rows),
            "score_scale": "0-1",
            "policy": (
                "Fill exact benchmark/model cells from official reports, model "
                "cards, papers, or public leaderboards. Do not use composite "
                "or inferred scores."
            ),
            "validation": "Run scripts/validate_score_backfill.py before any experiment run.",
        },
        "scores": [
            {
                "scope": row["scope"],
                "benchmark": row["benchmark"],
                "model": row["model"],
                "score": None,
                "metric": "TODO exact benchmark metric",
                "scale": "0-1",
                "source": {
                    "title": "Replace with official report or leaderboard title",
                    "url": "https://example.com/replace-with-source",
                    "date": "YYYY-MM-DD",
                },
                "notes": "TODO extraction note, score table name, or conversion detail.",
            }
            for row in rows
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--missing-csv",
        default=str(DEFAULT_MISSING_CSV),
        help="Missing-cell CSV plan. Defaults to data/score_backfill_missing.csv.",
    )
    parser.add_argument(
        "--out",
        default="-",
        help="Output path, or '-' for stdout.",
    )
    args = parser.parse_args()

    config = load_config()
    missing = load_missing_cell_plan(
        args.missing_csv,
        model_aliases=config.get("model_aliases") or {},
    )
    payload = build_score_backfill_skeleton(missing, missing_csv=args.missing_csv)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out == "-":
        print(text, end="")
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out} records={len(payload['scores'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
