"""Storage helpers for Part 2 run artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def make_run_dir(results_dir: str) -> Path:
    root = Path(results_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
