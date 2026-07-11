"""Diagnose pair-level failures in a v3 K-fold run without LLM calls."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from autotagging_loop.experiment.alignment import cosine_pair_matrix


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parse_pair_key(key: str) -> tuple[str, str]:
    parts = key.split("||")
    if len(parts) != 2:
        raise ValueError(f"invalid pair key: {key!r}")
    return (parts[0], parts[1])


def _pair_label(pair: tuple[str, str]) -> str:
    return f"{pair[0]} || {pair[1]}"


def _fmt(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "nan"
    return "nan" if not math.isfinite(v) else f"{v:+.4f}"


def _fold_rows(fold_dir: Path, *, scope: str) -> tuple[dict, list[dict]]:
    split_metrics = _load_json(fold_dir / "final" / "split_metrics.json")
    score_matrix = _load_json(fold_dir / "score_matrix.json")
    tag_vectors = _load_json(fold_dir / "final" / "T_star.json")

    benchmarks = split_metrics["benchmark_split"][scope]
    S = cosine_pair_matrix(tag_vectors, benchmark_names=benchmarks)
    R_raw = {
        _parse_pair_key(key): value
        for key, value in score_matrix.get("R_raw", {}).items()
    }
    common_count = {
        _parse_pair_key(key): value
        for key, value in score_matrix.get("common_count", {}).items()
    }

    rows: list[dict] = []
    for pair, s_val in S.items():
        r_val = R_raw.get(pair)
        if r_val is None:
            continue
        residual = abs(float(s_val) - float(r_val))
        rows.append(
            {
                "pair": pair,
                "S": float(s_val),
                "R": float(r_val),
                "residual": residual,
                "common": common_count.get(pair),
                "direction": (
                    "tag_similarity_too_high"
                    if float(s_val) > float(r_val)
                    else "tag_similarity_too_low"
                ),
            }
        )
    return split_metrics.get(scope, {}), sorted(
        rows,
        key=lambda row: row["residual"],
        reverse=True,
    )


def diagnose(parent_dir: Path, *, scope: str, top_k: int) -> int:
    fold_dirs = sorted(p for p in parent_dir.glob("fold*") if p.is_dir())
    if not fold_dirs:
        print(f"[pairs] FAIL no fold dirs under {parent_dir}")
        return 2

    for fold_dir in fold_dirs:
        metrics, rows = _fold_rows(fold_dir, scope=scope)
        print(
            f"[pairs] {fold_dir.name} {scope}: "
            f"n={metrics.get('n_pairs')} "
            f"L={_fmt(metrics.get('L_align'))} "
            f"rho_s={_fmt(metrics.get('rho_align_spearman'))} "
            f"delta={_fmt(metrics.get('delta_tag'))}"
        )
        for row in rows[:top_k]:
            print(
                "  "
                f"{_pair_label(row['pair']):36s} "
                f"S={row['S']:+.3f} "
                f"R={row['R']:+.3f} "
                f"resid={row['residual']:.3f} "
                f"common={row['common']} "
                f"{row['direction']}"
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_run_dir")
    parser.add_argument("--scope", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    return diagnose(Path(args.parent_run_dir), scope=args.scope, top_k=args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
