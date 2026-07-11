"""Build the pinned experiment target matrix from the leaderboard + correction overlay.

data/leaderboard_scores.json stays untouched (data/ is read-only); corrections live in
experiment/target_corrections_2026_07.json with per-cell provenance. Output matches the
score_matrix.json shape vloop_pilot.load_benchmarks expects ({"Y_norm": {bench: {model:
rank-normalized}}}) plus a provenance block with input hashes so runs can pin their target.

Usage:
  python -m experiment.build_target_matrix [--out results/target_v2/score_matrix.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from autotagging_loop.experiment import vloop

REPO = Path(__file__).resolve().parents[2]
LEADERBOARD = REPO / "data/leaderboard_scores.json"
CORRECTIONS = REPO / "experiment/target_corrections_2026_07.json"
DEFAULT_OUT = REPO / "results/target_v2/score_matrix.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def rank_normalize(scores: dict[str, float]) -> dict[str, float]:
    models = sorted(scores)
    ranks = vloop.ranks([scores[m] for m in models])  # average ranks, 0..n-1
    n = len(models)
    return {m: ranks[i] / (n - 1) for i, m in enumerate(models)}


def build() -> dict:
    lb = json.loads(LEADERBOARD.read_text())
    corr = json.loads(CORRECTIONS.read_text())
    applied, missing = [], []
    raw = {b: {m: v for m, v in lb[b].items() if isinstance(v, (int, float))}
           for b in lb if not b.startswith("_")}
    for c in corr["corrections"]:
        b, m = c["benchmark"], c["model"]
        if b in raw and m in raw[b]:
            if abs(raw[b][m] - c["old"]) > 1e-9:
                missing.append(f"{m}@{b}: leaderboard={raw[b][m]} != expected old={c['old']}")
            raw[b][m] = c["new"]
            applied.append(f"{m}@{b}")
        else:
            missing.append(f"{m}@{b}: cell not found")
    if missing:
        raise SystemExit(f"correction mismatch — leaderboard drifted?\n" + "\n".join(missing))
    return {
        "Y_norm": {b: rank_normalize(row) for b, row in raw.items()},
        "provenance": {
            "leaderboard_sha256": _sha(LEADERBOARD),
            "corrections_sha256": _sha(CORRECTIONS),
            "corrections_applied": applied,
            "normalize": "average-rank / (n-1), per benchmark over available models",
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="build pinned target score matrix")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    matrix = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(matrix, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")
    print(f"  benches={len(matrix['Y_norm'])} corrections={len(matrix['provenance']['corrections_applied'])}")
    print(f"  leaderboard_sha={matrix['provenance']['leaderboard_sha256']} "
          f"corrections_sha={matrix['provenance']['corrections_sha256']}")


if __name__ == "__main__":
    main()
