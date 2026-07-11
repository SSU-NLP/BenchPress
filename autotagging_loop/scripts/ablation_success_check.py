"""Per-run success criterion check for the 3-run ablation.

Success: for each fold,
  (a) best_iter.txt != "iter_000_baseline_static"
  (b) L_align(best) < L_align(iter_000_baseline_static)

Usage:
  uv run python scripts/ablation_success_check.py <run_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _l(p: Path) -> float | None:
    try:
        return float(json.loads(p.read_text())["L_align"])
    except Exception:
        return None


def check(run_dir: str) -> int:
    rd = Path(run_dir)
    folds = sorted(d for d in rd.iterdir() if d.is_dir() and d.name.startswith("fold"))
    rows = []
    pass_non_baseline = 0
    pass_l_align = 0
    for f in folds:
        best_p = f / "final" / "best_iter.txt"
        best = best_p.read_text().strip() if best_p.exists() else "?"
        base_L = _l(f / "iter_000_baseline_static" / "metrics.json")
        best_L = _l(f / best / "metrics.json") if best and (f / best).exists() else None
        non_base = best != "iter_000_baseline_static"
        beats = (best_L is not None and base_L is not None and best_L < base_L)
        if non_base:
            pass_non_baseline += 1
        if beats:
            pass_l_align += 1
        rows.append((f.name, best, base_L, best_L, non_base, beats))

    perm_p = rd / "agg" / "permutation_test.json"
    pooled = ""
    if perm_p.exists():
        try:
            pj = json.loads(perm_p.read_text())
            pl = pj.get("pooled", {})
            rs = pl.get("rho_spearman", {})
            rp = pl.get("rho_pearson", {})
            la = pl.get("L_align", {})
            pooled = (
                f"pooled rho_s={rs.get('observed', float('nan')):+.3f} "
                f"p_two={rs.get('p_two_sided', float('nan')):.4f} | "
                f"rho_p={rp.get('observed', float('nan')):+.3f} "
                f"p_two={rp.get('p_two_sided', float('nan')):.4f} | "
                f"L_align p_low={la.get('p_one_sided_low', float('nan')):.4f}"
            )
        except Exception as e:
            pooled = f"(pooled parse error: {e})"

    print(f"run_dir: {run_dir}")
    print(f"{'fold':<7}{'best_iter':<35}{'base_L':>10}{'best_L':>10}  non_base  beats_base")
    for name, best, bL, beL, nb, bt in rows:
        bL_s = f"{bL:.4f}" if bL is not None else "  n/a"
        beL_s = f"{beL:.4f}" if beL is not None else "  n/a"
        print(f"{name:<7}{best:<35}{bL_s:>10}{beL_s:>10}  {str(nb):>8}  {str(bt):>10}")
    print(
        f"\nFolds non-baseline: {pass_non_baseline}/{len(folds)}   "
        f"Folds L_align(best)<base: {pass_l_align}/{len(folds)}   {pooled}"
    )
    success = pass_non_baseline == len(folds) and pass_l_align == len(folds)
    print(f"SUCCESS: {success}")
    return 0 if success else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: ablation_success_check.py <run_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(check(sys.argv[1]))
