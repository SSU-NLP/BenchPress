"""
scripts/eval_smoke.py
=====================
smoke_mmlu.json 결과에 evaluator를 돌려 실제 metric 수치 확인.

Usage:
    uv run python scripts/eval_smoke.py
"""

from __future__ import annotations

import json

from dotenv import load_dotenv

load_dotenv()

from benchpress.evaluator import evaluate
from benchpress.researcher import research_benchmark


class _TagStub:
    def __init__(self, d: dict):
        self.benchmark_name = d["benchmark"]
        self.per_item_tags = d["per_item_tags"]


def main() -> None:
    with open("results/evaluation/smoke_mmlu.json", encoding="utf-8") as f:
        smoke = json.load(f)

    rc = research_benchmark("MMLU")
    res = evaluate(_TagStub(smoke), rc)

    print(f"\n== Eval: {res.benchmark} ==")
    print(f"  n_items={res.n_items}")
    print(f"  answer_format_acc  = {res.answer_format_acc:.3f}")
    print(f"  depth_per_item_acc = {res.depth_per_item_acc:.3f}")
    print(f"  depth_jsd_vs_gt    = {res.depth_jsd_vs_gt:.3f}")
    print(f"\n  gt_dist (full bench)  : {res.gt_distribution}")
    print(f"  pred_dist (50 items)  : {res.pred_distribution}")
    print(f"\n  mismatches: {len(res.mismatches)} (first 5)")
    for m in res.mismatches[:5]:
        print(f"    {m}")


if __name__ == "__main__":
    main()
