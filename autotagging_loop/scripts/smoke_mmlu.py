"""
scripts/smoke_mmlu.py
=====================
Tagger N=50 per-item canonical vocab 변경을 실 LLM으로 1회 검증한다.
MMLU 1개 벤치마크에 대해 tag_benchmark(method='single')를 돌려
per_item_tags가 50개 들어오는지, canonical vocab을 준수하는지 확인.

Usage:
    uv run python scripts/smoke_mmlu.py
"""

from __future__ import annotations

import json
from collections import Counter

from dotenv import load_dotenv

load_dotenv()

from benchpress.config import get_model_for_run
from benchpress.researcher import research_benchmark
from benchpress.tagger import CANONICAL_DEPTH, CANONICAL_FORMAT, tag_benchmark


def main() -> None:
    print("== research_benchmark('MMLU') ==")
    rc = research_benchmark("MMLU")
    print(
        f"  evidence_level={rc.evidence_level}, "
        f"gt_samples={len(rc.gt_samples)}, "
        f"qa_samples={len(rc.qa_samples)}, "
        f"label_stats_keys={list(rc.gt_label_stats.keys()) if rc.gt_label_stats else []}"
    )

    model, base_url = get_model_for_run(0)
    print(f"\n== tag_benchmark(method='single', model={model}) ==")
    tr = tag_benchmark(
        research_context=rc,
        method="single",
        feedback="",
        model=model,
        base_url=base_url,
    )

    print("\n== TagResult summary ==")
    print(f"  tags ({len(tr.tags)}): {tr.tags}")
    print(f"  axes keys: {list(tr.axes.keys())}")
    print(f"  per_item_tags count: {len(tr.per_item_tags)}")

    if tr.per_item_tags:
        depth_ctr: Counter = Counter()
        format_ctr: Counter = Counter()
        for tags in tr.per_item_tags.values():
            if not isinstance(tags, dict):
                continue
            depth_ctr[tags.get("reasoning_depth", "")] += 1
            format_ctr[tags.get("answer_format", "")] += 1

        n = len(tr.per_item_tags)
        depth_in_vocab = sum(
            v for k, v in depth_ctr.items() if k in CANONICAL_DEPTH
        )
        format_in_vocab = sum(
            v for k, v in format_ctr.items() if k in CANONICAL_FORMAT
        )
        print(
            f"\n  depth vocab compliance: {depth_in_vocab}/{n} "
            f"({100*depth_in_vocab/n:.1f}%)"
        )
        print(f"    distribution: {dict(depth_ctr.most_common())}")
        print(
            f"  format vocab compliance: {format_in_vocab}/{n} "
            f"({100*format_in_vocab/n:.1f}%)"
        )
        print(f"    distribution: {dict(format_ctr.most_common())}")

        print("\n  sample (first 3):")
        for i, (item_id, tags) in enumerate(tr.per_item_tags.items()):
            if i >= 3:
                break
            print(f"    {item_id}: {tags}")

    out_path = "results/evaluation/smoke_mmlu.json"
    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "benchmark": tr.benchmark_name,
                "tags": tr.tags,
                "axes": tr.axes,
                "per_item_tags": tr.per_item_tags,
                "method": tr.method,
                "iteration_count": tr.iteration_count,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  wrote → {out_path}")


if __name__ == "__main__":
    main()
