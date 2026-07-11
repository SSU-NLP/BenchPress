"""
scripts/label_tasks.py
======================
data/labels/ 아래 모든 tasks.jsonl의 gt_topic, gt_reasoning_depth,
gt_answer_format을 채우고 reviewer_status를 reviewed로 변경.

Usage:
    python scripts/label_tasks.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from benchpress.config import make_openai_kwargs

load_dotenv()
client = OpenAI(**make_openai_kwargs(None))

ROOT = Path(__file__).parent.parent.parent
LABELS_DIR = ROOT / "data" / "labels"

BATCH_SIZE = 20
MAX_WORKERS = 8  # parallel API calls

# ── Fixed rules per benchmark ─────────────────────────────────────────────────

FIXED: dict[str, dict] = {
    "GSM8K": {
        "gt_answer_format": "free_form_numeric",
        "gt_reasoning_depth": "medium",
        "gt_topic": "math_word_problem",
    },
    "HellaSwag": {
        "gt_answer_format": "multiple_choice",
        "gt_reasoning_depth": "shallow",
    },
    "GPQA": {
        "gt_answer_format": "multiple_choice",
        "gt_reasoning_depth": "deep",
    },
    "ARC Challenge": {
        "gt_answer_format": "multiple_choice",
        "gt_reasoning_depth": "medium",
    },
    "MMLU": {
        "gt_answer_format": "multiple_choice",
    },
}

TOPIC_OPTIONS: dict[str, list[str]] = {
    "ARC Challenge": [
        "earth_science", "biology", "chemistry", "physics",
        "engineering", "social_science",
    ],
    "GPQA": ["biology", "chemistry", "physics"],
    "HellaSwag": [
        "everyday_activity", "sports_fitness", "cooking_food",
        "arts_entertainment", "vehicle_transport",
        "nature_environment", "social_interaction",
    ],
    "MMLU": [
        "mathematics", "physics", "chemistry", "biology",
        "computer_science", "medicine", "law", "economics",
        "psychology", "history", "philosophy", "geography",
        "linguistics", "sociology", "other",
    ],
}

DEPTH_OPTIONS = ["shallow", "medium", "deep"]


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(benchmark: str, items: list[dict]) -> str:
    topic_opts = TOPIC_OPTIONS.get(benchmark, [])
    need_depth = benchmark == "MMLU"
    valid_topics = "|".join(topic_opts)

    if need_depth:
        valid_depths = "|".join(DEPTH_OPTIONS)
        elem_fmt = f'{{"topic": "<{valid_topics}>", "reasoning_depth": "<{valid_depths}>"}}'
        depth_hint = " (shallow=recall, medium=multi-step, deep=expert-level)"
    else:
        elem_fmt = f'{{"topic": "<{valid_topics}>"}}'
        depth_hint = ""

    lines = [
        f"Classify each '{benchmark}' question. Return a JSON array with exactly {len(items)} elements.",
        f"Each element must be exactly: {elem_fmt}{depth_hint}",
        "Use ONLY the allowed values shown. Output ONLY the raw JSON array, no markdown, no explanation.",
        "",
        "Questions:",
    ]
    for i, item in enumerate(items):
        q = item["question"][:200].replace("\n", " ")
        lines.append(f"{i+1}. {q}")

    return "\n".join(lines)


# ── LLM call ──────────────────────────────────────────────────────────────────

def _classify_single(benchmark: str, item: dict) -> dict:
    """Classify a single item as fallback for length-mismatch batches."""
    prompt = _build_prompt(benchmark, [item])
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=120,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            results = json.loads(raw)
            if isinstance(results, list) and len(results) == 1:
                return results[0]
            if isinstance(results, dict):
                return results
        except Exception:
            pass
    return {"topic": "other"}


def _classify_batch(benchmark: str, items: list[dict]) -> list[dict]:
    """Returns list of {topic, reasoning_depth?} for each item."""
    prompt = _build_prompt(benchmark, items)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=len(items) * 80 + 50,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            results = json.loads(raw)
            if len(results) == len(items):
                return results
            # length mismatch: fall through to retry
            print(f"  [warn] length mismatch {len(results)} vs {len(items)}, retrying", file=sys.stderr)
        except Exception as e:
            if attempt == 2:
                print(f"  [warn] batch failed: {e}", file=sys.stderr)
    # fallback: classify item by item
    return [_classify_single(benchmark, item) for item in items]


# ── Per-benchmark processing ──────────────────────────────────────────────────

def _apply_fixed(item: dict, fixed: dict) -> None:
    for k, v in fixed.items():
        item[k] = v


def _process_file(bench_dir: Path) -> None:
    path = bench_dir / "tasks.jsonl"
    items: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if not items:
        return

    benchmark = items[0]["benchmark"]
    fixed = FIXED.get(benchmark, {})
    needs_llm = benchmark in TOPIC_OPTIONS and "gt_topic" not in fixed
    needs_depth_llm = benchmark == "MMLU"

    print(f"  [{benchmark}] {len(items)} items, LLM={'yes' if needs_llm else 'no'}")

    # Apply fixed rules
    for item in items:
        _apply_fixed(item, fixed)

    # Batch LLM classification: only items still needing labels
    if needs_llm:
        valid_topics = set(TOPIC_OPTIONS.get(benchmark, []))

        def _needs_label(item: dict) -> bool:
            topic = item.get("gt_topic", "")
            # empty or not yet classified (fallback "other" is invalid unless it's in the allowed list)
            if not topic or (topic == "other" and "other" not in valid_topics):
                return True
            if needs_depth_llm and not item.get("gt_reasoning_depth"):
                return True
            return False

        pending = [item for item in items if _needs_label(item)]
        print(f"    {len(pending)}/{len(items)} items need LLM labeling")

        batches = [pending[i : i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]

        def process_batch(batch_idx: int) -> tuple[int, list[dict]]:
            batch = batches[batch_idx]
            return batch_idx, _classify_batch(benchmark, batch)

        done = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(process_batch, i): i for i in range(len(batches))}
            for fut in as_completed(futures):
                batch_idx, results = fut.result()
                batch = batches[batch_idx]
                for item, res in zip(batch, results):
                    item["gt_topic"] = res.get("topic", "other")
                    if needs_depth_llm and "reasoning_depth" in res:
                        item["gt_reasoning_depth"] = res["reasoning_depth"]
                done += 1
                print(f"    {done}/{len(batches)} batches done", end="\r")
        print()

    # Mark reviewed
    for item in items:
        item["reviewer_status"] = "reviewed"

    # Write back
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  [{benchmark}] done → {path.relative_to(ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    bench_dirs = sorted(LABELS_DIR.iterdir())
    print(f"Processing {len(bench_dirs)} benchmarks in {LABELS_DIR}")
    t0 = time.time()
    for bench_dir in bench_dirs:
        if bench_dir.is_dir() and (bench_dir / "tasks.jsonl").exists():
            _process_file(bench_dir)
    elapsed = time.time() - t0
    print(f"\nAll done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
