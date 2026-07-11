"""scripts/fetch_hf_benchmarks.py

Pull HF benchmarks and emit data/labels/<slug>/tasks.jsonl in the format the
experiment loader expects (see experiment/corpus.py). Labels are pre-filled
from native dataset fields, so scripts/label_tasks.py does not need to run.

Slug convention: matches `_name_key` in experiment/corpus.py (lowercase,
alphanumeric only) so the loader can join on the leaderboard key:
  MATH-500   -> math-500     (key: math500)
  GSM8K      -> gsm8k        (key: gsm8k)
  MMLU-Pro   -> mmlu-pro     (key: mmlupro)
  AIME       -> aime         (key: aime)
  MBPP       -> mbpp         (key: mbpp)
  HumanEval  -> humaneval    (key: humaneval)
  WinoGrande -> winogrande   (key: winogrande)
  TruthfulQA -> truthfulqa   (key: truthfulqa)

Usage:
    uv run python scripts/fetch_hf_benchmarks.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).parent.parent.parent
LABELS_DIR = ROOT / "data" / "labels"

GSM8K_FINAL = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def _strip_gsm8k_answer(raw: str) -> str:
    m = GSM8K_FINAL.search(raw or "")
    return m.group(1).strip() if m else (raw or "").strip()


def _strip_boxed(raw: str) -> str:
    m = BOXED.search(raw or "")
    return m.group(1).strip() if m else (raw or "").strip()


def _write_jsonl(rows: list[dict], slug: str) -> Path:
    out_dir = LABELS_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tasks.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def fetch_math500() -> Path:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows: list[dict] = []
    for i, ex in enumerate(ds):
        rows.append({
            "item_id": f"math-500_{i:05d}",
            "benchmark": "MATH-500",
            "question": ex["problem"],
            "answer": ex["answer"],
            "gt_topic": str(ex["subject"]).lower().replace(" ", "_"),
            "gt_reasoning_depth": "deep",
            "gt_answer_format": "free_form_numeric",
            "reviewer_status": "reviewed",
        })
    return _write_jsonl(rows, "math-500")


def fetch_gsm8k() -> Path:
    ds = load_dataset("tinyBenchmarks/tinyGSM8k", "main", split="test")
    rows: list[dict] = []
    for i, ex in enumerate(ds):
        rows.append({
            "item_id": f"gsm8k_{i:05d}",
            "benchmark": "GSM8K",
            "question": ex["question"],
            "answer": _strip_gsm8k_answer(ex["answer"]),
            "gt_topic": "math_word_problem",
            "gt_reasoning_depth": "medium",
            "gt_answer_format": "free_form_numeric",
            "reviewer_status": "reviewed",
        })
    return _write_jsonl(rows, "gsm8k")


def fetch_mmlu_pro() -> Path:
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rows: list[dict] = []
    for i, ex in enumerate(ds):
        rows.append({
            "item_id": f"mmlu-pro_{i:05d}",
            "benchmark": "MMLU-Pro",
            "question": ex["question"],
            "answer": ex["answer"],
            "choices": list(ex["options"]),
            "gt_topic": str(ex["category"]).lower().replace(" ", "_"),
            "gt_reasoning_depth": "deep",
            "gt_answer_format": "multiple_choice",
            "reviewer_status": "reviewed",
        })
    return _write_jsonl(rows, "mmlu-pro")


def fetch_aime() -> Path:
    ds = load_dataset("math-ai/aime24", split="test")
    rows: list[dict] = []
    for i, ex in enumerate(ds):
        rows.append({
            "item_id": f"aime_{i:05d}",
            "benchmark": "AIME",
            "question": ex["problem"],
            "answer": _strip_boxed(ex["solution"]),
            "gt_topic": "competition_math",
            "gt_reasoning_depth": "deep",
            "gt_answer_format": "free_form_numeric",
            "reviewer_status": "reviewed",
        })
    return _write_jsonl(rows, "aime")


def fetch_mbpp() -> Path:
    ds = load_dataset("google-research-datasets/mbpp", split="test")
    rows: list[dict] = []
    for i, ex in enumerate(ds):
        rows.append({
            "item_id": f"mbpp_{i:05d}",
            "benchmark": "MBPP",
            "question": ex["text"],
            "answer": ex["code"],
            "gt_topic": "programming",
            "gt_reasoning_depth": "medium",
            "gt_answer_format": "code_generation",
            "reviewer_status": "reviewed",
        })
    return _write_jsonl(rows, "mbpp")


def fetch_humaneval() -> Path:
    ds = load_dataset("openai/openai_humaneval", split="test")
    rows: list[dict] = []
    for i, ex in enumerate(ds):
        rows.append({
            "item_id": f"humaneval_{i:05d}",
            "benchmark": "HumanEval",
            "question": ex["prompt"],
            "answer": ex["canonical_solution"],
            "gt_topic": "programming",
            "gt_reasoning_depth": "medium",
            "gt_answer_format": "code_generation",
            "reviewer_status": "reviewed",
        })
    return _write_jsonl(rows, "humaneval")


def fetch_winogrande() -> Path:
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation",
                      trust_remote_code=True)
    rows: list[dict] = []
    for i, ex in enumerate(ds):
        sentence = ex["sentence"]
        opt1, opt2 = ex["option1"], ex["option2"]
        ans_label = str(ex.get("answer", "")).strip()
        ans_text = opt1 if ans_label == "1" else (opt2 if ans_label == "2" else "")
        rows.append({
            "item_id": f"winogrande_{i:05d}",
            "benchmark": "WinoGrande",
            "question": sentence,
            "choices": [opt1, opt2],
            "answer": ans_text,
            "gt_topic": "commonsense_coreference",
            "gt_reasoning_depth": "shallow",
            "gt_answer_format": "binary_choice",
            "reviewer_status": "reviewed",
        })
    return _write_jsonl(rows, "winogrande")


def fetch_truthfulqa() -> Path:
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    rows: list[dict] = []
    for i, ex in enumerate(ds):
        mc1 = ex.get("mc1_targets") or {}
        choices = list(mc1.get("choices") or [])
        labels = list(mc1.get("labels") or [])
        ans = ""
        for c, l in zip(choices, labels):
            if int(l) == 1:
                ans = c
                break
        rows.append({
            "item_id": f"truthfulqa_{i:05d}",
            "benchmark": "TruthfulQA",
            "question": ex["question"],
            "choices": choices,
            "answer": ans,
            "gt_topic": "factual_truthfulness",
            "gt_reasoning_depth": "medium",
            "gt_answer_format": "multiple_choice",
            "reviewer_status": "reviewed",
        })
    return _write_jsonl(rows, "truthfulqa")


def main() -> None:
    for fn in (
        fetch_math500, fetch_gsm8k, fetch_mmlu_pro, fetch_aime, fetch_mbpp,
        fetch_humaneval, fetch_winogrande, fetch_truthfulqa,
    ):
        path = fn()
        n = sum(1 for _ in open(path, encoding="utf-8"))
        print(f"  wrote {n:5d} rows -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
