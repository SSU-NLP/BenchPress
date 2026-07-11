"""experiment/corpus.py — scored 벤치마크 corpus 로딩.

`data/leaderboard_scores.json` 을 읽어 _alias dereference, _meta drop, composite reject,
exclude/min_models 필터링을 수행한다. composite/aggregate 컬럼은 외부 정의로 거른다 —
현재 v3 leaderboard 에는 composite 가 없지만 향후 추가에 대비.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# 명시적 composite/aggregate 벤치마크 차단 목록.
# `_meta` 는 키 prefix `_` 로 이미 제거되므로 별도 명시 불필요.
COMPOSITE_BENCHMARKS: set[str] = {
    "Intelligence Index",
    "intelligence_index",
    "Composite",
}


@dataclass
class Corpus:
    benchmark_names: list[str]
    model_names: list[str]
    Y: dict[str, dict[str, float]]
    descriptions: dict[str, str] = field(default_factory=dict)
    documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    drop_log: dict[str, str] = field(default_factory=dict)


def _resolve_aliases(raw: dict) -> dict[str, dict[str, float]]:
    """Dereference `_alias` and drop `_*` keys."""
    resolved: dict[str, dict[str, float]] = {}
    canonical_for: dict[str, str] = {}  # alias key -> canonical target

    # First pass: canonical entries (no _alias)
    for key, val in raw.items():
        if key.startswith("_"):
            continue
        if not isinstance(val, dict) or "_alias" in val:
            continue
        cleaned = {k: float(v) for k, v in val.items() if not k.startswith("_")}
        resolved[key] = cleaned

    # Second pass: alias entries map to canonical
    for key, val in raw.items():
        if key.startswith("_"):
            continue
        if not isinstance(val, dict):
            continue
        if "_alias" in val:
            target = val["_alias"]
            if target in resolved:
                canonical_for[key] = target

    return resolved


def _name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _iter_label_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("reviewer_status") == "reviewed":
                rows.append(row)
    return rows


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    ctr: Counter[str] = Counter()
    for row in rows:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            ctr[value.strip()] += 1
    return dict(ctr.most_common())


def _format_example(row: dict[str, Any], max_chars: int = 900) -> str:
    question = str(row.get("question") or "").strip()
    if len(question) > max_chars:
        question = question[: max_chars - 3].rstrip() + "..."
    parts = [f"Question: {question}"]
    choices = row.get("choices")
    if isinstance(choices, list) and choices:
        parts.append("Choices: " + " | ".join(str(c) for c in choices[:8]))
    answer = row.get("answer")
    if answer not in (None, ""):
        parts.append(f"Answer: {answer}")
    labels = []
    for key in ("gt_topic", "gt_reasoning_depth", "gt_answer_format"):
        value = row.get(key)
        if value:
            labels.append(f"{key}={value}")
    if labels:
        parts.append("Reviewed labels: " + ", ".join(labels))
    return "\n".join(parts)


def _example_rows(
    rows: list[dict[str, Any]],
    examples_per_benchmark: int | str | None,
) -> list[dict[str, Any]]:
    if examples_per_benchmark is None:
        return rows
    if isinstance(examples_per_benchmark, str):
        if examples_per_benchmark.lower() == "all":
            return rows
        try:
            limit = int(examples_per_benchmark)
        except ValueError:
            limit = 5
    else:
        limit = int(examples_per_benchmark)
    return rows[: max(0, limit)]


def _build_label_document(
    benchmark: str,
    slug: str,
    rows: list[dict[str, Any]],
    examples_per_benchmark: int | str | None,
    prompt_examples_per_benchmark: int | str | None = None,
    max_prompt_chars: int | None = None,
) -> dict[str, Any]:
    topic_counts = _counts(rows, "gt_topic")
    depth_counts = _counts(rows, "gt_reasoning_depth")
    format_counts = _counts(rows, "gt_answer_format")
    examples = [_format_example(row) for row in _example_rows(rows, examples_per_benchmark)]
    prompt_rows = _example_rows(
        rows,
        examples_per_benchmark if prompt_examples_per_benchmark is None else prompt_examples_per_benchmark,
    )
    prompt_examples = [_format_example(row) for row in prompt_rows]
    text_parts = [
        f"Benchmark: {benchmark}",
        "Metric description: individual public leaderboard quantitative score column.",
        "Task format summary from reviewed local benchmark rows:",
        f"- reviewed_rows: {len(rows)}",
        f"- stored_examples: {len(examples)}",
        f"- prompt_examples: {len(prompt_examples)}",
        f"- gt_topic distribution: {json.dumps(topic_counts, ensure_ascii=False)}",
        f"- gt_reasoning_depth distribution: {json.dumps(depth_counts, ensure_ascii=False)}",
        f"- gt_answer_format distribution: {json.dumps(format_counts, ensure_ascii=False)}",
    ]
    if len(prompt_examples) < len(examples):
        text_parts.append(
            "Prompt note: representative examples below are capped to fit model context; "
            "all reviewed rows are still represented in the aggregate distributions above."
        )
    if prompt_examples:
        text_parts.append("Representative examples:")
        for idx, example in enumerate(prompt_examples, start=1):
            text_parts.append(f"[example {idx}]\n{example}")
    text = "\n\n".join(text_parts)
    if max_prompt_chars is not None and max_prompt_chars > 0 and len(text) > max_prompt_chars:
        text = text[: max_prompt_chars - 120].rstrip()
        text += (
            "\n\n[truncated_for_prompt_budget]\n"
            "The prompt evidence was truncated after aggregate distributions and representative examples."
        )

    return {
        "benchmark": benchmark,
        "slug": slug,
        "source": f"data/labels/{slug}/tasks.jsonl",
        "reviewed_rows": len(rows),
        "metric_description": "individual public leaderboard quantitative score column",
        "topic_counts": topic_counts,
        "reasoning_depth_counts": depth_counts,
        "answer_format_counts": format_counts,
        "examples": examples,
        "prompt_examples": prompt_examples,
        "prompt_example_count": len(prompt_examples),
        "text": text,
    }


def load_label_documents(
    labels_dir: str,
    benchmark_names: list[str],
    examples_per_benchmark: int | str | None = 5,
    prompt_examples_per_benchmark: int | str | None = None,
    max_prompt_chars_per_benchmark: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Load reviewed local label rows as S_l document evidence when available."""
    if not labels_dir or not os.path.isdir(labels_dir):
        return {}

    by_key = {_name_key(name): name for name in benchmark_names}
    documents: dict[str, dict[str, Any]] = {}
    for slug in sorted(os.listdir(labels_dir)):
        path = os.path.join(labels_dir, slug, "tasks.jsonl")
        if not os.path.isfile(path):
            continue
        rows = _iter_label_rows(path)
        if not rows:
            continue
        row_benchmark = str(rows[0].get("benchmark") or slug)
        match_name = by_key.get(_name_key(row_benchmark)) or by_key.get(_name_key(slug))
        if not match_name:
            continue
        documents[match_name] = _build_label_document(
            match_name,
            slug,
            rows,
            examples_per_benchmark=examples_per_benchmark,
            prompt_examples_per_benchmark=prompt_examples_per_benchmark,
            max_prompt_chars=max_prompt_chars_per_benchmark,
        )
    return documents


def load_corpus(
    leaderboard_path: str,
    min_models_per_bench: int = 6,
    exclude: list[str] | None = None,
    labels_dir: str | None = None,
    examples_per_benchmark: int | str | None = 5,
    prompt_examples_per_benchmark: int | str | None = None,
    max_prompt_chars_per_benchmark: int | None = None,
) -> Corpus:
    with open(leaderboard_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    exclude_set = {e.lower() for e in (exclude or [])}
    drop_log: dict[str, str] = {}

    Y = _resolve_aliases(raw)

    # Composite reject
    for name in list(Y.keys()):
        if name in COMPOSITE_BENCHMARKS or name.lower() in {c.lower() for c in COMPOSITE_BENCHMARKS}:
            drop_log[name] = "composite"
            Y.pop(name)

    # Exclude list
    for name in list(Y.keys()):
        if name.lower() in exclude_set:
            drop_log[name] = "excluded"
            Y.pop(name)

    # min_models_per_bench
    for name in list(Y.keys()):
        if len(Y[name]) < min_models_per_bench:
            drop_log[name] = f"models<{min_models_per_bench} (have {len(Y[name])})"
            Y.pop(name)

    benchmark_names = sorted(Y.keys())
    model_names_set: set[str] = set()
    for scores in Y.values():
        model_names_set.update(scores.keys())
    model_names = sorted(model_names_set)
    documents = load_label_documents(
        labels_dir,
        benchmark_names,
        examples_per_benchmark=examples_per_benchmark,
        prompt_examples_per_benchmark=prompt_examples_per_benchmark,
        max_prompt_chars_per_benchmark=max_prompt_chars_per_benchmark,
    ) if labels_dir else {}
    descriptions = {
        name: documents.get(name, {}).get("text", "")
        for name in benchmark_names
    }

    return Corpus(
        benchmark_names=benchmark_names,
        model_names=model_names,
        Y=Y,
        descriptions=descriptions,
        documents=documents,
        drop_log=drop_log,
    )


def attach_descriptions(corpus: Corpus, descriptions: dict[str, str]) -> Corpus:
    corpus.descriptions = {b: descriptions.get(b, "") for b in corpus.benchmark_names}
    return corpus
