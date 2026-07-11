"""Corpus loading for the Part 2 main experiment."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from autotagging_loop.runner.aai_scores import aai_api_key, read_aai_scores, refresh_aai_scores

COMPOSITE_BENCHMARKS = {"Intelligence Index", "intelligence_index", "Composite"}
DEFAULT_MODEL_ALIASES = {
    "Claude Sonnet 4.6": "Claude-Sonnet-4.6",
    "DeepSeek v3": "DeepSeek-v3",
    "DeepSeek V3": "DeepSeek-v3",
    "DeepSeek-V3": "DeepSeek-v3",
    "GPT 5": "GPT-5",
    "Qwen 2.5": "Qwen2.5-72B",
    "Qwen-2.5-72B": "Qwen2.5-72B",
    "Qwen2.5": "Qwen2.5-72B",
}


@dataclass
class Corpus:
    benchmark_names: list[str]
    model_names: list[str]
    Y: dict[str, dict[str, float]]
    documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    descriptions: dict[str, str] = field(default_factory=dict)
    drop_log: dict[str, str] = field(default_factory=dict)


def name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def normalize_model_name(
    name: str,
    aliases: dict[str, str] | None = None,
) -> str:
    raw = str(name or "").strip()
    merged_aliases = dict(DEFAULT_MODEL_ALIASES)
    merged_aliases.update(aliases or {})
    return merged_aliases.get(raw, raw)


def load_leaderboard_scores(
    path: str,
    *,
    exclude: list[str] | None = None,
    model_aliases: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    exclude_set = {name.lower() for name in (exclude or [])}
    resolved: dict[str, dict[str, float]] = {}
    for key, value in raw.items():
        if key.startswith("_") or key in COMPOSITE_BENCHMARKS or key.lower() in exclude_set:
            continue
        if not isinstance(value, dict) or "_alias" in value:
            continue
        scores: dict[str, float] = {}
        for model, score in value.items():
            if str(model).startswith("_"):
                continue
            canonical_model = normalize_model_name(model, model_aliases)
            scores.setdefault(canonical_model, float(score))
        if scores:
            resolved[key] = scores
    return resolved


def merge_score_sources(
    primary: dict[str, dict[str, float]],
    secondary: dict[str, dict[str, float]],
    *,
    model_aliases: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    merged = {
        benchmark: {
            normalize_model_name(model, model_aliases): float(score)
            for model, score in scores.items()
        }
        for benchmark, scores in primary.items()
    }
    by_key = {name_key(benchmark): benchmark for benchmark in merged}
    for benchmark, scores in secondary.items():
        target = by_key.get(name_key(benchmark), benchmark)
        merged.setdefault(target, {})
        for model, score in scores.items():
            canonical_model = normalize_model_name(model, model_aliases)
            merged[target].setdefault(canonical_model, float(score))
        by_key.setdefault(name_key(benchmark), target)
    return merged


def _source_field(source: dict[str, Any], key: str) -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"curated score source.{key} must be a non-empty string")
    return value.strip()


def _looks_placeholder(text: str) -> bool:
    lower = str(text or "").strip().lower()
    return any(token in lower for token in ("todo", "replace", "placeholder"))


def _validate_source_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("curated score source.url must be an absolute http(s) URL")
    host = (parsed.hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "0.0.0.0"} or host.endswith(".local"):
        raise ValueError("curated score source.url must not point to a local host")
    if "example.com" in host or host.endswith(".invalid"):
        raise ValueError("curated score source.url appears to be a placeholder")


def _validate_source_date(value: Any, *, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"curated score source.{field} must be a non-empty string")
    raw = value.strip()
    if _looks_placeholder(raw):
        raise ValueError(f"curated score source.{field} appears to be a placeholder")
    try:
        if field == "date":
            date.fromisoformat(raw)
        else:
            datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"curated score source.{field} must be ISO formatted"
        ) from exc


def load_curated_score_backfill(
    path: str,
    *,
    exclude: list[str] | None = None,
    model_aliases: dict[str, str] | None = None,
    require_exists: bool = False,
) -> dict[str, dict[str, float]]:
    """Load manually curated score cells with per-cell provenance checks."""

    backfill_path = Path(path)
    if not backfill_path.exists():
        if require_exists:
            raise FileNotFoundError(str(backfill_path))
        return {}
    with open(backfill_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("curated score backfill must be a JSON object")
    records = raw.get("scores")
    if not isinstance(records, list):
        raise ValueError("curated score backfill must contain a 'scores' list")

    exclude_set = {name.lower() for name in (exclude or [])}
    by_key: dict[tuple[str, str], tuple[str, str, float]] = {}
    resolved: dict[str, dict[str, float]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"curated score record {index} must be an object")
        benchmark = str(record.get("benchmark") or "").strip()
        if not benchmark:
            raise ValueError(f"curated score record {index} missing benchmark")
        if benchmark.startswith("_") or benchmark in COMPOSITE_BENCHMARKS or benchmark.lower() in exclude_set:
            continue
        model = str(record.get("model") or "").strip()
        if not model:
            raise ValueError(f"curated score record {index} missing model")

        scale = str(record.get("scale") or "").strip()
        if scale != "0-1":
            raise ValueError(
                f"curated score record {index} must use scale='0-1'; got {scale!r}"
            )
        metric = str(record.get("metric") or "").strip()
        if not metric:
            raise ValueError(f"curated score record {index} missing metric")
        if _looks_placeholder(metric):
            raise ValueError(f"curated score record {index} metric appears to be a placeholder")
        try:
            score = float(record.get("score"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"curated score record {index} has non-numeric score") from exc
        if not math.isfinite(score) or score < 0.0 or score > 1.0:
            raise ValueError(
                f"curated score record {index} score must be finite and within [0, 1]"
            )

        source = record.get("source")
        if not isinstance(source, dict):
            raise ValueError(f"curated score record {index} missing source object")
        source_title = _source_field(source, "title")
        source_url = _source_field(source, "url")
        if _looks_placeholder(source_title):
            raise ValueError(
                f"curated score record {index} source appears to be a placeholder"
            )
        _validate_source_url(source_url)
        if source.get("date"):
            _validate_source_date(source.get("date"), field="date")
        elif source.get("retrieved_at"):
            _validate_source_date(source.get("retrieved_at"), field="retrieved_at")
        else:
            raise ValueError(
                f"curated score record {index} source must include date or retrieved_at"
            )

        canonical_model = normalize_model_name(model, model_aliases)
        key = (name_key(benchmark), canonical_model)
        existing = by_key.get(key)
        if existing is not None:
            prev_benchmark, prev_model, prev_score = existing
            raise ValueError(
                "duplicate curated score cell after normalization: "
                f"{prev_benchmark}/{prev_model}={prev_score} and "
                f"{benchmark}/{canonical_model}={score}"
            )
        by_key[key] = (benchmark, canonical_model, score)
        resolved.setdefault(benchmark, {})[canonical_model] = score
    return resolved


def _name_filter_set(values: list[str] | None) -> set[str]:
    return {name_key(value) for value in (values or []) if str(value).strip()}


def _model_filter_set(
    values: list[str] | None,
    *,
    aliases: dict[str, str] | None = None,
) -> set[str]:
    return {
        normalize_model_name(value, aliases)
        for value in (values or [])
        if str(value).strip()
    }


def filter_score_matrix(
    Y: dict[str, dict[str, float]],
    *,
    include_benchmarks: list[str] | None = None,
    include_models: list[str] | None = None,
    exclude_models: list[str] | None = None,
    model_aliases: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Apply explicit benchmark/model filters for diagnostic subsets."""

    include_bench_set = _name_filter_set(include_benchmarks)
    include_model_set = _model_filter_set(include_models, aliases=model_aliases)
    exclude_model_set = _model_filter_set(exclude_models, aliases=model_aliases)
    filtered: dict[str, dict[str, float]] = {}
    for benchmark, scores in Y.items():
        if include_bench_set and name_key(benchmark) not in include_bench_set:
            continue
        row: dict[str, float] = {}
        for model, score in scores.items():
            canonical_model = normalize_model_name(model, model_aliases)
            if include_model_set and canonical_model not in include_model_set:
                continue
            if canonical_model in exclude_model_set:
                continue
            row[canonical_model] = float(score)
        if row:
            filtered[benchmark] = row
    return filtered


def load_score_sources(config: dict) -> dict[str, dict[str, float]]:
    model_aliases = config.get("model_aliases") or {}
    Y = load_leaderboard_scores(
        config["leaderboard_path"],
        exclude=config.get("exclude", []),
        model_aliases=model_aliases,
    )
    aai_path = config.get("aai_scores_path")
    aai_scores: dict[str, dict[str, float]] = {}
    if config.get("use_aai_scores", True):
        if config.get("refresh_aai_scores") and aai_path and aai_api_key():
            try:
                aai_scores = refresh_aai_scores(aai_path, api_url=config.get("aai_api_url"))
            except Exception as exc:
                print(f"  [part2][aai] refresh failed: {exc}")
        if not aai_scores and aai_path:
            aai_scores = read_aai_scores(aai_path)
    Y = merge_score_sources(Y, aai_scores, model_aliases=model_aliases)

    if config.get("use_curated_score_backfill", True):
        backfill_path = config.get("curated_score_backfill_path")
        if backfill_path:
            curated_scores = load_curated_score_backfill(
                backfill_path,
                exclude=config.get("exclude", []),
                model_aliases=model_aliases,
            )
            Y = merge_score_sources(Y, curated_scores, model_aliases=model_aliases)
    Y = filter_score_matrix(
        Y,
        include_benchmarks=config.get("include_benchmarks"),
        include_models=config.get("include_models"),
        exclude_models=config.get("exclude_models"),
        model_aliases=model_aliases,
    )
    return Y


def iter_label_rows(path: Path) -> list[dict[str, Any]]:
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


def counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            counter[value.strip()] += 1
    return dict(counter.most_common())


def format_example(row: dict[str, Any], max_chars: int = 900) -> str:
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
    return "\n".join(parts)


def build_document(
    benchmark: str,
    slug: str,
    rows: list[dict[str, Any]],
    *,
    prompt_examples_per_benchmark: int = 20,
    max_prompt_chars: int | None = None,
) -> dict[str, Any]:
    examples = [format_example(row) for row in rows]
    prompt_examples = examples[: max(0, int(prompt_examples_per_benchmark))]
    topic_counts = counts(rows, "gt_topic")
    depth_counts = counts(rows, "gt_reasoning_depth")
    format_counts = counts(rows, "gt_answer_format")
    parts = [
        f"Benchmark: {benchmark}",
        "Metric description: individual public leaderboard quantitative score column.",
        f"- reviewed_rows: {len(rows)}",
        f"- stored_examples: {len(examples)}",
        f"- prompt_examples: {len(prompt_examples)}",
        f"- gt_topic distribution: {json.dumps(topic_counts, ensure_ascii=False)}",
        f"- gt_reasoning_depth distribution: {json.dumps(depth_counts, ensure_ascii=False)}",
        f"- gt_answer_format distribution: {json.dumps(format_counts, ensure_ascii=False)}",
    ]
    if prompt_examples:
        parts.append("Representative examples:")
        parts.extend(f"[example {i}]\n{ex}" for i, ex in enumerate(prompt_examples, start=1))
    text = "\n\n".join(parts)
    if max_prompt_chars and len(text) > max_prompt_chars:
        text = text[: max_prompt_chars - 80].rstrip() + "\n\n[truncated_for_prompt_budget]"
    return {
        "benchmark": benchmark,
        "slug": slug,
        "source": f"data/labels_part2/{slug}/tasks.jsonl",
        "reviewed_rows": len(rows),
        "topic_counts": topic_counts,
        "reasoning_depth_counts": depth_counts,
        "answer_format_counts": format_counts,
        "examples": examples,
        "prompt_examples": prompt_examples,
        "text": text,
    }


def load_documents(
    labels_dir: str,
    benchmark_names: list[str],
    *,
    prompt_examples_per_benchmark: int = 20,
    max_prompt_chars_per_benchmark: int | None = None,
) -> dict[str, dict[str, Any]]:
    root = Path(labels_dir)
    if not root.is_dir():
        return {}
    by_key = {name_key(name): name for name in benchmark_names}
    docs: dict[str, dict[str, Any]] = {}
    for task_path in sorted(root.glob("*/tasks.jsonl")):
        rows = iter_label_rows(task_path)
        if not rows:
            continue
        row_benchmark = str(rows[0].get("benchmark") or task_path.parent.name)
        benchmark = by_key.get(name_key(row_benchmark)) or by_key.get(name_key(task_path.parent.name))
        if not benchmark:
            continue
        docs[benchmark] = build_document(
            benchmark,
            task_path.parent.name,
            rows,
            prompt_examples_per_benchmark=prompt_examples_per_benchmark,
            max_prompt_chars=max_prompt_chars_per_benchmark,
        )
    return docs


def load_corpus(config: dict) -> Corpus:
    Y = load_score_sources(config)
    drop_log: dict[str, str] = {}
    min_models = int(config.get("min_common_models", 6))
    for benchmark in list(Y):
        if len(Y[benchmark]) < min_models:
            drop_log[benchmark] = f"models<{min_models}"
            Y.pop(benchmark)
    benchmark_names = sorted(Y)
    documents = load_documents(
        config["labels_dir"],
        benchmark_names,
        prompt_examples_per_benchmark=int(config.get("prompt_examples_per_benchmark", 20)),
        max_prompt_chars_per_benchmark=config.get("max_prompt_chars_per_benchmark"),
    )
    if documents:
        benchmark_names = [name for name in benchmark_names if name in documents]
        Y = {name: Y[name] for name in benchmark_names}
    model_names = sorted({model for scores in Y.values() for model in scores})
    descriptions = {name: documents.get(name, {}).get("text", "") for name in benchmark_names}
    return Corpus(
        benchmark_names=benchmark_names,
        model_names=model_names,
        Y=Y,
        documents=documents,
        descriptions=descriptions,
        drop_log=drop_log,
    )
