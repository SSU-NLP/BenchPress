"""HuggingFace dataset sampling for Part 2 benchmark corpora."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from urllib.parse import unquote

import requests

HF_DATASETS_API = "https://datasets-server.huggingface.co"
PARQUET_SKIP_COLUMNS = {"private_test_cases"}

Q_FIELD_CANDIDATES = [
    "question", "query", "input", "problem", "prompt", "instruction",
    "context", "text", "sentence", "user_scenario",
]
A_FIELD_CANDIDATES = [
    "answer", "response", "output", "solution", "label", "target",
    "choices", "options", "correct_answer", "answerKey", "gold",
]
THIN_ROW_KEYS = {
    *(key.lower() for key in Q_FIELD_CANDIDATES),
    *(key.lower() for key in A_FIELD_CANDIDATES),
    "choices", "options", "choice", "mc1_targets",
    "question_title", "question_content", "public_test_cases",
    "problem_name", "problem_description_main", "problem_io",
    "correct answer", "incorrect answer 1", "incorrect answer 2", "incorrect answer 3",
    "answer_type", "category", "raw_subject",
}


@dataclass(frozen=True)
class DatasetSpec:
    benchmark: str
    dataset_id: str
    config: str | None = None
    split: str | None = None


def name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def load_dataset_map(path: str | Path) -> dict[str, DatasetSpec]:
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    # Strip full-line // comments only — matching mid-line would eat URLs ("https://...").
    cleaned = re.sub(r"^\s*//.*$", "", raw_text, flags=re.MULTILINE)
    raw = json.loads(cleaned)
    out: dict[str, DatasetSpec] = {}
    for benchmark, entry in raw.items():
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        out[name_key(benchmark)] = DatasetSpec(
            benchmark=benchmark,
            dataset_id=str(entry["id"]),
            config=entry.get("config"),
            split=entry.get("split"),
        )
    return out


def headers(token: str | None = None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def clean_value(value: object, max_chars: int = 500) -> str:
    if isinstance(value, list):
        return " / ".join(str(v) for v in value[:8])[:max_chars].strip()
    if isinstance(value, dict):
        if isinstance(value.get("text"), list):
            return " / ".join(str(v) for v in value["text"][:8])[:max_chars].strip()
        return json.dumps(value, ensure_ascii=False)[:max_chars].strip()
    return str(value or "")[:max_chars].strip()


def thin_dataset_row(row: dict) -> dict:
    """Keep prompt fields only; avoids storing huge hidden-test payloads."""
    lowered = {str(k).lower(): v for k, v in row.items()}
    title = clean_value(lowered.get("question_title"), max_chars=500)
    content = clean_value(lowered.get("question_content"), max_chars=4000)
    if title or content:
        public_tests = clean_value(lowered.get("public_test_cases"), max_chars=1200)
        question = "\n\n".join(
            part for part in (
                f"Title: {title}" if title else "",
                content,
                f"Public tests:\n{public_tests}" if public_tests else "",
            )
            if part
        )
        return {"question": question, "answer": "program passes hidden/private test cases"}
    problem_name = clean_value(lowered.get("problem_name"), max_chars=500)
    problem_desc = clean_value(lowered.get("problem_description_main"), max_chars=4000)
    problem_io = clean_value(lowered.get("problem_io"), max_chars=2500)
    if problem_name or problem_desc or problem_io:
        question = "\n\n".join(
            part for part in (
                problem_name,
                problem_desc,
                problem_io,
            )
            if part
        )
        return {"question": question, "answer": ""}
    gpqa_question = clean_value(lowered.get("question"), max_chars=5000)
    gpqa_answer = clean_value(lowered.get("correct answer"), max_chars=1000)
    if gpqa_question and gpqa_answer:
        choices = [
            value
            for value in (
                gpqa_answer,
                clean_value(lowered.get("incorrect answer 1"), max_chars=1000),
                clean_value(lowered.get("incorrect answer 2"), max_chars=1000),
                clean_value(lowered.get("incorrect answer 3"), max_chars=1000),
            )
            if value
        ]
        row_out = {"question": gpqa_question, "answer": gpqa_answer}
        if choices:
            row_out["choices"] = choices
        return row_out
    return {
        key: value
        for key, value in row.items()
        if str(key).lower() in THIN_ROW_KEYS
    }


def find_field(row: dict, candidates: list[str]) -> str | None:
    row_lower = {str(k).lower(): v for k, v in row.items()}
    for candidate in candidates:
        if candidate.lower() in row_lower:
            value = clean_value(row_lower[candidate.lower()])
            if value:
                return value
    return None


def extract_choices(row: dict) -> list[str]:
    for key in ("choices", "options", "choice", "mc1_targets"):
        value = row.get(key)
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, dict):
            choices = value.get("choices") or value.get("text")
            if isinstance(choices, list):
                return [str(v) for v in choices]
    return []


def extract_rows(payload: dict, limit: int | None = None) -> list[dict]:
    rows = payload.get("rows") or []
    out: list[dict] = []
    for item in rows[:limit]:
        row = item.get("row") if isinstance(item, dict) else None
        if isinstance(row, dict):
            out.append(row)
    return out


def fetch_split_size(
    dataset_id: str,
    config: str,
    split: str,
    *,
    token: str | None = None,
) -> int | None:
    enc_dataset = requests.utils.quote(dataset_id, safe="")
    enc_config = requests.utils.quote(config, safe="")
    try:
        res = requests.get(
            f"{HF_DATASETS_API}/size?dataset={enc_dataset}&config={enc_config}",
            timeout=20,
            headers=headers(token),
        )
        if res.status_code != 200:
            return None
        raw_config = unquote(enc_config)
        raw_split = split
        for item in (res.json() or {}).get("size", {}).get("splits", []) or []:
            if item.get("config") == raw_config and item.get("split") == raw_split:
                n = item.get("num_rows")
                return n if isinstance(n, int) and n > 0 else None
    except Exception:
        return None
    return None


def plan_sample_ranges(total_size: int, n_target: int, max_chunk: int = 100) -> list[tuple[int, int]]:
    if total_size <= 0:
        return [(0, min(n_target, max_chunk))]
    if total_size <= n_target:
        return [(offset, min(max_chunk, total_size - offset)) for offset in range(0, total_size, max_chunk)]
    n_chunks = max(1, min(10, (n_target + max_chunk - 1) // max_chunk))
    chunk_size = max(1, min(max_chunk, n_target // n_chunks))
    stride = max(1, total_size // n_chunks)
    return [
        (min(i * stride, max(0, total_size - chunk_size)), chunk_size)
        for i in range(n_chunks)
    ]


def plan_full_ranges(total_size: int, max_chunk: int = 100) -> list[tuple[int, int]]:
    return [
        (offset, min(max_chunk, total_size - offset))
        for offset in range(0, max(0, total_size), max_chunk)
    ]


def parquet_columns_to_read(column_names: list[str]) -> list[str]:
    return [
        column
        for column in column_names
        if column.lower() not in PARQUET_SKIP_COLUMNS
    ]


def prefers_parquet_projection(spec: DatasetSpec) -> bool:
    return spec.dataset_id == "livecodebench/code_generation"


def fetch_rows_range(
    dataset_id: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    *,
    token: str | None = None,
) -> list[dict]:
    enc_dataset = requests.utils.quote(dataset_id, safe="")
    enc_config = requests.utils.quote(config, safe="")
    enc_split = requests.utils.quote(split, safe="")
    try:
        res = requests.get(
            f"{HF_DATASETS_API}/rows?dataset={enc_dataset}&config={enc_config}"
            f"&split={enc_split}&offset={offset}&length={length}",
            timeout=30,
            headers=headers(token),
        )
        if res.status_code == 200:
            return extract_rows(res.json(), length)
    except Exception:
        return []
    return []


def fetch_rows_streaming(spec: DatasetSpec, n: int | None, *, token: str | None = None) -> list[dict]:
    try:
        from datasets import load_dataset
    except Exception:
        return []

    args = [spec.dataset_id]
    if spec.config:
        args.append(spec.config)
    kwargs = {"split": spec.split or "test", "streaming": True}
    if token:
        kwargs["token"] = token
    try:
        ds = load_dataset(*args, **kwargs)
    except TypeError:
        if token:
            kwargs.pop("token", None)
            kwargs["use_auth_token"] = token
        try:
            ds = load_dataset(*args, **kwargs)
        except Exception:
            return []
    except Exception:
        return []

    iterator = iter(ds)
    if n is not None:
        iterator = islice(iterator, n)
    return [thin_dataset_row(dict(row)) for row in iterator]


def fetch_rows_parquet_projection(spec: DatasetSpec, n: int | None, *, token: str | None = None) -> list[dict]:
    try:
        import fsspec
        import pyarrow.parquet as pq
    except Exception:
        return []

    config = spec.config or "default"
    split = spec.split or "test"
    enc_dataset = requests.utils.quote(spec.dataset_id, safe="")
    try:
        res = requests.get(
            f"{HF_DATASETS_API}/parquet?dataset={enc_dataset}",
            timeout=30,
            headers=headers(token),
        )
        if res.status_code != 200:
            return []
    except Exception:
        return []

    parquet_files = [
        item
        for item in (res.json() or {}).get("parquet_files", []) or []
        if item.get("config") == config and item.get("split") == split and item.get("url")
    ]
    if not parquet_files:
        return []

    rows: list[dict] = []
    storage_options = {"headers": headers(token)} if token else {}
    try:
        for item in parquet_files:
            with fsspec.open(item["url"], "rb", block_size=2**20, **storage_options) as f:
                parquet_file = pq.ParquetFile(f)
                columns = parquet_columns_to_read(parquet_file.schema.names)
                table = parquet_file.read(columns=columns)
            rows.extend(thin_dataset_row(dict(row)) for row in table.to_pylist())
            if n is not None and len(rows) >= n:
                return rows[:n]
    except Exception:
        return []
    return rows


def fetch_rows(spec: DatasetSpec, n: int | None, *, token: str | None = None) -> list[dict]:
    config = spec.config or "default"
    split = spec.split or "test"
    total = fetch_split_size(spec.dataset_id, config, split, token=token)
    if n is None:
        if prefers_parquet_projection(spec):
            rows = fetch_rows_parquet_projection(spec, None, token=token)
            if rows:
                return rows
        rows = fetch_rows_streaming(spec, None, token=token)
        if rows:
            return rows
        rows: list[dict] = []
        if total:
            for offset, length in plan_full_ranges(total):
                rows.extend(fetch_rows_range(spec.dataset_id, config, split, offset, length, token=token))
            if len(rows) < total:
                parquet_rows = fetch_rows_parquet_projection(spec, None, token=token)
                if len(parquet_rows) > len(rows):
                    return parquet_rows
            return rows
        offset = 0
        while True:
            batch = fetch_rows_range(spec.dataset_id, config, split, offset, 100, token=token)
            if not batch:
                return rows
            rows.extend(batch)
            if len(batch) < 100:
                return rows
            offset += len(batch)
    if total:
        rows: list[dict] = []
        for offset, length in plan_sample_ranges(total, n):
            rows.extend(fetch_rows_range(spec.dataset_id, config, split, offset, length, token=token))
            if len(rows) >= n:
                break
        if rows:
            return rows[:n]
    rows = fetch_rows_range(spec.dataset_id, config, split, 0, n, token=token)
    if rows:
        return rows
    return fetch_rows_streaming(spec, n, token=token)


def rows_to_tasks(benchmark: str, rows: list[dict]) -> list[dict]:
    tasks: list[dict] = []
    slug = name_key(benchmark)
    for i, row in enumerate(rows):
        question = find_field(row, Q_FIELD_CANDIDATES)
        if not question:
            question = "[raw] " + json.dumps(row, ensure_ascii=False)[:800]
        answer = find_field(row, A_FIELD_CANDIDATES) or ""
        choices = extract_choices(row)
        item = {
            "item_id": f"{slug}_{i:05d}",
            "benchmark": benchmark,
            "question": question,
            "answer": answer,
            "gt_topic": "unknown",
            "gt_reasoning_depth": "unknown",
            "gt_answer_format": "multiple_choice" if choices else "unknown",
            "reviewer_status": "reviewed",
        }
        if choices:
            item["choices"] = choices
        tasks.append(item)
    return tasks
