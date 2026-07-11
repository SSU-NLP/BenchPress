"""MapReduce-style evidence extraction for Part 2 tag vectors."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from autotagging_loop.runner.config import make_openai_kwargs

_DEFAULT_REQUEST_TIMEOUT = float(os.getenv("BENCHPRESS_PART2_TIMEOUT", "180"))
_DEFAULT_MAX_RETRIES = int(os.getenv("BENCHPRESS_PART2_RETRIES", "2"))


def _extract_reasoning_text(message: Any) -> str:
    text = getattr(message, "reasoning", None)
    if isinstance(text, str) and text.strip():
        return text
    details = getattr(message, "reasoning_details", None)
    if isinstance(details, list):
        chunks: list[str] = []
        for item in details:
            t = getattr(item, "text", None) or (item.get("text") if isinstance(item, dict) else None)
            if isinstance(t, str) and t:
                chunks.append(t)
        if chunks:
            return "".join(chunks)
    if isinstance(message, dict):
        raw = message.get("reasoning")
        if isinstance(raw, str) and raw.strip():
            return raw
        details = message.get("reasoning_details") or []
        chunks = [d.get("text", "") for d in details if isinstance(d, dict)]
        if any(chunks):
            return "".join(chunks)
    return ""


def _extract_balanced_json(text: str) -> str:
    if not text:
        return ""
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""

ABILITY_LEVEL_SCORES = {
    "absent": 0.0,
    "weak": 0.25,
    "medium": 0.5,
    "strong": 0.75,
    "dominant": 1.0,
}

ChatFn = Callable[[str, str], str]


def parse_json(raw: str) -> dict:
    if not raw:
        return {}
    cleaned = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", raw, flags=re.DOTALL).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                return {}
    return {}


def default_chat_fn(model: str, base_url: str | None) -> ChatFn:
    client = OpenAI(**make_openai_kwargs(base_url), timeout=_DEFAULT_REQUEST_TIMEOUT)

    def call(system_msg: str, user_msg: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(1 + _DEFAULT_MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                )
                msg = resp.choices[0].message
                content = (msg.content or "").strip()
                if content:
                    return content
                reasoning = _extract_reasoning_text(msg)
                fallback = _extract_balanced_json(reasoning)
                if fallback:
                    print(
                        f"  [part2_mapreduce] recovered JSON from reasoning field "
                        f"(model={model}, attempt={attempt + 1})"
                    )
                    return fallback
                print(
                    f"  [part2_mapreduce] empty content (model={model}, attempt={attempt + 1}/"
                    f"{1 + _DEFAULT_MAX_RETRIES})"
                )
            except Exception as exc:
                last_exc = exc
                print(
                    f"  [part2_mapreduce] request error (model={model}, "
                    f"attempt={attempt + 1}/{1 + _DEFAULT_MAX_RETRIES}): {exc}"
                )
                time.sleep(min(2 ** attempt, 8))
        if last_exc is not None:
            raise RuntimeError(f"Part 2 LLM call failed for model={model}: {last_exc}") from last_exc
        return ""

    return call


def load_vocab(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("abilities"), list):
        return data["abilities"]
    if isinstance(data, list):
        return data
    raise ValueError(f"unsupported vocab format: {path}")


def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def chunk_examples(examples: list[str], chunk_size: int, max_chars: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for example in examples:
        ex = str(example)
        if current and (len(current) >= chunk_size or current_chars + len(ex) > max_chars):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(ex)
        current_chars += len(ex)
    if current:
        chunks.append(current)
    return chunks


def _prompt_cache_hash(system_msg: str, user_msg: str) -> str:
    return hashlib.sha256((system_msg + user_msg).encode("utf-8")).hexdigest()


def _model_slug(name: str | None) -> str:
    return str(name or "unknown-model").replace("/", "__")


def _cache_file_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    return stem or "benchmark"


def _cache_root(config: dict) -> Path:
    override = config.get("part2_mapreduce_cache_dir")
    if override:
        return Path(str(override))
    base = Path(str(config.get("results_dir", "results")))
    if base.name == "part2_experiment":
        base = base.parent
    return base / "_cache" / "part2_mapreduce"


def _read_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def coerce_level(value: Any) -> str:
    level = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"none": "absent", "low": "weak", "moderate": "medium", "high": "strong"}
    level = aliases.get(level, level)
    return level if level in ABILITY_LEVEL_SCORES else "absent"


def coerce_weight_to_level(value: Any) -> str:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return "absent"
    if weight >= 0.875:
        return "dominant"
    if weight >= 0.625:
        return "strong"
    if weight >= 0.375:
        return "medium"
    if weight >= 0.125:
        return "weak"
    return "absent"


def map_chunk(
    benchmark: str,
    examples: list[str],
    vocab: list[dict],
    prompt: str,
    chat_fn: ChatFn,
    *,
    model_name: str | None = None,
    chunk_index: int | None = None,
    cache_root: Path | None = None,
) -> dict:
    vocab_payload = [
        {"id": item["id"], "name": item.get("name", item["id"]), "definition": item.get("definition", "")}
        for item in vocab
    ]
    user_msg = json.dumps(
        {
            "benchmark": benchmark,
            "vocabulary": vocab_payload,
            "examples": examples,
        },
        ensure_ascii=False,
    )
    cache_path = None
    if cache_root is not None and chunk_index is not None:
        prompt_hash = _prompt_cache_hash(prompt, user_msg)
        cache_path = (
            cache_root
            / _model_slug(model_name)
            / f"prompt_{prompt_hash}"
            / f"{_cache_file_stem(benchmark)}_chunk{chunk_index}.json"
        )
        cached = _read_cache(cache_path)
        if cached is not None:
            cached["_cache_hit"] = True
            return cached
    parsed = parse_json(chat_fn(prompt, user_msg))
    raw_levels = parsed.get("ability_levels") if isinstance(parsed.get("ability_levels"), dict) else {}
    raw_weights = parsed.get("weights") if isinstance(parsed.get("weights"), dict) else {}
    raw_evidence = parsed.get("ability_evidence") if isinstance(parsed.get("ability_evidence"), dict) else {}
    if raw_levels:
        levels = {item["id"]: coerce_level(raw_levels.get(item["id"])) for item in vocab}
    else:
        levels = {item["id"]: coerce_weight_to_level(raw_weights.get(item["id"])) for item in vocab}
    rationale = str(parsed.get("rationale", ""))[:500]
    evidence = {item["id"]: str(raw_evidence.get(item["id"], rationale))[:500] for item in vocab}
    payload = {
        "n_examples": len(examples),
        "ability_levels": levels,
        "ability_evidence": evidence,
        "chunk_summary": str(parsed.get("chunk_summary", ""))[:1000],
    }
    if cache_path is not None:
        _write_cache(cache_path, payload)
    payload["_cache_hit"] = False
    return payload


def aggregate_chunks(benchmark: str, mapped: list[dict], vocab: list[dict]) -> tuple[dict[str, float], dict]:
    total = sum(max(1, int(item.get("n_examples", 0))) for item in mapped) or 1
    weights: dict[str, float] = {}
    level_counts: dict[str, dict[str, int]] = {}
    for tag in vocab:
        tag_id = tag["id"]
        weighted = 0.0
        counts = {level: 0 for level in ABILITY_LEVEL_SCORES}
        for item in mapped:
            n = max(1, int(item.get("n_examples", 0)))
            level = coerce_level((item.get("ability_levels") or {}).get(tag_id))
            counts[level] += n
            weighted += n * ABILITY_LEVEL_SCORES[level]
        weights[tag_id] = weighted / total
        level_counts[tag_id] = counts
    metadata = {
        "benchmark": benchmark,
        "n_chunks": len(mapped),
        "mapped_examples": total,
        "ability_level_counts": level_counts,
        "chunk_summaries": [item.get("chunk_summary", "") for item in mapped],
    }
    return weights, metadata


def reduce_benchmark(
    benchmark: str,
    aggregate_weights: dict[str, float],
    aggregate_metadata: dict,
    vocab: list[dict],
    config: dict,
    chat_fn: ChatFn,
) -> tuple[dict[str, float], dict]:
    reducer_cfg = config.get("mapreduce_reducer_model") or {}
    if not reducer_cfg.get("name"):
        return aggregate_weights, {"source": "deterministic_chunk_aggregate"}

    vocab_payload = [
        {"id": item["id"], "name": item.get("name", item["id"]), "definition": item.get("definition", "")}
        for item in vocab
    ]
    system_msg = (
        "You are synthesizing chunk-level benchmark evidence for auditability. "
        "Do not assign final cognitive ability levels or numeric tag weights; those are "
        "computed deterministically by code from aggregate counts. Return JSON only: "
        "{\"evidence_summary\": \"brief\", \"weight_cautions\": {\"<tag_id>\": \"brief or empty\"}}. "
        "Do not create new tag ids."
    )
    user_msg = json.dumps(
        {
            "benchmark": benchmark,
            "vocabulary": vocab_payload,
            "aggregate_chunk_weights": aggregate_weights,
            "ability_level_counts": aggregate_metadata.get("ability_level_counts", {}),
            "chunk_summaries": aggregate_metadata.get("chunk_summaries", []),
        },
        ensure_ascii=False,
    )
    parsed = parse_json(chat_fn(system_msg, user_msg))
    raw_cautions = parsed.get("weight_cautions") if isinstance(parsed.get("weight_cautions"), dict) else {}
    cautions = {item["id"]: str(raw_cautions.get(item["id"], ""))[:500] for item in vocab}
    return aggregate_weights, {
        "source": "llm_reducer_evidence_synthesis",
        "reducer_model": reducer_cfg.get("name"),
        "final_weight_policy": "deterministic_chunk_level_weighted_average",
        "evidence_summary": str(parsed.get("evidence_summary", ""))[:1000],
        "weight_cautions": cautions,
    }


def build_tag_vectors(
    documents: dict[str, dict[str, Any]],
    vocab: list[dict],
    config: dict,
    *,
    chat_fn: ChatFn | None = None,
) -> tuple[dict[str, dict[str, float]], dict]:
    model_cfg = config.get("mapreduce_model", {})
    fn = chat_fn or default_chat_fn(model_cfg.get("name"), model_cfg.get("base_url"))
    reducer_cfg = config.get("mapreduce_reducer_model") or {}
    reducer_fn = (
        fn
        if chat_fn is not None
        else default_chat_fn(reducer_cfg.get("name"), reducer_cfg.get("base_url"))
        if reducer_cfg.get("name")
        else fn
    )
    prompt = load_prompt(config["prompt_path"])
    chunk_size = int(config.get("mapreduce_chunk_examples", 25))
    max_chars = int(config.get("mapreduce_max_chunk_chars", 32000))
    max_workers = max(1, int(config.get("mapreduce_max_workers", 32)))
    cache_root = (
        None
        if chat_fn is not None or config.get("mapreduce_cache_enabled", True) is False
        else _cache_root(config)
    )
    T: dict[str, dict[str, float]] = {}
    metadata: dict[str, Any] = {
        "method": "part2_mapreduce_static",
        "mapper_model": model_cfg.get("name"),
        "reducer_model": reducer_cfg.get("name"),
        "mapreduce_max_workers": max_workers,
        "map_cache_root": str(cache_root) if cache_root is not None else None,
        "final_weight_policy": (
            "Mapper extracts chunk-level ordinal evidence; final benchmark tag weights "
            "are deterministic weighted averages over chunk-level evidence. The reducer "
            "may synthesize evidence for auditability but does not assign weights."
        ),
        "prompt_path": str(config["prompt_path"]),
        "level_scores": ABILITY_LEVEL_SCORES,
        "benchmarks": {},
    }
    chunks_by_benchmark: dict[str, list[list[str]]] = {}
    mapped_by_benchmark: dict[str, list[dict | None]] = {}
    example_counts: dict[str, int] = {}
    tasks: list[tuple[str, int, list[str]]] = []
    for benchmark, document in documents.items():
        examples = [str(item) for item in document.get("examples", [])]
        chunks = chunk_examples(examples, chunk_size, max_chars)
        chunks_by_benchmark[benchmark] = chunks
        mapped_by_benchmark[benchmark] = [None] * len(chunks)
        example_counts[benchmark] = len(examples)
        for idx, chunk in enumerate(chunks):
            tasks.append((benchmark, idx, chunk))

    if tasks:
        workers = min(max_workers, len(tasks))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_key = {
                executor.submit(
                    map_chunk,
                    benchmark,
                    chunk,
                    vocab,
                    prompt,
                    fn,
                    model_name=model_cfg.get("name"),
                    chunk_index=idx,
                    cache_root=cache_root,
                ): (benchmark, idx)
                for benchmark, idx, chunk in tasks
            }
            for future in as_completed(future_to_key):
                benchmark, idx = future_to_key[future]
                mapped_by_benchmark[benchmark][idx] = future.result()

    for benchmark in documents:
        chunks = chunks_by_benchmark[benchmark]
        mapped = [item for item in mapped_by_benchmark[benchmark] if item is not None]
        aggregate_weights, bench_meta = aggregate_chunks(benchmark, mapped, vocab)
        weights, reducer_meta = reduce_benchmark(
            benchmark,
            aggregate_weights,
            bench_meta,
            vocab,
            config,
            reducer_fn,
        )
        T[benchmark] = weights
        bench_meta["aggregate_weights"] = aggregate_weights
        bench_meta["reducer"] = reducer_meta
        bench_meta["final_weights"] = weights
        bench_meta["map_cache_hits"] = sum(1 for item in mapped if item.get("_cache_hit"))
        metadata["benchmarks"][benchmark] = bench_meta
        print(f"  [part2_mapreduce] {benchmark}: chunks={len(chunks)}, examples={example_counts[benchmark]}")
    return T, metadata
