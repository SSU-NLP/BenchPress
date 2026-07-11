"""v3 §2.2.4 Executer — produces vocabulary V from a set of source benchmarks.

Takes per-benchmark chunk evidence Z_src for one or more source benchmarks and
a prompt I_exec, and returns a vocabulary V = [{id, name, definition}, ...]
which the Maker then applies corpus-wide. v3 main loop calls this once per
iteration to let V evolve along with the prompt the Improver returns.

Cache key includes the canonical source-set signature so the same prompt over
different source-benchmark sets does not serve a stale V.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Callable

from autotagging_loop.experiment.config import (
    llm_debug_dump_dir,
    llm_empty_content_retries,
    llm_extra_body,
    llm_request_timeout_s,
    llm_sdk_exception_retries,
    role_cfg,
)
from autotagging_loop.experiment.json_contract import (
    JSONContractError,
    call_json_contract,
    json_contract_attempts,
    json_contract_enabled,
    parse_json_object_strict,
)
from autotagging_loop.experiment.maker import _aggregate_signature
from autotagging_loop.experiment.taxonomy_refiner import _coerce_vocab, vocab_quality_reasons


ExecuterChatFn = Callable[..., str]

EXECUTER_SCHEMA_VERSION = 10


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _source_set_sig(source_benchmarks: list[str]) -> str:
    """Order-insensitive stable hash of the source-benchmark set."""
    sorted_names = sorted(set(source_benchmarks))
    joined = "|".join(sorted_names)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _executer_cache_key(
    *,
    prompt_i_exec: str,
    z_src_sig: str,
    source_set_sig: str,
    executer_model: str | None,
    schema_version: int,
    target_count: int | None = None,
    seed: int | None = None,
) -> str:
    payload = {
        "schema_version": schema_version,
        "prompt_i_exec": prompt_i_exec or "",
        "z_src_sig": z_src_sig,
        "source_set_sig": source_set_sig,
        "executer_model": executer_model or "",
        "target_count": target_count,
        "seed": seed,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _executer_cache_path(run_dir: str, source_set_sig: str, cache_key: str) -> str:
    return os.path.join(
        run_dir,
        "executer_cache",
        f"sources_{source_set_sig}",
        f"{cache_key}.json",
    )


def _parse_json(raw: str) -> dict:
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


def _validate_executer_json(
    parsed: dict,
    benchmark_names: list[str] | None = None,
    target_count: int | None = None,
) -> None:
    required = {"vocab", "rationale"}
    keys = set(parsed)
    missing = sorted(required - keys)
    if missing:
        raise JSONContractError(f"missing_keys:{','.join(missing)}")
    extra = sorted(keys - required)
    if extra:
        raise JSONContractError(f"extra_keys:{','.join(extra)}")
    vocab, reasons = _coerce_vocab(parsed.get("vocab"))
    if reasons:
        raise JSONContractError(f"invalid_vocab:{','.join(reasons)}")
    if target_count is not None and len(vocab) != int(target_count):
        raise JSONContractError(
            f"target_count_mismatch:{len(vocab)}!={int(target_count)}"
        )
    quality_reasons = vocab_quality_reasons(vocab, benchmark_names)
    if quality_reasons:
        raise JSONContractError(f"invalid_vocab_quality:{','.join(quality_reasons)}")
    if not vocab:
        raise JSONContractError("empty_vocab")
    if not str(parsed.get("rationale") or "").strip():
        raise JSONContractError("empty_rationale")


def _executer_cache_payload_valid(
    payload: dict,
    benchmark_names: list[str] | None = None,
    target_count: int | None = None,
) -> bool:
    raw = payload.get("raw_response")
    if not raw:
        return False
    try:
        _validate_executer_json(
            parse_json_object_strict(str(raw)),
            benchmark_names=benchmark_names,
            target_count=target_count,
        )
    except JSONContractError:
        return False
    return True


def _build_z_src_evidence_one(aggregate: dict) -> dict:
    """Pool a single benchmark's chunk evidence for the Executer LLM."""
    return {
        "benchmark": aggregate.get("benchmark"),
        "reviewed_rows": aggregate.get("reviewed_rows"),
        "n_chunks": aggregate.get("n_chunks"),
        "mapped_examples": aggregate.get("mapped_examples"),
        "text": aggregate.get("text", ""),
        "justifications": aggregate.get("justifications", []),
    }


def _build_z_src_evidence_multi(
    source_benchmarks: list[str],
    source_aggregates: dict[str, dict],
) -> list[dict]:
    """Per-bench evidence list. Each entry retains its benchmark identity for
    section-headed prompt rendering. Benches missing from `source_aggregates`
    are dropped with a warning so the loop can still advance on a partial set.
    """
    out: list[dict] = []
    for name in source_benchmarks:
        agg = source_aggregates.get(name)
        if not isinstance(agg, dict) or not agg:
            print(
                f"  [executer] WARN: source benchmark {name!r} missing aggregate; "
                f"skipping"
            )
            continue
        ev = _build_z_src_evidence_one(agg)
        # Force the benchmark name from the input key — the aggregate may carry
        # a stale display name; the canonical id is what the loop hands us.
        ev["benchmark"] = name
        out.append(ev)
    return out


def _fallback_summary_text(evidence: dict, *, max_chunks: int = 60) -> str:
    chunks = evidence.get("chunk_evidence") or []
    lines: list[str] = []
    for ev in chunks[:max_chunks]:
        idx = ev.get("chunk_index")
        summary = str(ev.get("summary") or "").strip()
        if summary:
            lines.append(f"- chunk {idx}: {summary}")
    if len(chunks) > max_chunks:
        lines.append(f"[truncated_chunk_summaries: {len(chunks) - max_chunks} omitted]")
    return "\n".join(lines)


def _format_chunk_evidence_section(evidence: dict) -> str:
    text = str(evidence.get("text") or "").strip() or _fallback_summary_text(evidence)
    return json.dumps(
        {
            "summary_text": text,
            "representative_justifications": evidence.get("justifications", []),
        },
        ensure_ascii=False,
        indent=2,
    )


def _format_chunk_evidence_multi(evidence_list: list[dict]) -> str:
    """Render per-bench chunk evidence with `## Benchmark: <name>` headers."""
    sections: list[str] = []
    for ev in evidence_list:
        bench = ev.get("benchmark", "<unknown>")
        body = _format_chunk_evidence_section(ev)
        sections.append(f"## Benchmark: {bench}\n{body}")
    return "\n\n".join(sections)


def _aggregate_signature_multi(evidence_list: list[dict]) -> str:
    """Stable signature across all per-bench evidences (order-insensitive)."""
    sigs = sorted(
        f"{ev.get('benchmark', '')}::{_aggregate_signature(ev)}" for ev in evidence_list
    )
    joined = "|".join(sigs)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _default_chat_fn(
    model: str,
    base_url: str | None = None,
    *,
    base_url_env: str | None = None,
    api_key_env: str | None = None,
    empty_content_retries: int | None = None,
    request_timeout_s: float | int | None = None,
    sdk_exception_retries: int | None = None,
    debug_dump_dir: str | None = None,
    extra_body: dict | None = None,
) -> ExecuterChatFn:
    from autotagging_loop.experiment.llm_client import shared_factory

    return shared_factory().chat_fn(
        model=model,
        base_url=base_url,
        base_url_env=base_url_env,
        api_key_env=api_key_env,
        response_format={"type": "json_object"},
        error_label="executer",
        empty_content_retries=empty_content_retries,
        request_timeout_s=request_timeout_s,
        sdk_exception_retries=sdk_exception_retries,
        debug_dump_dir=debug_dump_dir,
        extra_body=extra_body,
    )


def run_executer(
    *,
    source_benchmarks: list[str],
    source_aggregates: dict[str, dict],
    prompt_i_exec: str,
    config: dict,
    run_dir: str,
    version: int,
    label: str,
    chat_fn: ExecuterChatFn | None = None,
    seed: int | None = None,
    target_count: int | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """v3 §2.2.4 Executer — produce V = [{id, name, definition}, ...] from a
    set of source benchmarks' Z_src + I_exec.

    Returns `(vocab, metadata)`. On parse/validation failure, returns
    `(vocab=[], metadata={"reasons": [...], ...})` so the caller can decide
    how to fall back (e.g. reuse the previous iteration's V).
    """
    if not source_benchmarks:
        raise ValueError("source_benchmarks must be a non-empty list")
    if not isinstance(source_aggregates, dict) or not source_aggregates:
        raise ValueError("source_aggregates must be a non-empty dict")
    if target_count is not None and int(target_count) <= 0:
        raise ValueError("target_count must be positive when supplied")
    target_count = int(target_count) if target_count is not None else None

    model_cfg = role_cfg(config, "executer_model") or role_cfg(config, "maker_model") or {}
    model = model_cfg.get("name")
    base_url = model_cfg.get("base_url")
    base_url_env = model_cfg.get("base_url_env")
    api_key_env = model_cfg.get("api_key_env")
    schema_version = int(config.get("executer_schema_version", EXECUTER_SCHEMA_VERSION))
    strict_json = json_contract_enabled(config)
    json_attempts = json_contract_attempts(config)
    empty_retries = llm_empty_content_retries(config)
    request_timeout = llm_request_timeout_s(config)
    sdk_exception_retries = llm_sdk_exception_retries(config)
    debug_dir = llm_debug_dump_dir(config)
    extra_body = llm_extra_body(config)

    evidence_list = _build_z_src_evidence_multi(source_benchmarks, source_aggregates)
    if not evidence_list:
        raise ValueError(
            "no source aggregates available — every member of source_benchmarks "
            "is missing from source_aggregates"
        )

    used_benchmarks = sorted({ev["benchmark"] for ev in evidence_list})
    source_set_sig = _source_set_sig(used_benchmarks)
    z_src_sig = _aggregate_signature_multi(evidence_list)
    cache_key = _executer_cache_key(
        prompt_i_exec=prompt_i_exec,
        z_src_sig=z_src_sig,
        source_set_sig=source_set_sig,
        executer_model=model,
        schema_version=schema_version,
        target_count=target_count,
        seed=seed,
    )
    cache_path = _executer_cache_path(run_dir, source_set_sig, cache_key)

    cache_hit = False
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_vocab = cached.get("vocab") or []
            cached_reasons = cached.get("reasons") or []
            cache_is_good = (
                cached.get("schema_version") == schema_version
                and cached.get("z_src_sig") == z_src_sig
                and cached.get("source_set_sig") == source_set_sig
                and cached.get("executer_model") == model
                and cached.get("target_count") == target_count
                and cached.get("prompt_hash") == _hash_text(prompt_i_exec or "")
                and cached_vocab
                and not cached_reasons
                and (
                    not strict_json
                    or _executer_cache_payload_valid(
                        cached,
                        benchmark_names=used_benchmarks,
                        target_count=target_count,
                    )
                )
            )
            if cache_is_good:
                metadata = {
                    "executer_model": model,
                    "executer_schema_version": schema_version,
                    "executer_label": label,
                    "executer_version": version,
                    "source_benchmarks": used_benchmarks,
                    "source_set_sig": source_set_sig,
                    "z_src_sig": z_src_sig,
                    "cache_hit": True,
                    "vocab_size": len(cached_vocab),
                    "target_count": target_count,
                    "reasons": [],
                    "raw_response": cached.get("raw_response", ""),
                }
                return cached_vocab, metadata
        except Exception:
            pass

    chunk_blob = _format_chunk_evidence_multi(evidence_list)
    summary_lines = []
    for ev in evidence_list:
        summary_lines.append(
            f"- {ev['benchmark']}: reviewed_rows={ev.get('reviewed_rows')}, "
            f"n_chunks={ev.get('n_chunks')}, mapped_examples={ev.get('mapped_examples')}"
        )
    summary_blob = "\n".join(summary_lines)

    target_rule = (
        f"14) This call is COUNT-CONDITIONED. Return exactly {target_count} "
        "vocab entries. Do not return fewer or more. If two abilities overlap, "
        "split or merge only enough to hit the requested count while keeping every "
        "entry a valid reusable cognitive operation.\n"
        if target_count is not None
        else ""
    )
    user_count_rule = (
        f"Target vocab count for this candidate: exactly {target_count}. "
        "Use the requested count to explore a different taxonomy granularity "
        "from other candidates; do not ignore it.\n\n"
        if target_count is not None
        else ""
    )

    system_msg = (
        "You are designing a cognitive ability vocabulary V from MULTIPLE source "
        "benchmarks' chunk evidence Z_src. Each benchmark's evidence appears under "
        "its own `## Benchmark: <name>` header — reason across them to abstract "
        "general cognitive dimensions that recur. Follow the instruction prompt "
        "I_exec exactly. Return JSON only.\n\n"
        "STRICT RULES:\n"
        "1) Output JSON: {\"vocab\": [{\"id\": snake_case, \"name\": str, \"definition\": str}, ...], "
        "\"rationale\": str}.\n"
        "2) Each vocab entry must be a general cognitive ability dimension reusable "
        "across benchmarks — NOT a benchmark name, dataset name, answer format, or "
        "leaderboard label.\n"
        "3) `id` must be snake_case, unique, and appear nowhere else as a duplicate.\n"
        "4) `definition` must be a non-empty short sentence.\n"
        "5) Prefer a compact vocabulary of broad reusable operations. Merge "
        "near-synonyms and avoid source-specific subskills unless the evidence "
        "shows a repeated operational distinction that Maker can later rate from "
        "absent to dominant using direct evidence.\n"
        "6) Do not split abilities merely because benchmarks differ in topic, "
        "domain, answer format, or perceived difficulty.\n"
        "7) Include dimensions that let downstream tagging express shared "
        "cognitive demands across different-looking benchmarks as well as clear "
        "contrasts. Do not let a simple, knowledge-heavy, coding, or math-heavy "
        "source become isolated by mostly unique dimensions when it shares "
        "operations such as comprehension, retrieval, rule application, "
        "algorithmic execution, or quantitative reasoning with other sources.\n"
        "8) Treat domain distance as weak evidence. If different domains still "
        "require the same model-level operation — careful parsing, retrieval, "
        "rule-constrained execution, decomposition, verification, or robust "
        "handling of tricky wording — keep that operation as a shared axis "
        "instead of creating domain-family axes.\n"
        "9) Do not use model names, score tables, leaderboard reputation, or "
        "benchmark identity as evidence for an ability.\n"
        "10) Do not create axes about model implementation, architecture, "
        "parameter scale, training setup, inference system behavior, benchmark "
        "scoring mechanics, or answer-output formatting. In vocab ids/names, "
        "avoid words such as model, implementation, architecture, parameter, "
        "leaderboard, score, dataset, benchmark, format, and output unless they "
        "are part of a clearly cognitive operation term.\n"
        "11) Avoid pure difficulty or frontier-status axes. Difficulty can inform "
        "how strongly an existing operation is exercised, but it is not itself a "
        "cognitive operation unless the evidence shows a reusable process such as "
        "decomposition, verification, robustness to distractors, or long-context "
        "integration.\n"
        "12) Definitions should state observable evidence cues for the operation "
        "without listing source benchmark names. The downstream Maker must be able "
        "to decide absent/weak/medium/strong/dominant from chunk evidence alone.\n"
        "13) The source set is sparse. Do not optimize the vocabulary to separate "
        "only these source benchmarks. A candidate ability should either appear as "
        "a reusable operation across multiple source patterns or be phrased broadly "
        "enough that held-out benchmark evidence can rate it fairly. If an apparent "
        "distinction is supported by only one source benchmark, merge it into a "
        "broader operation unless it is an obviously general process.\n"
        "14) Target roughly 6-12 abilities. Do not force compression to a fixed "
        "count: split only repeated operational distinctions that Maker can rate "
        "consistently, and merge rare, redundant, topic-, format-, or source-specific "
        "axes. Keep definitions short enough for Maker to apply consistently.\n"
        f"{target_rule}"
    )
    user_msg = (
        f"Source benchmarks ({len(evidence_list)}):\n{summary_blob}\n\n"
        f"I_exec (instruction prompt):\n{prompt_i_exec}\n\n"
        f"{user_count_rule}"
        "Design V for held-out benchmark evidence as well as the sources. The "
        "source set is small, so favor stable, transferable operations over "
        "narrow source-only categories. Let the final count be evidence-driven "
        "within the requested range rather than targeting a specific size. "
        "Before finalizing V, mentally check each "
        "candidate ability against three questions: is it an operation rather than "
        "a topic; can it apply outside the source benchmark family; can Maker rate "
        "it from evidence without seeing scores or benchmark reputation?\n\n"
        f"Chunk evidence Z_src (one section per benchmark):\n{chunk_blob}\n"
    )

    fn = chat_fn or _default_chat_fn(
        model,
        base_url,
        base_url_env=base_url_env,
        api_key_env=api_key_env,
        empty_content_retries=empty_retries,
        request_timeout_s=request_timeout,
        sdk_exception_retries=sdk_exception_retries,
        debug_dump_dir=debug_dir,
        extra_body=extra_body,
    )
    if strict_json:
        raw, parsed = call_json_contract(
            fn,
            system_msg,
            user_msg,
            role=f"executer:{label}",
            attempts=json_attempts,
            validate=lambda candidate: _validate_executer_json(
                candidate,
                benchmark_names=used_benchmarks,
                target_count=target_count,
            ),
            seed=seed,
        )
    else:
        raw = fn(system_msg, user_msg) if seed is None else fn(system_msg, user_msg, seed)
        parsed = _parse_json(raw or "")
    vocab, vocab_reasons = _coerce_vocab(parsed.get("vocab"))
    reasons = list(vocab_reasons)
    if target_count is not None and len(vocab) != target_count:
        reasons.append(f"target_count_mismatch:{len(vocab)}!={target_count}")
    reasons.extend(vocab_quality_reasons(vocab, used_benchmarks))
    if not vocab and not reasons:
        reasons.append("empty_vocab")

    cache_payload = {
        "schema_version": schema_version,
        "executer_model": model,
        "source_benchmarks": used_benchmarks,
        "source_set_sig": source_set_sig,
        "z_src_sig": z_src_sig,
        "target_count": target_count,
        "seed": seed,
        "prompt_hash": _hash_text(prompt_i_exec or ""),
        "vocab": vocab,
        "rationale": str(parsed.get("rationale") or "").strip()[:1200],
        "raw_response": raw,
        "reasons": reasons,
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_payload, f, ensure_ascii=False, indent=2)

    metadata: dict[str, Any] = {
        "executer_model": model,
        "executer_schema_version": schema_version,
        "executer_label": label,
        "executer_version": version,
        "source_benchmarks": used_benchmarks,
        "source_set_sig": source_set_sig,
        "z_src_sig": z_src_sig,
        "cache_hit": cache_hit,
        "vocab_size": len(vocab),
        "target_count": target_count,
        "seed": seed,
        "reasons": reasons,
        "raw_response": raw,
    }
    print(
        f"  [executer] {label}: source_benchmarks={used_benchmarks} "
        f"(n={len(used_benchmarks)}), vocab_size={len(vocab)}, "
        f"reasons={reasons or ['ok']}"
    )
    return vocab, metadata
