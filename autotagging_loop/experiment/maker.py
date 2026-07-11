"""v3 §2.2.5 Maker — applies a taxonomy V to corpus-wide chunk evidence.

Given per-benchmark vocab-free chunk evidence Z_l (built by Mapper) and a
vocabulary V, the Maker LLM assigns one ordinal cognitive-ability level per
ability id per benchmark. Numeric weights are produced downstream by
`build_static_tag_vectors_from_reducer_levels`.

This file owns the LLM-call body that previously lived in
`experiment/mapreduce_reducer.py`. The legacy module is kept as a
backward-compatible shim.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from tqdm.auto import tqdm

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
)
from autotagging_loop.experiment.mapreduce_evidence import _prompt_hash, _slug, _vocab_hash
from autotagging_loop.experiment.static_tag_weights import ABILITY_LEVEL_SCORES, coerce_ability_level


MakerChatFn = Callable[[str, str], str]
MAKER_SCHEMA_VERSION = 9

_GENERIC_BENCHMARK_TOKENS = {
    "benchmark",
    "bench",
    "dataset",
    "data",
    "test",
    "eval",
    "evaluation",
    "task",
    "tasks",
    "math",
    "code",
    "qa",
    "pro",
}
_NEGATION_CUES = {
    "not",
    "never",
    "without",
    "avoid",
    "avoids",
    "beyond",
    "rather",
    "instead",
}
_FORBIDDEN_RATIONALE_TOKEN_SEQUENCES = [
    ("multiple", "choice"),
    ("true", "false"),
    ("yes", "no"),
    ("short", "answer"),
    ("free", "form"),
    ("selected", "option"),
    ("answer", "format"),
    ("output", "format"),
    ("benchmark", "family"),
    ("dataset", "family"),
    ("domain", "family"),
    ("leaderboard",),
    ("model", "performance"),
    ("expected", "performance"),
    ("public", "reputation"),
    ("difficulty",),
    ("frontier",),
    ("hard", "benchmark"),
    ("easy", "benchmark"),
]
_NO_EVIDENCE_TOKEN_SEQUENCES = [
    ("no", "evidence"),
    ("no", "direct", "evidence"),
    ("missing", "evidence"),
    ("not", "evidenced"),
    ("not", "present"),
    ("absent",),
]
_SCORE_TOKENS = {"score", "scores"}
_EVAL_SCORE_CONTEXT_CUES = {
    "model",
    "models",
    "leaderboard",
    "performance",
    "evaluation",
    "eval",
    "benchmark",
    "benchmarks",
    "accuracy",
    "metric",
    "metrics",
    "grading",
    "graded",
    "rubric",
    "rank",
    "ranking",
}

_MAKER_JSON_RETRY_HINT = (
    "Maker repair rule: if validation reports invalid_maker_evidence or "
    "ability_rationale_leaks_non_evidence, rewrite ability_rationale strings "
    "from scratch using only observed cognitive operations from the chunk evidence. "
    "Remove every failed metadata/surface token from all rationale strings. "
    "If the validation error names a banned term, do not quote or reuse that "
    "term; replace it with operation-level evidence. Keep ability_levels "
    "unchanged unless the rationale cannot support the level."
)


def _default_chat_fn(
    model: str,
    base_url: str | None = None,
    *,
    base_url_env: str | None = None,
    api_key_env: str | None = None,
    error_label: str = "maker",
    empty_content_retries: int | None = None,
    request_timeout_s: float | int | None = None,
    sdk_exception_retries: int | None = None,
    debug_dump_dir: str | None = None,
    extra_body: dict | None = None,
) -> MakerChatFn:
    from autotagging_loop.experiment.llm_client import shared_factory

    return shared_factory().chat_fn(
        model=model,
        base_url=base_url,
        base_url_env=base_url_env,
        api_key_env=api_key_env,
        response_format={"type": "json_object"},
        error_label=error_label,
        empty_content_retries=empty_content_retries,
        request_timeout_s=request_timeout_s,
        sdk_exception_retries=sdk_exception_retries,
        debug_dump_dir=debug_dump_dir,
        extra_body=extra_body,
    )


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _contains_token_sequence(tokens: list[str], phrase: tuple[str, ...]) -> bool:
    if not phrase or len(tokens) < len(phrase):
        return False
    for idx in range(0, len(tokens) - len(phrase) + 1):
        if tuple(tokens[idx: idx + len(phrase)]) == phrase:
            return True
    return False


def _contains_unnegated_token_sequence(tokens: list[str], phrase: tuple[str, ...]) -> bool:
    if not phrase or len(tokens) < len(phrase):
        return False
    for idx in range(0, len(tokens) - len(phrase) + 1):
        if tuple(tokens[idx: idx + len(phrase)]) != phrase:
            continue
        context = tokens[max(0, idx - 4): idx]
        if any(token in _NEGATION_CUES for token in context):
            continue
        return True
    return False


def _score_eval_leak(tokens: list[str]) -> bool:
    for idx, token in enumerate(tokens):
        if token not in _SCORE_TOKENS:
            continue
        if any(t in _NEGATION_CUES for t in tokens[max(0, idx - 4): idx]):
            continue
        window = tokens[max(0, idx - 3): idx + 4]
        if any(cue in _EVAL_SCORE_CONTEXT_CUES for cue in window):
            return True
    return False


def _benchmark_token_sequences(benchmark: str | None) -> list[tuple[str, ...]]:
    tokens = _tokens(benchmark or "")
    sequences: set[tuple[str, ...]] = set()
    if len(tokens) > 1:
        sequences.add(tuple(tokens))
    for token in tokens:
        if token.isdigit() or len(token) < 3 or token in _GENERIC_BENCHMARK_TOKENS:
            continue
        sequences.add((token,))
    return sorted(sequences, key=lambda item: (len(item), item))


def _maker_evidence_quality_reasons(
    parsed: dict,
    vocab_ids: list[str],
    *,
    benchmark: str | None = None,
) -> list[str]:
    levels = parsed.get("ability_levels") or {}
    rationale = parsed.get("ability_rationale") or {}
    benchmark_sequences = _benchmark_token_sequences(benchmark)
    reasons: list[str] = []
    for tag_id in vocab_ids:
        text = str(rationale.get(tag_id) or "")
        tokens = _tokens(text)
        for phrase in _FORBIDDEN_RATIONALE_TOKEN_SEQUENCES:
            if _contains_unnegated_token_sequence(tokens, phrase):
                reasons.append(
                    f"ability_rationale_leaks_non_evidence:{tag_id}:{'_'.join(phrase)}"
                )
                break
        if _score_eval_leak(tokens):
            reasons.append(f"ability_rationale_leaks_non_evidence:{tag_id}:eval_score")
        for phrase in benchmark_sequences:
            if _contains_token_sequence(tokens, phrase):
                reasons.append(
                    f"ability_rationale_mentions_benchmark:{tag_id}:{'_'.join(phrase)}"
                )
                break
        level = str(levels.get(tag_id) or "").strip().lower()
        if level and level != "absent":
            for phrase in _NO_EVIDENCE_TOKEN_SEQUENCES:
                if _contains_unnegated_token_sequence(tokens, phrase):
                    reasons.append(
                        f"ability_level_rationale_contradiction:{tag_id}:{level}:{'_'.join(phrase)}"
                    )
                    break
    return reasons


def _validate_maker_json(
    parsed: dict,
    vocab_ids: list[str],
    *,
    benchmark: str | None = None,
) -> None:
    required = {"benchmark_summary", "ability_levels", "ability_rationale"}
    keys = set(parsed)
    missing = sorted(required - keys)
    if missing:
        raise JSONContractError(f"missing_keys:{','.join(missing)}")
    extra = sorted(keys - required)
    if extra:
        raise JSONContractError(f"extra_keys:{','.join(extra)}")
    if not str(parsed.get("benchmark_summary") or "").strip():
        raise JSONContractError("empty_benchmark_summary")
    levels = parsed.get("ability_levels")
    if not isinstance(levels, dict):
        raise JSONContractError("ability_levels_not_object")
    rationale = parsed.get("ability_rationale")
    if not isinstance(rationale, dict):
        raise JSONContractError("ability_rationale_not_object")

    expected = set(vocab_ids)
    level_keys = set(str(k) for k in levels)
    rationale_keys = set(str(k) for k in rationale)
    missing_levels = sorted(expected - level_keys)
    extra_levels = sorted(level_keys - expected)
    if missing_levels:
        raise JSONContractError(f"ability_levels_missing:{','.join(missing_levels)}")
    if extra_levels:
        raise JSONContractError(f"ability_levels_extra:{','.join(extra_levels)}")
    missing_rationale = sorted(expected - rationale_keys)
    extra_rationale = sorted(rationale_keys - expected)
    if missing_rationale:
        raise JSONContractError(f"ability_rationale_missing:{','.join(missing_rationale)}")
    if extra_rationale:
        raise JSONContractError(f"ability_rationale_extra:{','.join(extra_rationale)}")

    allowed = set(ABILITY_LEVEL_SCORES)
    invalid_levels = [
        f"{tag_id}:{levels.get(tag_id)}"
        for tag_id in vocab_ids
        if str(levels.get(tag_id) or "").strip().lower() not in allowed
    ]
    if invalid_levels:
        raise JSONContractError(f"invalid_ability_levels:{','.join(invalid_levels)}")
    empty_rationale = [
        tag_id for tag_id in vocab_ids if not str(rationale.get(tag_id) or "").strip()
    ]
    if empty_rationale:
        raise JSONContractError(f"empty_ability_rationale:{','.join(empty_rationale)}")

    quality_reasons = _maker_evidence_quality_reasons(
        parsed,
        vocab_ids,
        benchmark=benchmark,
    )
    if quality_reasons:
        raise JSONContractError(f"invalid_maker_evidence:{','.join(quality_reasons)}")


def _normalize_maker_json_for_validation(parsed: dict, vocab_ids: list[str], aggregate: dict) -> None:
    levels = parsed.get("ability_levels")
    if isinstance(levels, dict):
        parsed["ability_levels"] = {
            str(tag_id): coerce_ability_level(value)
            for tag_id, value in levels.items()
        }
    if not str(parsed.get("benchmark_summary") or "").strip():
        summaries = [
            str(ev.get("summary") or "").strip()
            for ev in (aggregate.get("chunk_evidence") or [])
            if str(ev.get("summary") or "").strip()
        ]
        if summaries:
            parsed["benchmark_summary"] = " ".join(summaries[:2])[:1200]


def _maker_cache_payload_valid(
    payload: dict,
    vocab_ids: list[str],
    *,
    benchmark: str | None = None,
) -> bool:
    try:
        parsed = {
            "benchmark_summary": payload.get("benchmark_summary"),
            "ability_levels": payload.get("ability_levels"),
            "ability_rationale": payload.get("ability_rationale"),
        }
        _validate_maker_json(
            parsed,
            vocab_ids,
            benchmark=benchmark,
        )
    except JSONContractError:
        return False
    return True


def _maker_root(run_dir: str, prompt: str | None) -> str:
    # Directory name kept as `mapreduce_reducer/` for cache continuity with
    # pre-Phase-2 runs. The Maker writes under the same prompt-keyed tree.
    return os.path.join(run_dir, "mapreduce_reducer", f"prompt_{_prompt_hash(prompt)}")


def _aggregate_signature(evidence: dict) -> str:
    return _hash_text(json.dumps(evidence, ensure_ascii=False, sort_keys=True))


def _maker_cache_key(
    *,
    benchmark: str,
    prompt: str | None,
    vocab_hash: str,
    model: str | None,
    aggregate_hash: str,
    schema_version: int,
    seed: int | None = None,
) -> str:
    payload = {
        "schema_version": schema_version,
        "benchmark": benchmark,
        "prompt": prompt or "",
        "vocab_hash": vocab_hash,
        "model": model or "",
        "aggregate_hash": aggregate_hash,
        "seed": seed,
    }
    return _hash_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _evidence_for_maker(aggregate: dict) -> dict:
    """Build a vocab-free per-chunk evidence package for the Maker LLM."""
    return {
        "benchmark": aggregate.get("benchmark"),
        "reviewed_rows": aggregate.get("reviewed_rows"),
        "n_chunks": aggregate.get("n_chunks"),
        "mapped_examples": aggregate.get("mapped_examples"),
        "text": aggregate.get("text", ""),
        "justifications": aggregate.get("justifications", []),
    }


def _coerce_maker_output(
    benchmark: str,
    raw_response: str,
    parsed: dict,
    vocab_ids: list[str],
    aggregate: dict,
) -> dict:
    from autotagging_loop.experiment.static_tag_weights import coerce_ability_level

    raw_levels = (
        parsed.get("ability_levels")
        or parsed.get("final_ability_levels")
        or parsed.get("levels")
    )
    if not isinstance(raw_levels, dict):
        raw_levels = {}
    levels = {
        tag_id: coerce_ability_level(raw_levels.get(tag_id))
        for tag_id in vocab_ids
    }

    raw_rationale = parsed.get("ability_rationale") or parsed.get("rationale")
    if isinstance(raw_rationale, dict):
        rationale = {
            tag_id: str(raw_rationale.get(tag_id, "")).strip()[:800]
            for tag_id in vocab_ids
        }
    else:
        shared = str(raw_rationale or "").strip()[:800]
        rationale = {tag_id: shared for tag_id in vocab_ids}

    return {
        "benchmark": benchmark,
        "n_chunks": aggregate.get("n_chunks"),
        "mapped_examples": aggregate.get("mapped_examples"),
        "ability_levels": levels,
        "ability_rationale": rationale,
        "benchmark_summary": str(parsed.get("benchmark_summary") or "").strip()[:1200],
        "raw_response": raw_response,
    }


def _apply_one(
    *,
    benchmark: str,
    aggregate: dict,
    vocab: list[dict],
    prompt: str,
    model: str,
    base_url: str | None,
    cache_path: str,
    aggregate_hash: str,
    vocab_hash: str,
    schema_version: int,
    chat_fn: MakerChatFn | None,
    base_url_env: str | None = None,
    api_key_env: str | None = None,
    seed: int | None = None,
    json_contract_strict: bool = True,
    json_contract_max_attempts: int = 3,
    empty_content_retries: int | None = None,
    request_timeout_s: float | int | None = None,
    sdk_exception_retries: int | None = None,
    debug_dump_dir: str | None = None,
    extra_body: dict | None = None,
) -> dict:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            vocab_ids = [v["id"] for v in vocab]
            if (
                cached.get("aggregate_hash") == aggregate_hash
                and cached.get("prompt_hash") == _prompt_hash(prompt)
                and cached.get("vocab_hash") == vocab_hash
                and cached.get("model") == model
                and (
                    not json_contract_strict
                    or _maker_cache_payload_valid(
                        cached,
                        vocab_ids,
                        benchmark=benchmark,
                    )
                )
            ):
                cached["_cache_hit"] = True
                return cached
        except Exception:
            pass

    vocab_ids = [v["id"] for v in vocab]
    ability_lines = "\n".join(
        f"- {v['id']}: {v.get('definition') or v.get('name') or v['id']}"
        for v in vocab
    )
    levels_skeleton = json.dumps(
        {vid: "absent|weak|medium|strong|dominant" for vid in vocab_ids},
        ensure_ascii=False,
        indent=2,
    )
    rationale_skeleton = json.dumps(
        {vid: "short justification; use no evidence if absent" for vid in vocab_ids},
        ensure_ascii=False,
        indent=2,
    )
    vocab_id_list = json.dumps(vocab_ids, ensure_ascii=False)
    evidence = _evidence_for_maker(aggregate)
    chunk_blob = json.dumps(
        {
            "summary_text": evidence.get("text", ""),
            "representative_justifications": evidence.get("justifications", []),
        },
        ensure_ascii=False,
        indent=2,
    )
    system_msg = (
        "You are a benchmark-level tagger. Read the descriptive chunk evidence below "
        "and assign one ordinal cognitive-ability level per ability id. Tag broad "
        "transferable operations, not benchmark identity, domain reputation, response-interface "
        "cues, external outcome cues, or perceived difficulty. Domain mismatch is not evidence of absence "
        "when the benchmark still directly exercises the same operation. At the same "
        "time, do not inflate every ability because a benchmark is broadly hard; each "
        "level must be justified by direct evidence for that ability's definition. "
        "Output JSON only."
    )
    user_msg = (
        f"Benchmark: {benchmark}\n"
        f"reviewed_rows: {evidence.get('reviewed_rows')}\n"
        f"n_chunks: {evidence.get('n_chunks')}\n"
        f"mapped_examples: {evidence.get('mapped_examples')}\n\n"
        f"Maker prompt candidate:\n{prompt}\n\n"
        f"Cognitive ability vocabulary (the only valid ids):\n{ability_lines}\n\n"
        "Return JSON with exactly these top-level keys and no extra keys. "
        f"The exact valid ability id set is {vocab_id_list}. "
        "The object keys inside `ability_levels` and `ability_rationale` must "
        "match that id set exactly. Do not add, rename, split, merge, or invent "
        "ability ids, even if the prompt candidate or evidence mentions a useful "
        "concept outside this vocabulary. Every listed id must appear exactly once "
        "in both nested objects; do not omit ids. "
        "If the benchmark has no evidence for an ability, set its level to `absent`.\n"
        "{\n"
        '  "benchmark_summary": "2-4 sentence synthesis",\n'
        f'  "ability_levels": {levels_skeleton},\n'
        f'  "ability_rationale": {rationale_skeleton}\n'
        "}\n"
        "Level meanings: absent=no direct evidence for the operation; weak=occasional "
        "or peripheral cue in a minority of chunks; medium=repeated and clear but "
        "secondary requirement; strong=major requirement across many chunks; "
        "dominant=primary operation across most chunks. Use dominant sparingly.\n"
        "Calibration rules: assign similar levels when different-looking tasks require "
        "the same operation; do not mark a directly visible secondary operation as "
        "absent; avoid uniformly high or uniformly low vectors; use absent only for "
        "missing evidence, not for domain mismatch. Rate each ability against its "
        "own definition using absolute evidence, not relative to the hardest "
        "benchmark or the benchmark's headline domain. For directly visible shared "
        "operations, prefer weak or medium over absent when the operation is real "
        "but secondary. For simple or knowledge-heavy evidence, still mark shared "
        "operations such as comprehension, retrieval, rule use, or multi-step "
        "reasoning when directly present. When the evidence is from a different "
        "topic family than the ability name suggests, still assign a non-absent "
        "level if the underlying operation is visible; do not zero out common "
        "operations merely because the benchmark is math, code, factual QA, or "
        "general reasoning. Level discipline: weak means visible but not load-bearing; "
        "medium means repeated and materially useful; strong requires the operation "
        "to be a main bottleneck in many chunks; dominant requires that most chunks "
        "would fail without that operation. Do not assign strong or dominant just "
        "because an ability is generally useful, broadly associated with the topic, "
        "or present in only one salient example. If two ability definitions overlap, "
        "give the higher level to the more specific directly evidenced operation and "
        "keep the broader overlapping one one level lower unless it is independently "
        "central. Rationale strings must cite the observed operation patterns. "
        "Forbidden rationale wording: cite only observed cognitive "
        "operations. Do not copy surface-presentation, public-outcome, model-outcome, "
        "grading/measurement, reputation, or generic-difficulty terms from the prompt, "
        "evidence, or validation feedback. If evidence contains alternatives or "
        "distractors, describe the operation as plausible-alternative filtering, "
        "semantic precision, or verification burden without naming the presentation "
        "style. If evidence contains numbers, describe arithmetic, comparison, or "
        "quantitative calculation without naming grading artifacts. For sports or "
        "event records, describe point totals, counts, margins, entities, or temporal "
        "relations; do not use evaluation/outcome wording. If evidence describes a "
        "requested response shape, describe schema mapping or value extraction instead. If evidence "
        "supports only presentation or topic but not the ability's operation, keep "
        "the level absent or weak.\n\n"
        f"Chunk evidence:\n{chunk_blob}"
    )
    fn = chat_fn or _default_chat_fn(
        model,
        base_url,
        base_url_env=base_url_env,
        api_key_env=api_key_env,
        error_label=f"maker:{benchmark}",
        empty_content_retries=empty_content_retries,
        request_timeout_s=request_timeout_s,
        sdk_exception_retries=sdk_exception_retries,
        debug_dump_dir=debug_dump_dir,
        extra_body=extra_body,
    )
    if seed is None:
        if json_contract_strict:
            raw, parsed = call_json_contract(
                fn,
                system_msg,
                user_msg,
                role=f"maker:{benchmark}",
                attempts=json_contract_max_attempts,
                validate=lambda payload: (
                    _normalize_maker_json_for_validation(payload, vocab_ids, aggregate),
                    _validate_maker_json(payload, vocab_ids, benchmark=benchmark),
                ),
                retry_hint=_MAKER_JSON_RETRY_HINT,
            )
        else:
            raw = fn(system_msg, user_msg)
            parsed = _parse_json(raw)
    else:
        if json_contract_strict:
            raw, parsed = call_json_contract(
                fn,
                system_msg,
                user_msg,
                role=f"maker:{benchmark}",
                attempts=json_contract_max_attempts,
                validate=lambda payload: (
                    _normalize_maker_json_for_validation(payload, vocab_ids, aggregate),
                    _validate_maker_json(payload, vocab_ids, benchmark=benchmark),
                ),
                seed=seed,
                retry_hint=_MAKER_JSON_RETRY_HINT,
            )
        else:
            raw = fn(system_msg, user_msg, seed)
            parsed = _parse_json(raw)
    payload = _coerce_maker_output(
        benchmark=benchmark,
        raw_response=raw,
        parsed=parsed,
        vocab_ids=vocab_ids,
        aggregate=aggregate,
    )
    payload.update({
        "schema_version": schema_version,
        "aggregate_hash": aggregate_hash,
        "prompt_hash": _prompt_hash(prompt),
        "vocab_hash": vocab_hash,
        "model": model,
        "seed": seed,
    })
    payload["_cache_hit"] = False
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in payload.items() if not k.startswith("_")}, f, ensure_ascii=False, indent=2)
    return payload


def run_maker(
    benchmark_names: list[str],
    vocab: list[dict],
    aggregates: dict[str, dict],
    config: dict,
    run_dir: str,
    prompt: str,
    version: int,
    label: str,
    chat_fn: MakerChatFn | None = None,
    *,
    seed: int | None = None,
) -> tuple[dict[str, dict], dict[str, Any]]:
    """v3 §2.2.5 Maker. Apply taxonomy `vocab` to every benchmark's evidence.

    Returns `(outputs, metadata)` where `outputs[benchmark]` is the per-benchmark
    Maker result dict (`benchmark_summary`, `ability_levels`, `ability_rationale`,
    plus cache provenance fields).
    """
    model_cfg = (
        config.get("mapreduce_reducer_model")
        or config.get("maker_model")
        or role_cfg(config, "maker_model")
    )
    model = model_cfg.get("name")
    base_url = model_cfg.get("base_url")
    base_url_env = model_cfg.get("base_url_env")
    api_key_env = model_cfg.get("api_key_env")
    schema_version = int(
        config.get(
            "maker_schema_version",
            config.get("mapreduce_reducer_schema_version", MAKER_SCHEMA_VERSION),
        )
    )
    vocab_sig = _vocab_hash(vocab)
    strict_json = json_contract_enabled(config)
    json_attempts = json_contract_attempts(config)
    empty_retries = llm_empty_content_retries(config)
    request_timeout = llm_request_timeout_s(config)
    sdk_exception_retries = llm_sdk_exception_retries(config)
    debug_dir = llm_debug_dump_dir(config)
    extra_body = llm_extra_body(config)

    outputs: dict[str, dict] = {}
    cache_hits = 0
    counter_lock = threading.Lock()
    max_workers = int(config.get("maker_max_workers", 8))

    # codex 2026-05-10 #4: warm-bind `.chat` cached_property before fan-out so
    # concurrent first-access on the OpenAI client does not race.
    if chat_fn is None:
        from autotagging_loop.experiment.llm_client import shared_factory

        client = shared_factory().get(
            base_url=base_url, base_url_env=base_url_env, api_key_env=api_key_env
        )
        _ = client.chat

    work: list[tuple[str, dict, str, str]] = []  # (benchmark, aggregate, agg_hash, cache_path)
    for benchmark in benchmark_names:
        aggregate = aggregates.get(benchmark)
        if not aggregate:
            continue
        evidence = _evidence_for_maker(aggregate)
        aggregate_hash = _aggregate_signature(evidence)
        cache_key = _maker_cache_key(
            benchmark=benchmark,
            prompt=prompt,
            vocab_hash=vocab_sig,
            model=model,
            aggregate_hash=aggregate_hash,
            schema_version=schema_version,
            seed=seed,
        )
        cache_path = os.path.join(
            _maker_root(run_dir, prompt),
            _slug(benchmark),
            f"{cache_key}.json",
        )
        work.append((benchmark, aggregate, aggregate_hash, cache_path))

    workers = max(1, min(max_workers, len(work))) if work else 1

    def _do(benchmark: str, aggregate: dict, agg_hash: str, cache_path: str) -> tuple[str, dict]:
        applied = _apply_one(
            benchmark=benchmark,
            aggregate=aggregate,
            vocab=vocab,
            prompt=prompt,
            model=model,
            base_url=base_url,
            cache_path=cache_path,
            aggregate_hash=agg_hash,
            vocab_hash=vocab_sig,
            schema_version=schema_version,
            chat_fn=chat_fn,
            base_url_env=base_url_env,
            api_key_env=api_key_env,
            seed=seed,
            json_contract_strict=strict_json,
            json_contract_max_attempts=json_attempts,
            empty_content_retries=empty_retries,
            request_timeout_s=request_timeout,
            sdk_exception_retries=sdk_exception_retries,
            debug_dump_dir=debug_dir,
            extra_body=extra_body,
        )
        return benchmark, applied

    pbar = tqdm(
        total=len(work),
        desc=f"  [{label} maker]",
        unit="bench",
        leave=False,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_do, b, a, h, c) for (b, a, h, c) in work]
        for fut in as_completed(futures):
            benchmark, applied = fut.result()
            outputs[benchmark] = applied
            if applied.get("_cache_hit"):
                with counter_lock:
                    cache_hits += 1
            pbar.set_postfix_str(benchmark[:40])
            pbar.update(1)
    pbar.close()

    # Deterministic dict order regardless of completion order.
    outputs = {k: outputs[k] for k in sorted(outputs)}

    metadata: dict[str, Any] = {
        # Legacy `reducer_*` keys preserved for downstream consumers
        # (loop.py merges them into tag-weight metadata; renaming is deferred to Phase 8).
        "reducer_model": model,
        "reducer_prompt_hash": _prompt_hash(prompt),
        "reducer_prompt_version": version,
        "reducer_schema_version": schema_version,
        "reducer_cache_hits": cache_hits,
        "reducer_cache_misses": len(outputs) - cache_hits,
        "reducer_output_count": len(outputs),
        "maker_model": model,
        "maker_schema_version": schema_version,
        "maker_cache_hits": cache_hits,
        "maker_cache_misses": len(outputs) - cache_hits,
        "maker_output_count": len(outputs),
    }
    print(
        f"  [maker] {label}: maked={len(outputs)}, "
        f"cache_hits={cache_hits}/{len(outputs)}"
    )
    return outputs, metadata
