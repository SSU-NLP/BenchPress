"""No-seed taxonomy induction for ablation experiments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from autotagging_loop.experiment.corpus import Corpus
from autotagging_loop.experiment.prompt_improver import validate_prompt
from autotagging_loop.experiment.taxonomy_refiner import _coerce_vocab, vocab_quality_reasons


ChatFn = Callable[[str, str], str]
_MAX_NO_SEED_PROMPT_CHARS = 3200
_MAKER_SCHEMA_CONFLICT_PATTERNS = [
    r"\bconfidence\s+weight\b",
    r"\bfloat\s+weights?\b",
    r"\bnumeric\s+weights?\b",
    r"\bweights?\s+between\s+0(?:\.0)?\s+and\s+1(?:\.0)?\b",
]


@dataclass
class NoSeedTaxonomyResult:
    vocab: list[dict]
    prompt: str
    accepted: bool
    reasons: list[str]
    raw_response: str
    rationale: str = ""


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
) -> ChatFn:
    from autotagging_loop.experiment.llm_client import shared_factory

    return shared_factory().chat_fn(
        model=model,
        base_url=base_url,
        base_url_env=base_url_env,
        api_key_env=api_key_env,
        response_format={"type": "json_object"},
        error_label="no_seed_taxonomy",
        empty_content_retries=empty_content_retries,
        request_timeout_s=request_timeout_s,
        sdk_exception_retries=sdk_exception_retries,
        debug_dump_dir=debug_dump_dir,
        extra_body=extra_body,
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


def _truncate(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80].rstrip() + "\n[truncated]"


def _anonymize_benchmark_text(text: str, benchmark: str, benchmark_ref: str) -> str:
    text = str(text or "")
    if not text:
        return ""
    text = re.sub(
        r"(?im)^\s*Benchmark\s*:\s*.*$",
        f"Benchmark: {benchmark_ref}",
        text,
    )
    name = str(benchmark or "").strip()
    if name:
        text = re.sub(re.escape(name), benchmark_ref, text, flags=re.IGNORECASE)
    return text


def _benchmark_briefs(
    corpus: Corpus,
    *,
    examples_per_benchmark: int,
    max_chars_per_benchmark: int,
) -> list[dict]:
    briefs: list[dict] = []
    for idx, benchmark in enumerate(corpus.benchmark_names, start=1):
        benchmark_ref = f"benchmark_{idx:03d}"
        document = corpus.documents.get(benchmark, {})
        examples = document.get("prompt_examples") or document.get("examples") or []
        n_examples = max(0, int(examples_per_benchmark))
        per_example_chars = max(500, int(max_chars_per_benchmark) // max(n_examples + 2, 1))
        examples = [
            _truncate(
                _anonymize_benchmark_text(str(example), benchmark, benchmark_ref),
                per_example_chars,
            )
            for example in examples[:n_examples]
        ]
        brief = {
            "benchmark_ref": benchmark_ref,
            "reviewed_rows": document.get("reviewed_rows"),
            "topic_counts": document.get("topic_counts", {}),
            "reasoning_depth_counts": document.get("reasoning_depth_counts", {}),
            "answer_format_counts": document.get("answer_format_counts", {}),
            "description": _truncate(
                _anonymize_benchmark_text(
                    corpus.descriptions.get(benchmark, ""),
                    benchmark,
                    benchmark_ref,
                ),
                per_example_chars,
            ),
            "examples": examples,
        }
        briefs.append(brief)
    return briefs


def _ensure_prompt_lists_vocab(prompt: str, vocab: list[dict]) -> str:
    return _canonical_no_seed_prompt(vocab)


def _definition_snippet(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    evidence_marker = "Observable evidence includes"
    marker_pos = text.lower().find(evidence_marker.lower())
    if marker_pos >= 0:
        text = text[:marker_pos].strip()
    text = text.rstrip(" .;:")
    if not text:
        return ""
    if len(text) <= max_chars:
        return text + "."
    clipped = text[: max_chars - 1].rsplit(" ", 1)[0].rstrip(" .;:")
    return clipped + "."


def _canonical_no_seed_prompt(vocab: list[dict]) -> str:
    vocab_ids = [str(v.get("id") or "") for v in vocab if v.get("id")]
    if not vocab_ids:
        return ""
    ids = ", ".join(vocab_ids)
    per_definition_chars = max(
        80,
        min(220, (_MAX_NO_SEED_PROMPT_CHARS - 900) // max(len(vocab_ids), 1)),
    )
    ability_lines = "\n".join(
        "- "
        + str(item.get("id") or "")
        + ": "
        + _definition_snippet(
            str(item.get("definition") or item.get("name") or item.get("id") or ""),
            per_definition_chars,
        )
        for item in vocab
        if item.get("id")
    )
    prompt = (
        f"Use exactly these cognitive ability tag ids: {ids}.\n"
        "These ids are reusable cognitive operations, not benchmark families, topics, "
        "presentation styles, score proxies, or difficulty labels. For every listed "
        "id, rate only direct operation evidence as absent, weak, medium, strong, or "
        "dominant using the caller's required output schema.\n\n"
        "Calibration rules: use absent only when direct evidence is missing; use weak "
        "for occasional or peripheral evidence; use medium when the operation recurs "
        "and materially contributes; use strong when it is central across many chunks; "
        "use dominant sparingly when most evidence depends on that operation. Do not "
        "create, rename, split, merge, or omit ids. Do not use scores, model information, "
        "benchmark names, public reputation, surface presentation cues, or generic "
        "difficulty as evidence. Different domains can share the same operation; similar "
        "domains can differ when their observable operations differ.\n\n"
        "Ability definitions:\n"
        f"{ability_lines}"
    )
    return prompt[:_MAX_NO_SEED_PROMPT_CHARS].rstrip()


def validate_no_seed_taxonomy(
    vocab: list[dict],
    prompt: str,
    benchmark_names: list[str],
    *,
    min_tags: int = 8,
    max_tags: int = 14,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    vocab_ids = [str(v.get("id") or "") for v in vocab]
    if len(vocab_ids) < int(min_tags):
        reasons.append(f"too_few_tags:{len(vocab_ids)}<{int(min_tags)}")
    if len(vocab_ids) > int(max_tags):
        reasons.append(f"too_many_tags:{len(vocab_ids)}>{int(max_tags)}")
    if len(set(vocab_ids)) != len(vocab_ids):
        reasons.append("duplicate_tag_ids")

    reasons.extend(vocab_quality_reasons(vocab, benchmark_names))

    prompt_ok, prompt_reasons = validate_prompt(
        prompt,
        prompt,
        vocab_ids,
        benchmark_names,
        allow_taxonomy_changes=True,
    )
    if not prompt_ok:
        reasons.extend(prompt_reasons)

    prompt_lower = str(prompt or "").lower()
    for pat in _MAKER_SCHEMA_CONFLICT_PATTERNS:
        if re.search(pat, prompt_lower):
            reasons.append(f"prompt_conflicts_with_maker_schema:{pat}")
            break

    return (len(reasons) == 0, reasons)


def induce_no_seed_taxonomy(
    corpus: Corpus,
    benchmark_names: list[str],
    model: str,
    base_url: str | None,
    *,
    min_tags: int = 8,
    max_tags: int = 14,
    max_attempts: int = 3,
    examples_per_benchmark: int = 3,
    max_chars_per_benchmark: int = 4000,
    chat_fn: ChatFn | None = None,
    base_url_env: str | None = None,
    api_key_env: str | None = None,
    empty_content_retries: int | None = None,
    request_timeout_s: float | int | None = None,
    sdk_exception_retries: int | None = None,
    debug_dump_dir: str | None = None,
    extra_body: dict | None = None,
    seed: int | str | None = None,
) -> NoSeedTaxonomyResult:
    """Induce a cognitive ability taxonomy without using the seed vocabulary."""
    system_msg = (
        "You are designing a no-seed cognitive ability taxonomy for a benchmark-tagging "
        "autotagging_loop.experiment. Infer general reusable ability axes from task evidence only. "
        "Do not use or reference any pre-existing seed taxonomy.\n\n"
        "STRICT RULES:\n"
        "1) Output JSON: {\"vocab\": [...], \"new_prompt\": \"...\", \"rationale\": \"...\"}.\n"
        "2) Each vocab item must be {\"id\": snake_case, \"name\": string, \"definition\": string}.\n"
        f"3) Return between {int(min_tags)} and {int(max_tags)} ability tags.\n"
        "4) Tags must be general cognitive ability dimensions, not benchmark-specific labels, "
        "dataset names, answer formats, or leaderboard clusters.\n"
        "   Do not use answer-format phrases such as multiple-choice, true/false, "
        "short-answer, free-form, answer format, task format, benchmark family, "
        "leaderboard, score, model, hard, easy, or difficulty in vocab ids, names, "
        "definitions, or prompt text.\n"
        "5) Do not mention benchmark names, benchmark_ref ids, scores, model names, rankings, "
        "correlation, Spearman, Pearson, rho, or numeric thresholds in the new prompt.\n"
        "6) The new prompt must instruct the tagger to use only the returned vocab ids and "
        "rate every tag id from benchmark evidence. Do not hard-code top-level JSON keys, "
        "numeric float weights, or a schema that conflicts with the caller's required "
        "output. For the v3 Maker role, ask for ordinal ability levels "
        "(absent, weak, medium, strong, dominant) for every tag id.\n"
        "7) Separate surface task format from reusable reasoning operations. Do not create "
        "axes that only describe topic, modality, answer format, benchmark family, or public "
        "difficulty.\n"
        "8) Prefer broad operations that can make different-looking benchmarks comparable "
        "when evidence shows the same underlying demand, such as careful parsing, retrieval, "
        "decomposition, rule application, algorithmic execution, verification, quantitative "
        "reasoning, or handling negations and irrelevant clauses.\n"
        "9) Every definition must include observable evidence cues so a separate tagging role "
        "can rate the axis from absent to dominant using only benchmark evidence.\n"
        "10) Do not define axes by accuracy, robustness, error resistance, avoiding mistakes, "
        "performance, or outcome quality. If an axis involves misleading wording, define the "
        "operation itself, such as parsing negations, filtering irrelevant clauses, or resolving "
        "ambiguous references."
    )
    payload = {
        "benchmark_evidence_anonymized": _benchmark_briefs(
            corpus,
            examples_per_benchmark=examples_per_benchmark,
            max_chars_per_benchmark=max_chars_per_benchmark,
        ),
        "instruction": (
            "Return a complete no-seed cognitive ability vocabulary and tag-generation prompt. "
            "The axes should be reusable across benchmarks and should avoid task-format labels. "
            "Do not include benchmark names, benchmark_ref ids, raw scores, or model information "
            "in the prompt. Do not isolate simple, knowledge-heavy, math, coding, or reasoning "
            "sources with mostly unique axes when they share directly visible operations with "
            "other sources. Avoid pure difficulty tags; encode difficulty only through "
            "observable operations such as longer dependency chains, stricter constraints, "
            "verification burden, or integration of multiple knowledge pieces. In vocab text and "
            "prompt text, avoid literal surface-format terms; use operation names such as "
            "distractor discrimination, verification burden, or semantic precision instead. "
            "Do not use accuracy, robustness, error resistance, avoiding mistakes, performance, "
            "or outcome quality as vocab axes; describe the observable operation instead."
        ),
    }

    fn = chat_fn or _default_chat_fn(
        model,
        base_url,
        base_url_env=base_url_env,
        api_key_env=api_key_env,
        empty_content_retries=empty_content_retries,
        request_timeout_s=request_timeout_s,
        sdk_exception_retries=sdk_exception_retries,
        debug_dump_dir=debug_dump_dir,
        extra_body=extra_body,
    )
    attempts = max(1, int(max_attempts))
    user_msg = json.dumps(payload, ensure_ascii=False)
    last_result: NoSeedTaxonomyResult | None = None
    for attempt in range(attempts):
        raw = fn(system_msg, user_msg) if seed is None else fn(system_msg, user_msg, seed)
        parsed = _parse_json(raw)
        vocab, vocab_reasons = _coerce_vocab(parsed.get("vocab"))
        prompt = _ensure_prompt_lists_vocab(parsed.get("new_prompt", "") or "", vocab)
        rationale = parsed.get("rationale", "") or ""
        ok, reasons = validate_no_seed_taxonomy(
            vocab,
            prompt,
            benchmark_names,
            min_tags=min_tags,
            max_tags=max_tags,
        )
        reasons = [*vocab_reasons, *reasons]
        accepted = not reasons and ok
        last_result = NoSeedTaxonomyResult(
            vocab=vocab,
            prompt=prompt,
            accepted=accepted,
            reasons=reasons,
            raw_response=raw,
            rationale=rationale,
        )
        if accepted or attempt + 1 >= attempts:
            return last_result
        repair_payload = dict(payload)
        repair_payload["previous_rejection"] = {
            "reasons": reasons,
            "rejected_vocab": vocab,
            "rejected_prompt": prompt,
        }
        repair_payload["repair_instruction"] = (
            "Return corrected JSON only. Preserve useful cognitive operations, but remove "
            "all rejected surface-format, benchmark, score, model, difficulty, and schema "
            "conflict language from vocab ids, names, definitions, and the prompt. Do not "
            "mention multiple-choice, answer format, benchmark_ref ids, scores, models, "
            "rankings, accuracy, robustness, error resistance, outcome quality, or numeric "
            "float weights."
        )
        user_msg = json.dumps(repair_payload, ensure_ascii=False)
    return last_result or NoSeedTaxonomyResult(
        vocab=[],
        prompt="",
        accepted=False,
        reasons=["no_seed_taxonomy_no_attempts"],
        raw_response="",
    )
