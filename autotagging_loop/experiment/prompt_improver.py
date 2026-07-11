"""experiment/prompt_improver.py — A_imp: 에러 리포트 + 메트릭으로 I_{i+1} 합성.

4 가드를 통과해야 새 프롬프트로 채택. 모두 fail 시 best 프롬프트 유지.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable

from autotagging_loop.experiment.alignment import ErrorPair, error_pairs_to_dicts
from autotagging_loop.experiment.json_contract import (
    JSONContractError,
    call_json_contract,
    call_with_optional_seed,
)


_FORBIDDEN_LITERAL_PATTERNS = [
    r"spearman",
    r"pearson",
    r"\brho\b",
    r"correlation",
    r"\br_raw\b",
    r"\br01\b",
    r"\bρ\b",
]

_FORBIDDEN_TAXONOMY_PATTERNS = [
    r"\badd\s+(?:a\s+)?(?:new\s+)?(?:tag|ability|dimension|category|taxonomy)\b",
    r"\bremove\s+(?:a\s+)?(?:tag|ability|dimension|category)\b",
    r"\bdelete\s+(?:a\s+)?(?:tag|ability|dimension|category)\b",
    r"\brename\s+(?:a\s+)?(?:tag|ability|dimension|category)\b",
    r"\bnew\s+(?:tag|ability|dimension|category|taxonomy)\b",
    r"\badditional\s+(?:tag|ability|dimension|category)\b",
    r"\bextend\s+(?:the\s+)?(?:tag|ability|dimension|category|taxonomy)\b",
    r"\bexpand\s+(?:the\s+)?(?:tag|ability|dimension|category|taxonomy)\b",
    r"\brefine\s+(?:the\s+)?(?:tag\s+list|taxonomy|vocabulary)\b",
]

_FORBIDDEN_SHORTCUT_PATTERNS = [
    r"\bleaderboard\b",
    r"\branking(?:s)?\b",
    r"\bpublic\s+reputation\b",
    r"\bbenchmark\s+reputation\b",
    r"\bmodel\s+performance\b",
    r"\bexpected\s+performance\b",
    r"\bmodel\s+score(?:s)?\b",
    r"\bscore\s+table(?:s)?\b",
    r"\bbenchmark\s+difficulty\b",
    r"\bpublic\s+difficulty\b",
    r"\bfrontier\s+(?:status|difficulty|tier)\b",
    r"\banswer\s+format\b",
    r"\boutput\s+format\b",
    r"\btask\s+format\b",
    r"\bmultiple\s+choice\b",
]

_SCORE_LITERAL_PATTERN = re.compile(r"\b\d+\.\d{2,}\b")

_MAX_PROMPT_CHARS = 3500
_PROMPT_LENGTH_MARGIN = 1100

_PROHIBITION_PATTERN = re.compile(
    r"\b(?:do\s+not|don't|never|must\s+not|shall\s+not|cannot|can'?t|"
    r"forbidden|disallowed|not\s+allowed|avoid|without\s+(?:using|mentioning)|"
    r"not\s+(?:a|an|the)\b)\b"
)

_NEGATED_PROHIBITION_ACTION_PATTERN = re.compile(
    r"\b(?:ignore|ignoring|ignored|disregard|disregarding|disregarded|"
    r"overlook|overlooking|overlooked)\b"
)


def _public_validation_reason(reason: str) -> str:
    if reason.startswith("shortcut_instruction_present:"):
        return "shortcut_instruction_present:surface_or_outcome_cue"
    if reason.startswith("forbidden_label:"):
        return "forbidden_label:alignment_metric_word"
    if reason.startswith("taxonomy_change_requested:"):
        return "taxonomy_change_requested:fixed_vocabulary_phase"
    return reason


def _strip_prohibition_clauses(text: str) -> str:
    """Drop sentence-like clauses that explicitly prohibit a term, so banned-word
    matchers don't false-fire when the prompt is *forbidding* the term itself
    (e.g. 'Do not mention correlation')."""
    parts = re.split(r"(?<=[.!?])\s+|\n+|;", text)
    kept: list[str] = []
    for part in parts:
        lower = part.lower()
        is_prohibition = _PROHIBITION_PATTERN.search(lower)
        has_negated_action = _NEGATED_PROHIBITION_ACTION_PATTERN.search(lower)
        if is_prohibition and not has_negated_action:
            continue
        kept.append(part)
    return " ".join(kept)


@dataclass
class ImproverResult:
    new_prompt: str
    accepted: bool
    reasons: list[str]
    raw_response: str
    rationale: str = ""


ChatFn = Callable[[str, str], str]


def _default_chat_fn(
    model: str,
    base_url: str | None = None,
    *,
    base_url_env: str | None = None,
    api_key_env: str | None = None,
    temperature: float = 0.0,
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
        temperature=temperature,
        response_format={"type": "json_object"},
        error_label="prompt_improver",
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


def _validate_improver_json(parsed: dict) -> None:
    required = {"new_prompt", "rationale"}
    keys = set(parsed)
    missing = sorted(required - keys)
    if missing:
        raise JSONContractError(f"missing_keys:{','.join(missing)}")
    extra = sorted(keys - required)
    if extra:
        raise JSONContractError(f"extra_keys:{','.join(extra)}")
    if not isinstance(parsed.get("new_prompt"), str):
        raise JSONContractError("new_prompt_not_string")
    if not isinstance(parsed.get("rationale"), str):
        raise JSONContractError("rationale_not_string")


def _validate_improver_json_with_length(parsed: dict) -> None:
    _validate_improver_json(parsed)
    new_prompt = str(parsed.get("new_prompt", "")).strip()
    if len(new_prompt) > _MAX_PROMPT_CHARS:
        raise JSONContractError(
            "new_prompt_too_long:"
            f"{len(new_prompt)}>{_MAX_PROMPT_CHARS}:"
            f"rewrite_under_{_MAX_PROMPT_CHARS}_chars"
        )


def _min_rewrite_prompt_chars(base_prompt: str) -> int:
    """Minimum accepted rewrite length after applying the hard prompt cap.

    Some active prompts are longer than ``_MAX_PROMPT_CHARS``. Requiring a
    rewrite to be at least as long as those prompts makes valid compact
    rewrites impossible, because JSON-contract validation rejects candidates
    above the hard cap first.
    """

    base_len = len(str(base_prompt or "").strip())
    capped_target = max(1, _MAX_PROMPT_CHARS - _PROMPT_LENGTH_MARGIN)
    return min(base_len, capped_target)


def validate_prompt(
    new_prompt: str,
    base_prompt: str,
    vocab_ids: list[str],
    benchmark_names: list[str],
    allow_taxonomy_changes: bool = False,
) -> tuple[bool, list[str]]:
    """4 guards. Returns (accepted, reasons-of-rejection)."""
    reasons: list[str] = []

    if not new_prompt:
        reasons.append("empty_new_prompt")
    elif len(new_prompt.strip()) > _MAX_PROMPT_CHARS:
        reasons.append(f"prompt_too_long:{len(new_prompt.strip())}>{_MAX_PROMPT_CHARS}")
    elif base_prompt and len(new_prompt.strip()) < _min_rewrite_prompt_chars(base_prompt):
        reasons.append("shorter_than_I0")

    missing = [tid for tid in vocab_ids if tid not in new_prompt]
    if missing:
        reasons.append(f"vocab_missing:{','.join(missing)}")

    # Case-sensitive word-boundary match. Benchmark names use canonical casing
    # (MATH, MMLU, HellaSwag, AIME 2024); matching case-insensitively would
    # reject common English words ("math problem", "athletes" containing "hle")
    # and freeze v_loop. The intent is to catch actual benchmark references,
    # not generic prose that shares letters with an acronym.
    found_names = [
        b for b in benchmark_names
        if b and re.search(r"\b" + re.escape(b) + r"\b", new_prompt)
    ]
    if found_names:
        reasons.append(f"benchmark_names_present:{','.join(found_names[:5])}")

    sanitized_lower = _strip_prohibition_clauses(new_prompt).lower()

    if _SCORE_LITERAL_PATTERN.search(_strip_prohibition_clauses(new_prompt)):
        reasons.append("score_literal_present")

    if not allow_taxonomy_changes:
        for pat in _FORBIDDEN_TAXONOMY_PATTERNS:
            if re.search(pat, sanitized_lower):
                reasons.append(f"taxonomy_change_requested:{pat}")
                break

    for pat in _FORBIDDEN_LITERAL_PATTERNS:
        if re.search(pat, sanitized_lower):
            reasons.append(f"forbidden_label:{pat}")
            break

    for pat in _FORBIDDEN_SHORTCUT_PATTERNS:
        if re.search(pat, sanitized_lower):
            reasons.append(f"shortcut_instruction_present:{pat}")
            break

    return (len(reasons) == 0, reasons)


def _sanitize_description_for_improver(text: str, benchmark_names: list[str]) -> str:
    """Remove benchmark identifiers before descriptions reach A_imp."""
    cleaned_lines: list[str] = []
    for line in str(text or "").splitlines():
        if line.strip().lower().startswith("benchmark:"):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)
    for name in sorted((b for b in benchmark_names if b), key=len, reverse=True):
        cleaned = re.sub(r"\b" + re.escape(name) + r"\b", "[benchmark]", cleaned)
    cleaned = _SCORE_LITERAL_PATTERN.sub("[number]", cleaned)
    return cleaned.strip()


def _validation_feedback(reasons: list[str]) -> dict[str, Any]:
    """Convert guard failures into concise repair hints for the next sample."""

    hints: list[str] = []
    blocked_terms: list[str] = []
    for reason in reasons:
        if reason == "shorter_than_I0":
            hints.append(
                "Make the rewrite complete enough to preserve role, evidence, scoring, "
                "schema, and vocabulary instructions, but stay near target_prompt_chars."
            )
        elif reason.startswith("shortcut_instruction_present:"):
            blocked_terms.append("surface_or_outcome_cue")
            hints.append(
                "Remove shortcut cue wording entirely. Use neutral phrases such as "
                "'surface presentation cues' or 'external outcome cues' only when "
                "contrasting them with cognitive operations."
            )
        elif reason.startswith("forbidden_label:"):
            blocked_terms.append("alignment_metric_word")
            hints.append(
                "Remove alignment-metric wording entirely; describe only "
                "general cognitive-operation calibration."
            )
        elif reason.startswith("benchmark_names_present:"):
            hints.append(
                "Remove all benchmark names and pair-specific references; keep the "
                "rule applicable to unseen benchmark evidence."
            )
        elif reason.startswith("taxonomy_change_requested:"):
            hints.append(
                "Keep the active vocabulary fixed; do not add, remove, rename, split, "
                "or merge tags in this phase."
            )
        elif reason.startswith("vocab_missing:"):
            hints.append(
                "Mention every supplied vocab id exactly at least once in new_prompt."
            )
        elif reason == "score_literal_present":
            hints.append(
                "Remove numeric outcome-like literals from the prompt."
            )
        elif "new_prompt_too_long" in reason or "prompt_too_long" in reason:
            hints.append(
                "Rewrite shorter, near target_prompt_chars. Do not copy the full "
                "previous_prompt; compress repeated criteria, keep one concise rule "
                "per behavior, and preserve schema plus active vocabulary coverage."
            )
    return {
        "previous_candidate_rejected": True,
        "validation_reasons": [_public_validation_reason(reason) for reason in reasons],
        "blocked_patterns": blocked_terms,
        "repair_hints": list(dict.fromkeys(hints)),
    }


def _loss_bucket(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(v):
        return "unknown"
    if v < 0.03:
        return "low"
    if v < 0.08:
        return "moderate"
    return "high"


def _signed_bucket(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(v):
        return "unknown"
    if v > 0.15:
        return "strong_positive"
    if v > 0.0:
        return "positive"
    if v < -0.15:
        return "strong_negative"
    if v < 0.0:
        return "negative"
    return "neutral"


def _residual_bucket(value: Any) -> str:
    try:
        v = abs(float(value))
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(v):
        return "unknown"
    if v < 0.10:
        return "small"
    if v < 0.30:
        return "medium"
    return "large"


def _error_mode_summary(error_report: list[ErrorPair]) -> dict[str, Any]:
    false_dis = sum(1 for e in error_report if e.type == "false_dis")
    false_sim = sum(1 for e in error_report if e.type == "false_sim")
    if false_dis > false_sim:
        dominant = "false_dis"
        interpretation = (
            "tag_similarity_too_low: paired evidence appears similar but "
            "the prompt separates it too much"
        )
        preferred_fix = (
            "emphasize shared transferable operations across domains, formats, "
            "and difficulty levels; keep directly evidenced common operations "
            "non-absent instead of separating pairs by topic family"
        )
    elif false_sim > false_dis:
        dominant = "false_sim"
        interpretation = (
            "tag_similarity_too_high: paired evidence appears different but "
            "the prompt collapses it too much"
        )
        preferred_fix = "sharpen evidence-grounded distinctions between operations"
    elif false_dis or false_sim:
        dominant = "mixed"
        interpretation = "mixed false_dis and false_sim residual modes"
        preferred_fix = (
            "balance shared-operation calibration with sharper operation definitions"
        )
    else:
        dominant = "none"
        interpretation = "no residual-pair mode supplied"
        preferred_fix = "keep the prompt concise and evidence-grounded"
    return {
        "false_dis_count": false_dis,
        "false_sim_count": false_sim,
        "dominant_error_type": dominant,
        "dominant_interpretation": interpretation,
        "preferred_fix": preferred_fix,
    }


def improve_prompt(
    prev_prompt: str,
    base_prompt: str,
    error_report: list[ErrorPair],
    metrics: dict,
    bench_descriptions: dict[str, str],
    vocab: list[dict],
    benchmark_names: list[str],
    model: str,
    base_url: str | None,
    chat_fn: ChatFn | None = None,
    allow_taxonomy_changes: bool = False,
    base_url_env: str | None = None,
    api_key_env: str | None = None,
    temperature: float = 0.0,
    n_samples: int = 1,
    json_contract_strict: bool = True,
    json_contract_max_attempts: int = 3,
    empty_content_retries: int | None = None,
    request_timeout_s: float | int | None = None,
    sdk_exception_retries: int | None = None,
    debug_dump_dir: str | None = None,
    extra_body: dict | None = None,
    seed: int | None = None,
) -> ImproverResult:
    """Call A_imp + run guards. If guards fail, accepted=False (caller keeps prev).

    With ``n_samples>1`` and ``temperature>0`` the Improver draws multiple
    candidates and returns the first one that (a) passes all guards and
    (b) is not byte-identical to ``prev_prompt`` — required to escape the
    Improver fixed-point observed on 2026-05-11 (`run_20260511_134315`):
    at temperature=0, after iter_2 the Improver kept returning the same
    prompt hash so the loop stalled before exploring better candidates.
    """
    vocab_ids = [v["id"] for v in vocab]

    taxonomy_rule = (
        "5) The current phase keeps the seed cognitive ability vocabulary fixed. Do not add, "
        "remove, rename, split, merge, refine, or expand the tag list or taxonomy. If residual "
        "errors remain, treat them as evidence for a later taxonomy-unlocked phase.\n"
        if not allow_taxonomy_changes
        else
        "5) This is a taxonomy-unlocked phase. Treat the supplied active vocabulary as a "
        "diagnostic seed, not as an exact final list. The rewrite may instruct the next "
        "vocabulary generator to split, merge, retire, or replace seed abilities when "
        "persistent residual modes show over-compression or over-splitting. Do not hard-code "
        "the active ids as the only allowed ids.\n"
    )
    vocab_reference_rule = (
        "2) The new_prompt MUST mention every one of the cognitive ability ids supplied "
        "and keep them as the fixed active ids.\n"
        if not allow_taxonomy_changes
        else
        "2) The new_prompt MUST mention every supplied seed cognitive ability id as "
        "diagnostic context, but must make clear that these ids are revisable examples "
        "rather than an exact final vocabulary.\n"
    )
    active_vocab_rule = (
        "Use the supplied active_vocabulary definitions to preserve each id's intended "
        "operation. Do not change an id into a benchmark-family label or a difficulty label. "
        "If false_sim dominates, tighten level criteria and evidence requirements for the "
        "existing ids; if false_dis dominates, widen cross-domain applicability for directly "
        "evidenced shared operations.\n"
        if not allow_taxonomy_changes
        else
        "Use the supplied active_vocabulary definitions as seed examples of intended "
        "operations, then write taxonomy-generation guidance that can revise them. When "
        "false_sim dominates, tell the next generator to create sharper reusable operation "
        "axes instead of keeping broad seed axes that collapse unlike evidence. When "
        "false_dis dominates, tell it to merge or broaden axes only where the same operation "
        "is directly evidenced across domains. Do not turn any seed id into a benchmark-family "
        "label or a difficulty label.\n"
    )

    system_msg = (
        "You are a prompt-refinement assistant. Your job is to improve a benchmark evidence "
        "extraction/tagging prompt. The prompt guides an LLM to identify cognitive-ability "
        "evidence; final benchmark weights may be computed later by a deterministic reducer.\n\n"
        "STRICT RULES:\n"
        "1) Output JSON: {\"new_prompt\": \"...\", \"rationale\": \"...\"}.\n"
        f"{vocab_reference_rule}"
        f"3) Keep the new_prompt CONCISE. Hard cap: {_MAX_PROMPT_CHARS} characters. "
        f"Target at most {_MAX_PROMPT_CHARS - _PROMPT_LENGTH_MARGIN} characters to leave margin. "
        "Rewrite compactly instead of appending to previous_prompt. Prefer sharper definitions "
        "over verbose ones; trim redundant text from previous_prompt. Lengthening is only "
        "justified when adding a genuinely new, specific guideline derived from the "
        "error_pairs_anonymized signal.\n"
        "4) NEVER mention specific benchmark names, outcome tables, public status cues, "
        "alignment metric names/symbols, or numeric thresholds. State only general rules "
        "about how to identify cognitive-ability evidence for any benchmark. Do not instruct downstream "
        "roles to use reputation, difficulty, model outcome, response-interface, or public-outcome "
        "shortcuts as evidence. In new_prompt, use neutral category phrases such as surface "
        "presentation cues or external outcome cues, and do not include parenthetical examples "
        "of forbidden shortcut words, even in negative examples. Do not copy diagnostic "
        "payload key names or validation category labels into new_prompt.\n"
        f"{taxonomy_rule}"
        "6) Improve the prompt by clarifying the cognitive distinctions surfaced by the error "
        "pairs (without naming the benchmarks). Keep instructions general.\n"
        "7) Interpret error types explicitly: false_dis means the current tag space separates "
        "a pair too much, so prioritize shared-operation calibration; false_sim means the tag "
        "space collapses a pair too much, so sharpen evidence-grounded operation definitions. "
        "If one error type dominates, prioritize that failure mode. Prefer general calibration "
        "rules over exceptions. For false_dis, the safest fix is usually to make existing "
        "ability ids apply across topic families when the same operation is directly evidenced, "
        "not to invent narrower domain-specific criteria.\n"
        f"8) {active_vocab_rule}"
        "9) Do not add pair-specific exceptions, hidden benchmark clusters, or instructions "
        "that would only help the supplied error pairs. Prefer short calibration rules that "
        "a Maker can apply to unseen benchmark evidence. Preserve the caller's schema: do "
        "not hard-code a conflicting JSON format, numeric weights, or outcome-derived criteria. "
        "Prefer distinctions that would stay stable across unseen benchmark subsets and "
        "held-out model subsets, not rules that only explain the current residual sample."
    )

    err_payload = []
    for e in error_report[:10]:
        err_payload.append(
            {
                "type": e.type,
                "residual_bucket": _residual_bucket(e.delta),
                "desc_p": _sanitize_description_for_improver(
                    bench_descriptions.get(e.p, ""), benchmark_names,
                ),
                "desc_q": _sanitize_description_for_improver(
                    bench_descriptions.get(e.q, ""), benchmark_names,
                ),
            }
        )

    user_payload = {
        "base_prompt": base_prompt,
        "previous_prompt": prev_prompt,
        "active_vocabulary": [
            {
                "id": str(item.get("id") or ""),
                "name": str(item.get("name") or item.get("id") or ""),
                "definition": str(item.get("definition") or ""),
            }
            for item in vocab
        ],
        "vocab_ids": vocab_ids,
        "fixed_vocabulary_policy": (
            "This phase holds the seed tag vocabulary fixed. Improve only the prompt instructions; "
            "do not propose taxonomy/tag-list changes. Persistent residuals can trigger a later taxonomy-unlocked phase."
            if not allow_taxonomy_changes
            else
            "This phase is taxonomy-unlocked because fixed-vocabulary residuals remained large."
        ),
        "metrics_summary": {
            "loss_level": _loss_bucket(metrics.get("L_align")),
            "rho_pearson_direction": _signed_bucket(metrics.get("rho_align_pearson")),
            "rho_spearman_direction": _signed_bucket(metrics.get("rho_align_spearman")),
            "delta_tag_direction": _signed_bucket(metrics.get("delta_tag")),
        },
        "error_mode_summary": _error_mode_summary(error_report),
        "error_pairs_anonymized": err_payload,
        "max_prompt_chars": _MAX_PROMPT_CHARS,
        "target_prompt_chars": _MAX_PROMPT_CHARS - _PROMPT_LENGTH_MARGIN,
        "min_prompt_chars": _min_rewrite_prompt_chars(base_prompt),
        "instruction": (
            "Return JSON {\"new_prompt\": \"...\", \"rationale\": \"...\"}. "
            "The new_prompt must be a compact rewrite, not an appended copy. "
            "Aim for target_prompt_chars or less; prompts near max_prompt_chars "
            "are invalid unless every sentence is essential. "
            "It should improve broad-operation calibration and avoid source-specific "
            "or benchmark-specific exceptions. Favor stable cognitive axes over "
            "single-split repairs. Do not copy forbidden shortcut cue "
            "terms into new_prompt; describe cognitive operations instead."
        ),
    }

    fn = chat_fn or _default_chat_fn(
        model,
        base_url,
        base_url_env=base_url_env,
        api_key_env=api_key_env,
        temperature=temperature,
        empty_content_retries=empty_content_retries,
        request_timeout_s=request_timeout_s,
        sdk_exception_retries=sdk_exception_retries,
        debug_dump_dir=debug_dump_dir,
        extra_body=extra_body,
    )

    attempts = max(1, int(n_samples))
    last_raw = ""
    last_parsed: dict = {}
    last_reasons: list[str] = []
    last_new_prompt = ""
    for sample_idx in range(attempts):
        active_payload = dict(user_payload)
        if last_reasons:
            active_payload["previous_candidate_feedback"] = _validation_feedback(last_reasons)
        user_msg = json.dumps(active_payload, ensure_ascii=False)
        sample_seed = (
            None
            if seed is None
            else int(seed) + sample_idx * max(1, json_contract_max_attempts)
        )
        if json_contract_strict:
            try:
                raw, parsed = call_json_contract(
                    fn,
                    system_msg,
                    user_msg,
                    role="prompt_improver",
                    attempts=json_contract_max_attempts,
                    validate=_validate_improver_json_with_length,
                    seed=sample_seed,
                )
            except JSONContractError as exc:
                if "new_prompt_too_long" not in str(exc):
                    raise
                last_reasons = [str(exc)]
                continue
        else:
            raw = call_with_optional_seed(fn, system_msg, user_msg, sample_seed)
            parsed = _parse_json(raw)
        new_prompt = parsed.get("new_prompt", "") or ""
        rationale = parsed.get("rationale", "") or ""
        accepted, reasons = validate_prompt(
            new_prompt,
            base_prompt,
            vocab_ids,
            benchmark_names,
            allow_taxonomy_changes=allow_taxonomy_changes,
        )
        last_raw, last_parsed, last_reasons, last_new_prompt = (
            raw, parsed, reasons, new_prompt,
        )
        if accepted and new_prompt.strip() != prev_prompt.strip():
            return ImproverResult(
                new_prompt=new_prompt,
                accepted=True,
                reasons=reasons,
                raw_response=raw,
                rationale=rationale,
            )

    final_rationale = last_parsed.get("rationale", "") or ""
    return ImproverResult(
        new_prompt=prev_prompt,
        accepted=False,
        reasons=last_reasons or ["duplicate_of_prev_prompt"],
        raw_response=last_raw,
        rationale=final_rationale,
    )
