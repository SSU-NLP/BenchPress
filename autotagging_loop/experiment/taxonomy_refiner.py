"""Taxonomy-unlocked refinement for residual-heavy Part 1 runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from autotagging_loop.experiment.prompt_improver import validate_prompt


ChatFn = Callable[[str, str], str]


@dataclass
class TaxonomyRefinementResult:
    vocab: list[dict]
    prompt: str
    accepted: bool
    reasons: list[str]
    raw_response: str
    rationale: str = ""


_TAG_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
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
_FORBIDDEN_VOCAB_TOKEN_SEQUENCES = [
    ("multiple", "choice"),
    ("true", "false"),
    ("yes", "no"),
    ("short", "answer"),
    ("free", "form"),
    ("numeric", "answer"),
    ("selected", "option"),
    ("answer", "format"),
    ("output", "format"),
    ("task", "format"),
    ("benchmark", "family"),
    ("dataset", "family"),
    ("domain", "family"),
    ("task", "family"),
    ("source", "specific"),
    ("public", "reputation"),
    ("leaderboard",),
    ("score",),
    ("scores",),
    ("accuracy",),
    ("performance",),
    ("model",),
    ("difficulty",),
    ("hardness",),
    ("frontier",),
    ("hard",),
    ("easy",),
]


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
        error_label="taxonomy_refiner",
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


def _coerce_vocab(raw_vocab: Any) -> tuple[list[dict], list[str]]:
    reasons: list[str] = []
    if not isinstance(raw_vocab, list):
        return [], ["vocab_not_list"]

    out: list[dict] = []
    seen: set[str] = set()
    for idx, item in enumerate(raw_vocab):
        if not isinstance(item, dict):
            reasons.append(f"vocab_item_not_object:{idx}")
            continue
        tag_id = str(item.get("id") or "").strip()
        name = str(item.get("name") or tag_id).strip()
        definition = str(item.get("definition") or "").strip()
        if not tag_id or not _TAG_ID_PATTERN.match(tag_id):
            reasons.append(f"invalid_tag_id:{tag_id or idx}")
            continue
        if tag_id in seen:
            reasons.append(f"duplicate_tag_id:{tag_id}")
            continue
        if not definition:
            reasons.append(f"missing_definition:{tag_id}")
            continue
        seen.add(tag_id)
        coerced = {"id": tag_id, "name": name, "definition": definition}
        if item.get("abbr"):
            coerced["abbr"] = str(item["abbr"]).strip()
        out.append(coerced)
    return out, reasons


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _contains_token_sequence(tokens: list[str], phrase: tuple[str, ...]) -> bool:
    if not phrase or len(tokens) < len(phrase):
        return False
    for idx in range(0, len(tokens) - len(phrase) + 1):
        if tuple(tokens[idx: idx + len(phrase)]) == phrase:
            return True
    return False


def _contains_forbidden_vocab_phrase(tokens: list[str], phrase: tuple[str, ...]) -> bool:
    if not phrase or len(tokens) < len(phrase):
        return False
    for idx in range(0, len(tokens) - len(phrase) + 1):
        if tuple(tokens[idx: idx + len(phrase)]) != phrase:
            continue
        if phrase == ("model",) and idx > 0 and tokens[idx - 1] == "mental":
            continue
        return True
    return False


def _benchmark_token_sequences(benchmark_names: list[str]) -> list[tuple[str, ...]]:
    sequences: set[tuple[str, ...]] = set()
    for name in benchmark_names:
        tokens = _tokens(name)
        if not tokens:
            continue
        if len(tokens) > 1:
            sequences.add(tuple(tokens))
        for token in tokens:
            if len(token) < 3 or token in _GENERIC_BENCHMARK_TOKENS:
                continue
            # Catch acronym/family leaks such as mmlu_reasoning or gpqa_skill
            # while avoiding broad domains like "math" and "code".
            sequences.add((token,))
    return sorted(sequences, key=lambda item: (len(item), item))


def vocab_quality_reasons(
    vocab: list[dict],
    benchmark_names: list[str] | None = None,
) -> list[str]:
    """Return reasons for vocabulary entries that leak non-cognitive axes.

    This is intentionally narrow. It rejects explicit benchmark/dataset,
    answer-format, leaderboard, model-performance, and pure-difficulty axes
    without banning legitimate domain-general abilities such as
    quantitative_reasoning or long_term_knowledge_recall.
    """
    reasons: list[str] = []
    benchmark_sequences = _benchmark_token_sequences(benchmark_names or [])
    for idx, item in enumerate(vocab):
        tag_id = str(item.get("id") or f"item_{idx}")
        text = " ".join(
            str(item.get(key) or "")
            for key in ("id", "name", "definition", "abbr")
        )
        tokens = _tokens(text)
        for phrase in _FORBIDDEN_VOCAB_TOKEN_SEQUENCES:
            if _contains_forbidden_vocab_phrase(tokens, phrase):
                reasons.append(
                    f"vocab_leaks_non_cognitive_axis:{tag_id}:{'_'.join(phrase)}"
                )
                break
        for phrase in benchmark_sequences:
            if _contains_token_sequence(tokens, phrase):
                reasons.append(
                    f"vocab_mentions_benchmark:{tag_id}:{'_'.join(phrase)}"
                )
                break
    return reasons


def validate_refined_taxonomy(
    vocab: list[dict],
    prompt: str,
    seed_vocab: list[dict],
    benchmark_names: list[str],
    *,
    retain_seed_tags: bool = True,
    max_new_tags: int = 4,
    base_prompt: str = "",
) -> tuple[bool, list[str]]:
    """Validate a taxonomy-unlocked proposal before running a second prompt phase."""
    reasons: list[str] = []
    seed_ids = [str(v["id"]) for v in seed_vocab]
    vocab_ids = [str(v["id"]) for v in vocab]

    if retain_seed_tags:
        missing_seed = [tag_id for tag_id in seed_ids if tag_id not in vocab_ids]
        if missing_seed:
            reasons.append(f"seed_tags_missing:{','.join(missing_seed)}")

    if len(vocab_ids) > len(seed_ids) + int(max_new_tags):
        reasons.append(f"too_many_new_tags:{len(vocab_ids) - len(seed_ids)}")

    if len(vocab_ids) < len(seed_ids):
        reasons.append("vocab_smaller_than_seed")

    if len(set(vocab_ids)) != len(vocab_ids):
        reasons.append("duplicate_tag_ids")

    reasons.extend(vocab_quality_reasons(vocab, benchmark_names))

    prompt_ok, prompt_reasons = validate_prompt(
        prompt,
        base_prompt or prompt,
        vocab_ids,
        benchmark_names,
        allow_taxonomy_changes=True,
    )
    if not prompt_ok:
        reasons.extend(prompt_reasons)

    return (len(reasons) == 0, reasons)


def refine_taxonomy(
    seed_vocab: list[dict],
    base_prompt: str,
    best_prompt: str,
    residual_report: list[dict],
    metrics: dict,
    benchmark_names: list[str],
    model: str,
    base_url: str | None,
    *,
    retain_seed_tags: bool = True,
    max_new_tags: int = 4,
    protected_pairs: list[dict] | None = None,
    chat_fn: ChatFn | None = None,
    base_url_env: str | None = None,
    api_key_env: str | None = None,
    empty_content_retries: int | None = None,
    request_timeout_s: float | int | None = None,
    sdk_exception_retries: int | None = None,
    debug_dump_dir: str | None = None,
    extra_body: dict | None = None,
) -> TaxonomyRefinementResult:
    """Use persistent residuals to propose V1 and a matching tag-generation prompt."""
    seed_ids = [v["id"] for v in seed_vocab]
    system_msg = (
        "You are a taxonomy-refinement assistant for a benchmark-tagging experiment.\n\n"
        "The fixed seed vocabulary has already been pushed through prompt refinement. "
        "Residual mismatches remain, so this taxonomy-unlocked phase may refine definitions "
        "and add a small number of new cognitive ability tags.\n\n"
        "STRICT RULES:\n"
        "1) Output JSON: {\"vocab\": [...], \"new_prompt\": \"...\", \"rationale\": \"...\"}.\n"
        "2) Each vocab item must be {\"id\": snake_case, \"name\": string, \"definition\": string}.\n"
        f"3) Retain the seed tags unless explicitly impossible: {', '.join(seed_ids)}.\n"
        f"4) Add at most {int(max_new_tags)} new tags.\n"
        "5) Do not mention specific benchmark names, scores, leaderboard rankings, correlation, "
        "Spearman, Pearson, rho, or numeric thresholds in the new prompt.\n"
        "6) The new prompt must instruct the active tagging role to use only the returned "
        "vocab ids. For the v3 Maker role, request ordinal ability_levels for every tag id; "
        "do not hard-code a conflicting JSON schema inside the prompt body.\n"
        "7) New or revised tags must describe reusable cognitive operations visible in the "
        "evidence, not benchmark families, topical domains, answer formats, leaderboard "
        "difficulty, or frontier-status labels.\n"
        "8) Prefer tightening definitions and level criteria for existing seed tags before "
        "adding new tags. Add a tag only when persistent residuals expose a repeated "
        "operation that the seed vocabulary cannot express.\n"
        "9) Definitions must include observable evidence cues so the downstream Maker can "
        "rate absent/weak/medium/strong/dominant from benchmark evidence alone.\n"
        "10) Do not break protected high-similarity pairs by splitting shared operations "
        "into domain-specific variants; preserve cross-domain operations when directly "
        "evidenced."
    )
    user_msg = json.dumps(
        {
            "seed_vocab": seed_vocab,
            "base_prompt": base_prompt,
            "best_fixed_vocab_prompt": best_prompt,
            "metrics": {
                "L_align": metrics.get("L_align"),
                "rho_align_pearson": metrics.get("rho_align_pearson"),
                "rho_align_spearman": metrics.get("rho_align_spearman"),
                "delta_tag": metrics.get("delta_tag"),
                "residual_mean": metrics.get("residual_mean"),
                "residual_max": metrics.get("residual_max"),
            },
            "largest_residuals": residual_report[:10],
            "protected_high_similarity_pairs": (protected_pairs or [])[:10],
            "instruction": (
                "Return a refined or extended vocabulary and a complete tag-generation prompt. "
                "Reduce the largest residuals without artificially pushing apart the protected "
                "high score-pattern similarity pairs. The taxonomy result will only be adopted "
                "if it passes the overall alignment gate. Do not include benchmark names or raw "
                "scores in the prompt. Treat domain mismatch as weak evidence: if different "
                "domains share an operation such as parsing, retrieval, decomposition, "
                "rule-constrained execution, verification, or quantitative reasoning, keep that "
                "operation as a shared axis instead of creating benchmark-family axes. Avoid "
                "pure difficulty tags; express difficulty only through observable operations."
            ),
        },
        ensure_ascii=False,
    )

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
    raw = fn(system_msg, user_msg)
    parsed = _parse_json(raw)
    vocab, vocab_reasons = _coerce_vocab(parsed.get("vocab"))
    prompt = parsed.get("new_prompt", "") or ""
    rationale = parsed.get("rationale", "") or ""
    ok, reasons = validate_refined_taxonomy(
        vocab,
        prompt,
        seed_vocab,
        benchmark_names,
        retain_seed_tags=retain_seed_tags,
        max_new_tags=max_new_tags,
        base_prompt=base_prompt,
    )
    reasons = [*vocab_reasons, *reasons]
    accepted = not reasons and ok
    return TaxonomyRefinementResult(
        vocab=vocab if accepted else seed_vocab,
        prompt=prompt if accepted else best_prompt,
        accepted=accepted,
        reasons=reasons,
        raw_response=raw,
        rationale=rationale,
    )
