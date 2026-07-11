"""Strict JSON contract helpers for LLM role outputs."""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable


class JSONContractError(RuntimeError):
    """Raised when an LLM response is not valid JSON for the role contract."""


ValidateFn = Callable[[dict[str, Any]], None]


def _accepts_seed_arg(fn: Callable[..., str]) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return True
    positional = 0
    for param in params:
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            return True
        if param.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            positional += 1
    return positional >= 3


def call_with_optional_seed(
    fn: Callable[..., str],
    system_msg: str,
    user_msg: str,
    seed: int | None = None,
) -> str:
    if seed is not None and _accepts_seed_arg(fn):
        return fn(system_msg, user_msg, seed)
    return fn(system_msg, user_msg)


def json_contract_enabled(config: dict | None = None) -> bool:
    value = (config or {}).get("llm_json_contract_strict", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no"}
    return bool(value)


def json_contract_attempts(config: dict | None = None) -> int:
    try:
        return max(1, int((config or {}).get("llm_json_contract_max_attempts", 3)))
    except (TypeError, ValueError):
        return 3


def parse_json_object_strict(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        raise JSONContractError("empty_response")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JSONContractError(f"invalid_json:{exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise JSONContractError(f"json_not_object:{type(parsed).__name__}")
    return parsed


def call_json_contract(
    fn: Callable[..., str],
    system_msg: str,
    user_msg: str,
    *,
    role: str,
    attempts: int = 1,
    validate: ValidateFn | None = None,
    seed: int | None = None,
    retry_hint: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Call an LLM until it returns a strict JSON object satisfying `validate`."""
    errors: list[str] = []
    max_attempts = max(1, int(attempts))
    for attempt_idx in range(max_attempts):
        active_user_msg = user_msg
        if errors:
            recent = "; ".join(errors[-3:])
            active_user_msg = (
                f"{user_msg}\n\n"
                "STRICT JSON RETRY:\n"
                f"Your previous response failed validation: {recent}.\n"
                "Return one corrected JSON object only. Do not explain. "
                "Do not add ids, keys, or fields outside the requested schema."
            )
            if retry_hint:
                active_user_msg += f"\n{retry_hint.strip()}"
        active_seed = None if seed is None else seed + attempt_idx
        raw = call_with_optional_seed(fn, system_msg, active_user_msg, active_seed)
        try:
            parsed = parse_json_object_strict(raw)
            if validate is not None:
                validate(parsed)
        except JSONContractError as exc:
            errors.append(str(exc))
            continue
        return raw, parsed

    detail = "; ".join(errors[-3:]) if errors else "unknown_contract_error"
    raise JSONContractError(
        f"{role}: JSON contract failed after {max_attempts} attempt(s): {detail}"
    )
