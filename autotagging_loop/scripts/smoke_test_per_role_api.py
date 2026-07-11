"""scripts/smoke_test_per_role_api.py — Phase 1 사전 sanity check.

`benchpress_config.json` 의 역할별 모델 항목 (mapper / executer / maker / improver) 을 읽어
각각 `base_url_env` / `api_key_env` 를 env 에서 해석한 뒤, 최소 chat completion 호출로
endpoint 가 살아 있는지 본다. 키 값은 절대 출력하지 않는다 — prefix(앞 4자) + *** 만.

성공 조건: 모든 역할이 200 응답 + 비어 있지 않은 텍스트.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "benchpress_config.json")

ROLE_KEYS = ["mapper_model", "executer_model", "maker_model", "improver_model"]

PROMPT_USER = "Reply with the single token: ok"
PROMPT_SYS = "You are a smoke test responder. Reply with 'ok' and nothing else."


def _redact(value: str | None) -> str:
    if not value:
        return "<missing>"
    return value[:4] + "***" + f"(len={len(value)})"


def _resolve(entry: dict[str, Any]) -> tuple[str | None, str | None, str, str]:
    name = entry.get("name", "")
    base_url_env = entry.get("base_url_env")
    api_key_env = entry.get("api_key_env")
    base_url = os.environ.get(base_url_env) if base_url_env else None
    api_key = os.environ.get(api_key_env) if api_key_env else None
    return base_url, api_key, name, f"base_url_env={base_url_env} api_key_env={api_key_env}"


def _ping(role: str, entry: dict[str, Any]) -> dict[str, Any]:
    base_url, api_key, model_name, env_desc = _resolve(entry)
    result = {
        "role": role,
        "model": model_name,
        "base_url": base_url,
        "api_key_redacted": _redact(api_key),
        "env": env_desc,
    }

    if not base_url or not api_key:
        result["status"] = "config_missing"
        result["error"] = f"base_url={base_url!r} api_key_present={bool(api_key)}"
        return result
    if api_key in {"REPLACE_ME", "replace-me"} or base_url == "REPLACE_ME":
        result["status"] = "config_placeholder"
        result["error"] = "still has REPLACE_ME placeholder"
        return result

    kwargs = {"api_key": api_key, "base_url": base_url, "timeout": 120.0, "max_retries": 0}
    client = OpenAI(**kwargs)

    try:
        models = client.models.list()
        served = [m.id for m in (models.data or [])][:10]
        result["served_models_sample"] = served
        result["model_match"] = model_name in served
    except Exception as exc:
        result["models_list_error"] = f"{type(exc).__name__}: {exc}"

    started = time.time()
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": PROMPT_SYS},
                {"role": "user", "content": PROMPT_USER},
            ],
            max_tokens=512,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        elapsed_ms = int((time.time() - started) * 1000)
        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        result["status"] = "ok" if text else "empty_response"
        result["latency_ms"] = elapsed_ms
        result["response_text"] = text[:120]
        result["finish_reason"] = resp.choices[0].finish_reason
        if reasoning:
            result["reasoning_len"] = len(reasoning)
        if hasattr(resp, "usage") and resp.usage:
            result["usage"] = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["latency_ms"] = int((time.time() - started) * 1000)
    return result


def main() -> int:
    if not os.path.exists(ENV_PATH):
        print(f"[smoke] missing {ENV_PATH}", file=sys.stderr)
        return 2
    if not os.path.exists(CONFIG_PATH):
        print(f"[smoke] missing {CONFIG_PATH}", file=sys.stderr)
        return 2

    load_dotenv(ENV_PATH, override=False)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    section = cfg.get("experiment", {})

    rows: list[dict[str, Any]] = []
    for role in ROLE_KEYS:
        entry = section.get(role)
        if not entry:
            rows.append({"role": role, "status": "missing_config_key"})
            continue
        rows.append(_ping(role, entry))

    print(json.dumps(rows, indent=2, ensure_ascii=False))

    failures = [r for r in rows if r.get("status") != "ok"]
    if failures:
        print(f"\n[smoke] FAIL — {len(failures)}/{len(rows)} role(s) failed", file=sys.stderr)
        return 1
    print(f"\n[smoke] PASS — all {len(rows)} roles responded", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
