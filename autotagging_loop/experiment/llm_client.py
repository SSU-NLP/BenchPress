"""experiment/llm_client.py — Phase 1 per-role API isolation + Phase A concurrency caps.

`LLMClientFactory` 가 `(base_url, api_key_env)` 키로 OpenAI 클라이언트를 캐시하고,
`chat_fn(...)` 으로 모듈별 호출에 쓸 closure 를 만든다.

신호 흐름:
- `make_openai_kwargs(base_url, base_url_env, api_key_env)` 가 URL/key 를 .env / 인자에서 해석
- 같은 `(resolved_url, api_key_env_name)` 짝은 같은 OpenAI 인스턴스를 공유 (커넥션 재사용)
- `configure_limit(...)` 로 endpoint 별 동시성 한도(`Semaphore`)를 등록하면 `chat_fn` closure
  가 자동으로 그 endpoint 키의 semaphore 안에서 LLM 호출을 수행한다.
- closure 시그니처: `(system: str, user: str, seed: int | str | None = None) -> str`
  - 5개 모듈은 2-positional 로 호출 (tag_generator 만 3-positional `seed_str`)
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import queue
import re
import threading
import time
import uuid
from typing import Any, Callable

import httpx
from openai import OpenAI

from autotagging_loop.experiment.config import make_openai_kwargs

ChatFn = Callable[..., str]
EndpointKey = tuple[str | None, str | None]
_REAL_OPENAI_CLASS = OpenAI

# Per-request HTTP timeout (seconds). Without this, a half-dead TCP connection
# (CLOSE_WAIT on the remote, no FIN propagated) leaves `socket.recv` blocked
# in `poll()` indefinitely — a real-run incident on 2026-05-11 stalled the
# v3 main loop for >30 min mid-iter. 300s is well above any legitimate
# completion time while still bounded. SDK internal retries default to zero so
# stalls/errors surface through our logged retry/fallback path instead.
_DEFAULT_LLM_TIMEOUT_S = float(os.getenv("BENCHPRESS_LLM_TIMEOUT_S", "300"))
_DEFAULT_LLM_MAX_RETRIES = int(os.getenv("BENCHPRESS_LLM_MAX_RETRIES", "0"))
_DEFAULT_LLM_EXCEPTION_RETRIES = int(os.getenv("BENCHPRESS_LLM_EXCEPTION_RETRIES", "1"))
_DEFAULT_LLM_RETRY_COOLDOWN_S = float(os.getenv("BENCHPRESS_LLM_RETRY_COOLDOWN_S", "15"))
_DEFAULT_LLM_RETRY_COOLDOWN_MAX_S = float(
    os.getenv("BENCHPRESS_LLM_RETRY_COOLDOWN_MAX_S", "120")
)
_DEFAULT_FAIL_ON_FALLBACK = os.getenv(
    "BENCHPRESS_FAIL_ON_LLM_FALLBACK", "1"
).strip().lower() not in {"0", "false", "no"}

# Empty-content retry: reasoning models (K2.6 etc.) occasionally finish a
# completion with reasoning populated but `message.content` empty. Same prompt
# re-issued normally returns content. Bounded retries before falling back.
_DEFAULT_EMPTY_RETRIES = int(os.getenv("BENCHPRESS_EMPTY_CONTENT_RETRIES", "2"))

# Process-wide counter of LLM-call failures that fell through to `error_fallback`
# (e.g. Cloudflare 524 from a self-hosted backend after retries are exhausted).
# Without this counter, fallbacks silently degrade Mapper/Maker/Improver evidence
# and the run finishes with no visible signal that anything went wrong.
_FALLBACK_LOCK = threading.Lock()
_FALLBACK_COUNTS: dict[str, int] = {}
_DEBUG_DUMP_LOCK = threading.Lock()


class LLMFallbackError(RuntimeError):
    """Raised when an LLM call exhausts retries and fallback output is disabled."""


def _bounded_delay(seconds: float | int | None) -> float:
    try:
        delay = float(seconds)
    except (TypeError, ValueError):
        return 0.0
    if delay <= 0:
        return 0.0
    return min(delay, _DEFAULT_LLM_RETRY_COOLDOWN_MAX_S)


def _http_timeout(seconds: float | int | None) -> httpx.Timeout | None:
    if seconds is None:
        return None
    try:
        timeout = float(seconds)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return httpx.Timeout(timeout)


def _header_retry_after(headers: Any) -> float:
    if not headers:
        return 0.0
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        return 0.0
    return _bounded_delay(value)


def _exception_status_code(exc: Exception) -> int | None:
    for obj in (exc, getattr(exc, "response", None)):
        for attr in ("status_code", "status"):
            try:
                value = getattr(obj, attr)
            except Exception:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    text = str(exc)
    match = re.search(r"\b(?:status|Error code)\D{0,20}(\d{3})\b", text)
    if match:
        return int(match.group(1))
    return None


def _exception_retry_after(exc: Exception) -> float:
    delay = _header_retry_after(getattr(getattr(exc, "response", None), "headers", None))
    if delay:
        return delay
    match = re.search(r"['\"]?retry_after['\"]?\s*:\s*([0-9]+(?:\.[0-9]+)?)", str(exc))
    return _bounded_delay(match.group(1) if match else None)


def _retry_delay_for_exception(exc: Exception, attempt: int) -> float:
    status = _exception_status_code(exc)
    if status is None:
        exc_name = type(exc).__name__.lower()
        if "timeout" in exc_name or "connection" in exc_name:
            return _bounded_delay(_DEFAULT_LLM_RETRY_COOLDOWN_S * (2**attempt))
        return 0.0
    if status not in {408, 409, 429, 500, 502, 503, 504, 524}:
        return 0.0
    return _exception_retry_after(exc) or _bounded_delay(
        _DEFAULT_LLM_RETRY_COOLDOWN_S * (2**attempt)
    )


class _ResponseProxy:
    def __init__(self, payload: Any) -> None:
        self._payload = payload if isinstance(payload, dict) else {}

    def model_dump(self) -> dict[str, Any]:
        return dict(self._payload)

    def __getattr__(self, name: str) -> Any:
        if name not in self._payload:
            raise AttributeError(name)
        value = self._payload[name]
        if name == "choices" and isinstance(value, list):
            return [_ResponseProxy(v) for v in value]
        if name == "message" and isinstance(value, dict):
            return _ResponseProxy(value)
        return value


def _subprocess_entry(target: Callable[..., Any], args: tuple[Any, ...], out_q: Any) -> None:
    try:
        out_q.put(("ok", target(*args)))
    except BaseException as exc:
        out_q.put(("err", type(exc).__name__, str(exc)))


def _run_with_process_deadline(
    target: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    timeout_s: float,
) -> Any:
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_subprocess_entry, args=(target, args, out_q))
    proc.daemon = True
    proc.start()
    deadline = time.monotonic() + float(timeout_s)
    while proc.is_alive() and time.monotonic() < deadline:
        time.sleep(0.1)
    if proc.is_alive():
        proc.terminate()
        stop_deadline = time.monotonic() + 2.0
        while proc.is_alive() and time.monotonic() < stop_deadline:
            time.sleep(0.1)
        if proc.is_alive():
            proc.kill()
            kill_deadline = time.monotonic() + 2.0
            while proc.is_alive() and time.monotonic() < kill_deadline:
                time.sleep(0.1)
        raise TimeoutError(f"LLM call exceeded hard deadline ({timeout_s:.1f}s)")
    proc.join(0)
    try:
        status, *payload = out_q.get(timeout=1.0)
    except queue.Empty as exc:
        raise RuntimeError(f"LLM subprocess exited without result (exitcode={proc.exitcode})") from exc
    if status == "ok":
        return payload[0]
    error_type, message = payload
    raise RuntimeError(f"LLM subprocess {error_type}: {message}")


def _openai_create_payload(
    client_kwargs: dict[str, Any],
    create_kwargs: dict[str, Any],
    timeout_s: float,
    max_retries: int,
) -> Any:
    child_create_kwargs = dict(create_kwargs)
    if "timeout" in child_create_kwargs:
        child_create_kwargs["timeout"] = _http_timeout(timeout_s)
    child = _REAL_OPENAI_CLASS(
        timeout=_http_timeout(timeout_s),
        max_retries=max_retries,
        **client_kwargs,
    )
    response = child.chat.completions.create(**child_create_kwargs)
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return _safe_model_dump(response)


def _create_with_hard_deadline(
    create_fn: Callable[..., Any],
    kwargs: dict[str, Any],
    *,
    timeout_s: float | None,
    client_kwargs: dict[str, Any] | None = None,
    use_subprocess: bool = False,
) -> Any:
    """Bound main-thread SDK calls even when socket read timeout is ineffective."""
    if (
        timeout_s is None
        or timeout_s <= 0
        or not use_subprocess
        or client_kwargs is None
    ):
        return create_fn(**kwargs)
    child_kwargs = dict(kwargs)
    if "timeout" in child_kwargs:
        child_kwargs["timeout"] = float(timeout_s)
    payload = _run_with_process_deadline(
        _openai_create_payload,
        (dict(client_kwargs), child_kwargs, float(timeout_s), _DEFAULT_LLM_MAX_RETRIES),
        timeout_s=float(timeout_s),
    )
    return _ResponseProxy(payload)


def llm_fallback_counts() -> dict[str, int]:
    """Return a snapshot of per-error_label fallback counts (e.g. {'mapper': 3})."""
    with _FALLBACK_LOCK:
        return dict(_FALLBACK_COUNTS)


def reset_llm_fallback_counts() -> None:
    with _FALLBACK_LOCK:
        _FALLBACK_COUNTS.clear()


def _reasoning_text(message: Any) -> str:
    """Pull text out of OpenRouter reasoning fields.

    OpenRouter documents `reasoning_content` as an alias for `reasoning`, and
    some providers put both fields under `provider_specific_fields`.
    """
    try:
        d = message.model_dump() if hasattr(message, "model_dump") else dict(message)
    except Exception:
        return ""
    parts: list[str] = []

    def _add_text(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            parts.append(value)

    def _add_details(value: Any) -> None:
        if isinstance(value, list):
            for it in value:
                if isinstance(it, dict):
                    _add_text(it.get("text"))

    _add_text(d.get("reasoning"))
    _add_text(d.get("reasoning_content"))
    _add_details(d.get("reasoning_details"))

    provider_fields = d.get("provider_specific_fields")
    if isinstance(provider_fields, dict):
        _add_text(provider_fields.get("reasoning"))
        _add_text(provider_fields.get("reasoning_content"))
        _add_details(provider_fields.get("reasoning_details"))

    # Preserve older behavior: when no direct reasoning string is available,
    # concatenate reasoning_details text.
    if not parts:
        details = d.get("reasoning_details") or []
        if isinstance(details, list):
            parts = [it.get("text", "") for it in details if isinstance(it, dict)]
    return "\n".join(p for p in parts if p) or ""


def _extract_balanced_json(text: str) -> str:
    """Return the first brace-balanced `{...}` substring that parses as JSON.

    Used as a fallback when a reasoning model emits its final answer inside
    the reasoning trace and leaves `message.content` empty.
    """
    if not text:
        return ""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return ""


def _safe_model_dump(obj: Any) -> Any:
    try:
        if hasattr(obj, "model_dump"):
            dumped = obj.model_dump()
            if isinstance(dumped, (dict, list, tuple, str, int, float, bool)) or dumped is None:
                return dumped
            return repr(dumped)
        if hasattr(obj, "dict"):
            dumped = obj.dict()
            if isinstance(dumped, (dict, list, tuple, str, int, float, bool)) or dumped is None:
                return dumped
            return repr(dumped)
        if isinstance(obj, (dict, list, tuple, str, int, float, bool)) or obj is None:
            return obj
        return repr(obj)
    except Exception as exc:
        return {"_dump_error": f"{type(exc).__name__}: {exc}", "_repr": repr(obj)}


def _truncate_debug_value(value: Any, *, max_str: int = 4000, depth: int = 0) -> Any:
    if depth > 8:
        return "<max_depth>"
    if isinstance(value, str):
        if len(value) <= max_str:
            return value
        return value[:max_str] + f"...<truncated {len(value) - max_str} chars>"
    if isinstance(value, dict):
        return {
            str(k): _truncate_debug_value(v, max_str=max_str, depth=depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        out = [
            _truncate_debug_value(v, max_str=max_str, depth=depth + 1)
            for v in list(value)[:20]
        ]
        if len(value) > 20:
            out.append(f"...<truncated {len(value) - 20} items>")
        return out
    return value


def _message_fingerprint(text: str) -> dict[str, Any]:
    payload = (text or "").encode("utf-8")
    return {
        "sha256": hashlib.sha256(payload).hexdigest()[:16],
        "chars": len(text or ""),
    }


def _write_debug_dump(
    *,
    debug_dump_dir: str | None,
    error_label: str,
    model: str,
    endpoint_key: EndpointKey,
    attempt: int,
    max_attempts: int,
    reason: str,
    request_kwargs: dict[str, Any],
    response: Any = None,
    message: Any = None,
    exception: Exception | None = None,
) -> None:
    if not debug_dump_dir:
        return
    try:
        system_msg = ""
        user_msg = ""
        for item in request_kwargs.get("messages", []) or []:
            if item.get("role") == "system":
                system_msg = str(item.get("content") or "")
            elif item.get("role") == "user":
                user_msg = str(item.get("content") or "")
        payload = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "error_label": error_label,
            "model": model,
            "endpoint": {
                "base_url": endpoint_key[0],
                "api_key_env": endpoint_key[1],
            },
            "attempt": attempt + 1,
            "max_attempts": max_attempts,
            "reason": reason,
            "request": {
                "model": request_kwargs.get("model"),
                "temperature": request_kwargs.get("temperature"),
                "response_format": request_kwargs.get("response_format"),
                "seed": request_kwargs.get("seed"),
                "system": _message_fingerprint(system_msg),
                "user": _message_fingerprint(user_msg),
            },
            "response": _truncate_debug_value(_safe_model_dump(response)),
            "message": _truncate_debug_value(_safe_model_dump(message)),
            "exception": (
                {
                    "type": type(exception).__name__,
                    "message": str(exception),
                }
                if exception is not None
                else None
            ),
        }
        safe_label = re.sub(r"[^A-Za-z0-9_.:-]+", "_", error_label)[:80] or "llm"
        filename = (
            f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}_"
            f"{safe_label}_attempt{attempt + 1}_{reason}.json"
        )
        path = os.path.join(debug_dump_dir, safe_label, filename)
        with _DEBUG_DUMP_LOCK:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  [{error_label}] wrote LLM debug dump: {path}")
    except Exception as exc:
        print(f"  [{error_label}] failed to write LLM debug dump: {type(exc).__name__}: {exc}")


class LLMClientFactory:
    """Cache OpenAI clients keyed by `(resolved_base_url, api_key_env_name)`.

    Phase A: per-endpoint `threading.Semaphore` for bounded concurrency. The
    cache itself is guarded by a `Lock` so concurrent cold-starts do not
    construct two `OpenAI` instances for the same endpoint.
    """

    def __init__(self) -> None:
        self._cache: dict[EndpointKey, OpenAI] = {}
        self._cache_lock = threading.Lock()
        self._sem_cache: dict[EndpointKey, threading.Semaphore] = {}
        self._sem_caps: dict[EndpointKey, int] = {}
        self._cooldown_until: dict[EndpointKey, float] = {}

    @staticmethod
    def _endpoint_key(
        resolved_base_url: str | None, api_key_env: str | None
    ) -> EndpointKey:
        return (resolved_base_url, api_key_env or "_legacy")

    def get(
        self,
        *,
        base_url: str | None = None,
        base_url_env: str | None = None,
        api_key_env: str | None = None,
    ) -> OpenAI:
        kwargs = make_openai_kwargs(
            base_url=base_url, base_url_env=base_url_env, api_key_env=api_key_env
        )
        cache_key = self._endpoint_key(kwargs.get("base_url"), api_key_env)
        client = self._cache.get(cache_key)
        if client is None:
            with self._cache_lock:
                client = self._cache.get(cache_key)
                if client is None:
                    client = OpenAI(
                        timeout=_http_timeout(_DEFAULT_LLM_TIMEOUT_S),
                        max_retries=_DEFAULT_LLM_MAX_RETRIES,
                        **kwargs,
                    )
                    self._cache[cache_key] = client
        return client

    def configure_limit(
        self,
        *,
        base_url: str | None = None,
        base_url_env: str | None = None,
        api_key_env: str | None = None,
        max_concurrent: int,
    ) -> None:
        """Cap concurrent in-flight requests for an endpoint.

        Idempotent. If two roles share an endpoint and declare different caps,
        the **minimum** wins (logged once per endpoint).
        """
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be > 0")
        kwargs = make_openai_kwargs(
            base_url=base_url, base_url_env=base_url_env, api_key_env=api_key_env
        )
        cache_key = self._endpoint_key(kwargs.get("base_url"), api_key_env)
        with self._cache_lock:
            existing = self._sem_caps.get(cache_key)
            cap = max_concurrent if existing is None else min(existing, max_concurrent)
            if existing is not None and cap != existing:
                print(
                    f"  [llm_client] endpoint {cache_key[0]} cap lowered "
                    f"{existing} -> {cap} (min of declared role caps)"
                )
            if cap == existing and cache_key in self._sem_cache:
                return
            self._sem_caps[cache_key] = cap
            self._sem_cache[cache_key] = threading.Semaphore(cap)

    def _semaphore_for(self, cache_key: EndpointKey) -> threading.Semaphore | None:
        return self._sem_cache.get(cache_key)

    def _wait_for_cooldown(self, cache_key: EndpointKey) -> None:
        with self._cache_lock:
            delay = self._cooldown_until.get(cache_key, 0.0) - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def _record_cooldown(
        self, cache_key: EndpointKey, delay: float, error_label: str
    ) -> None:
        if delay <= 0:
            return
        until = time.monotonic() + delay
        with self._cache_lock:
            if until <= self._cooldown_until.get(cache_key, 0.0):
                return
            self._cooldown_until[cache_key] = until
        print(
            f"  [llm_client] endpoint {cache_key[0]} cooldown {delay:.1f}s "
            f"after {error_label}"
        )

    def chat_fn(
        self,
        *,
        model: str,
        base_url: str | None = None,
        base_url_env: str | None = None,
        api_key_env: str | None = None,
        temperature: float = 0.0,
        response_format: dict | None = None,
        error_label: str = "llm",
        error_fallback: str = "{}",
        fail_on_fallback: bool | None = None,
        empty_content_retries: int | None = None,
        request_timeout_s: float | int | None = None,
        sdk_exception_retries: int | None = None,
        debug_dump_dir: str | None = None,
        **passthrough: Any,
    ) -> ChatFn:
        client = self.get(
            base_url=base_url, base_url_env=base_url_env, api_key_env=api_key_env
        )
        # codex 2026-05-10 #2: warm-bind `.chat` cached_property before any
        # downstream thread fan-out so concurrent first-access does not race.
        _ = client.chat
        kwargs = make_openai_kwargs(
            base_url=base_url, base_url_env=base_url_env, api_key_env=api_key_env
        )
        endpoint_key = self._endpoint_key(kwargs.get("base_url"), api_key_env)
        sem_lookup = self._semaphore_for
        use_subprocess_deadline = isinstance(client, _REAL_OPENAI_CLASS)

        def call(system_msg: str, user_msg: str, seed: int | str | None = None) -> str:
            base_kwargs: dict[str, Any] = {
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            }
            if response_format is not None:
                base_kwargs["response_format"] = response_format
            seed_int: int | None = None
            if seed is not None and seed != "":
                try:
                    seed_int = int(seed)
                except (TypeError, ValueError):
                    seed_int = None
            base_kwargs.update({k: v for k, v in passthrough.items() if v is not None})
            timeout = _http_timeout(request_timeout_s)
            hard_timeout_s = _DEFAULT_LLM_TIMEOUT_S
            if timeout is not None:
                base_kwargs["timeout"] = timeout
                hard_timeout_s = float(timeout.read)
            sem = sem_lookup(endpoint_key)
            should_fail_on_fallback = (
                _DEFAULT_FAIL_ON_FALLBACK
                if fail_on_fallback is None
                else bool(fail_on_fallback)
            )

            try:
                retries = (
                    _DEFAULT_EMPTY_RETRIES
                    if empty_content_retries is None
                    else max(0, int(empty_content_retries))
                )
            except (TypeError, ValueError):
                retries = _DEFAULT_EMPTY_RETRIES
            try:
                exception_retries = (
                    _DEFAULT_LLM_EXCEPTION_RETRIES
                    if sdk_exception_retries is None
                    else max(0, int(sdk_exception_retries))
                )
            except (TypeError, ValueError):
                exception_retries = _DEFAULT_LLM_EXCEPTION_RETRIES
            max_attempts = 1 + max(retries, exception_retries)
            last_exc: Exception | None = None
            sdk_exception_count = 0
            for attempt in range(max_attempts):
                create_kwargs = dict(base_kwargs)
                # Perturb seed on retry so a sticky-cached provider response
                # cannot keep returning the same empty content.
                if seed_int is not None:
                    create_kwargs["seed"] = seed_int + attempt
                try:
                    self._wait_for_cooldown(endpoint_key)
                    if sem is not None:
                        with sem:
                            resp = _create_with_hard_deadline(
                                client.chat.completions.create,
                                create_kwargs,
                                timeout_s=hard_timeout_s,
                                client_kwargs=kwargs,
                                use_subprocess=use_subprocess_deadline,
                            )
                    else:
                        resp = _create_with_hard_deadline(
                            client.chat.completions.create,
                            create_kwargs,
                            timeout_s=hard_timeout_s,
                            client_kwargs=kwargs,
                            use_subprocess=use_subprocess_deadline,
                        )
                except Exception as exc:
                    last_exc = exc
                    sdk_exception_count += 1
                    _write_debug_dump(
                        debug_dump_dir=debug_dump_dir,
                        error_label=error_label,
                        model=model,
                        endpoint_key=endpoint_key,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        reason="sdk_exception",
                        request_kwargs=create_kwargs,
                        exception=exc,
                    )
                    if sdk_exception_count > exception_retries:
                        break
                    retry_delay = _retry_delay_for_exception(exc, attempt)
                    if retry_delay:
                        self._record_cooldown(endpoint_key, retry_delay, error_label)
                    continue
                choices = getattr(resp, "choices", None) or []
                if not choices:
                    last_exc = ValueError("LLM response missing choices")
                    _write_debug_dump(
                        debug_dump_dir=debug_dump_dir,
                        error_label=error_label,
                        model=model,
                        endpoint_key=endpoint_key,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        reason="missing_choices",
                        request_kwargs=create_kwargs,
                        response=resp,
                    )
                    if attempt + 1 < max_attempts:
                        print(
                            f"  [{error_label}] response missing choices, retrying "
                            f"({attempt + 1}/{max_attempts - 1})"
                        )
                        continue
                    break
                msg = getattr(choices[0], "message", None)
                if msg is None:
                    last_exc = ValueError("LLM response choice missing message")
                    _write_debug_dump(
                        debug_dump_dir=debug_dump_dir,
                        error_label=error_label,
                        model=model,
                        endpoint_key=endpoint_key,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        reason="missing_message",
                        request_kwargs=create_kwargs,
                        response=resp,
                    )
                    if attempt + 1 < max_attempts:
                        print(
                            f"  [{error_label}] response missing message, retrying "
                            f"({attempt + 1}/{max_attempts - 1})"
                        )
                        continue
                    break
                content = (getattr(msg, "content", None) or "").strip()
                if content:
                    return content
                # Empty content — try recovering JSON from the reasoning trace
                # before paying for a retry. Common with K2.6 / reasoning models.
                recovered = _extract_balanced_json(_reasoning_text(msg))
                if recovered:
                    print(
                        f"  [{error_label}] recovered JSON from reasoning field "
                        f"(attempt {attempt + 1}/{max_attempts})"
                    )
                    return recovered
                _write_debug_dump(
                    debug_dump_dir=debug_dump_dir,
                    error_label=error_label,
                    model=model,
                    endpoint_key=endpoint_key,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    reason="empty_content",
                    request_kwargs=create_kwargs,
                    response=resp,
                    message=msg,
                )
                if attempt + 1 < max_attempts:
                    print(
                        f"  [{error_label}] empty content, retrying "
                        f"({attempt + 1}/{max_attempts - 1})"
                    )
                    continue
            # All attempts produced empty content or raised exceptions.
            with _FALLBACK_LOCK:
                _FALLBACK_COUNTS[error_label] = _FALLBACK_COUNTS.get(error_label, 0) + 1
                count = _FALLBACK_COUNTS[error_label]
            if last_exc is not None:
                message = (
                    f"[{error_label}] LLM error after {max_attempts} attempts "
                    f"({type(last_exc).__name__}, fallback #{count}): {last_exc}"
                )
            else:
                message = (
                    f"[{error_label}] empty content after {max_attempts} attempts, "
                    f"fallback #{count}"
                )
            print(f"  {message}")
            if should_fail_on_fallback:
                raise LLMFallbackError(message)
            return error_fallback

        return call


_SHARED_FACTORY: LLMClientFactory | None = None


def shared_factory() -> LLMClientFactory:
    """Process-wide factory so two roles sharing an endpoint reuse the same OpenAI client."""
    global _SHARED_FACTORY
    if _SHARED_FACTORY is None:
        _SHARED_FACTORY = LLMClientFactory()
    return _SHARED_FACTORY


def reset_shared_factory() -> None:
    """Tests only — drop the cached factory (so monkeypatched OpenAI takes effect)."""
    global _SHARED_FACTORY
    _SHARED_FACTORY = None
