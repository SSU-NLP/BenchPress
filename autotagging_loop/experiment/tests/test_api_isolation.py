"""Phase 1 — per-role API isolation tests.

Monkeypatch ``OpenAI`` and verify that ``LLMClientFactory`` threads distinct
``(base_url, api_key)`` pairs per role, caches clients per endpoint, falls back
to legacy env keys when ``api_key_env`` is missing, and forwards ``seed`` /
``temperature`` into ``chat.completions.create`` kwargs.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from autotagging_loop.experiment import config as cfg_mod
from autotagging_loop.experiment import llm_client as lc


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def __init__(self, parent: "_FakeOpenAI") -> None:
        self._parent = parent

    def create(self, **kwargs: Any) -> _FakeResponse:
        self._parent.create_calls.append(kwargs)
        return _FakeResponse('{"ok": true}')


class _FakeChatNamespace:
    def __init__(self, parent: "_FakeOpenAI") -> None:
        self.completions = _FakeChatCompletions(parent)


class _FakeOpenAI:
    """Records init kwargs and create() calls so tests can assert on them."""

    instances: list["_FakeOpenAI"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.create_calls: list[dict[str, Any]] = []
        self.chat = _FakeChatNamespace(self)
        _FakeOpenAI.instances.append(self)


@pytest.fixture(autouse=True)
def _patch_openai_and_env(monkeypatch):
    _FakeOpenAI.instances = []
    monkeypatch.setattr(lc, "OpenAI", _FakeOpenAI)
    lc.reset_shared_factory()

    # Wipe any host-side env so tests are deterministic.
    for var in (
        "OPENROUTER_BASE_URL",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "SELFHOST_BASE_URL",
        "SELFHOST_API_KEY",
        "ROUTER_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    # Also stub the dotenv snapshot so legacy fallback tests don't see real .env values.
    monkeypatch.setattr(cfg_mod, "_DOTENV_VALUES", {})

    yield
    lc.reset_shared_factory()


def test_distinct_kwargs_per_role(monkeypatch):
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://router.example.test/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setenv("SELFHOST_BASE_URL", "https://selfhost.example.test/v1")
    monkeypatch.setenv("SELFHOST_API_KEY", "selfhost-key")

    factory = lc.LLMClientFactory()
    mapper = factory.get(base_url_env="OPENROUTER_BASE_URL", api_key_env="OPENROUTER_API_KEY")
    executer = factory.get(base_url_env="SELFHOST_BASE_URL", api_key_env="SELFHOST_API_KEY")

    assert mapper is not executer
    assert mapper.init_kwargs["base_url"] == "https://router.example.test/v1"
    assert mapper.init_kwargs["api_key"] == "router-key"
    assert executer.init_kwargs["base_url"] == "https://selfhost.example.test/v1"
    assert executer.init_kwargs["api_key"] == "selfhost-key"
    assert executer.init_kwargs["timeout"].read == lc._DEFAULT_LLM_TIMEOUT_S
    assert executer.init_kwargs["max_retries"] == 0


def test_hard_deadline_used_in_worker_thread(monkeypatch):
    calls = []

    def fake_deadline(target, args, *, timeout_s):
        calls.append((target, args, timeout_s))
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    monkeypatch.setattr(lc, "_run_with_process_deadline", fake_deadline)

    def worker():
        return lc._create_with_hard_deadline(
            lambda **_kwargs: pytest.fail("deadline path was skipped"),
            {"model": "m"},
            timeout_s=1.0,
            client_kwargs={"api_key": "k"},
            use_subprocess=True,
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        response = pool.submit(worker).result()

    assert calls
    assert response.choices[0].message.content == '{"ok": true}'


def test_cache_reuse_when_role_shares_endpoint(monkeypatch):
    monkeypatch.setenv("SELFHOST_BASE_URL", "https://selfhost.example.test/v1")
    monkeypatch.setenv("SELFHOST_API_KEY", "selfhost-key")

    factory = lc.LLMClientFactory()
    executer = factory.get(base_url_env="SELFHOST_BASE_URL", api_key_env="SELFHOST_API_KEY")
    maker = factory.get(base_url_env="SELFHOST_BASE_URL", api_key_env="SELFHOST_API_KEY")
    improver = factory.get(base_url_env="SELFHOST_BASE_URL", api_key_env="SELFHOST_API_KEY")

    assert executer is maker is improver
    # exactly one OpenAI() construction
    assert len(_FakeOpenAI.instances) == 1


def test_legacy_resolution_without_api_key_env(monkeypatch):
    """A config dict missing `api_key_env` (legacy `mapreduce_model`) still resolves via
    OPENROUTER_API_KEY."""
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://legacy.example.test/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "legacy-key")

    factory = lc.LLMClientFactory()
    client = factory.get(base_url=None, base_url_env=None, api_key_env=None)

    assert client.init_kwargs["base_url"] == "https://legacy.example.test/v1"
    assert client.init_kwargs["api_key"] == "legacy-key"


def test_base_url_env_resolution(monkeypatch):
    monkeypatch.setenv("SELFHOST_BASE_URL", "https://selfhost.example.test/v1")
    monkeypatch.setenv("SELFHOST_API_KEY", "selfhost-key")

    factory = lc.LLMClientFactory()
    client = factory.get(base_url=None, base_url_env="SELFHOST_BASE_URL", api_key_env="SELFHOST_API_KEY")

    assert client.init_kwargs["base_url"] == "https://selfhost.example.test/v1"


def test_explicit_base_url_overrides_env(monkeypatch):
    monkeypatch.setenv("SELFHOST_BASE_URL", "https://wrong.example.test/v1")
    monkeypatch.setenv("SELFHOST_API_KEY", "selfhost-key")

    factory = lc.LLMClientFactory()
    client = factory.get(
        base_url="https://explicit.example.test/v1",
        base_url_env="SELFHOST_BASE_URL",
        api_key_env="SELFHOST_API_KEY",
    )

    assert client.init_kwargs["base_url"] == "https://explicit.example.test/v1"


def test_missing_key_raises(monkeypatch):
    """URL resolves but no key in any env source → must raise."""
    monkeypatch.setenv("SELFHOST_BASE_URL", "https://selfhost.example.test/v1")
    # SELFHOST_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY all unset.

    factory = lc.LLMClientFactory()
    with pytest.raises(RuntimeError, match="API key"):
        factory.get(base_url_env="SELFHOST_BASE_URL", api_key_env="SELFHOST_API_KEY")


def test_chat_fn_forwards_seed_and_temperature(monkeypatch):
    monkeypatch.setenv("SELFHOST_BASE_URL", "https://selfhost.example.test/v1")
    monkeypatch.setenv("SELFHOST_API_KEY", "selfhost-key")

    factory = lc.LLMClientFactory()
    fn = factory.chat_fn(
        model="dummy-model",
        base_url_env="SELFHOST_BASE_URL",
        api_key_env="SELFHOST_API_KEY",
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    out = fn("sys", "user", 42)

    assert out == '{"ok": true}'
    instance = _FakeOpenAI.instances[-1]
    assert len(instance.create_calls) == 1
    call = instance.create_calls[0]
    assert call["model"] == "dummy-model"
    assert call["temperature"] == 0.3
    assert call["seed"] == 42
    assert call["response_format"] == {"type": "json_object"}
    # 2-positional callers (seed=None) must not inject seed= into the API call
    fn("sys2", "user2")
    call2 = instance.create_calls[1]
    assert "seed" not in call2


def test_concurrent_cold_start_constructs_one_client(monkeypatch):
    """codex 2026-05-10 #5 — fresh factory + 16 racing get() calls must build OpenAI() once."""
    monkeypatch.setenv("SELFHOST_BASE_URL", "https://selfhost.example.test/v1")
    monkeypatch.setenv("SELFHOST_API_KEY", "selfhost-key")

    factory = lc.LLMClientFactory()

    barrier = threading.Barrier(16)

    def call_get() -> Any:
        barrier.wait()
        return factory.get(base_url_env="SELFHOST_BASE_URL", api_key_env="SELFHOST_API_KEY")

    with ThreadPoolExecutor(max_workers=16) as pool:
        clients = list(pool.map(lambda _: call_get(), range(16)))

    # Exactly one OpenAI() construction shared by all 16 threads.
    assert len(_FakeOpenAI.instances) == 1
    assert all(c is clients[0] for c in clients)


def test_configure_limit_caps_concurrent_inflight(monkeypatch):
    """codex 2026-05-10 #5 — semaphore cap must bound observed peak in-flight."""
    monkeypatch.setenv("SELFHOST_BASE_URL", "https://selfhost.example.test/v1")
    monkeypatch.setenv("SELFHOST_API_KEY", "selfhost-key")

    factory = lc.LLMClientFactory()
    factory.configure_limit(
        base_url_env="SELFHOST_BASE_URL",
        api_key_env="SELFHOST_API_KEY",
        max_concurrent=4,
    )
    fn = factory.chat_fn(
        model="dummy",
        base_url_env="SELFHOST_BASE_URL",
        api_key_env="SELFHOST_API_KEY",
    )

    inflight = 0
    peak = 0
    counter_lock = threading.Lock()

    instance = _FakeOpenAI.instances[-1]
    real_create = instance.chat.completions.create

    def slow_create(**kwargs: Any) -> _FakeResponse:
        nonlocal inflight, peak
        with counter_lock:
            inflight += 1
            if inflight > peak:
                peak = inflight
        try:
            time.sleep(0.05)
            return real_create(**kwargs)
        finally:
            with counter_lock:
                inflight -= 1

    instance.chat.completions.create = slow_create  # type: ignore[assignment]

    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(lambda _: fn("sys", "user"), range(32)))

    assert peak <= 4, f"peak inflight {peak} exceeded cap 4"
    assert peak >= 2, f"semaphore should permit some parallelism, got peak={peak}"


def test_configure_limit_min_wins_for_shared_endpoint(monkeypatch):
    """Two roles sharing one endpoint → effective cap = min of the two."""
    monkeypatch.setenv("SELFHOST_BASE_URL", "https://selfhost.example.test/v1")
    monkeypatch.setenv("SELFHOST_API_KEY", "selfhost-key")

    factory = lc.LLMClientFactory()
    factory.configure_limit(
        base_url_env="SELFHOST_BASE_URL", api_key_env="SELFHOST_API_KEY", max_concurrent=8
    )
    factory.configure_limit(
        base_url_env="SELFHOST_BASE_URL", api_key_env="SELFHOST_API_KEY", max_concurrent=4
    )

    key = lc.LLMClientFactory._endpoint_key("https://selfhost.example.test/v1", "SELFHOST_API_KEY")
    assert factory._sem_caps[key] == 4


def test_chat_fn_no_semaphore_when_unconfigured(monkeypatch):
    """If `configure_limit` was never called, behavior is the legacy (unbounded) path."""
    monkeypatch.setenv("SELFHOST_BASE_URL", "https://selfhost.example.test/v1")
    monkeypatch.setenv("SELFHOST_API_KEY", "selfhost-key")

    factory = lc.LLMClientFactory()
    fn = factory.chat_fn(
        model="dummy",
        base_url_env="SELFHOST_BASE_URL",
        api_key_env="SELFHOST_API_KEY",
    )

    # Just confirm it works — no semaphore registered.
    assert fn("sys", "user") == '{"ok": true}'
    key = lc.LLMClientFactory._endpoint_key(
        "https://selfhost.example.test/v1", "SELFHOST_API_KEY"
    )
    assert factory._semaphore_for(key) is None


def test_role_cfg_prefers_new_key_then_legacy():
    config = {
        "mapper_model": {"name": "new-mapper", "base_url_env": "X"},
        "mapreduce_model": {"name": "legacy-mapper"},
        "model_a": {"name": "legacy-a"},
        "model_imp": {"name": "legacy-imp"},
    }
    assert cfg_mod.role_cfg(config, "mapper_model")["name"] == "new-mapper"
    assert cfg_mod.role_cfg(config, "executer_model")["name"] == "legacy-a"
    assert cfg_mod.role_cfg(config, "maker_model")["name"] == "legacy-a"
    assert cfg_mod.role_cfg(config, "improver_model")["name"] == "legacy-imp"
    assert cfg_mod.role_cfg({}, "mapper_model") == {}
