from __future__ import annotations

import pytest

import autotagging_loop.runner.config as config


def test_make_openai_kwargs_uses_openrouter_base_url_and_auth_header(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(config, "DOTENV_VALUES", {
        "OPENROUTER_API_KEY": "test-key",
        "OPENROUTER_BASE_URL": "https://router.example/v1",
    })

    kwargs = config.make_openai_kwargs()

    assert kwargs["api_key"] == "test-key"
    assert kwargs["base_url"] == "https://router.example/v1"
    assert kwargs["default_headers"]["Authorization"] == "Bearer test-key"


def test_make_openai_kwargs_fails_without_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(config, "DOTENV_VALUES", {})

    with pytest.raises(RuntimeError, match="Missing API key"):
        config.make_openai_kwargs()
