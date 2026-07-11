"""Configuration for the Part 2 main experiment."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE = PROJECT_ROOT / "benchpress_config.json"
DOTENV_FILE = PROJECT_ROOT / ".env"
PART1_BEST_PROMPT_PATH = PROJECT_ROOT / "results" / "experiment" / "run_20260504_224616" / "final" / "I_star.txt"
PART2_FALLBACK_PROMPT_PATH = PROJECT_ROOT / "part2_experiment" / "prompts" / "tag_mapper.txt"
load_dotenv(dotenv_path=DOTENV_FILE)
DOTENV_VALUES = dotenv_values(DOTENV_FILE)


def default_prompt_path() -> str:
    if PART1_BEST_PROMPT_PATH.exists():
        return str(PART1_BEST_PROMPT_PATH)
    return str(PART2_FALLBACK_PROMPT_PATH)


DEFAULT_CONFIG: dict[str, Any] = {
    "leaderboard_path": str(PROJECT_ROOT / "data" / "leaderboard_scores.json"),
    "aai_scores_path": str(PROJECT_ROOT / "data" / "aai_scores.json"),
    "curated_score_backfill_path": str(PROJECT_ROOT / "data" / "curated_score_backfill.json"),
    "use_curated_score_backfill": True,
    "aai_api_url": "https://artificialanalysis.ai/api/v2/data/llms/models",
    "use_aai_scores": True,
    "refresh_aai_scores": False,
    "model_aliases": {
        "Claude Sonnet 4.6": "Claude-Sonnet-4.6",
        "DeepSeek v3": "DeepSeek-v3",
        "DeepSeek V3": "DeepSeek-v3",
        "DeepSeek-V3": "DeepSeek-v3",
        "GPT 5": "GPT-5",
        "Qwen 2.5": "Qwen2.5-72B",
        "Qwen-2.5-72B": "Qwen2.5-72B",
        "Qwen2.5": "Qwen2.5-72B",
    },
    "hf_dataset_map_path": str(PROJECT_ROOT / "data" / "hf_dataset_map.json"),
    "labels_dir": str(PROJECT_ROOT / "data" / "labels_part2"),
    "vocab_path": str(PROJECT_ROOT / "data" / "cognitive_abilities.json"),
    "prompt_path": default_prompt_path(),
    "results_dir": str(PROJECT_ROOT / "results" / "part2_experiment"),
    "hf_sample_n": 100,
    "hf_full_scored_only": True,
    "prompt_examples_per_benchmark": 20,
    "max_prompt_chars_per_benchmark": 24000,
    "min_common_models": 6,
    "min_common_models_warn": 5,
    "exclude": ["arena_hard", "ifeval"],
    "include_benchmarks": None,
    "include_models": None,
    "exclude_models": None,
    "normalize": "rank",
    "bootstrap_B": 200,
    "seed": 42,
    "mapreduce_model": {"name": "qwen/qwen3.5-9b", "base_url": None},
    "mapreduce_reducer_model": {"name": "qwen/qwen3.5-35b-a3b", "base_url": None},
    "model_imp": {"name": "qwen/qwen3.5-35b-a3b", "base_url": None},
    "mapreduce_chunk_examples": 25,
    "mapreduce_max_chunk_chars": 32000,
    "weight_bounds": [0.0, 1.0],
    "run_random_baseline": True,
    "wandb": False,
    "wandb_project": "bench experiment",
    "wandb_entity": None,
    "wandb_mode": None,
    # v3 main loop opt-in. When True, run_part2 delegates to
    # experiment.loop.run_part1 (Executer→Maker→Improver loop with split-aware
    # signals). Executer evidence comes from the full per-fold train split.
    # Default False keeps the legacy single-shot path.
    "enable_v_loop": False,
    "max_iter": 5,
}


def deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(overrides: dict | None = None, config_path: str | None = None) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    file_path = Path(config_path) if config_path else CONFIG_FILE
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        section = raw.get("part2_experiment", {})
        if isinstance(section, dict):
            config = deep_merge(config, section)
    if overrides:
        config = deep_merge(config, overrides)
    return config


def _env_value(name: str) -> str | None:
    value = os.getenv(name) or DOTENV_VALUES.get(name)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def make_openai_kwargs(base_url: str | None = None) -> dict:
    api_key = (
        _env_value("OPENROUTER_API_KEY")
        or _env_value("OPENAI_API_KEY")
        or _env_value("openai_api_key")
    )
    url = (
        base_url
        or _env_value("OPENROUTER_BASE_URL")
        or _env_value("OPENAI_BASE_URL")
    )
    if not api_key:
        raise RuntimeError(
            "Missing API key for Part 2 LLM calls. Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env."
        )
    kwargs: dict = {"api_key": api_key}
    if url:
        kwargs["base_url"] = url
    elif _env_value("OPENROUTER_API_KEY"):
        kwargs["base_url"] = "https://openrouter.ai/api/v1"
    # Some OpenAI-compatible routers require the header explicitly even when api_key is set.
    kwargs["default_headers"] = {"Authorization": f"Bearer {api_key}"}
    return kwargs
