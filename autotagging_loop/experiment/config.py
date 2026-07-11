"""experiment/config.py — Part 1 전용 격리 설정.

`benchpress/config.py` 와 독립적으로 동작한다. 동일한 .env 를 로드하지만
`DEFAULT_EXPERIMENT_CONFIG` + 선택적 `benchpress_config.json` 의 `"experiment"` 섹션만
참조한다.
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import dotenv_values, load_dotenv

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CONFIG_FILE = os.path.join(_PROJECT_ROOT, "benchpress_config.json")
_DOTENV_FILE = os.path.join(_PROJECT_ROOT, ".env")
load_dotenv(dotenv_path=_DOTENV_FILE)
_DOTENV_VALUES = dotenv_values(_DOTENV_FILE)

LEADERBOARD_PATH = os.path.join(_PROJECT_ROOT, "data", "leaderboard_scores.json")
LABELS_DIR = os.path.join(_PROJECT_ROOT, "data", "labels")
VOCAB_PATH = os.path.join(_PROJECT_ROOT, "data", "cognitive_abilities.json")
PROMPT_I0_PATH = os.path.join(os.path.dirname(__file__), "prompts", "I0.txt")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "experiment")

DEFAULT_EXPERIMENT_CONFIG: dict[str, Any] = {
    "max_iter": 5,
    "eps": 0.005,
    "early_stop_consecutive": 2,
    "min_common_models": 6,
    "min_common_models_warn": 5,
    "theta_mode": "quantile",
    "theta_p_q": 0.80,
    "theta_n_q": 0.20,
    "bootstrap_B": 200,
    "normalize": "rank",
    "exclude": ["arena_hard", "ifeval"],
    "error_top_k": 20,
    "seed": 42,
    "optimize_tag_weights": False,
    "run_weight_calibration_ablation": False,
    "weight_target_scale": "raw",
    "weight_bounds": [0.0, 1.0],
    "weight_l2_lambda": 0.01,
    "weight_max_iter": 200,
    "tag_weight_mode": "static_from_mapreduce",
    "static_tag_require_mapreduce": True,
    "static_tag_restrict_to_labeled_benchmarks": True,
    "use_mapreduce_evidence": True,
    "mapreduce_model": {"name": "qwen/qwen3.5-9b", "base_url": None},
    "mapreduce_chunk_examples": 25,
    "mapreduce_max_chunk_chars": 48000,
    "mapreduce_max_evidence_chars": 24000,
    "taxonomy_refinement_enabled": False,
    "taxonomy_refinement_min_pairs": 3,
    "taxonomy_refinement_residual_max_threshold": 0.35,
    "taxonomy_refinement_residual_mean_threshold": None,
    "taxonomy_refinement_l_align_threshold": None,
    "taxonomy_refinement_max_new_tags": 4,
    "taxonomy_refinement_max_iter": 3,
    "taxonomy_refinement_retain_seed_tags": True,
    "taxonomy_refinement_adoption_enabled": True,
    "taxonomy_refinement_min_l_align_improvement": 0.0,
    "taxonomy_refinement_max_rho_pearson_drop": 0.02,
    "taxonomy_refinement_max_delta_tag_drop": 0.05,
    "taxonomy_refinement_max_residual_increase": 0.0,
    "taxonomy_refinement_protected_pairs_top_k": 10,
    "taxonomy_refinement_protected_pairs_min_r": 0.80,
    "no_seed_taxonomy_enabled": False,
    "no_seed_taxonomy_min_tags": 8,
    "no_seed_taxonomy_max_tags": 14,
    "no_seed_taxonomy_examples_per_benchmark": 3,
    "no_seed_taxonomy_max_chars_per_benchmark": 4000,
    "no_seed_taxonomy_max_attempts": 3,
    "no_seed_taxonomy_fallback_to_seed": False,
    "no_seed_taxonomy_model": None,
    "model_a": {"name": "qwen/qwen3.5-35b-a3b", "base_url": None},
    "model_imp": {"name": "qwen/qwen3.5-35b-a3b", "base_url": None},
    # Improver exploration knobs. temperature>0 + n_samples>1 break the
    # deterministic fixed-point seen on 2026-05-11 (Improver returned the
    # same prompt hash from iter_2 onwards, stalling v_loop).
    "improver_temperature": 0.7,
    "improver_n_samples": 3,
    # Fallback responses are deterministic placeholders from the LLM client.
    # The default client raises immediately when fallback would be returned;
    # this threshold remains as a backstop for custom clients / legacy paths.
    "llm_fallback_fail_threshold": 0,
    # Role outputs must be strict JSON objects matching the role schema. Invalid
    # JSON/schema drift is retried, then raises to keep experiment artifacts clean.
    "llm_json_contract_strict": True,
    "llm_json_contract_max_attempts": 3,
    # Additional retries for provider responses with empty content / missing
    # choices before the call is treated as fallback. Total attempts = 1 + this.
    "llm_empty_content_retries": 2,
    # Optional per-request timeout and retries after SDK exceptions such as
    # request/read timeouts. Empty-content retries remain separate because
    # provider JSON omissions often recover quickly while socket stalls should
    # fail fast enough to keep K-fold runs moving.
    "llm_request_timeout_s": 180.0,
    "llm_sdk_exception_retries": 1,
    # Optional anomaly dump root. When set, empty-content / missing-choice /
    # SDK-error attempts are written as redacted JSON for provider debugging.
    "llm_debug_dump_dir": None,
    # Optional OpenRouter/OpenAI-compatible extra_body.reasoning config. For
    # JSON-only role calls, {"effort": "none", "exclude": True} prevents
    # thinking-only responses from leaving message.content empty.
    "llm_reasoning": None,
    # Direct LLM tag generation legacy fallback. Pipeline defaults to fail-fast;
    # tests/debug callers can opt back into the old uniform-vector fallback.
    "tag_generator_allow_uniform_fallback": False,
    # Executer owns V_i generation. If it emits invalid JSON or a rejected
    # schema, fail the run instead of silently reusing the seed vocabulary.
    "executer_fallback_to_seed": False,
    # Optional local-optimum probe: cycle Executer through exact target
    # vocabulary counts across v-loop iterations.
    "executer_candidate_counts": None,
    # Best-iter selection. v3 default is "train_l_align" — argmin train L_align
    # among Δ_tag>0 candidates. "dev_l_align" switches the primary criterion to
    # dev L_align (ML-hygiene: dev never sees Improver gradient).
    # "dev_stability_l_align" compares the worst non-leaky L across dev/train/
    # model-probe and rewards the weakest Spearman signal. The optional
    # `best_iter_dev_rho_floor` rejects dev_l_align candidates whose dev ρ_s
    # falls below the floor (catastrophic-collapse guard). Optional
    # `best_iter_dev_rho_drop_tolerance` rejects candidates that improve dev
    # L_align by sacrificing too much dev ρ_s relative to the current best.
    # The train_* guards are optional generalization checks for dev_l_align:
    # they reject dev-improving candidates that destabilize train L/rho too far
    # relative to the current best.
    # Only meaningful with `enable_v_loop=True`; otherwise dev metrics are
    # absent and legacy fixed-V runs normalize this setting back to train_l_align.
    "best_iter_selection": "train_l_align",
    "best_iter_dev_rho_floor": 0.0,
    "best_iter_dev_rho_drop_tolerance": None,
    "best_iter_train_l_increase_tolerance": None,
    "best_iter_train_rho_drop_tolerance": None,
    "best_iter_train_rho_floor": None,
    "best_iter_stability_rho_weight": 0.30,
    # Optional non-leaky model-stability guard. When enabled with
    # v_loop_score_model_scope=seen, selection computes leave-one-seen-model-out
    # dev metrics and can reject candidates whose proxy rho collapses.
    "best_iter_model_probe_enabled": False,
    "best_iter_model_probe_min_common": None,
    "best_iter_model_probe_dev_rho_floor": None,
    "best_iter_model_probe_dev_rho_drop_tolerance": None,
    "best_iter_model_probe_dev_l_increase_tolerance": None,
    # Optional taxonomy-cardinality objective for v_loop selection. When
    # enabled, candidates can be rejected outside [min,max] tag counts and
    # compared by L_align plus a small distance-from-target penalty.
    "taxonomy_selection_enabled": False,
    "taxonomy_selection_min_tags": None,
    "taxonomy_selection_max_tags": None,
    "taxonomy_selection_target_tags": None,
    "taxonomy_selection_count_penalty": 0.0,
    "delta_tag_threshold": 0.0,
    "leaderboard_path": LEADERBOARD_PATH,
    "labels_dir": LABELS_DIR,
    "examples_per_benchmark": "all",
    "prompt_examples_per_benchmark": 20,
    "max_prompt_chars_per_benchmark": 24000,
    "vocab_path": VOCAB_PATH,
    "prompt_i0_path": PROMPT_I0_PATH,
    "results_dir": RESULTS_DIR,
    "subset_profile_sizes": [1, 2, 3, 5],
    "run_baseline": True,
    "wandb": False,
    # v3 §2.2.4 — V loop. When True, every main iteration calls Executer to
    # produce V^(i) from the per-fold train split's chunk evidence Z_src and
    # the current I_exec; dev split drives Improver selection + Δ_tag gate;
    # D_test stays untouched until end-of-run reporting. Default False keeps
    # the legacy fixed-V path bit-identical.
    "enable_v_loop": False,
    # Split preflight guard: each split must retain at least this many
    # score-comparable benchmark pairs after min_common filtering. Without this,
    # selection can silently run on NaN dev metrics.
    "v_loop_min_train_valid_pairs": 1,
    "v_loop_min_dev_valid_pairs": 1,
    "v_loop_min_test_valid_pairs": 1,
    # Optional stricter guard: enough pairs can still mean only a small subset
    # of benchmarks are actually connected by valid pairs.
    "v_loop_min_train_effective_benchmarks": 0,
    "v_loop_min_dev_effective_benchmarks": 0,
    "v_loop_min_test_effective_benchmarks": 0,
    "v_loop_require_held_model_test": False,
    "v_loop_score_model_scope": "all",
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_experiment_config(overrides: dict | None = None) -> dict:
    """Build the experiment config from defaults + benchpress_config.json + overrides."""
    config = dict(DEFAULT_EXPERIMENT_CONFIG)
    if os.path.exists(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                user = json.load(f)
            section = user.get("experiment")
            if isinstance(section, dict):
                config = _deep_merge(config, section)
        except Exception as exc:
            print(f"  [experiment.config] benchpress_config.json 로딩 실패: {exc}")
    if overrides:
        config = _deep_merge(config, overrides)
    return config


def _env_value(name: str) -> str | None:
    val = os.getenv(name)
    if val and val.strip():
        return val.strip()
    val = _DOTENV_VALUES.get(name)
    if val is None:
        return None
    val = val.strip()
    return val or None


def make_openai_kwargs(
    base_url: str | None = None,
    base_url_env: str | None = None,
    api_key_env: str | None = None,
) -> dict:
    """Build kwargs for `openai.OpenAI(...)`.

    URL precedence: explicit `base_url` > `os.environ[base_url_env]` > legacy `OPENROUTER_BASE_URL`.
    Key precedence: `os.environ[api_key_env]` > legacy `OPENROUTER_API_KEY` > legacy `OPENAI_API_KEY`.
    Raises if a URL is resolved but no key is found (existing invariant).
    """
    kwargs: dict = {}

    api_key: str | None = None
    if api_key_env:
        api_key = _env_value(api_key_env)
    if not api_key:
        api_key = _env_value("OPENROUTER_API_KEY") or _env_value("OPENAI_API_KEY")

    url: str | None = base_url
    if not url and base_url_env:
        url = _env_value(base_url_env)
    if not url:
        url = _env_value("OPENROUTER_BASE_URL")

    if url and not api_key:
        raise RuntimeError(
            f"experiment/config: base_url 이 설정됐지만 (resolved={url!r}) API key 가 없습니다. "
            f".env 의 api_key_env={api_key_env!r} 또는 OPENROUTER_API_KEY / OPENAI_API_KEY 를 확인하세요."
        )

    if api_key:
        kwargs["api_key"] = api_key
    if url:
        kwargs["base_url"] = url
    if api_key:
        kwargs["default_headers"] = {"Authorization": f"Bearer {api_key}"}
    return kwargs


_ROLE_LEGACY_FALLBACK: dict[str, str] = {
    "mapper_model": "mapreduce_model",
    "executer_model": "model_a",
    "maker_model": "model_a",
    "improver_model": "model_imp",
}


def role_cfg(config: dict, role: str) -> dict:
    """Return the role's model config, preferring the v3 key with legacy fallback.

    v3 keys: `mapper_model` / `executer_model` / `maker_model` / `improver_model`.
    Legacy fallbacks: `mapreduce_model` / `model_a` / `model_a` / `model_imp` respectively.
    Returns `{}` if neither is configured (caller must validate `name`).
    """
    cfg = config.get(role)
    if isinstance(cfg, dict) and cfg:
        return cfg
    legacy_key = _ROLE_LEGACY_FALLBACK.get(role)
    if legacy_key:
        legacy = config.get(legacy_key)
        if isinstance(legacy, dict):
            return legacy
    return {}


def llm_empty_content_retries(config: dict | None = None) -> int | None:
    """Return configured retries for empty-content provider responses."""
    value = (config or {}).get("llm_empty_content_retries")
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def llm_debug_dump_dir(config: dict | None = None) -> str | None:
    """Return optional LLM anomaly dump directory."""
    value = (config or {}).get("llm_debug_dump_dir")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def llm_request_timeout_s(config: dict | None = None) -> float | None:
    """Return per-request LLM timeout seconds, or None for client default."""
    value = (config or {}).get("llm_request_timeout_s")
    if value is None:
        return None
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return None
    return timeout if timeout > 0 else None


def llm_sdk_exception_retries(config: dict | None = None) -> int | None:
    """Return retries after SDK exceptions such as provider timeouts."""
    value = (config or {}).get("llm_sdk_exception_retries")
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def llm_extra_body(config: dict | None = None) -> dict | None:
    """Return optional OpenAI SDK extra_body for provider-specific controls."""
    reasoning = (config or {}).get("llm_reasoning")
    if not isinstance(reasoning, dict) or not reasoning:
        return None
    return {"reasoning": dict(reasoning)}
