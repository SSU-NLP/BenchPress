"""Artificial Analysis score import helpers for Part 2."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

AAI_API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"

AAI_EVALUATION_BY_BENCHMARK = {
    "aime2025": "aime",
    "gpqa": "gpqa",
    "hle": "hle",
    "humanityslastexam": "hle",
    "math500": "math_500",
    "mmlu-pro": "mmlu_pro",
    "mmlupro": "mmlu_pro",
    "mmmu-pro": "mmmu_pro",
    "mmmupro": "mmmu_pro",
    "livecodebench": "livecodebench",
    "scicode": "scicode",
    "terminalbenchhard": "terminal_bench_hard",
    "ifbench": "ifbench",
    "critpt": "critpt",
    "aalcr": "aa_lcr",
    "aaomniscience": "aa_omniscience",
}


def _name_key(name: str) -> str:
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def _env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def aai_api_key() -> str | None:
    return _env_value("ARTIFICIAL_ANALYSIS_API_KEY") or _env_value("AAI_API_KEY")


def benchmark_to_aai_eval_key(benchmark: str) -> str | None:
    return AAI_EVALUATION_BY_BENCHMARK.get(_name_key(benchmark))


def fetch_aai_payload(*, api_key: str | None = None, api_url: str = AAI_API_URL) -> dict[str, Any]:
    key = api_key or aai_api_key()
    if not key:
        raise RuntimeError("Missing Artificial Analysis API key. Set ARTIFICIAL_ANALYSIS_API_KEY or AAI_API_KEY.")
    res = requests.get(api_url, headers={"x-api-key": key}, timeout=30)
    res.raise_for_status()
    payload = res.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Artificial Analysis API returned a non-object payload.")
    return payload


def extract_aai_scores(
    payload: dict[str, Any],
    benchmarks: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    targets = benchmarks or sorted({
        "AIME 2025",
        "GPQA",
        "HLE",
        "MATH-500",
        "MMLU-Pro",
        "MMMU-Pro",
    })
    by_eval = {
        eval_key: benchmark
        for benchmark in targets
        if (eval_key := benchmark_to_aai_eval_key(benchmark))
    }
    scores: dict[str, dict[str, float]] = {benchmark: {} for benchmark in by_eval.values()}
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return {}
    for item in data:
        if not isinstance(item, dict):
            continue
        model = str(item.get("name") or item.get("slug") or item.get("id") or "").strip()
        evaluations = item.get("evaluations")
        if not model or not isinstance(evaluations, dict):
            continue
        for eval_key, benchmark in by_eval.items():
            value = evaluations.get(eval_key)
            if isinstance(value, (int, float)):
                score = float(value)
                if score > 1.0 and score <= 100.0:
                    score /= 100.0
                if 0.0 <= score <= 1.0:
                    scores[benchmark][model] = score
    return {benchmark: values for benchmark, values in scores.items() if values}


def read_aai_scores(path: str | Path) -> dict[str, dict[str, float]]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return {}
    data = raw.get("scores", raw)
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for benchmark, scores in data.items():
        if not isinstance(scores, dict):
            continue
        clean = {
            str(model): float(score)
            for model, score in scores.items()
            if isinstance(score, (int, float))
        }
        if clean:
            out[str(benchmark)] = clean
    return out


def write_aai_scores(path: str | Path, scores: dict[str, dict[str, float]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "source": "Artificial Analysis API /api/v2/data/llms/models",
            "attribution": "https://artificialanalysis.ai/",
            "note": "Individual evaluation columns only; composite index scores are not used.",
        },
        "scores": scores,
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def refresh_aai_scores(
    path: str | Path,
    *,
    benchmarks: list[str] | None = None,
    api_key: str | None = None,
    api_url: str = AAI_API_URL,
) -> dict[str, dict[str, float]]:
    payload = fetch_aai_payload(api_key=api_key, api_url=api_url)
    scores = extract_aai_scores(payload, benchmarks=benchmarks)
    write_aai_scores(path, scores)
    return scores
