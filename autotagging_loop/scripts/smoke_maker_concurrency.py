"""Live Maker concurrency calibration.

Usage:
    uv run python scripts/smoke_maker_concurrency.py --concurrency 32

This script isolates the Maker role: it reuses existing Part2 map-evidence
aggregate.json files, applies the active vocabulary to every Part2 benchmark,
and records wall/per-call latency under a fixed Maker concurrency cap.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autotagging_loop.experiment.config import (
    llm_debug_dump_dir,
    llm_empty_content_retries,
    llm_extra_body,
    load_experiment_config,
    role_cfg,
)
from autotagging_loop.experiment.json_contract import parse_json_object_strict
from autotagging_loop.experiment.llm_client import llm_fallback_counts, reset_llm_fallback_counts, shared_factory
from autotagging_loop.experiment.maker import _validate_maker_json, run_maker
from autotagging_loop.experiment.mapreduce_evidence import _slug
from autotagging_loop.experiment.static_tag_weights import ABILITY_LEVEL_SCORES
from autotagging_loop.runner.config import load_config as load_part2_config
from autotagging_loop.runner.corpus import load_corpus


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_vocab(path: Path) -> list[dict[str, Any]]:
    vocab = _read_json(path)
    if isinstance(vocab, dict):
        vocab = vocab.get("vocab")
    if not isinstance(vocab, list) or not vocab:
        raise ValueError(f"vocab must be a non-empty list: {path}")
    out: list[dict[str, Any]] = []
    for item in vocab:
        if not isinstance(item, dict) or not item.get("id"):
            raise ValueError(f"invalid vocab item in {path}: {item!r}")
        out.append(dict(item))
    return out


def _candidate_aggregate_roots(results_dir: Path) -> list[Path]:
    paths = sorted(results_dir.rglob("map_evidence/**/aggregate.json"))
    roots: set[Path] = set()
    for path in paths:
        # root/<benchmark-slug>/aggregate.json
        roots.add(path.parent.parent)
    return sorted(roots, key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)


def _load_aggregates_from_root(
    root: Path,
    benchmark_names: list[str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    aggregates: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for benchmark in benchmark_names:
        path = root / _slug(benchmark) / "aggregate.json"
        if not path.exists():
            missing.append(benchmark)
            continue
        payload = _read_json(path)
        if not isinstance(payload, dict):
            missing.append(benchmark)
            continue
        payload = dict(payload)
        payload["benchmark"] = benchmark
        aggregates[benchmark] = payload
    return aggregates, missing


def _find_full_coverage_root(
    results_dir: Path,
    benchmark_names: list[str],
) -> tuple[Path, dict[str, dict[str, Any]]]:
    best_root: Path | None = None
    best_aggregates: dict[str, dict[str, Any]] = {}
    best_missing: list[str] = benchmark_names
    for root in _candidate_aggregate_roots(results_dir):
        aggregates, missing = _load_aggregates_from_root(root, benchmark_names)
        if len(missing) < len(best_missing):
            best_root, best_aggregates, best_missing = root, aggregates, missing
        if not missing:
            return root, aggregates
    if best_root is None:
        raise FileNotFoundError(
            f"no aggregate roots found under {results_dir}/**/map_evidence"
        )
    raise FileNotFoundError(
        "no full-coverage aggregate root found; "
        f"best={best_root} covered={len(best_aggregates)}/{len(benchmark_names)} "
        f"missing={best_missing}"
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - pos) + ordered[high] * (pos - low)


def _configure_maker_limit(config: dict[str, Any], concurrency: int) -> None:
    maker_cfg = role_cfg(config, "maker_model")
    shared_factory().configure_limit(
        base_url=maker_cfg.get("base_url"),
        base_url_env=maker_cfg.get("base_url_env"),
        api_key_env=maker_cfg.get("api_key_env"),
        max_concurrent=concurrency,
    )


def _maker_chat_fn(config: dict[str, Any], error_label: str = "maker") -> Callable[..., str]:
    maker_cfg = role_cfg(config, "maker_model")
    return shared_factory().chat_fn(
        model=maker_cfg["name"],
        base_url=maker_cfg.get("base_url"),
        base_url_env=maker_cfg.get("base_url_env"),
        api_key_env=maker_cfg.get("api_key_env"),
        response_format={"type": "json_object"},
        error_label=error_label,
        empty_content_retries=llm_empty_content_retries(config),
        debug_dump_dir=llm_debug_dump_dir(config),
        extra_body=llm_extra_body(config),
    )


def _benchmark_from_user_msg(user_msg: str) -> str:
    marker = "Benchmark: "
    if marker not in user_msg:
        return ""
    return user_msg.split(marker, 1)[1].split("\n", 1)[0].strip()


def _timed_chat_fn(config: dict[str, Any], latencies_ms: list[int], lock: Lock) -> Callable[..., str]:
    fn_cache: dict[str, Callable[..., str]] = {}

    def call(system_msg: str, user_msg: str, seed: int | str | None = None) -> str:
        benchmark = _benchmark_from_user_msg(user_msg)
        error_label = f"maker:{benchmark}" if benchmark else "maker"
        if error_label not in fn_cache:
            fn_cache[error_label] = _maker_chat_fn(config, error_label)
        base_fn = fn_cache[error_label]
        started = time.time()
        try:
            return base_fn(system_msg, user_msg) if seed is None else base_fn(system_msg, user_msg, seed)
        finally:
            elapsed = int((time.time() - started) * 1000)
            with lock:
                latencies_ms.append(elapsed)

    return call


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument(
        "--empty-retries",
        type=int,
        default=6,
        help="Additional empty-content/missing-choice retries before strict failure.",
    )
    parser.add_argument(
        "--debug-dump-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "_debug" / "llm_responses",
        help="Where to write redacted anomalous LLM responses.",
    )
    parser.add_argument(
        "--aggregate-root",
        type=Path,
        default=None,
        help="Optional map_evidence root containing <benchmark-slug>/aggregate.json.",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "cognitive_abilities.json",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "maker_concurrency",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=PROJECT_ROOT / "experiment" / "prompts" / "I_exec_seed.txt",
    )
    args = parser.parse_args()

    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")

    run_dir = args.results_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    part2_config = load_part2_config()
    part2_corpus = load_corpus(part2_config)
    benchmark_names = list(part2_corpus.benchmark_names)
    if not benchmark_names:
        raise ValueError("Part2 corpus has no benchmarks")

    if args.aggregate_root is not None:
        aggregate_root = args.aggregate_root
        aggregates, missing = _load_aggregates_from_root(aggregate_root, benchmark_names)
        if missing:
            raise FileNotFoundError(
                f"aggregate root missing {len(missing)} benchmark(s): {missing}"
            )
    else:
        aggregate_root, aggregates = _find_full_coverage_root(
            PROJECT_ROOT / "results" / "part2_experiment",
            benchmark_names,
        )

    vocab = _load_vocab(args.vocab_path)
    prompt = args.prompt_path.read_text(encoding="utf-8")
    config = load_experiment_config({
        "maker_max_concurrent": args.concurrency,
        "maker_max_workers": args.concurrency,
        "llm_json_contract_strict": True,
        "llm_json_contract_max_attempts": 3,
        "llm_empty_content_retries": args.empty_retries,
        "llm_debug_dump_dir": str(args.debug_dump_dir),
    })

    reset_llm_fallback_counts()
    _configure_maker_limit(config, args.concurrency)
    latencies_ms: list[int] = []
    lock = Lock()
    chat_fn = _timed_chat_fn(config, latencies_ms, lock)

    started = time.time()
    outputs, metadata = run_maker(
        benchmark_names=benchmark_names,
        vocab=vocab,
        aggregates=aggregates,
        config=config,
        run_dir=str(run_dir),
        prompt=prompt,
        version=1,
        label=f"maker_concurrency_{args.concurrency}",
        chat_fn=chat_fn,
    )
    total_ms = int((time.time() - started) * 1000)

    vocab_ids = [str(item["id"]) for item in vocab]
    invalid_outputs: list[str] = []
    for benchmark, payload in outputs.items():
        try:
            parsed = parse_json_object_strict(str(payload.get("raw_response") or ""))
            _validate_maker_json(parsed, vocab_ids)
        except Exception as exc:
            invalid_outputs.append(f"{benchmark}: {type(exc).__name__}: {exc}")

    fallback_counts = llm_fallback_counts()
    summary = {
        "status": "ok" if not invalid_outputs and not fallback_counts else "fail",
        "run_dir": str(run_dir),
        "aggregate_root": str(aggregate_root),
        "model": role_cfg(config, "maker_model").get("name"),
        "concurrency": args.concurrency,
        "empty_content_retries": llm_empty_content_retries(config),
        "debug_dump_dir": llm_debug_dump_dir(config),
        "benchmark_count": len(benchmark_names),
        "maker_output_count": metadata.get("maker_output_count"),
        "cache_hits": metadata.get("maker_cache_hits"),
        "cache_misses": metadata.get("maker_cache_misses"),
        "total_ms": total_ms,
        "total_s": round(total_ms / 1000.0, 3),
        "latency_ms": latencies_ms,
        "latency_p50_ms": int(_percentile(latencies_ms, 0.50)) if latencies_ms else None,
        "latency_p95_ms": int(_percentile(latencies_ms, 0.95)) if latencies_ms else None,
        "latency_max_ms": max(latencies_ms) if latencies_ms else None,
        "fallback_counts": fallback_counts,
        "invalid_outputs": invalid_outputs,
        "level_vocab": sorted(ABILITY_LEVEL_SCORES),
        "benchmarks": benchmark_names,
    }
    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    status = summary["status"]
    print(
        f"[maker-concurrency] {status.upper()} concurrency={args.concurrency} "
        f"benchmarks={len(benchmark_names)} outputs={metadata.get('maker_output_count')} "
        f"total={summary['total_s']}s p50={summary['latency_p50_ms']}ms "
        f"p95={summary['latency_p95_ms']}ms max={summary['latency_max_ms']}ms "
        f"summary={summary_path}",
        file=sys.stderr,
    )
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
