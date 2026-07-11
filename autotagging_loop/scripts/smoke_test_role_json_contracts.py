"""Live smoke for strict JSON contracts across v3 LLM roles.

Usage:
    uv run python scripts/smoke_test_role_json_contracts.py

This uses the active `benchpress_config.json` experiment model settings and a
fresh temporary run directory. It disables MapReduce persistent cache so the
current backend response is what gets validated.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autotagging_loop.experiment.alignment import ErrorPair
from autotagging_loop.experiment.config import (
    llm_debug_dump_dir,
    llm_empty_content_retries,
    llm_extra_body,
    llm_request_timeout_s,
    llm_sdk_exception_retries,
    load_experiment_config,
    role_cfg,
)
from autotagging_loop.experiment.corpus import Corpus
from autotagging_loop.experiment.executer import _validate_executer_json, run_executer
from autotagging_loop.experiment.json_contract import parse_json_object_strict
from autotagging_loop.experiment.maker import _validate_maker_json, run_maker
from autotagging_loop.experiment.mapreduce_evidence import (
    _validate_mapper_json,
    build_mapreduce_descriptions,
)
from autotagging_loop.experiment.prompt_improver import _validate_improver_json, improve_prompt


SEED_VOCAB = [
    {
        "id": "deductive_reasoning",
        "name": "Deductive Reasoning",
        "definition": "Apply explicit rules to derive a conclusion.",
    },
    {
        "id": "quantitative_reasoning",
        "name": "Quantitative Reasoning",
        "definition": "Use arithmetic or symbolic quantities to solve a task.",
    },
    {
        "id": "semantic_comprehension",
        "name": "Semantic Comprehension",
        "definition": "Interpret natural language meaning and constraints.",
    },
]


def _aggregate(benchmark: str) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "reviewed_rows": 2,
        "n_chunks": 1,
        "mapped_examples": 2,
        "chunk_evidence": [
            {
                "chunk_index": 0,
                "n_examples": 2,
                "summary": (
                    "The examples require reading a short problem, identifying explicit "
                    "constraints, and applying a small reasoning chain."
                ),
                "task_patterns": [
                    "short answer selection",
                    "constraint-based problem solving",
                ],
                "reasoning_patterns": [
                    "deductive rule application",
                    "simple quantitative comparison",
                ],
                "justifications": [
                    "The answer follows from applying stated conditions rather than recall.",
                    "Some items require comparing small quantities.",
                ],
                "_banned_drift": [],
            }
        ],
        "justifications": [
            "The benchmark stresses explicit rule use and semantic interpretation."
        ],
    }


def _corpus() -> Corpus:
    examples = [
        "Question: If every red token is worth 2 points and Ana has three red tokens, "
        "how many points does she have?\nAnswer: 6",
        "Question: All squares are blue. The tile is a square. What color is the tile?\n"
        "Answer: blue",
    ]
    return Corpus(
        benchmark_names=["ContractMapperBench"],
        model_names=["model_a", "model_b"],
        Y={"ContractMapperBench": {"model_a": 0.8, "model_b": 0.6}},
        descriptions={"ContractMapperBench": "contract smoke benchmark"},
        documents={
            "ContractMapperBench": {
                "reviewed_rows": len(examples),
                "topic_counts": {"logic": 1, "math": 1},
                "reasoning_depth_counts": {"single_step": 2},
                "answer_format_counts": {"short_answer": 2},
                "examples": examples,
            }
        },
    )


def _read_first_mapper_raw(run_dir: Path) -> str:
    candidates = [
        p
        for p in (run_dir / "map_evidence").rglob("*.json")
        if p.name not in {"aggregate.json", "chunks_manifest.json"}
    ]
    if not candidates:
        raise RuntimeError("mapper cache payload not found")
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    raw = payload.get("raw_response")
    if not raw:
        raise RuntimeError("mapper raw_response missing")
    return str(raw)


def _role_model(config: dict[str, Any], role: str) -> str | None:
    cfg = role_cfg(config, role)
    return cfg.get("name") if isinstance(cfg, dict) else None


def _run_timed(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.time()
    try:
        detail = fn()
    except Exception as exc:
        return {
            "role": name,
            "status": "fail",
            "latency_ms": int((time.time() - started) * 1000),
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "role": name,
        "status": "ok",
        "latency_ms": int((time.time() - started) * 1000),
        **detail,
    }


def _mapper_smoke(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    _, aggregates = build_mapreduce_descriptions(
        _corpus(),
        SEED_VOCAB,
        config,
        str(run_dir),
    )
    raw = _read_first_mapper_raw(run_dir)
    parsed = parse_json_object_strict(raw)
    _validate_mapper_json(parsed)
    aggregate = aggregates["ContractMapperBench"]
    return {
        "model": _role_model(config, "mapper_model"),
        "strict_json": True,
        "mapped_examples": aggregate.get("mapped_examples"),
        "n_chunks": aggregate.get("n_chunks"),
        "raw_keys": sorted(parsed),
    }


def _maker_smoke(
    config: dict[str, Any],
    run_dir: Path,
    vocab: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    active_vocab = (vocab or SEED_VOCAB)[:3]
    outputs, metadata = run_maker(
        benchmark_names=["ContractMakerBench"],
        vocab=active_vocab,
        aggregates={"ContractMakerBench": _aggregate("ContractMakerBench")},
        config=config,
        run_dir=str(run_dir),
        prompt=(
            "Assign ordinal ability levels for every supplied ability id using only "
            "the chunk evidence."
        ),
        version=1,
        label="json_contract_maker",
    )
    raw = outputs["ContractMakerBench"].get("raw_response") or ""
    parsed = parse_json_object_strict(str(raw))
    _validate_maker_json(parsed, [item["id"] for item in active_vocab])
    return {
        "model": _role_model(config, "maker_model"),
        "maker_output_count": metadata.get("maker_output_count"),
        "vocab_ids": [item["id"] for item in active_vocab],
        "raw_keys": sorted(parsed),
    }


def _improver_smoke(config: dict[str, Any], vocab: list[dict[str, Any]] | None) -> dict[str, Any]:
    active_vocab = (vocab or SEED_VOCAB)[:3]
    vocab_ids = [item["id"] for item in active_vocab]
    base_prompt = (
        "Use exactly these cognitive ability ids: "
        + ", ".join(vocab_ids)
        + ". Distinguish explicit rule application, quantitative comparison, and "
        "natural-language constraint interpretation when reading benchmark evidence."
    )
    result = improve_prompt(
        prev_prompt=base_prompt,
        base_prompt=base_prompt,
        error_report=[
            ErrorPair(
                p="ContractLogicBench",
                q="ContractMathBench",
                s_pq=0.1,
                r_pq_raw=0.9,
                r_pq_01=0.95,
                delta=0.4,
                type="false_dis",
            )
        ],
        metrics={
            "L_align": 0.1,
            "rho_align_pearson": 0.2,
            "rho_align_spearman": 0.2,
            "delta_tag": 0.1,
        },
        bench_descriptions={
            "ContractLogicBench": "Uses short logic chains and explicit constraints.",
            "ContractMathBench": "Uses small arithmetic comparisons and rule following.",
        },
        vocab=active_vocab,
        benchmark_names=["ContractLogicBench", "ContractMathBench"],
        model=_role_model(config, "improver_model"),
        base_url=role_cfg(config, "improver_model").get("base_url"),
        base_url_env=role_cfg(config, "improver_model").get("base_url_env"),
        api_key_env=role_cfg(config, "improver_model").get("api_key_env"),
        temperature=0.0,
        n_samples=1,
        json_contract_strict=True,
        json_contract_max_attempts=int(config.get("llm_json_contract_max_attempts", 3)),
        empty_content_retries=llm_empty_content_retries(config),
        request_timeout_s=llm_request_timeout_s(config),
        sdk_exception_retries=llm_sdk_exception_retries(config),
        debug_dump_dir=llm_debug_dump_dir(config),
        extra_body=llm_extra_body(config),
    )
    parsed = parse_json_object_strict(result.raw_response)
    _validate_improver_json(parsed)
    return {
        "model": _role_model(config, "improver_model"),
        "accepted": result.accepted,
        "guard_reasons": result.reasons,
        "raw_keys": sorted(parsed),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Role-level JSON contract attempts before failing.",
    )
    args = parser.parse_args()

    parent = Path(tempfile.mkdtemp(prefix="benchpress_json_contract_"))
    run_dir = parent / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = load_experiment_config({
        "llm_json_contract_strict": True,
        "llm_json_contract_max_attempts": args.attempts,
        "mapreduce_cache_enabled": False,
        "mapreduce_write_run_cache_copy": True,
        "mapreduce_chunk_examples": 2,
        "mapreduce_max_workers": 1,
        "maker_max_workers": 1,
    })

    rows: list[dict[str, Any]] = []
    rows.append(_run_timed("mapper", lambda: _mapper_smoke(config, run_dir)))

    executer_vocab: list[dict[str, Any]] | None = None

    def _run_executer() -> dict[str, Any]:
        nonlocal executer_vocab
        prompt_path = PROJECT_ROOT / "experiment" / "prompts" / "I_exec_seed.txt"
        prompt = prompt_path.read_text(encoding="utf-8")
        source_aggregates = {
            "ContractLogicBench": _aggregate("ContractLogicBench"),
            "ContractMathBench": _aggregate("ContractMathBench"),
        }
        vocab, metadata = run_executer(
            source_benchmarks=list(source_aggregates),
            source_aggregates=source_aggregates,
            prompt_i_exec=prompt,
            config=config,
            run_dir=str(run_dir),
            version=1,
            label="json_contract_executer",
        )
        parsed = parse_json_object_strict(str(metadata.get("raw_response") or ""))
        _validate_executer_json(parsed)
        if metadata.get("reasons"):
            raise RuntimeError(f"executer validation reasons: {metadata['reasons']}")
        executer_vocab = vocab
        return {
            "model": _role_model(config, "executer_model"),
            "vocab_size": len(vocab),
            "sample_ids": [item["id"] for item in vocab[:5]],
            "raw_keys": sorted(parsed),
        }

    rows.append(_run_timed("executer", _run_executer))
    rows.append(_run_timed("maker", lambda: _maker_smoke(config, run_dir, executer_vocab)))
    rows.append(_run_timed("improver", lambda: _improver_smoke(config, executer_vocab)))

    summary = {
        "run_dir": str(run_dir),
        "attempts": args.attempts,
        "llm_empty_content_retries": llm_empty_content_retries(config),
        "llm_request_timeout_s": llm_request_timeout_s(config),
        "llm_sdk_exception_retries": llm_sdk_exception_retries(config),
        "llm_debug_dump_dir": llm_debug_dump_dir(config),
        "llm_extra_body": llm_extra_body(config),
        "rows": rows,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    failures = [row for row in rows if row.get("status") != "ok"]
    if failures:
        print(
            f"[json-contract-smoke] FAIL - {len(failures)}/{len(rows)} role(s) failed",
            file=sys.stderr,
        )
        return 1
    print(
        f"[json-contract-smoke] PASS - all {len(rows)} role contracts passed",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
