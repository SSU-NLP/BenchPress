"""Stage 1 tagger calibration — pick the cheapest item-tagger model whose
per-item cognitive-ability tags match the teacher's. See docs/calibration_plan.md.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import threading
from pathlib import Path

import numpy as np
from tqdm import tqdm

from autotagging_loop.experiment.llm_client import shared_factory

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "labels_part2_full"
RESULTS_DIR = REPO_ROOT / "results" / "calibration"
ABILITIES_PATH = REPO_ROOT / "data" / "cognitive_abilities.json"
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "tagger_calibration.txt"

BENCHMARKS = ["arcchallenge", "mmlupro", "math500", "humaneval", "gsm8k", "gpqa", "drop", "bbh"]
N_ITEMS_PER_BENCH = 100
LEVEL_VALUES = {"absent": 0.0, "weak": 0.25, "medium": 0.5, "strong": 0.75, "dominant": 1.0}
N_BOOTSTRAP = 2000
CI_PCTL = [2.5, 97.5]
MAX_WORKERS = 100  # ponytail: concurrent gateway calls (pool cap is 1000); lower if it 429s
REASONING_ENABLED = True  # set False (--no-reasoning) to strip reasoning tokens; ~50x cheaper output

THRESHOLDS = {
    "json_valid_rate": 0.995,
    "teacher_cosine": 0.90,
    "top5_overlap": 0.75,
    "pair_structure": 0.85,
    "decoding_stability": 0.95,
    "cost_ratio": 5.0,
}

# USD-per-token prices (input, output) — real SELFHOST gateway rates.
PRICES: dict[str, tuple[float, float]] = {
    "openrouter/qwen/qwen3.5-397b-a17b": (3.85e-7, 2.45e-6),  # $0.385 / $2.45 per 1M
    "openrouter/qwen/qwen3.5-9b": (1.0e-7, 1.5e-7),  # $0.10 / $0.15 per 1M
    "openrouter/qwen/qwen3.5-35b-a3b": (1.4e-7, 1.0e-6),  # $0.14 / $1.00 per 1M
    "openrouter/qwen/qwen3.5-122b-a10b": (2.6e-7, 2.08e-6),  # $0.26 / $2.08 per 1M
    "openrouter/qwen/qwen3.5-27b": (1.95e-7, 1.56e-6),  # $0.195 / $1.56 per 1M (35% off)
    "openrouter/openai/gpt-4.1-nano": (1.0e-7, 4.0e-7),  # $0.10 / $0.40 per 1M
    "openrouter/openai/gpt-4.1-mini": (4.0e-7, 1.6e-6),  # $0.40 / $1.60 per 1M
    "openrouter/mistralai/ministral-3b-2512": (1.0e-7, 1.0e-7),  # $0.10 / $0.10 per 1M
    "openrouter/mistralai/ministral-8b-2512": (1.5e-7, 1.5e-7),  # $0.15 / $0.15 per 1M
    "openrouter/mistralai/mistral-small-3.2-24b-instruct": (7.5e-8, 2.0e-7),  # $0.075 / $0.20 per 1M
    "openrouter/deepseek/deepseek-chat-v3.1": (2.1e-7, 7.9e-7),  # $0.21 / $0.79 per 1M
    "openrouter/google/gemini-2.5-flash-lite": (1.0e-7, 4.0e-7),  # $0.10 / $0.40 per 1M
    "openrouter/meta-llama/llama-3.1-8b-instruct": (2.0e-8, 3.0e-8),  # $0.02 / $0.03 per 1M
    "openrouter/meta-llama/llama-3.3-70b-instruct": (1.0e-7, 3.2e-7),  # $0.10 / $0.32 per 1M
}
_DEFAULT_PRICE = (0.0, 0.0)

AXES: list[dict] = json.loads(ABILITIES_PATH.read_text())
AXIS_IDS: list[str] = [a["id"] for a in AXES]
TAGGER_INSTRUCTIONS = PROMPT_PATH.read_text()


def _vocab_block() -> str:
    lines = ["## Vocabulary"]
    for axis in AXES:
        lines.append(f"- {axis['id']} ({axis['name']}): {axis['definition']}")
    return "\n".join(lines)


_SCHEMA_INSTRUCTION = (
    'Return strict JSON only, no prose, no markdown fences: {"tags": '
    '{"<axis_id>": "<level>"}} with exactly these axis ids and each value '
    f"one of {list(LEVEL_VALUES)}."
)
_VOCAB_BLOCK = _vocab_block()


_MAX_ITEM_CHARS = 4000  # ponytail: hard truncation; raise if long DROP passages matter


def _item_block(item: dict) -> str:
    parts = [str(item.get("question", ""))]
    choices = item.get("choices")
    if choices:
        parts.append("Choices: " + " | ".join(str(c) for c in choices))
    if item.get("answer"):
        parts.append(f"Answer: {item['answer']}")
    return "\n".join(parts)[:_MAX_ITEM_CHARS]


def build_user_prompt(item: dict) -> str:
    return f"{_VOCAB_BLOCK}\n\n## Item\n{_item_block(item)}\n\n{_SCHEMA_INSTRUCTION}"


def sample_items(bench: str, seed: int, n: int | None = None) -> list[dict]:
    """Seeded uniform sample of n items from the full unbiased task pool."""
    n = N_ITEMS_PER_BENCH if n is None else n
    path = DATA_DIR / bench / "tasks.jsonl"
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return random.Random(seed).sample(items, min(n, len(items)))


def write_manifest(run_dir: Path, seed: int, samples: dict[str, list[dict]]) -> None:
    manifest = {
        bench: {"seed": seed, "item_ids": [it["item_id"] for it in samples[bench]]}
        for bench in BENCHMARKS
    }
    (run_dir / "sample_manifest.json").write_text(json.dumps(manifest, indent=2))


def parse_tags(raw: str) -> dict[str, str] | None:
    """Validate a tagging response; None if invalid (missing/extra axis, bad level)."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    tags = data.get("tags") if isinstance(data, dict) else None
    if not isinstance(tags, dict) or set(tags.keys()) != set(AXIS_IDS):
        return None
    if any(v not in LEVEL_VALUES for v in tags.values()):
        return None
    return tags


def _call_model(model: str, system: str, user: str) -> tuple[str, dict[str, int]]:
    """One raw chat completion; returns (content, usage). Empty content on any failure."""
    try:
        client = shared_factory().get(
            base_url_env="SELFHOST_BASE_URL", api_key_env="SELFHOST_API_KEY"
        )
        extra = {} if REASONING_ENABLED else {"extra_body": {"reasoning": {"enabled": False}}}
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **extra,
        )
        content = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        tok_in = getattr(usage, "prompt_tokens", 0) or 0
        tok_out = getattr(usage, "completion_tokens", 0) or 0
        return content, {"prompt_tokens": tok_in, "completion_tokens": tok_out}
    except Exception:
        return "", {"prompt_tokens": 0, "completion_tokens": 0}


def _fake_tags(model: str, bench: str, item_id: str, pass_k: int) -> dict[str, str]:
    rng = random.Random(f"{model}|{bench}|{item_id}|{pass_k}")
    return {axis_id: rng.choice(list(LEVEL_VALUES)) for axis_id in AXIS_IDS}


def tag_item(model: str, bench: str, pass_k: int, item: dict, *, dry_run: bool) -> dict:
    """Tag one item (1 try + 1 retry on invalid output); returns a cache row."""
    item_id = item["item_id"]
    if dry_run:
        tags = _fake_tags(model, bench, item_id, pass_k)
        zero_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        return {"item_id": item_id, "valid": True, "tags": tags, "usage": zero_usage}

    user = build_user_prompt(item)
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    tags = None
    for _attempt in range(2):
        content, usage = _call_model(model, TAGGER_INSTRUCTIONS, user)
        usage_total["prompt_tokens"] += usage["prompt_tokens"]
        usage_total["completion_tokens"] += usage["completion_tokens"]
        tags = parse_tags(content)
        if tags is not None:
            break
    row = {"item_id": item_id, "valid": tags is not None, "tags": tags, "usage": usage_total}
    if tags is None:
        row["raw"] = content[:500]  # ponytail: keep failing output so invalids are triageable
    return row


def _tag_path(run_dir: Path, model: str, bench: str, pass_k: int) -> Path:
    d = run_dir / "tags" / model
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{bench}_pass{pass_k}.jsonl"


def _load_cached(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    cached = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            cached[row["item_id"]] = row
    return cached


_write_lock = threading.Lock()


def _append_row(path: Path, row: dict) -> None:
    with _write_lock:
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")


def run_tagging(
    run_dir: Path, models: dict[str, str], samples: dict[str, list[dict]], *, dry_run: bool
) -> None:
    """Fan out uncached (model, bench, pass, item) tagging calls; resume-safe."""
    tasks = []
    for name, model in models.items():
        n_passes = 1 if name == "teacher" else 2
        for bench in BENCHMARKS:
            for pass_k in range(1, n_passes + 1):
                path = _tag_path(run_dir, name, bench, pass_k)
                cached = _load_cached(path)
                for item in samples[bench]:
                    prev = cached.get(item["item_id"])
                    # ponytail: re-tag on resume if never done or previously invalid;
                    # append-only file, _load_cached dedups by item_id (last row wins)
                    if prev is None or not prev.get("valid"):
                        tasks.append((name, model, bench, pass_k, item, path))

    def _run(task: tuple) -> None:
        name, model, bench, pass_k, item, path = task
        row = tag_item(model, bench, pass_k, item, dry_run=dry_run)
        _append_row(path, row)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_run, t) for t in tasks]
        for f in tqdm(
            concurrent.futures.as_completed(futures), total=len(tasks), desc="tagging"
        ):
            f.result()


def collect_model_data(run_dir: Path, models: dict[str, str]) -> dict[str, dict]:
    """Load cached tag rows into per-model vectors/usage/valid-rate counters."""
    data = {}
    for name in models:
        n_passes = 1 if name == "teacher" else 2
        vectors: dict[int, dict[str, dict[str, list[float]]]] = {}
        n_total = 0
        n_valid = 0
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        for pass_k in range(1, n_passes + 1):
            vectors[pass_k] = {}
            for bench in BENCHMARKS:
                rows = _load_cached(_tag_path(run_dir, name, bench, pass_k))
                n_total += len(rows)
                bench_vecs = {}
                for item_id, row in rows.items():
                    row_usage = row.get("usage") or {}
                    usage["prompt_tokens"] += row_usage.get("prompt_tokens", 0)
                    usage["completion_tokens"] += row_usage.get("completion_tokens", 0)
                    if row.get("valid") and row.get("tags"):
                        n_valid += 1
                        bench_vecs[item_id] = [LEVEL_VALUES[row["tags"][a]] for a in AXIS_IDS]
                vectors[pass_k][bench] = bench_vecs
        data[name] = {
            "model_id": models[name],
            "vectors": vectors,
            "n_total": n_total,
            "n_valid": n_valid,
            "usage": usage,
        }
    return data


def _aligned_pair(
    a_vecs: dict[str, list[float]], b_vecs: dict[str, list[float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Row-align two item_id->vector dicts on their shared valid ids (paired design)."""
    ids = sorted(set(a_vecs) & set(b_vecs))
    if not ids:
        empty = np.zeros((0, len(AXIS_IDS)))
        return empty, empty
    a = np.array([a_vecs[i] for i in ids], dtype=float)
    b = np.array([b_vecs[i] for i in ids], dtype=float)
    return a, b


def _point_T_b(arr: np.ndarray) -> np.ndarray:
    return arr.mean(axis=0) if arr.size else np.zeros(len(AXIS_IDS))


def _resample_pair(
    a_arr: np.ndarray, b_arr: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """One paired bootstrap replicate: the same item draw applied to both models."""
    if a_arr.shape[0] == 0:
        zero = np.zeros(len(AXIS_IDS))
        return zero, zero
    idx = rng.integers(0, a_arr.shape[0], size=a_arr.shape[0])
    return a_arr[idx].mean(axis=0), b_arr[idx].mean(axis=0)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _top5(vec: np.ndarray) -> set[int]:
    return set(np.argsort(vec)[-5:].tolist())


def _pair_cosine_vector(Tbs: list[np.ndarray]) -> np.ndarray:
    """Raw cosine similarity of every benchmark pair, upper-triangular order."""
    out = []
    for i in range(len(Tbs)):
        for j in range(i + 1, len(Tbs)):
            a, b = Tbs[i], Tbs[j]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            out.append(float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0)
    return np.array(out)


Pairs = dict[str, tuple[np.ndarray, np.ndarray]]


def metric_teacher_cosine(pairs: Pairs, rng: np.random.Generator) -> dict[str, dict]:
    """Criterion 2: per-bench Pearson between teacher and candidate T_b, paired bootstrap CI."""
    out = {}
    for bench, (t_arr, c_arr) in pairs.items():
        point = _pearson(_point_T_b(t_arr), _point_T_b(c_arr))
        samples = []
        for _ in range(N_BOOTSTRAP):
            t_b, c_b = _resample_pair(t_arr, c_arr, rng)
            samples.append(_pearson(t_b, c_b))
        lo, hi = np.percentile(samples, CI_PCTL)
        out[bench] = {"point": point, "ci_lo": float(lo), "ci_hi": float(hi)}
    return out


def metric_top5_overlap(pairs: Pairs, rng: np.random.Generator) -> dict:
    """Criterion 3: mean top-5 tag overlap across benches, paired bootstrap CI."""

    def _mean_overlap(profiles: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
        vals = [len(_top5(t) & _top5(c)) / 5.0 for t, c in profiles.values()]
        return float(np.mean(vals))

    point = _mean_overlap({b: (_point_T_b(t), _point_T_b(c)) for b, (t, c) in pairs.items()})
    samples = []
    for _ in range(N_BOOTSTRAP):
        profiles = {b: _resample_pair(t, c, rng) for b, (t, c) in pairs.items()}
        samples.append(_mean_overlap(profiles))
    lo, hi = np.percentile(samples, CI_PCTL)
    return {"point": point, "ci_lo": float(lo), "ci_hi": float(hi)}


def metric_pair_structure(pairs: Pairs, rng: np.random.Generator) -> dict:
    """Criterion 4: Pearson between teacher/candidate pair-cosine vectors, paired bootstrap CI."""

    def _structure_corr(profiles: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
        t_vec = _pair_cosine_vector([t for t, _ in profiles.values()])
        c_vec = _pair_cosine_vector([c for _, c in profiles.values()])
        return _pearson(t_vec, c_vec)

    point = _structure_corr({b: (_point_T_b(t), _point_T_b(c)) for b, (t, c) in pairs.items()})
    samples = []
    for _ in range(N_BOOTSTRAP):
        profiles = {b: _resample_pair(t, c, rng) for b, (t, c) in pairs.items()}
        samples.append(_structure_corr(profiles))
    lo, hi = np.percentile(samples, CI_PCTL)
    return {"point": point, "ci_lo": float(lo), "ci_hi": float(hi)}


def metric_decoding_stability(stab_pairs: Pairs) -> dict:
    """Criterion 5: per-bench Pearson between pass1 and pass2 T_b (real repeat-pass noise, no bootstrap)."""
    return {
        bench: {"point": _pearson(_point_T_b(p1), _point_T_b(p2))}
        for bench, (p1, p2) in stab_pairs.items()
    }


def _cost(usage: dict, model_id: str) -> float:
    p_in, p_out = PRICES.get(model_id, _DEFAULT_PRICE)
    return usage["prompt_tokens"] * p_in + usage["completion_tokens"] * p_out


def _valid_rate(model_data: dict) -> float:
    return model_data["n_valid"] / model_data["n_total"] if model_data["n_total"] else 0.0


def _stable_seed_offset(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def compute_metrics(models: dict[str, str], data: dict, *, seed: int) -> dict:
    teacher_vecs = data["teacher"]["vectors"][1]
    teacher_cost_per_item = _cost(data["teacher"]["usage"], models["teacher"]) / max(
        data["teacher"]["n_total"], 1
    )
    out: dict = {
        "teacher": {
            "json_valid_rate": _valid_rate(data["teacher"]),
            "cost_per_item": teacher_cost_per_item,
        }
    }
    for name in models:
        if name == "teacher":
            continue
        rng = np.random.default_rng(seed + _stable_seed_offset(name) % (2**31))
        cand_p1 = data[name]["vectors"][1]
        cand_p2 = data[name]["vectors"][2]
        pairs = {b: _aligned_pair(teacher_vecs[b], cand_p1[b]) for b in BENCHMARKS}
        stab_pairs = {b: _aligned_pair(cand_p1[b], cand_p2[b]) for b in BENCHMARKS}
        teacher_cosine = metric_teacher_cosine(pairs, rng)
        top5 = metric_top5_overlap(pairs, rng)
        pair_structure = metric_pair_structure(pairs, rng)
        decoding = metric_decoding_stability(stab_pairs)
        valid_rate = _valid_rate(data[name])
        cand_cost_per_item = _cost(data[name]["usage"], models[name]) / max(data[name]["n_total"], 1)
        cost_ratio = (
            teacher_cost_per_item / cand_cost_per_item if cand_cost_per_item > 0 else float("nan")
        )
        passes = {
            "json_valid_rate": valid_rate >= THRESHOLDS["json_valid_rate"],
            "teacher_cosine": all(v["ci_lo"] >= THRESHOLDS["teacher_cosine"] for v in teacher_cosine.values()),
            "top5_overlap": top5["ci_lo"] >= THRESHOLDS["top5_overlap"],
            "pair_structure": pair_structure["ci_lo"] >= THRESHOLDS["pair_structure"],
            "decoding_stability": all(
                v["point"] >= THRESHOLDS["decoding_stability"] for v in decoding.values()
            ),
            "cost_ratio": cost_ratio >= THRESHOLDS["cost_ratio"] if cost_ratio == cost_ratio else False,
        }
        out[name] = {
            "json_valid_rate": valid_rate,
            "teacher_cosine": teacher_cosine,
            "top5_overlap": top5,
            "pair_structure": pair_structure,
            "decoding_stability": decoding,
            "cost_per_item": {
                "teacher": teacher_cost_per_item,
                "candidate": cand_cost_per_item,
                "ratio": cost_ratio,
            },
            "pass": passes,
            "overall_pass": all(passes.values()),
        }
    return out


def render_report(metrics: dict) -> str:
    lines = ["# Tagger Calibration Report", ""]
    lines.append("| model | valid% | teacher_cos(min) | top5 | pair | decode(min) | cost_ratio | PASS |")
    lines.append("|---|---|---|---|---|---|---|---|")
    winner, winner_cost = None, float("inf")
    for name, m in metrics.items():
        if name == "teacher":
            lines.append(f"| teacher | {m['json_valid_rate']:.3f} | - | - | - | - | - | - |")
            continue
        tc_min = min(v["ci_lo"] for v in m["teacher_cosine"].values())
        dec_min = min(v["point"] for v in m["decoding_stability"].values())
        lines.append(
            f"| {name} | {m['json_valid_rate']:.3f} | {tc_min:.3f} | "
            f"{m['top5_overlap']['ci_lo']:.3f} | {m['pair_structure']['ci_lo']:.3f} | "
            f"{dec_min:.3f} | {m['cost_per_item']['ratio']:.2f} | "
            f"{'PASS' if m['overall_pass'] else 'fail'} |"
        )
        if m["overall_pass"] and m["cost_per_item"]["candidate"] < winner_cost:
            winner, winner_cost = name, m["cost_per_item"]["candidate"]
    lines.append("")
    lines.append(f"**Winner (cheapest passing model):** {winner or 'none — no candidate passed all criteria'}")
    return "\n".join(lines) + "\n"


def parse_models_arg(pairs: list[str]) -> dict[str, str]:
    models = {}
    for pair in pairs:
        name, _, model_id = pair.partition("=")
        if not model_id:
            raise ValueError(f"--models entry must be name=model_id, got: {pair!r}")
        models[name] = model_id
    if "teacher" not in models:
        raise ValueError("--models must include a 'teacher=<model_id>' entry")
    return models


def main(argv: list[str] | None = None) -> None:
    global REASONING_ENABLED, MAX_WORKERS
    parser = argparse.ArgumentParser(description="Stage 1 tagger calibration")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--models", nargs="+", required=True, help="name=model_id pairs; must include teacher=..."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-reasoning", action="store_true", help="disable model reasoning (strips ~50x output tokens)"
    )
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="concurrent gateway calls")
    args = parser.parse_args(argv)

    REASONING_ENABLED = not args.no_reasoning
    MAX_WORKERS = args.workers
    models = parse_models_arg(args.models)
    run_dir = RESULTS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    samples = {bench: sample_items(bench, args.seed) for bench in BENCHMARKS}
    write_manifest(run_dir, args.seed, samples)

    run_tagging(run_dir, models, samples, dry_run=args.dry_run)

    data = collect_model_data(run_dir, models)
    metrics = compute_metrics(models, data, seed=args.seed)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "report.md").write_text(render_report(metrics))


if __name__ == "__main__":
    main()
