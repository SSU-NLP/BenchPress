"""experiment/loop.py — Part 1 orchestrator.

Step C–H + baselines + best-so-far rollback. Pure-Python; LLM clients are injected via
chat_fn callables for testability.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import os
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from tqdm.auto import tqdm

from autotagging_loop.experiment.alignment import (
    alignment_corr,
    alignment_loss,
    bootstrap_metrics,
    build_error_report,
    build_residual_report,
    cosine_pair_matrix,
    error_pairs_to_dicts,
    intra_inter_gap,
    quantile_thresholds,
)
from autotagging_loop.experiment.config import (
    llm_debug_dump_dir,
    llm_empty_content_retries,
    llm_extra_body,
    llm_request_timeout_s,
    llm_sdk_exception_retries,
    role_cfg,
)
from autotagging_loop.experiment.corpus import Corpus, load_corpus
from autotagging_loop.experiment.executer import run_executer
from autotagging_loop.experiment.mapreduce_evidence import build_mapreduce_descriptions
from autotagging_loop.experiment.maker import run_maker
from autotagging_loop.experiment.no_seed_taxonomy import NoSeedTaxonomyResult, induce_no_seed_taxonomy
from autotagging_loop.experiment.prompt_improver import ImproverResult, improve_prompt
from autotagging_loop.experiment.score_matrix import (
    normalize_matrix,
    spearman_pair_matrix,
    to_R01,
)
from autotagging_loop.experiment.split_diagnostics import (
    benchmark_split_from_config,
    split_pair_count_failures,
    split_valid_pair_counts,
)
from autotagging_loop.experiment.splits import (
    induced_pair_set,
    restrict_pair_dict,
    split_models,
)
from autotagging_loop.experiment.split_metrics import (
    compute_held_model_test_metrics,
    compute_split_metrics,
    write_split_metrics_json,
)
from autotagging_loop.experiment.storage import (
    make_run_dir,
    save_config,
    save_corpus,
    save_final,
    save_iteration,
    save_no_seed_taxonomy,
    save_profile_support,
    save_score_matrix,
    save_selection,
    save_taxonomy_final,
    save_taxonomy_refinement,
    save_taxonomy_status,
    write_json,
)
from autotagging_loop.experiment.static_tag_weights import build_static_tag_vectors_from_reducer_levels
from autotagging_loop.experiment.tag_generator import (
    TagVector,
    generate_tag_vector,
    random_tag_vectors,
)
from autotagging_loop.experiment.taxonomy_refiner import TaxonomyRefinementResult, refine_taxonomy
from autotagging_loop.experiment.weight_optimizer import optimize_tag_weights
from autotagging_loop.experiment.pipeline.selection import (
    build_protected_pairs as _build_protected_pairs,
    finite_float as _finite_float,
    finite_ge as _finite_ge,
    is_better as _is_better,
    metric_delta as _metric_delta,
    passes_delta_tag_gate as _passes_delta_tag_gate,
    probe_l_for_selection as _probe_l_for_selection,
    taxonomy_adoption_decision as _taxonomy_adoption_decision,
    taxonomy_trigger_status as _taxonomy_trigger_status,
)


TagFn = Callable[[str, str, list[dict], str, int], TagVector]
ImproverCallable = Callable[..., ImproverResult]
TaxonomyRefinerCallable = Callable[..., TaxonomyRefinementResult]
NoSeedTaxonomyCallable = Callable[..., NoSeedTaxonomyResult]
MapReduceChatFn = Callable[[str, str], str]
ExecuterChatFn = Callable[..., str]


@dataclass
class IterationResult:
    label: str
    iter: int
    prompt: str
    T: dict[str, dict[str, float]]
    S: dict[tuple[str, str], float]
    L_align: float
    L_align_01: float
    rho_align_pearson: float
    rho_align_spearman: float
    delta_tag: float
    bootstrap: dict
    error_report_size: int
    improver: dict | None = None
    tag_weight_metadata: dict | None = None
    # codex 2026-05-10 #5 — needed so the final phase can resurrect the V used
    # by the best iteration. Both fields are None on the legacy fixed-V path.
    vocab: list[dict] | None = None
    vocab_hash: str | None = None
    dev_metrics: dict | None = None
    train_metrics: dict | None = None
    model_probe_dev_metrics: dict | None = None


def _vocab_hash(vocab: list[dict] | None) -> str | None:
    if not vocab:
        return None
    payload = json.dumps(
        [{"id": v.get("id"), "definition": v.get("definition")} for v in vocab],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class TagGenerationResult:
    T: dict[str, dict[str, float]]
    descriptions: dict[str, str]
    mapreduce_aggregates: dict[str, dict] = field(default_factory=dict)
    tag_weight_metadata: dict | None = None


def _load_vocab(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _build_default_tag_fn(
    model: str,
    base_url: str | None,
    weight_bounds: tuple[float, float],
    *,
    base_url_env: str | None = None,
    api_key_env: str | None = None,
    allow_uniform_fallback: bool = False,
) -> TagFn:
    def tag(benchmark: str, description: str, vocab: list[dict], prompt: str, version: int) -> TagVector:
        return generate_tag_vector(
            benchmark=benchmark,
            description=description,
            samples=None,
            vocab=vocab,
            prompt=prompt,
            model=model,
            base_url=base_url,
            seed=0,
            prompt_version=version,
            weight_bounds=weight_bounds,
            base_url_env=base_url_env,
            api_key_env=api_key_env,
            allow_uniform_fallback=allow_uniform_fallback,
        )
    return tag


def _compute_metrics(
    T: dict[str, dict[str, float]],
    benchmark_names: list[str],
    R_raw: dict,
    R01: dict,
    q_p: float,
    q_n: float,
    bootstrap_B: int,
    seed: int,
) -> tuple[dict, dict, dict]:
    """Returns (S, point_metrics, bootstrap_summary)."""
    S = cosine_pair_matrix(T, benchmark_names=benchmark_names)
    L = alignment_loss(S, R_raw)
    L01 = alignment_loss(S, R01)
    pr, sp = alignment_corr(S, R_raw)
    # All point metrics must operate on the same comparable-pair universe Ω.
    # L/rho already drop pairs whose score-pattern similarity is undefined;
    # Δ_tag must not let those undefined-R pairs influence its S quantiles.
    S_valid = {k: v for k, v in S.items() if R_raw.get(k) is not None}
    theta_p, theta_n = quantile_thresholds(S_valid, q_p=q_p, q_n=q_n)
    gap = intra_inter_gap(S_valid, R_raw, theta_p, theta_n)
    boot = bootstrap_metrics(S, R_raw, R01, B=bootstrap_B, seed=seed, q_p=q_p, q_n=q_n)
    residuals = [
        abs(float(sv) - float(R_raw[k]))
        for k, sv in S_valid.items()
    ]
    metrics = {
        "L_align": L,
        "L_align_01": L01,
        "rho_align_pearson": pr,
        "rho_align_spearman": sp,
        "theta_p": theta_p,
        "theta_n": theta_n,
        "intra_tag_sim": gap["intra"],
        "inter_tag_sim": gap["inter"],
        "delta_tag": gap["delta"],
        "n_pos": gap["n_pos"],
        "n_neg": gap["n_neg"],
        "n_pairs": len(S_valid),
        "residual_mean": float(sum(residuals) / len(residuals)) if residuals else float("nan"),
        "residual_max": float(max(residuals)) if residuals else float("nan"),
    }
    return S, metrics, boot


def _generate_T_via_prompt(
    benchmark_names: list[str],
    descriptions: dict[str, str],
    vocab: list[dict],
    prompt: str,
    version: int,
    tag_fn: TagFn,
    *,
    desc: str = "tagging",
    max_workers: int = 8,
) -> dict[str, dict[str, float]]:
    """Phase C — concurrent per-benchmark tag generation.

    `tag_fn` for each benchmark is independent. The endpoint Semaphore
    registered by Phase A is the real concurrency ceiling; `max_workers`
    is just the pool size (default 8). Output dict is sorted on return so
    downstream consumers see byte-identical order regardless of completion.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    T: dict[str, dict[str, float]] = {}
    if not benchmark_names:
        return T
    workers = max(1, min(int(max_workers), len(benchmark_names)))
    pbar = tqdm(total=len(benchmark_names), desc=f"  [{desc}]", unit="bench", leave=False)

    def _do(b: str) -> tuple[str, dict[str, float]]:
        tv = tag_fn(b, descriptions.get(b, ""), vocab, prompt, version)
        return b, tv.weights

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_do, b) for b in benchmark_names]
        for fut in as_completed(futures):
            b, weights = fut.result()
            T[b] = weights
            pbar.set_postfix_str(b[:40])
            pbar.update(1)
    pbar.close()
    return {k: T[k] for k in sorted(T)}


def _mapper_prompt_for_static(
    config: dict,
    base_prompt: str | None,
    current_prompt: str,
) -> str | None:
    source = str(config.get("mapreduce_mapper_prompt_source", "base")).strip().lower()
    if source in {"base", "i0"}:
        return base_prompt or current_prompt
    if source in {"none", "default"}:
        return None
    if source in {"iteration", "current"}:
        return current_prompt
    if source == "custom":
        custom = config.get("mapreduce_mapper_prompt")
        if not custom:
            raise ValueError(
                "mapreduce_mapper_prompt_source='custom' requires mapreduce_mapper_prompt"
            )
        return str(custom)
    raise ValueError(
        "mapreduce_mapper_prompt_source must be one of: base, iteration, none, custom"
    )


def _generate_T_for_prompt(
    *,
    corpus: Corpus,
    benchmark_names: list[str],
    descriptions: dict[str, str],
    vocab: list[dict],
    prompt: str,
    version: int,
    tag_fn: TagFn | None,
    config: dict,
    run_dir: str,
    label: str,
    static_from_mapreduce: bool,
    base_prompt: str | None = None,
    mapreduce_chat_fn: MapReduceChatFn | None = None,
    mapreduce_reducer_chat_fn: MapReduceChatFn | None = None,
) -> TagGenerationResult:
    if static_from_mapreduce:
        mapper_prompt = _mapper_prompt_for_static(config, base_prompt, prompt)
        descriptions_i, aggregates = build_mapreduce_descriptions(
            corpus=corpus,
            vocab=vocab,
            config=config,
            run_dir=run_dir,
            prompt=mapper_prompt,
            chat_fn=mapreduce_chat_fn,
        )
        reducer_outputs, reducer_metadata = run_maker(
            benchmark_names=benchmark_names,
            vocab=vocab,
            aggregates=aggregates,
            config=config,
            run_dir=run_dir,
            prompt=prompt,
            version=version,
            label=label,
            chat_fn=mapreduce_reducer_chat_fn,
            seed=_role_iteration_seed(config, "maker", version),
        )
        T_i, metadata = build_static_tag_vectors_from_reducer_levels(
            benchmark_names=benchmark_names,
            vocab=vocab,
            reducer_outputs=reducer_outputs,
            config=config,
        )
        metadata = {
            **metadata,
            **reducer_metadata,
            "iteration_label": label,
            "prompt_version": version,
            "mapper_prompt_source": config.get("mapreduce_mapper_prompt_source", "base"),
            "prompt_drives": (
                "current prompt drives benchmark-level LLM reduction; "
                "mapper evidence prompt is fixed and final weights are deterministic"
            ),
            "mapreduce_aggregate_count": len(aggregates),
        }
        print(
            f"  [static_weights] {label}: computed T for {len(T_i)} benchmarks "
            f"from MapReduce reducer levels"
        )
        return TagGenerationResult(
            T=T_i,
            descriptions=descriptions_i,
            mapreduce_aggregates=aggregates,
            tag_weight_metadata=metadata,
        )

    if tag_fn is None:
        raise ValueError("tag_fn is required when tag_weight_mode is not static_from_mapreduce")
    return TagGenerationResult(
        T=_generate_T_via_prompt(
            benchmark_names, descriptions, vocab, prompt, version, tag_fn,
            desc=label,
            max_workers=int(config.get("taggen_max_workers", 8)),
        ),
        descriptions=descriptions,
    )


def _subset_corpus(corpus: Corpus, keep: list[str], reason: str) -> Corpus:
    keep_set = set(keep)
    dropped = [b for b in corpus.benchmark_names if b not in keep_set]
    drop_log = dict(corpus.drop_log)
    for benchmark in dropped:
        drop_log[benchmark] = reason
    return Corpus(
        benchmark_names=[b for b in corpus.benchmark_names if b in keep_set],
        model_names=corpus.model_names,
        Y={b: corpus.Y[b] for b in corpus.benchmark_names if b in keep_set},
        descriptions={b: corpus.descriptions.get(b, "") for b in corpus.benchmark_names if b in keep_set},
        documents={b: corpus.documents[b] for b in corpus.benchmark_names if b in corpus.documents and b in keep_set},
        drop_log=drop_log,
    )


def _maybe_optimize_T(
    T_initial: dict[str, dict[str, float]],
    benchmark_names: list[str],
    R_raw: dict,
    vocab: list[dict],
    config: dict,
) -> tuple[dict[str, dict[str, float]], dict]:
    """Fine-tune weights w so pairwise tag similarity follows score similarity."""
    if not config.get("optimize_tag_weights", False):
        return T_initial, {"enabled": False}

    result = optimize_tag_weights(
        T_initial,
        R_raw,
        benchmark_names,
        [v["id"] for v in vocab],
        target_scale=config.get("weight_target_scale", "raw"),
        bounds=tuple(config.get("weight_bounds", [0.0, 1.0])),
        l2_lambda=float(config.get("weight_l2_lambda", 0.01)),
        max_iter=int(config.get("weight_max_iter", 200)),
    )
    return result.T, {
        "enabled": True,
        "initial_loss": result.initial_loss,
        "optimized_loss": result.optimized_loss,
        "n_pairs": result.n_pairs,
        "target_scale": result.target_scale,
        "bounds": list(result.bounds),
        "success": result.success,
        "message": result.message,
        "iterations": result.iterations,
        "clipped_negative_targets": result.clipped_negative_targets,
    }


def _calibration_enabled(config: dict) -> bool:
    return bool(
        config.get("run_weight_calibration_ablation", False)
        or config.get("optimize_tag_weights", False)
    )


def _maybe_build_calibration(
    T_raw: dict[str, dict[str, float]],
    benchmark_names: list[str],
    R_raw: dict,
    R01: dict,
    vocab: list[dict],
    config: dict,
) -> dict | None:
    """Optional ablation: calibrate weights after raw prompt scoring."""
    if not _calibration_enabled(config):
        return None

    T_cal, opt = _maybe_optimize_T(
        T_raw,
        benchmark_names,
        R_raw,
        vocab,
        {**config, "optimize_tag_weights": True},
    )
    S_cal, m_cal, boot_cal = _compute_metrics(
        T_cal,
        benchmark_names,
        R_raw,
        R01,
        config["theta_p_q"],
        config["theta_n_q"],
        config["bootstrap_B"],
        config["seed"],
    )
    return {
        "T": T_cal,
        "S": S_cal,
        "metrics": {**m_cal, "bootstrap": boot_cal, "weight_optimization": opt},
    }


def _profile_cosine(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) | set(b))
    if not keys:
        return float("nan")
    dot = sum(float(a.get(k, 0.0)) * float(b.get(k, 0.0)) for k in keys)
    norm_a = math.sqrt(sum(float(a.get(k, 0.0)) ** 2 for k in keys))
    norm_b = math.sqrt(sum(float(b.get(k, 0.0)) ** 2 for k in keys))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return float("nan")
    return dot / (norm_a * norm_b)


def _profile_for_model(
    model: str,
    benchmarks: list[str],
    Y_norm: dict[str, dict[str, float]],
    T: dict[str, dict[str, float]],
) -> dict[str, float]:
    profile: dict[str, float] = {}
    for bench in benchmarks:
        score = Y_norm.get(bench, {}).get(model)
        weights = T.get(bench, {})
        if score is None or not weights:
            continue
        for tag_id, weight in weights.items():
            profile[tag_id] = profile.get(tag_id, 0.0) + float(score) * float(weight)
    return profile


def _select_tag_cover_subset(
    T: dict[str, dict[str, float]],
    benchmark_names: list[str],
    k: int,
) -> list[str]:
    """Deterministic greedy subset that covers the final tag matrix dimensions."""
    remaining = [b for b in benchmark_names if b in T]
    selected: list[str] = []
    covered: dict[str, float] = {}
    for _ in range(min(max(k, 0), len(remaining))):
        best_name = ""
        best_score = -1.0
        for bench in remaining:
            vec = T.get(bench, {})
            gain = 0.0
            norm = 0.0
            for tag_id, weight in vec.items():
                w = max(0.0, float(weight))
                norm += w * w
                gain += max(covered.get(tag_id, 0.0), w) - covered.get(tag_id, 0.0)
            score = gain + 1e-6 * math.sqrt(norm)
            if score > best_score or (score == best_score and bench < best_name):
                best_score = score
                best_name = bench
        if not best_name:
            break
        selected.append(best_name)
        remaining.remove(best_name)
        for tag_id, weight in T.get(best_name, {}).items():
            covered[tag_id] = max(covered.get(tag_id, 0.0), max(0.0, float(weight)))
    return selected


def _select_kmedoids_subset(
    T: dict[str, dict[str, float]],
    benchmark_names: list[str],
    k: int,
    *,
    seed: int = 0,
    max_iter: int = 64,
) -> list[str]:
    """v3 §2.2.10 k-medoids subset selection on tag-vector cosine distance.

    PAM-style alternation: pick `k` initial medoids by `random.Random(seed)`,
    assign every benchmark to its closest medoid by `1 - cos(T[a], T[b])`,
    then for each cluster swap in the member that minimizes the total
    intra-cluster distance. Iterate until medoids stabilize or `max_iter`.

    Returns medoids in lex-sorted order so tie-breaks are deterministic across
    runs that produce the same medoid set.
    """
    candidates = sorted(b for b in benchmark_names if b in T)
    if k <= 0 or not candidates:
        return []
    if k >= len(candidates):
        return list(candidates)

    rng = random.Random(int(seed))

    def vec(b: str) -> dict[str, float]:
        return T.get(b, {})

    def dist(a: str, b: str) -> float:
        v1 = vec(a)
        v2 = vec(b)
        if not v1 or not v2:
            return 1.0
        n1 = math.sqrt(sum(x * x for x in v1.values()))
        n2 = math.sqrt(sum(x * x for x in v2.values()))
        if n1 == 0.0 or n2 == 0.0:
            return 1.0
        keys = set(v1) | set(v2)
        dot = sum(float(v1.get(k_, 0.0)) * float(v2.get(k_, 0.0)) for k_ in keys)
        cos = dot / (n1 * n2)
        return 1.0 - max(-1.0, min(1.0, cos))

    medoids = sorted(rng.sample(candidates, k))
    for _ in range(int(max_iter)):
        clusters: dict[str, list[str]] = {m: [] for m in medoids}
        for b in candidates:
            best_m = medoids[0]
            best_d = dist(b, medoids[0])
            for m in medoids[1:]:
                d = dist(b, m)
                if d < best_d or (d == best_d and m < best_m):
                    best_d = d
                    best_m = m
            clusters[best_m].append(b)

        new_medoids: list[str] = []
        for m, members in clusters.items():
            if not members:
                new_medoids.append(m)
                continue
            best_member = members[0]
            best_total = float("inf")
            for cand in members:
                total = sum(dist(cand, other) for other in members)
                if total < best_total or (total == best_total and cand < best_member):
                    best_total = total
                    best_member = cand
            new_medoids.append(best_member)
        new_medoids = sorted(set(new_medoids))
        if new_medoids == medoids:
            break
        # Pad if a duplicate medoid collapsed two clusters.
        while len(new_medoids) < k:
            for cand in candidates:
                if cand not in new_medoids:
                    new_medoids.append(cand)
                    break
        medoids = sorted(new_medoids[:k])
    return list(medoids)


def _build_profile_support(
    Y_norm: dict[str, dict[str, float]],
    T: dict[str, dict[str, float]],
    benchmark_names: list[str],
    model_names: list[str],
    subset_sizes: list[int],
    methods: list[str] | None = None,
    kmedoids_seed: int = 0,
) -> dict:
    """Check whether T* can support model profiles from selected benchmark subsets.

    `methods` is a v3 §2.2.10 gate: `"greedy"` runs the original tag-coverage
    selector; `"kmedoids"` runs the new PAM-style alternation. Both write to
    `final/subset_profiling.json` under their own section.
    """
    requested = list(methods) if methods else ["greedy"]
    full_profiles = {
        model: _profile_for_model(model, benchmark_names, Y_norm, T)
        for model in model_names
    }

    def evaluate(selected: list[str]) -> dict:
        sims: dict[str, float] = {}
        for model in model_names:
            subset_profile = _profile_for_model(model, selected, Y_norm, T)
            sim = _profile_cosine(full_profiles.get(model, {}), subset_profile)
            if not math.isnan(sim):
                sims[model] = sim
        vals = list(sims.values())
        return {
            "selected_benchmarks": selected,
            "model_profile_cosine": sims,
            "mean_profile_cosine": float(sum(vals) / len(vals)) if vals else float("nan"),
            "min_profile_cosine": float(min(vals)) if vals else float("nan"),
            "n_models_compared": len(vals),
        }

    selectors: dict[str, callable] = {
        "greedy": lambda k: _select_tag_cover_subset(T, benchmark_names, k),
        "kmedoids": lambda k: _select_kmedoids_subset(
            T, benchmark_names, k, seed=kmedoids_seed
        ),
    }
    method_descriptions = {
        "greedy": "greedy tag-coverage subset; compare P_full and P_subset by cosine",
        "kmedoids": "k-medoids on (1 - cos(T)) distance with deterministic init seed",
    }

    method_results: dict[str, dict] = {}
    for method in requested:
        selector = selectors.get(method)
        if selector is None:
            continue
        subsets: dict[str, dict] = {}
        for raw_k in subset_sizes:
            k = int(raw_k)
            if k <= 0:
                continue
            subsets[str(k)] = evaluate(selector(k))
        method_results[method] = {
            "method": method_descriptions[method],
            "subsets": subsets,
        }

    primary = requested[0] if requested else "greedy"
    primary_result = method_results.get(primary, {})
    return {
        "method": method_descriptions.get(primary, primary_result.get("method", "")),
        "full_profile_formula": "P_r = Y_norm[r, :] @ T_star",
        "full_profiles": full_profiles,
        "subsets": primary_result.get("subsets", {}),
        "methods": method_results,
    }


def _write_split_metrics_artifact(
    *,
    run_dir: str,
    config: dict,
    S: dict,
    R_raw: dict,
    R01: dict,
    Y_norm: dict[str, dict[str, float]],
    Y_norm_for_held: dict[str, dict[str, float]] | None = None,
    benchmark_names: list[str],
    model_names: list[str],
    split_required_pair_dicts: list[dict] | None = None,
) -> None:
    """v3 §2.2.7 split-aware metrics writer.

    Writes `final/split_metrics.json` with train / dev / test / held_model_test
    blocks. Splits are deterministic given `splits.benchmark_seed` /
    `splits.model_seed` (defaults: 0). Skipped quietly if the corpus is too
    small to split.
    """
    if len(benchmark_names) < 3:
        return
    splits_cfg = config.get("splits", {}) or {}
    model_ratios = tuple(splits_cfg.get("model_ratios", (0.8, 0.2)))
    model_seed = int(splits_cfg.get("model_seed", 0))
    fold = int(splits_cfg.get("fold", 0))
    bootstrap_B = int(splits_cfg.get("bootstrap_B", config.get("bootstrap_B", 1000)))
    q_p = float(config.get("q_p", 0.80))
    q_n = float(config.get("q_n", 0.20))
    min_common_models = int(
        config.get("min_common_models", config.get("min_common", 8))
    )

    bench_split = benchmark_split_from_config(
        benchmark_names,
        splits_cfg,
        score_pair_dict=R_raw,
        required_pair_dicts=split_required_pair_dicts,
        min_test_valid_pairs=int(config.get("v_loop_min_test_valid_pairs", 0)),
        min_test_effective_benchmarks=int(
            config.get("v_loop_min_test_effective_benchmarks", 0)
        ),
    )
    model_split = split_models(
        model_names,
        ratios=model_ratios,
        seed=model_seed,
        strategy=splits_cfg.get("model_split_strategy", "random"),
    ) if model_names else None

    blocks = compute_split_metrics(
        S=S, R_raw=R_raw, R01=R01,
        benchmark_split=bench_split,
        q_p=q_p, q_n=q_n,
        bootstrap_B=bootstrap_B,
        seed=bench_split.seed,
    )
    held = (
        compute_held_model_test_metrics(
            S=S, Y_norm=Y_norm_for_held or Y_norm,
            benchmark_split=bench_split, model_split=model_split,
            q_p=q_p, q_n=q_n, bootstrap_B=bootstrap_B, seed=model_seed,
            min_common=min_common_models,
        )
        if model_split is not None
        else None
    )
    write_split_metrics_json(
        run_dir,
        fold=fold, seed=bench_split.seed,
        benchmark_split=bench_split,
        model_split=model_split,
        train_dev_test=blocks,
        held_model_test=held,
    )


def _model_split_for_score_matrix(config: dict, model_names: list[str]):
    splits_cfg = config.get("splits", {}) or {}
    if not model_names:
        return None
    return split_models(
        model_names,
        ratios=tuple(splits_cfg.get("model_ratios", (0.8, 0.2))),
        seed=int(splits_cfg.get("model_seed", 0)),
        strategy=splits_cfg.get("model_split_strategy", "random"),
    )


def _v_loop_score_model_scope(config: dict) -> str:
    scope = str(config.get("v_loop_score_model_scope", "all")).strip().lower()
    if bool(config.get("enable_v_loop", False)) and bool(
        config.get("v_loop_require_held_model_test", False)
    ):
        scope = "seen"
    if scope not in {"all", "seen"}:
        raise ValueError(
            "v_loop_score_model_scope must be one of {'all', 'seen'}, "
            f"got {scope!r}"
        )
    return scope


def _filter_corpus_scores_by_models(
    Y: dict[str, dict[str, float]],
    models: list[str] | set[str],
) -> dict[str, dict[str, float]]:
    keep = set(models)
    return {
        bench: {
            model: float(score)
            for model, score in scores.items()
            if model in keep
        }
        for bench, scores in Y.items()
    }


def _setup_model_probe_state(
    *,
    config: dict,
    corpus: Corpus,
    score_model_split: Any,
) -> dict[str, Any] | None:
    """Build leave-one-seen-model-out score matrices for selection stability."""

    if not bool(config.get("best_iter_model_probe_enabled", False)):
        return None
    if score_model_split is None:
        print("  [model_probe] disabled: no model split")
        return None
    seen_models = sorted(score_model_split.seen)
    if len(seen_models) < 3:
        print(f"  [model_probe] disabled: seen_models={len(seen_models)}<3")
        return None

    configured_min_common = config.get("best_iter_model_probe_min_common")
    if configured_min_common is None:
        min_common = max(
            2,
            min(int(config.get("min_common_models", 6)), len(seen_models) - 1),
        )
    else:
        min_common = max(2, int(configured_min_common))

    probes: list[dict[str, Any]] = []
    for dropped_model in seen_models:
        probe_models = [model for model in seen_models if model != dropped_model]
        if len(probe_models) < min_common:
            continue
        Y_probe = _filter_corpus_scores_by_models(corpus.Y, probe_models)
        Y_norm_probe = normalize_matrix(Y_probe, method=config["normalize"])
        R_raw_probe, _common_count = spearman_pair_matrix(
            Y_norm_probe,
            corpus.benchmark_names,
            min_common=min_common,
            warn_below=10**9,
        )
        if not any(value is not None for value in R_raw_probe.values()):
            continue
        probes.append(
            {
                "dropped_model": dropped_model,
                "score_model_names": probe_models,
                "R_raw": R_raw_probe,
                "R01": to_R01(R_raw_probe),
            }
        )

    if not probes:
        print(
            "  [model_probe] disabled: no comparable leave-one-out "
            f"seen-model matrices (seen={len(seen_models)}, min_common={min_common})"
        )
        return None
    print(
        f"  [model_probe] enabled: probes={len(probes)}, "
        f"seen={len(seen_models)}, min_common={min_common}"
    )
    return {
        "min_common": min_common,
        "seen_model_names": seen_models,
        "probes": probes,
    }


def _model_probe_summary(state: dict[str, Any] | None) -> dict | None:
    if state is None:
        return None
    return {
        "enabled": True,
        "min_common": state["min_common"],
        "seen_model_names": list(state["seen_model_names"]),
        "probes": [
            {
                "dropped_model": probe["dropped_model"],
                "score_model_names": list(probe["score_model_names"]),
                "n_pairs": int(sum(1 for value in probe["R_raw"].values() if value is not None)),
            }
            for probe in state["probes"]
        ],
    }


def _mean_finite_metric(rows: list[dict], key: str) -> float:
    vals = [
        value
        for value in (_finite_float(row.get(key)) for row in rows)
        if value is not None
    ]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _min_finite_metric(rows: list[dict], key: str) -> float:
    vals = [
        value
        for value in (_finite_float(row.get(key)) for row in rows)
        if value is not None
    ]
    return float(min(vals)) if vals else float("nan")


def _max_finite_metric(rows: list[dict], key: str) -> float:
    vals = [
        value
        for value in (_finite_float(row.get(key)) for row in rows)
        if value is not None
    ]
    return float(max(vals)) if vals else float("nan")


def _compute_model_probe_dev_metrics(
    *,
    T: dict[str, dict[str, float]],
    benchmark_names: list[str],
    model_probe_state: dict[str, Any] | None,
    config: dict,
) -> dict | None:
    if model_probe_state is None:
        return None

    rows: list[dict] = []
    for idx, probe in enumerate(model_probe_state["probes"]):
        _S_probe, metrics, _boot = _compute_metrics(
            T,
            benchmark_names,
            probe["R_raw"],
            probe["R01"],
            config["theta_p_q"],
            config["theta_n_q"],
            0,
            int(config["seed"]) + idx,
        )
        if int(metrics.get("n_pairs", 0) or 0) <= 0:
            continue
        rows.append(
            {
                "dropped_model": probe["dropped_model"],
                "L_align": metrics.get("L_align"),
                "rho_align_pearson": metrics.get("rho_align_pearson"),
                "rho_align_spearman": metrics.get("rho_align_spearman"),
                "delta_tag": metrics.get("delta_tag"),
                "n_pairs": metrics.get("n_pairs"),
            }
        )

    if not rows:
        return None
    return {
        "n_probes": len(rows),
        "min_common": model_probe_state["min_common"],
        "L_align_mean": _mean_finite_metric(rows, "L_align"),
        "L_align_max": _max_finite_metric(rows, "L_align"),
        "rho_align_pearson_mean": _mean_finite_metric(rows, "rho_align_pearson"),
        "rho_align_pearson_min": _min_finite_metric(rows, "rho_align_pearson"),
        "rho_align_spearman_mean": _mean_finite_metric(rows, "rho_align_spearman"),
        "rho_align_spearman_min": _min_finite_metric(rows, "rho_align_spearman"),
        "delta_tag_mean": _mean_finite_metric(rows, "delta_tag"),
        "n_pairs_min": int(min(int(row.get("n_pairs", 0) or 0) for row in rows)),
    }


def _write_llm_fallbacks(run_dir: str) -> dict:
    """Persist per-role counts of LLM calls that fell through to `error_fallback`.

    Surfaces silent degradation (e.g. Cloudflare 524 from a self-hosted backend
    after retries exhausted). Empty `{}` means no fallbacks occurred. Counts are
    process-wide since the most recent `reset_llm_fallback_counts()` at run start.
    """
    import json
    import os

    from autotagging_loop.experiment.llm_client import llm_fallback_counts

    counts = llm_fallback_counts()
    final_dir = os.path.join(run_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    payload = {
        "counts": counts,
        "total": int(sum(counts.values())),
    }
    with open(os.path.join(final_dir, "llm_fallbacks.json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    if payload["total"] > 0:
        print(
            f"  [loop] WARN: {payload['total']} LLM call(s) fell back to "
            f"error_fallback during this run: {counts}"
        )
    return payload


def _write_stop_reason(
    run_dir: str,
    *,
    stalled_delta_tag: bool,
    consecutive_no_improve: int,
    threshold: int,
    status: str | None = None,
    details: dict | None = None,
) -> None:
    """Persist v3 §2.2.6 stop reason next to final/. Created only inside main loop."""
    import json
    import os

    final_dir = os.path.join(run_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    payload = {
        "stalled_delta_tag": bool(stalled_delta_tag),
        "consecutive_no_improve": int(consecutive_no_improve),
        "early_stop_consecutive_threshold": int(threshold),
        "status": status or ("stalled_delta_tag" if stalled_delta_tag else "ok"),
    }
    if details:
        payload["details"] = details
    with open(os.path.join(final_dir, "stop_reason.json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _next_prompt_from_improver(
    improver_result: ImproverResult,
    current_prompt: str,
) -> tuple[str | None, str | None]:
    """Return a usable next prompt, or the stop status when none exists."""
    if not improver_result.accepted:
        return None, "improver_rejected"
    new_prompt = str(improver_result.new_prompt or "")
    if not new_prompt.strip() or new_prompt.strip() == str(current_prompt or "").strip():
        return None, "improver_no_change"
    return improver_result.new_prompt, None


def _can_continue_with_same_prompt_after_improver_stop(
    config: dict,
    *,
    v_loop_active: bool,
) -> bool:
    """Allow exploration to continue when Executer will vary the next candidate."""

    if not v_loop_active:
        return False
    counts = config.get("executer_candidate_counts")
    return isinstance(counts, list) and len(counts) > 1


def _flatten_bootstrap(boot: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(boot, dict):
        return out
    for metric_key, stats in boot.items():
        if not isinstance(stats, dict):
            continue
        for stat_key, value in stats.items():
            try:
                out[f"bootstrap/{metric_key}/{stat_key}"] = float(value)
            except (TypeError, ValueError):
                continue
    return out


def _wandb_log(wandb_run, payload: dict, step: int | None = None) -> None:
    if wandb_run is None:
        return
    try:
        if step is not None:
            wandb_run.log(payload, step=step)
        else:
            wandb_run.log(payload)
    except Exception as exc:
        print(f"  [wandb] log failed (step={step}): {exc}")


def _taxonomy_metrics_for_selection(
    result: IterationResult,
    R_raw: dict[tuple[str, str], float],
) -> dict:
    residuals = [
        abs(float(sv) - float(R_raw[k]))
        for k, sv in result.S.items()
        if R_raw.get(k) is not None
    ]
    return {
        **_metrics_payload(result),
        "residual_mean": (
            float(sum(residuals) / len(residuals)) if residuals else float("nan")
        ),
        "residual_max": float(max(residuals)) if residuals else float("nan"),
        "n_pairs": int(sum(1 for v in R_raw.values() if v is not None)),
    }


def _supports_kwarg(fn: Callable, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD or param_name == name
        for param_name, param in sig.parameters.items()
    )


def _metrics_payload(result: IterationResult) -> dict:
    return {
        "L_align": result.L_align,
        "L_align_01": result.L_align_01,
        "rho_align_pearson": result.rho_align_pearson,
        "rho_align_spearman": result.rho_align_spearman,
        "delta_tag": result.delta_tag,
        "bootstrap": result.bootstrap,
    }


def _result_vocab_source(result: IterationResult, config: dict) -> str:
    if str(result.label).startswith("taxonomy_refinement/"):
        return "taxonomy_refinement"
    if result.vocab is not None:
        return "executer"
    active_source = str(config.get("active_vocab_source") or "seed")
    if active_source == "no_seed_taxonomy":
        return "no_seed_taxonomy"
    return "seed"


def _selected_source_label(vocab_source: str) -> str:
    if vocab_source == "seed":
        return "fixed"
    return vocab_source


def _result_tag_count(result: IterationResult, seed_vocab: list[dict]) -> int:
    active_vocab = result.vocab if result.vocab is not None else seed_vocab
    return len(active_vocab or [])


def _tag_count_penalty(tag_count: int, config: dict) -> float:
    if not bool(config.get("taxonomy_selection_enabled", False)):
        return 0.0
    try:
        weight = float(config.get("taxonomy_selection_count_penalty", 0.0) or 0.0)
    except (TypeError, ValueError):
        weight = 0.0
    target = config.get("taxonomy_selection_target_tags")
    if target is None or weight <= 0.0:
        return 0.0
    try:
        target_count = int(target)
    except (TypeError, ValueError):
        return 0.0
    if target_count <= 0:
        return 0.0
    return float(weight * abs(int(tag_count) - target_count))


def _finite_selection_float(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _selection_stability_components(
    *,
    dev_metrics: dict,
    train_metrics: dict,
    model_probe_dev_metrics: dict,
    tag_count_penalty: float,
    config: dict,
) -> dict:
    l_values = [
        _finite_selection_float(dev_metrics.get("L_align")),
        _finite_selection_float(train_metrics.get("L_align")),
        _probe_l_for_selection(
            {
                "model_probe_dev_L_align_max": model_probe_dev_metrics.get("L_align_max"),
                "model_probe_dev_L_align_mean": model_probe_dev_metrics.get("L_align_mean"),
            }
        ),
    ]
    rho_values = [
        _finite_selection_float(dev_metrics.get("rho_align_spearman")),
        _finite_selection_float(train_metrics.get("rho_align_spearman")),
        _finite_selection_float(model_probe_dev_metrics.get("rho_align_spearman_min")),
    ]
    l_values = [v for v in l_values if v is not None]
    rho_values = [v for v in rho_values if v is not None]
    l_max = max(l_values) + tag_count_penalty if l_values else float("nan")
    rho_min = min(rho_values) if rho_values else None
    try:
        rho_weight = float(config.get("best_iter_stability_rho_weight", 0.30))
    except (TypeError, ValueError):
        rho_weight = 0.30
    score = l_max
    if rho_min is not None:
        score -= rho_weight * rho_min
    return {
        "score": score,
        "l_max": l_max,
        "rho_min": rho_min,
        "rho_weight": rho_weight,
    }


def _selection_candidate(
    result: IterationResult,
    *,
    seed_vocab: list[dict],
    config: dict,
) -> dict:
    tag_count = _result_tag_count(result, seed_vocab)
    penalty = _tag_count_penalty(tag_count, config)
    dev_metrics = result.dev_metrics or {}
    train_metrics = result.train_metrics or {}
    model_probe_dev_metrics = result.model_probe_dev_metrics or {}
    dev_l = dev_metrics.get("L_align")
    try:
        dev_selection_score = float(dev_l) + penalty
    except (TypeError, ValueError):
        dev_selection_score = float("nan")
    try:
        selection_score = float(result.L_align) + penalty
    except (TypeError, ValueError):
        selection_score = float("nan")
    stability = _selection_stability_components(
        dev_metrics=dev_metrics,
        train_metrics=train_metrics,
        model_probe_dev_metrics=model_probe_dev_metrics,
        tag_count_penalty=penalty,
        config=config,
    )
    return {
        "L_align": result.L_align,
        "rho_align_pearson": result.rho_align_pearson,
        "delta_tag": result.delta_tag,
        "dev_L_align": dev_metrics.get("L_align"),
        "dev_rho_pearson": dev_metrics.get("rho_align_pearson"),
        "dev_rho_spearman": dev_metrics.get("rho_align_spearman"),
        "train_L_align": train_metrics.get("L_align"),
        "train_rho_pearson": train_metrics.get("rho_align_pearson"),
        "train_rho_spearman": train_metrics.get("rho_align_spearman"),
        "train_delta_tag": train_metrics.get("delta_tag"),
        "model_probe_dev_L_align_mean": model_probe_dev_metrics.get("L_align_mean"),
        "model_probe_dev_L_align_max": model_probe_dev_metrics.get("L_align_max"),
        "model_probe_dev_rho_pearson_mean": model_probe_dev_metrics.get("rho_align_pearson_mean"),
        "model_probe_dev_rho_pearson_min": model_probe_dev_metrics.get("rho_align_pearson_min"),
        "model_probe_dev_rho_spearman_mean": model_probe_dev_metrics.get("rho_align_spearman_mean"),
        "model_probe_dev_rho_spearman_min": model_probe_dev_metrics.get("rho_align_spearman_min"),
        "model_probe_dev_delta_tag_mean": model_probe_dev_metrics.get("delta_tag_mean"),
        "model_probe_dev_n_probes": model_probe_dev_metrics.get("n_probes"),
        "model_probe_dev_n_pairs_min": model_probe_dev_metrics.get("n_pairs_min"),
        "model_probe_dev_min_common": model_probe_dev_metrics.get("min_common"),
        "selection_score": selection_score,
        "dev_selection_score": dev_selection_score,
        "stability_selection_score": stability["score"],
        "stability_selection_l_max": stability["l_max"],
        "stability_selection_rho_min": stability["rho_min"],
        "stability_selection_rho_weight": stability["rho_weight"],
        "selection_penalty_tag_count": penalty,
        "tag_count": tag_count,
        "vocab_source": _result_vocab_source(result, config),
        "label": result.label,
    }


def _selection_objective_key(selection_cfg: dict) -> str:
    if selection_cfg.get("objective_key"):
        return str(selection_cfg["objective_key"])
    if selection_cfg.get("mode") == "dev_stability_l_align":
        return "stability_selection_rho_min"
    if selection_cfg.get("mode") == "dev_l_align":
        return "dev_L_align"
    return "L_align"


def _selection_record(
    candidate: dict,
    *,
    objective_key: str,
    gate_pass: bool,
    is_better: bool,
    reason: str,
) -> dict:
    record = dict(candidate)
    record["selection_objective_key"] = objective_key
    record["selection_objective_value"] = candidate.get(objective_key)
    record["gate_pass"] = bool(gate_pass)
    record["selected_at_step"] = bool(is_better)
    record["decision"] = "selected" if is_better else reason
    return record


def _write_selection_candidates(
    run_dir: str,
    *,
    candidates: list[dict],
    selection_cfg: dict,
    selected_label: str | None = None,
) -> None:
    selected_label = str(selected_label) if selected_label is not None else None
    payload_candidates = []
    for candidate in candidates:
        item = dict(candidate)
        item["selected_final"] = (
            selected_label is not None and item.get("label") == selected_label
        )
        payload_candidates.append(item)
    write_json(
        os.path.join(run_dir, "selection_candidates.json"),
        {
            "selection_cfg": selection_cfg,
            "objective_key": _selection_objective_key(selection_cfg),
            "candidates": payload_candidates,
        },
    )


def _selection_delta_tag(metrics: dict | None) -> float:
    if metrics is None:
        return float("nan")
    value = metrics.get("delta_tag", float("nan"))
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _improver_selection_guard_skip_reason(
    *,
    m_dev: dict | None,
    m_train: dict | None,
    model_probe_dev: dict | None,
    config: dict,
) -> str | None:
    """Skip costly prompt rewrites for candidates that cannot pass selection."""

    delta = _selection_delta_tag(m_dev)
    if not _passes_delta_tag_gate(
        {"delta_tag": delta},
        threshold=float(config.get("delta_tag_threshold", 0.0)),
    ):
        return "delta_tag_gate_failed"

    dev_floor = config.get("best_iter_dev_rho_floor")
    if dev_floor is not None:
        dev_rho = _finite_selection_float(
            (m_dev or {}).get("rho_align_spearman")
        )
        if dev_rho is None or dev_rho < float(dev_floor):
            return "dev_rho_floor_failed"

    train_floor = config.get("best_iter_train_rho_floor")
    if train_floor is not None:
        train_rho = _finite_selection_float(
            (m_train or {}).get("rho_align_spearman")
        )
        if train_rho is None or train_rho < float(train_floor):
            return "train_rho_floor_failed"

    probe_floor = config.get("best_iter_model_probe_dev_rho_floor")
    if probe_floor is not None:
        probe_rho = _finite_selection_float(
            (model_probe_dev or {}).get("rho_align_spearman_min")
        )
        if probe_rho is None or probe_rho < float(probe_floor):
            return "model_probe_dev_rho_floor_failed"

    return None


def _executer_target_count_for_iter(config: dict, iteration: int) -> int | None:
    counts = config.get("executer_candidate_counts") or []
    if not isinstance(counts, (list, tuple)) or not counts:
        return None
    clean: list[int] = []
    for value in counts:
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0 and count not in clean:
            clean.append(count)
    if not clean:
        return None
    return clean[(max(1, int(iteration)) - 1) % len(clean)]


def _role_iteration_seed(config: dict, role: str, iteration: int | None) -> int | None:
    """Stable per-role LLM seed for reproducible v-loop candidates."""
    value = config.get(f"{role}_seed", config.get("llm_seed", config.get("seed")))
    if value is None:
        return None
    try:
        base = int(value)
    except (TypeError, ValueError):
        return None
    offsets = {
        "executer": 100_000,
        "maker": 200_000,
        "improver": 300_000,
        "taxonomy": 400_000,
    }
    try:
        iter_num = int(iteration or 0)
    except (TypeError, ValueError):
        iter_num = 0
    return base + offsets.get(str(role), 0) + iter_num


def _candidate_improvement_status(
    candidate: dict,
    best: dict | None,
    *,
    selection_cfg: dict | None = None,
    delta_tag_threshold: float = 0.0,
) -> tuple[bool, bool, str]:
    """Return (is_better, gate_pass, reason) for one candidate.

    A gate-passing candidate can still be a non-improvement when the seeded
    static baseline is better. Count that as no-improvement so v_loop does not
    burn the full max_iter budget on worse generated vocabularies.
    """
    gate_pass = _passes_delta_tag_gate(candidate, threshold=delta_tag_threshold)
    is_better = _is_better(
        candidate,
        best,
        selection_cfg=selection_cfg,
        delta_tag_threshold=delta_tag_threshold,
    )
    if is_better:
        return True, gate_pass, "new_best"
    if not gate_pass:
        return False, gate_pass, "delta_tag_gate_failed"
    if best is None:
        return False, gate_pass, "selection_gate_failed"
    return False, gate_pass, "not_better_than_current_best"


def _fmt_metric(value: float | None) -> str:
    if value is None:
        return "nan"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "nan"
    return "nan" if math.isnan(v) else f"{v:.4f}"


def _log_metrics(label: str, metrics: dict, extra: str = "") -> None:
    suffix = f" {extra}" if extra else ""
    print(
        f"  [metrics] {label}: "
        f"L_align={_fmt_metric(metrics.get('L_align'))}, "
        f"rho_p={_fmt_metric(metrics.get('rho_align_pearson'))}, "
        f"rho_s={_fmt_metric(metrics.get('rho_align_spearman'))}, "
        f"delta={_fmt_metric(metrics.get('delta_tag'))}, "
        f"residual_max={_fmt_metric(metrics.get('residual_max'))}, "
        f"pairs={metrics.get('n_pairs')}{suffix}"
    )


def _run_taxonomy_unlocked_phase(
    *,
    run_dir: str,
    config: dict,
    corpus: Corpus,
    descriptions: dict[str, str],
    seed_vocab: list[dict],
    base_prompt: str,
    fixed_best: IterationResult,
    fixed_metrics: dict,
    fixed_residuals: list[dict],
    protected_pairs: list[dict],
    Y_norm: dict[str, dict[str, float]],
    R_raw: dict,
    R01: dict,
    tag_fn: TagFn | None,
    improver_fn: ImproverCallable | None,
    taxonomy_refiner_fn: TaxonomyRefinerCallable | None,
    static_from_mapreduce: bool = False,
    mapreduce_chat_fn: MapReduceChatFn | None = None,
    mapreduce_reducer_chat_fn: MapReduceChatFn | None = None,
    wandb_run=None,
) -> IterationResult | None:
    """Run Part 1-B after fixed-vocabulary residuals trigger taxonomy unlock."""
    mi = role_cfg(config, "improver_model")
    print("  [taxonomy_refinement] residual trigger met; requesting refined taxonomy")
    if taxonomy_refiner_fn is not None:
        kwargs = {
            "seed_vocab": seed_vocab,
            "base_prompt": base_prompt,
            "best_prompt": fixed_best.prompt,
            "residual_report": fixed_residuals,
            "metrics": fixed_metrics,
            "benchmark_names": corpus.benchmark_names,
            "model": mi["name"],
            "base_url": mi.get("base_url"),
            "retain_seed_tags": bool(config.get("taxonomy_refinement_retain_seed_tags", True)),
            "max_new_tags": int(config.get("taxonomy_refinement_max_new_tags", 4)),
        }
        if _supports_kwarg(taxonomy_refiner_fn, "protected_pairs"):
            kwargs["protected_pairs"] = protected_pairs
        if _supports_kwarg(taxonomy_refiner_fn, "base_url_env"):
            kwargs["base_url_env"] = mi.get("base_url_env")
        if _supports_kwarg(taxonomy_refiner_fn, "api_key_env"):
            kwargs["api_key_env"] = mi.get("api_key_env")
        if _supports_kwarg(taxonomy_refiner_fn, "empty_content_retries"):
            kwargs["empty_content_retries"] = llm_empty_content_retries(config)
        if _supports_kwarg(taxonomy_refiner_fn, "request_timeout_s"):
            kwargs["request_timeout_s"] = llm_request_timeout_s(config)
        if _supports_kwarg(taxonomy_refiner_fn, "sdk_exception_retries"):
            kwargs["sdk_exception_retries"] = llm_sdk_exception_retries(config)
        if _supports_kwarg(taxonomy_refiner_fn, "debug_dump_dir"):
            kwargs["debug_dump_dir"] = llm_debug_dump_dir(config)
        if _supports_kwarg(taxonomy_refiner_fn, "extra_body"):
            kwargs["extra_body"] = llm_extra_body(config)
        refinement = taxonomy_refiner_fn(**kwargs)
    else:
        refinement = refine_taxonomy(
            seed_vocab=seed_vocab,
            base_prompt=base_prompt,
            best_prompt=fixed_best.prompt,
            residual_report=fixed_residuals,
            metrics=fixed_metrics,
            benchmark_names=corpus.benchmark_names,
            model=mi["name"],
            base_url=mi.get("base_url"),
            retain_seed_tags=bool(config.get("taxonomy_refinement_retain_seed_tags", True)),
            max_new_tags=int(config.get("taxonomy_refinement_max_new_tags", 4)),
            protected_pairs=protected_pairs,
            base_url_env=mi.get("base_url_env"),
            api_key_env=mi.get("api_key_env"),
            empty_content_retries=llm_empty_content_retries(config),
            request_timeout_s=llm_request_timeout_s(config),
            sdk_exception_retries=llm_sdk_exception_retries(config),
            debug_dump_dir=llm_debug_dump_dir(config),
            extra_body=llm_extra_body(config),
        )

    save_taxonomy_refinement(
        run_dir,
        {
            "accepted": refinement.accepted,
            "reasons": refinement.reasons,
            "rationale": refinement.rationale,
            "seed_vocab": seed_vocab,
            "refined_vocab": refinement.vocab,
            "raw_response": refinement.raw_response,
            "source_best_iter": fixed_best.label,
            "protected_high_similarity_pairs": protected_pairs,
        },
    )
    print(
        "  [taxonomy_refinement] proposal "
        f"accepted={refinement.accepted}, reasons={refinement.reasons or ['ok']}, "
        f"tags={len(refinement.vocab)}"
    )
    if not refinement.accepted:
        return None

    vocab = refinement.vocab
    current_prompt = refinement.prompt
    phase_best: dict | None = None
    phase_best_result: IterationResult | None = None
    prev_L: list[float] = []
    consecutive_small = 0
    consecutive_no_improve = 0
    stalled_delta_tag = False
    max_iter = int(config.get("taxonomy_refinement_max_iter", 3))

    tax_pbar = tqdm(
        range(1, max_iter + 1),
        desc="[taxonomy_refinement]",
        unit="iter",
        total=max_iter,
    )
    for i in tax_pbar:
        tax_pbar.set_postfix_str(
            f"benchmarks={len(corpus.benchmark_names)} tags={len(vocab)}"
        )
        gen_i = _generate_T_for_prompt(
            corpus=corpus,
            benchmark_names=corpus.benchmark_names,
            descriptions=descriptions,
            vocab=vocab,
            prompt=current_prompt,
            version=10_000 + i,
            tag_fn=tag_fn,
            config=config,
            run_dir=run_dir,
            label=f"taxonomy_refinement/iter_{i:03d}",
            static_from_mapreduce=static_from_mapreduce,
            base_prompt=refinement.prompt,
            mapreduce_chat_fn=mapreduce_chat_fn,
            mapreduce_reducer_chat_fn=mapreduce_reducer_chat_fn,
        )
        T_i = gen_i.T
        descriptions_i = gen_i.descriptions
        S_i, m_i, boot_i = _compute_metrics(
            T_i,
            corpus.benchmark_names,
            R_raw,
            R01,
            config["theta_p_q"],
            config["theta_n_q"],
            config["bootstrap_B"],
            config["seed"],
        )
        err_i = build_error_report(
            S_i,
            R_raw,
            R01,
            top_k=config["error_top_k"],
            q_p_s=config["theta_p_q"],
            q_n_s=config["theta_n_q"],
        )
        res_i = build_residual_report(S_i, R_raw, top_k=config["error_top_k"])
        _log_metrics(f"taxonomy_refinement/iter_{i:03d}", m_i)

        improver_result: ImproverResult | None = None
        improver_payload: dict | None = None
        if i < max_iter:
            print(f"  [taxonomy_refinement] iter {i}: requesting prompt improvement")
            if improver_fn is not None:
                improver_kwargs = dict(
                    prev_prompt=current_prompt,
                    base_prompt=refinement.prompt,
                    error_report=err_i,
                    metrics=m_i,
                    bench_descriptions=descriptions_i,
                    vocab=vocab,
                    benchmark_names=corpus.benchmark_names,
                    model=mi["name"],
                    base_url=mi.get("base_url"),
                )
                if _supports_kwarg(improver_fn, "base_url_env"):
                    improver_kwargs["base_url_env"] = mi.get("base_url_env")
                if _supports_kwarg(improver_fn, "api_key_env"):
                    improver_kwargs["api_key_env"] = mi.get("api_key_env")
                if _supports_kwarg(improver_fn, "json_contract_strict"):
                    improver_kwargs["json_contract_strict"] = bool(
                        config.get("llm_json_contract_strict", True)
                    )
                if _supports_kwarg(improver_fn, "json_contract_max_attempts"):
                    improver_kwargs["json_contract_max_attempts"] = int(
                        config.get("llm_json_contract_max_attempts", 3)
                    )
                if _supports_kwarg(improver_fn, "empty_content_retries"):
                    improver_kwargs["empty_content_retries"] = llm_empty_content_retries(
                        config
                    )
                if _supports_kwarg(improver_fn, "request_timeout_s"):
                    improver_kwargs["request_timeout_s"] = llm_request_timeout_s(config)
                if _supports_kwarg(improver_fn, "sdk_exception_retries"):
                    improver_kwargs["sdk_exception_retries"] = llm_sdk_exception_retries(
                        config
                    )
                if _supports_kwarg(improver_fn, "debug_dump_dir"):
                    improver_kwargs["debug_dump_dir"] = llm_debug_dump_dir(config)
                if _supports_kwarg(improver_fn, "extra_body"):
                    improver_kwargs["extra_body"] = llm_extra_body(config)
                improver_result = improver_fn(**improver_kwargs)
            else:
                improver_result = improve_prompt(
                    prev_prompt=current_prompt,
                    base_prompt=refinement.prompt,
                    error_report=err_i,
                    metrics=m_i,
                    bench_descriptions=descriptions_i,
                    vocab=vocab,
                    benchmark_names=corpus.benchmark_names,
                    model=mi["name"],
                    base_url=mi.get("base_url"),
                    allow_taxonomy_changes=False,
                    base_url_env=mi.get("base_url_env"),
                    api_key_env=mi.get("api_key_env"),
                    temperature=float(config.get("improver_temperature", 0.0)),
                    n_samples=int(config.get("improver_n_samples", 1)),
                    json_contract_strict=bool(config.get("llm_json_contract_strict", True)),
                    json_contract_max_attempts=int(
                        config.get("llm_json_contract_max_attempts", 3)
                    ),
                    empty_content_retries=llm_empty_content_retries(config),
                    request_timeout_s=llm_request_timeout_s(config),
                    sdk_exception_retries=llm_sdk_exception_retries(config),
                    debug_dump_dir=llm_debug_dump_dir(config),
                    extra_body=llm_extra_body(config),
                )
            improver_payload = {
                "accepted": improver_result.accepted,
                "reasons": improver_result.reasons,
                "rationale": improver_result.rationale,
                "raw_response": improver_result.raw_response,
            }
            print(
                f"  [taxonomy_refinement] iter {i}: improver "
                f"accepted={improver_result.accepted}, reasons={improver_result.reasons or ['ok']}"
            )

        label = f"taxonomy_refinement/iter_{i:03d}"
        save_iteration(
            run_dir,
            i,
            current_prompt,
            T_i,
            S_i,
            {**m_i, "bootstrap": boot_i},
            error_pairs_to_dicts(err_i),
            improver_payload,
            label=label,
            residual_report=res_i,
            tag_weight_metadata=gen_i.tag_weight_metadata,
        )
        ir = IterationResult(
            label=label,
            iter=i,
            prompt=current_prompt,
            T=T_i,
            S=S_i,
            L_align=m_i["L_align"],
            L_align_01=m_i["L_align_01"],
            rho_align_pearson=m_i["rho_align_pearson"],
            rho_align_spearman=m_i["rho_align_spearman"],
            delta_tag=m_i["delta_tag"],
            bootstrap=boot_i,
            error_report_size=len(err_i),
            improver=improver_payload,
            tag_weight_metadata=gen_i.tag_weight_metadata,
        )
        candidate = {
            "L_align": ir.L_align,
            "rho_align_pearson": ir.rho_align_pearson,
            "delta_tag": ir.delta_tag,
        }
        _dt_thr = float(config.get("delta_tag_threshold", 0.0))
        is_better, gate_pass, no_improve_reason = _candidate_improvement_status(
            candidate,
            phase_best,
            delta_tag_threshold=_dt_thr,
        )
        if is_better:
            phase_best = candidate
            phase_best_result = ir
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            if not gate_pass:
                print(
                    f"  [taxonomy_refinement] iter {i}: Δ_tag={ir.delta_tag} fails gate "
                    f"(>{_dt_thr} required); consecutive_no_improve={consecutive_no_improve}"
                )
            else:
                print(
                    f"  [taxonomy_refinement] iter {i}: no selection improvement "
                    f"({no_improve_reason}); consecutive_no_improve={consecutive_no_improve}"
                )

        _wandb_log(
            wandb_run,
            {
                "phase": "taxonomy_refinement",
                "phase_iter": i,
                "label": label,
                **{f"taxonomy/{k}": v for k, v in m_i.items()},
                **{f"taxonomy/{k}": v for k, v in _flatten_bootstrap(boot_i).items()},
                "taxonomy/delta_tag_gate_pass": int(gate_pass),
                "taxonomy/consecutive_no_improve": consecutive_no_improve,
            },
        )

        if (
            phase_best_result is not None
            and consecutive_no_improve >= int(config.get("early_stop_consecutive", 2))
        ):
            stalled_delta_tag = not gate_pass
            stall_status = (
                "stalled_delta_tag" if stalled_delta_tag else "stalled_no_improvement"
            )
            print(
                f"  [taxonomy_refinement] {stall_status}: "
                f"{consecutive_no_improve} consecutive non-improving rounds"
            )
            break

        if prev_L:
            if not math.isnan(prev_L[-1]) and not math.isnan(ir.L_align):
                if abs(ir.L_align - prev_L[-1]) < float(config["eps"]):
                    consecutive_small += 1
                else:
                    consecutive_small = 0
        prev_L.append(ir.L_align)
        if consecutive_small >= int(config.get("early_stop_consecutive", 2)):
            print(
                f"  [taxonomy_refinement] early stop at iter {i} "
                f"(Δ<{config['eps']} for {consecutive_small} iters)"
            )
            break

        if improver_result is not None:
            next_prompt, improver_stop = _next_prompt_from_improver(
                improver_result,
                current_prompt,
            )
            if next_prompt is not None:
                current_prompt = next_prompt
            else:
                print(
                    f"  [taxonomy_refinement] stop at iter {i}: "
                    f"{improver_stop or 'improver_rejected'}; no valid next prompt"
                )
                break
        elif phase_best_result is not None:
            current_prompt = phase_best_result.prompt or refinement.prompt

    if phase_best_result is None:
        return None

    phase_residuals = build_residual_report(phase_best_result.S, R_raw, top_k=config["error_top_k"])
    save_taxonomy_final(
        run_dir,
        best_iter_label=phase_best_result.label,
        prompt=phase_best_result.prompt,
        vocab=vocab,
        tag_vectors=phase_best_result.T,
        metrics_with_bootstrap=_metrics_payload(phase_best_result),
        residual_report=phase_residuals,
        profile_support=_build_profile_support(
            Y_norm=Y_norm,
            T=phase_best_result.T,
            benchmark_names=corpus.benchmark_names,
            model_names=corpus.model_names,
            subset_sizes=list(config.get("subset_profile_sizes", [])),
            methods=list(config.get("subset_selection_methods", ["greedy"])),
            kmedoids_seed=int(config.get("kmedoids_seed", 0)),
        ),
        tag_weight_metadata=phase_best_result.tag_weight_metadata,
    )
    print(
        f"  [taxonomy_refinement] best={phase_best_result.label}, "
        f"L_align={_fmt_metric(phase_best_result.L_align)}, "
        f"saved under taxonomy_refinement/final"
    )
    return phase_best_result


def _setup_v_loop_state(
    *,
    config: dict,
    corpus: Corpus,
    run_dir: str,
    vocab: list[dict],
    mapreduce_aggregates: dict[str, dict],
    mapreduce_chat_fn: MapReduceChatFn | None,
    R_raw: dict,
    R01: dict,
    split_required_pair_dicts: list[dict] | None = None,
) -> dict[str, Any]:
    """Validate enable_v_loop config and pre-compute split-aware pair sets.

    Source = full per-fold train split. Each iteration the Executer reads
    Z_src from every benchmark in `bench_split.train`. dev gates Δ_tag /
    early-stop, test stays untouched until end-of-run.
    """
    splits_cfg = config.get("splits", {}) or {}
    cv_folds = int(splits_cfg.get("cv_folds", 1))
    fold = int(splits_cfg.get("fold", 0))
    bench_split = benchmark_split_from_config(
        corpus.benchmark_names,
        splits_cfg,
        score_pair_dict=R_raw,
        required_pair_dicts=split_required_pair_dicts,
        min_test_valid_pairs=int(config.get("v_loop_min_test_valid_pairs", 0)),
        min_test_effective_benchmarks=int(
            config.get("v_loop_min_test_effective_benchmarks", 0)
        ),
    )

    if not bench_split.train:
        raise ValueError(
            f"v_loop requires non-empty train split; got train=[] "
            f"(cv_folds={cv_folds}, fold={fold})"
        )
    if not bench_split.dev:
        raise ValueError(
            f"v_loop requires non-empty dev split for selection signal; got dev=[] "
            f"(cv_folds={cv_folds}, fold={fold}). Check splits.dev_train_split "
            f"and corpus size."
        )

    source_benchmarks = list(bench_split.train)

    train_pairs = set(induced_pair_set(bench_split.train))
    dev_pairs = set(induced_pair_set(bench_split.dev))
    test_pairs = set(induced_pair_set(bench_split.test))
    loop_benchmark_names = list(dict.fromkeys([*bench_split.train, *bench_split.dev]))
    loop_pairs = set(induced_pair_set(loop_benchmark_names))
    R_raw_train = restrict_pair_dict(R_raw, train_pairs)
    R01_train = restrict_pair_dict(R01, train_pairs)
    R_raw_dev = restrict_pair_dict(R_raw, dev_pairs)
    R01_dev = restrict_pair_dict(R01, dev_pairs)
    R_raw_test = restrict_pair_dict(R_raw, test_pairs)
    R_raw_loop = restrict_pair_dict(R_raw, loop_pairs)
    R01_loop = restrict_pair_dict(R01, loop_pairs)

    split_pair_counts = split_valid_pair_counts(R_raw, bench_split)
    split_thresholds = {
        "train": int(config.get("v_loop_min_train_valid_pairs", 1)),
        "dev": int(config.get("v_loop_min_dev_valid_pairs", 1)),
        "test": int(config.get("v_loop_min_test_valid_pairs", 1)),
    }
    insufficient = split_pair_count_failures(split_pair_counts, split_thresholds)
    if insufficient:
        raise ValueError(
            "v_loop split has insufficient score-comparable pairs after "
            f"min_common filtering: {', '.join(insufficient)} "
            f"(cv_folds={cv_folds}, fold={fold}, split={bench_split})"
        )

    # Ensure mapreduce_aggregates are available (Executer needs Z_src).
    descriptions: dict[str, str] = {}
    if not mapreduce_aggregates and corpus.documents:
        print(
            "  [v_loop] pre-computing mapreduce aggregates for Executer "
            f"(sources={source_benchmarks})"
        )
        source_corpus = _subset_corpus(
            corpus,
            source_benchmarks,
            reason="v_loop_source_only",
        )
        descriptions, mapreduce_aggregates = build_mapreduce_descriptions(
            corpus=source_corpus,
            vocab=vocab,
            config=config,
            run_dir=run_dir,
            chat_fn=mapreduce_chat_fn,
        )
    source_aggregates: dict[str, dict] = {}
    missing_sources: list[str] = []
    for name in source_benchmarks:
        agg = mapreduce_aggregates.get(name)
        if isinstance(agg, dict) and agg:
            source_aggregates[name] = agg
        else:
            missing_sources.append(name)
    if missing_sources:
        print(
            f"  [v_loop] WARN: source benchmarks missing mapreduce aggregate: "
            f"{missing_sources} — they will be skipped by the Executer"
        )
    if not source_aggregates:
        raise ValueError(
            f"no source aggregate available for any train benchmark "
            f"({source_benchmarks}); ensure use_mapreduce_evidence=True or "
            f"pass corpus.documents"
        )

    print(
        f"  [v_loop] enabled: sources={sorted(source_aggregates.keys())} "
        f"(n={len(source_aggregates)}), "
        f"train={len(bench_split.train)}, dev={len(bench_split.dev)}, "
        f"test={len(bench_split.test)} "
        f"valid_pairs={split_pair_counts} (test held until end)"
    )
    return {
        "source_benchmarks": sorted(source_aggregates.keys()),
        "source_aggregates": source_aggregates,
        "bench_split": bench_split,
        "bench_seed": bench_split.seed,
        "bench_ratios": bench_split.ratios,
        "loop_benchmark_names": loop_benchmark_names,
        "loop_pairs": loop_pairs,
        "train_pairs": train_pairs,
        "dev_pairs": dev_pairs,
        "test_pairs": test_pairs,
        "R_raw_loop": R_raw_loop,
        "R01_loop": R01_loop,
        "R_raw_train": R_raw_train,
        "R01_train": R01_train,
        "R_raw_dev": R_raw_dev,
        "R01_dev": R01_dev,
        "R_raw_test": R_raw_test,
        "mapreduce_aggregates": mapreduce_aggregates,
        "descriptions": descriptions,
        "test_split_eval_count": 0,
    }


def run_part1(
    config: dict,
    corpus: Corpus | None = None,
    descriptions: dict[str, str] | None = None,
    tag_fn: TagFn | None = None,
    improver_fn: ImproverCallable | None = None,
    taxonomy_refiner_fn: TaxonomyRefinerCallable | None = None,
    no_seed_taxonomy_fn: NoSeedTaxonomyCallable | None = None,
    mapreduce_chat_fn: MapReduceChatFn | None = None,
    mapreduce_reducer_chat_fn: MapReduceChatFn | None = None,
    executer_chat_fn: ExecuterChatFn | None = None,
    run_dir: str | None = None,
    wandb_run=None,
) -> tuple[list[IterationResult], IterationResult]:
    """Execute baselines + iterative prompt refinement loop. Returns (history, best)."""
    print("  [loop] starting BenchPress Part 1")
    if corpus is None:
        print(
            f"  [loop] loading corpus from {config['leaderboard_path']} "
            f"with labels_dir={config.get('labels_dir')}"
        )
        corpus = load_corpus(
            leaderboard_path=config["leaderboard_path"],
            min_models_per_bench=config["min_common_models"],
            exclude=config.get("exclude", []),
            labels_dir=config.get("labels_dir"),
            examples_per_benchmark=config.get("examples_per_benchmark", 5),
            prompt_examples_per_benchmark=config.get("prompt_examples_per_benchmark"),
            max_prompt_chars_per_benchmark=config.get("max_prompt_chars_per_benchmark"),
        )
    vocab = _load_vocab(config["vocab_path"])
    base_prompt = _load_prompt(config["prompt_i0_path"])
    static_from_mapreduce = (
        tag_fn is None
        and config.get("tag_weight_mode", "llm_direct") == "static_from_mapreduce"
    )
    if static_from_mapreduce and config.get("static_tag_restrict_to_labeled_benchmarks", True):
        labeled = [b for b in corpus.benchmark_names if b in corpus.documents]
        if labeled and len(labeled) < len(corpus.benchmark_names):
            print(
                "  [loop] static weight mode: restricting Part 1 to "
                f"{len(labeled)} benchmarks with full label documents "
                f"(dropped {len(corpus.benchmark_names) - len(labeled)} without documents)"
            )
            corpus = _subset_corpus(
                corpus,
                labeled,
                reason="no_label_document_for_static_mapreduce_weight",
            )
        elif not labeled:
            raise ValueError(
                "tag_weight_mode=static_from_mapreduce requires full label documents, "
                "but none matched the scored benchmarks."
            )

    descriptions = descriptions or corpus.descriptions or {b: "" for b in corpus.benchmark_names}
    print(
        f"  [loop] corpus ready: benchmarks={len(corpus.benchmark_names)}, "
        f"models={len(corpus.model_names)}, docs={len(corpus.documents)}, tags={len(vocab)}"
    )
    if corpus.descriptions:
        longest_desc = max((len(text) for text in corpus.descriptions.values()), default=0)
        print(
            f"  [loop] prompt evidence budget: "
            f"max_description_chars={longest_desc}, "
            f"prompt_examples_per_benchmark={config.get('prompt_examples_per_benchmark')}"
        )
    _mapper_cfg = role_cfg(config, "mapper_model")
    _maker_cfg = role_cfg(config, "maker_model")
    _executer_cfg = role_cfg(config, "executer_model") or _maker_cfg
    _improver_cfg = role_cfg(config, "improver_model")
    _reducer_cfg = config.get("mapreduce_reducer_model") or _maker_cfg
    print(
        f"  [loop] models: mapreduce={_mapper_cfg.get('name')}, "
        f"reducer={_reducer_cfg.get('name')}, "
        f"tagger={_maker_cfg.get('name')}, improver={_improver_cfg.get('name')}, "
        f"tag_weight_mode={config.get('tag_weight_mode', 'llm_direct')}"
    )

    # Phase A — register per-endpoint concurrency caps once. Two roles sharing an
    # endpoint resolve to min(cap1, cap2). No-op if shared_factory is unused.
    from autotagging_loop.experiment.llm_client import shared_factory as _shared_factory

    _factory = _shared_factory()
    _role_caps: list[tuple[dict, str, int]] = [
        (_mapper_cfg, "mapper_max_concurrent", int(config.get("mapper_max_concurrent", 16))),
        (_executer_cfg, "executer_max_concurrent", int(config.get("executer_max_concurrent", 8))),
        (_maker_cfg, "maker_max_concurrent", int(config.get("maker_max_concurrent", 8))),
        (_improver_cfg, "improver_max_concurrent", int(config.get("improver_max_concurrent", 8))),
    ]
    for _cfg, _key, _cap in _role_caps:
        if not _cfg:
            continue
        try:
            _factory.configure_limit(
                base_url=_cfg.get("base_url"),
                base_url_env=_cfg.get("base_url_env"),
                api_key_env=_cfg.get("api_key_env"),
                max_concurrent=int(config.get(_key, _cap)),
            )
        except Exception as exc:  # missing env / key — surface but don't crash startup
            print(f"  [loop] configure_limit({_key}) skipped: {exc}")

    # Score matrix (immutable across iterations). In strict v-loop mode, the
    # train/dev selection matrix must be computed only on F_seen so F_held is
    # a real model holdout for final split_metrics["held_model_test"].
    score_model_scope = _v_loop_score_model_scope(config)
    score_model_split = (
        _model_split_for_score_matrix(config, corpus.model_names)
        if bool(config.get("enable_v_loop", False))
        else None
    )
    score_model_names = list(corpus.model_names)
    score_Y = corpus.Y
    if (
        bool(config.get("enable_v_loop", False))
        and score_model_scope == "seen"
        and score_model_split is not None
    ):
        score_model_names = list(score_model_split.seen)
        score_Y = _filter_corpus_scores_by_models(corpus.Y, score_model_names)
    print(
        f"  [loop] computing score-pattern similarities "
        f"(normalize={config['normalize']}, min_common={config['min_common_models']}, "
        f"model_scope={score_model_scope}, models={len(score_model_names)})"
    )
    Y_norm_full = normalize_matrix(corpus.Y, method=config["normalize"])
    Y_norm = normalize_matrix(score_Y, method=config["normalize"])
    R_raw, common_count = spearman_pair_matrix(
        Y_norm,
        corpus.benchmark_names,
        min_common=config["min_common_models"],
        warn_below=config["min_common_models_warn"],
    )
    R01 = to_R01(R_raw)
    split_required_pair_dicts = [R_raw]
    if (
        bool(config.get("enable_v_loop", False))
        and bool(config.get("v_loop_require_held_model_test", False))
        and score_model_split is not None
        and len(score_model_split.held) >= int(config["min_common_models"])
    ):
        held = set(score_model_split.held)
        Y_held_for_split = {
            bench: {
                model: score
                for model, score in scores.items()
                if model in held
            }
            for bench, scores in Y_norm_full.items()
        }
        R_held_for_split, _ = spearman_pair_matrix(
            Y_held_for_split,
            corpus.benchmark_names,
            min_common=config["min_common_models"],
            warn_below=10**9,
        )
        split_required_pair_dicts.append(R_held_for_split)
    n_pairs = sum(1 for v in R_raw.values() if v is not None)
    print(f"  [loop] score matrix ready: comparable_pairs={n_pairs}")

    # Storage
    if run_dir is None:
        run_dir = make_run_dir(config["results_dir"])
    print(f"  [loop] saving run artifacts to {run_dir}")
    # Reset process-wide LLM fallback counter so `final/llm_fallbacks.json`
    # only reflects this run, not whatever leaked from a prior in-process call.
    from autotagging_loop.experiment.llm_client import reset_llm_fallback_counts
    reset_llm_fallback_counts()
    vocab_source = "seed"
    if bool(config.get("no_seed_taxonomy_enabled", False)):
        model_cfg = config.get("no_seed_taxonomy_model") or role_cfg(config, "improver_model")
        print(
            "  [no_seed_taxonomy] enabled; inducing taxonomy "
            f"with {model_cfg.get('name')}"
        )
        if no_seed_taxonomy_fn is not None:
            ns_kwargs = dict(
                corpus=corpus,
                benchmark_names=corpus.benchmark_names,
                model=model_cfg.get("name"),
                base_url=model_cfg.get("base_url"),
                min_tags=int(config.get("no_seed_taxonomy_min_tags", 8)),
                max_tags=int(config.get("no_seed_taxonomy_max_tags", 14)),
                max_attempts=int(config.get("no_seed_taxonomy_max_attempts", 3)),
                examples_per_benchmark=int(config.get("no_seed_taxonomy_examples_per_benchmark", 3)),
                max_chars_per_benchmark=int(config.get("no_seed_taxonomy_max_chars_per_benchmark", 4000)),
            )
            if _supports_kwarg(no_seed_taxonomy_fn, "base_url_env"):
                ns_kwargs["base_url_env"] = model_cfg.get("base_url_env")
            if _supports_kwarg(no_seed_taxonomy_fn, "api_key_env"):
                ns_kwargs["api_key_env"] = model_cfg.get("api_key_env")
            if _supports_kwarg(no_seed_taxonomy_fn, "empty_content_retries"):
                ns_kwargs["empty_content_retries"] = llm_empty_content_retries(config)
            if _supports_kwarg(no_seed_taxonomy_fn, "request_timeout_s"):
                ns_kwargs["request_timeout_s"] = llm_request_timeout_s(config)
            if _supports_kwarg(no_seed_taxonomy_fn, "sdk_exception_retries"):
                ns_kwargs["sdk_exception_retries"] = llm_sdk_exception_retries(config)
            if _supports_kwarg(no_seed_taxonomy_fn, "debug_dump_dir"):
                ns_kwargs["debug_dump_dir"] = llm_debug_dump_dir(config)
            if _supports_kwarg(no_seed_taxonomy_fn, "extra_body"):
                ns_kwargs["extra_body"] = llm_extra_body(config)
            if _supports_kwarg(no_seed_taxonomy_fn, "seed"):
                ns_kwargs["seed"] = config.get("no_seed_taxonomy_seed", config.get("seed"))
            no_seed_result = no_seed_taxonomy_fn(**ns_kwargs)
        else:
            no_seed_result = induce_no_seed_taxonomy(
                corpus=corpus,
                benchmark_names=corpus.benchmark_names,
                model=model_cfg.get("name"),
                base_url=model_cfg.get("base_url"),
                min_tags=int(config.get("no_seed_taxonomy_min_tags", 8)),
                max_tags=int(config.get("no_seed_taxonomy_max_tags", 14)),
                max_attempts=int(config.get("no_seed_taxonomy_max_attempts", 3)),
                examples_per_benchmark=int(config.get("no_seed_taxonomy_examples_per_benchmark", 3)),
                max_chars_per_benchmark=int(config.get("no_seed_taxonomy_max_chars_per_benchmark", 4000)),
                base_url_env=model_cfg.get("base_url_env"),
                api_key_env=model_cfg.get("api_key_env"),
                empty_content_retries=llm_empty_content_retries(config),
                request_timeout_s=llm_request_timeout_s(config),
                sdk_exception_retries=llm_sdk_exception_retries(config),
                debug_dump_dir=llm_debug_dump_dir(config),
                extra_body=llm_extra_body(config),
                seed=config.get("no_seed_taxonomy_seed", config.get("seed")),
            )
        save_no_seed_taxonomy(
            run_dir,
            {
                "accepted": no_seed_result.accepted,
                "reasons": no_seed_result.reasons,
                "rationale": no_seed_result.rationale,
                "vocab": no_seed_result.vocab,
                "prompt": no_seed_result.prompt,
                "raw_response": no_seed_result.raw_response,
                "model": model_cfg.get("name"),
            },
        )
        print(
            "  [no_seed_taxonomy] proposal "
            f"accepted={no_seed_result.accepted}, "
            f"reasons={no_seed_result.reasons or ['ok']}, "
            f"tags={len(no_seed_result.vocab)}"
        )
        if no_seed_result.accepted:
            vocab = no_seed_result.vocab
            base_prompt = no_seed_result.prompt
            vocab_source = "no_seed_taxonomy"
            config = {**config, "active_vocab_source": vocab_source}
        elif not bool(config.get("no_seed_taxonomy_fallback_to_seed", False)):
            raise RuntimeError(
                "no_seed_taxonomy proposal was rejected: "
                + ", ".join(no_seed_result.reasons or ["unknown"])
            )
        else:
            config = {**config, "active_vocab_source": "seed_fallback_after_no_seed_reject"}
    else:
        config = {**config, "active_vocab_source": vocab_source}
    mapreduce_aggregates: dict[str, dict] = {}
    if (
        not static_from_mapreduce
        and config.get("use_mapreduce_evidence", False)
        and not bool(config.get("enable_v_loop", False))
        and corpus.documents
    ):
        print(
            "  [mapreduce] enabled: using chunk evidence Z_l instead of raw full examples "
            "for benchmark-level tagging"
        )
        descriptions, mapreduce_aggregates = build_mapreduce_descriptions(
            corpus=corpus,
            vocab=vocab,
            config=config,
            run_dir=run_dir,
            chat_fn=mapreduce_chat_fn,
        )
        longest_desc = max((len(text) for text in descriptions.values()), default=0)
        print(f"  [mapreduce] complete: max_aggregate_description_chars={longest_desc}")

    # v3 §2.2.4 — V loop state. Validated once before iteration so a
    # misconfigured run fails fast (codex 2026-05-10 #1, #2). When disabled,
    # v_loop_state is None and the iteration body uses the legacy fixed-V path.
    v_loop_state: dict[str, Any] | None = None
    model_probe_state: dict[str, Any] | None = None
    if bool(config.get("enable_v_loop", False)):
        v_loop_state = _setup_v_loop_state(
            config=config,
            corpus=corpus,
            run_dir=run_dir,
            vocab=vocab,
            mapreduce_aggregates=mapreduce_aggregates,
            mapreduce_chat_fn=mapreduce_chat_fn,
            R_raw=R_raw,
            R01=R01,
            split_required_pair_dicts=split_required_pair_dicts,
        )
        # Pre-compute may have populated mapreduce_aggregates; pull them back.
        mapreduce_aggregates = v_loop_state["mapreduce_aggregates"]
        if not descriptions:
            descriptions = v_loop_state["descriptions"] or descriptions
        model_probe_state = _setup_model_probe_state(
            config=config,
            corpus=corpus,
            score_model_split=score_model_split,
        )
        v_loop_state["model_probe_state"] = model_probe_state
    loop_benchmark_names = corpus.benchmark_names
    loop_corpus = corpus
    loop_R_raw = R_raw
    loop_R01 = R01
    if v_loop_state is not None:
        loop_benchmark_names = list(v_loop_state["loop_benchmark_names"])
        loop_corpus = _subset_corpus(
            corpus,
            loop_benchmark_names,
            reason="v_loop_holdout_test_until_final",
        )
        loop_R_raw = v_loop_state["R_raw_loop"]
        loop_R01 = v_loop_state["R01_loop"]
        if (
            not static_from_mapreduce
            and config.get("use_mapreduce_evidence", False)
            and loop_corpus.documents
        ):
            print(
                "  [mapreduce] v_loop: building train/dev evidence for "
                "benchmark-level tagging"
            )
            descriptions, loop_mapreduce_aggregates = build_mapreduce_descriptions(
                corpus=loop_corpus,
                vocab=vocab,
                config=config,
                run_dir=run_dir,
                chat_fn=mapreduce_chat_fn,
            )
            mapreduce_aggregates = {
                **loop_mapreduce_aggregates,
                **v_loop_state["source_aggregates"],
            }
    save_config(run_dir, config)
    save_corpus(
        run_dir,
        {
            "benchmark_names": corpus.benchmark_names,
            "model_names": corpus.model_names,
            "drop_log": corpus.drop_log,
            "document_coverage": {
                "with_documents": sorted(corpus.documents.keys()),
                "without_documents": sorted(
                    b for b in corpus.benchmark_names if b not in corpus.documents
                ),
            },
            "documents": corpus.documents,
            "tag_weight_mode": config.get("tag_weight_mode", "llm_direct"),
            "active_vocab_source": vocab_source,
            "active_vocab": vocab,
            "mapreduce_aggregates": mapreduce_aggregates,
        },
    )
    save_score_matrix(
        run_dir,
        {
            "Y_norm": Y_norm,
            "R_raw": R_raw,
            "R01": R01,
            "common_count": common_count,
            "normalize": config["normalize"],
            "model_scope": score_model_scope,
            "score_model_names": score_model_names,
            "model_probe": _model_probe_summary(model_probe_state),
            "model_split": (
                {
                    "seen": list(score_model_split.seen),
                    "held": list(score_model_split.held),
                    "ratios": list(score_model_split.ratios),
                    "strategy": getattr(score_model_split, "strategy", "random"),
                }
                if score_model_split is not None
                else None
            ),
        },
    )

    if tag_fn is None and not static_from_mapreduce:
        ma = role_cfg(config, "maker_model")
        tag_fn = _build_default_tag_fn(
            ma["name"],
            ma.get("base_url"),
            tuple(config.get("weight_bounds", [0.0, 1.0])),
            base_url_env=ma.get("base_url_env"),
            api_key_env=ma.get("api_key_env"),
            allow_uniform_fallback=bool(
                config.get("tag_generator_allow_uniform_fallback", False)
            ),
        )

    history: list[IterationResult] = []
    best: dict | None = None
    best_result: IterationResult | None = None
    selection_candidates: list[dict] = []

    _sel_mode = str(config.get("best_iter_selection", "train_l_align"))
    if _sel_mode not in ("train_l_align", "dev_l_align", "dev_stability_l_align"):
        print(
            f"  [loop] WARN: unknown best_iter_selection={_sel_mode!r}; "
            f"falling back to train_l_align"
        )
        _sel_mode = "train_l_align"
    _sel_uses_dev = _sel_mode in ("dev_l_align", "dev_stability_l_align")
    if _sel_uses_dev and not bool(config.get("enable_v_loop", False)):
        print(
            f"  [loop] WARN: best_iter_selection={_sel_mode!r} requires "
            "enable_v_loop=True; dev metrics are unavailable, using "
            "train_l_align for this fixed-V run"
        )
        _sel_mode = "train_l_align"
    selection_cfg = {
        "mode": _sel_mode,
        "dev_rho_floor": config.get("best_iter_dev_rho_floor"),
        "dev_rho_drop_tolerance": config.get("best_iter_dev_rho_drop_tolerance"),
        "train_l_increase_tolerance": config.get(
            "best_iter_train_l_increase_tolerance"
        ),
        "train_rho_drop_tolerance": config.get(
            "best_iter_train_rho_drop_tolerance"
        ),
        "train_rho_floor": config.get("best_iter_train_rho_floor"),
    }
    if model_probe_state is not None:
        selection_cfg.update(
            {
                "model_probe_dev_rho_floor": config.get(
                    "best_iter_model_probe_dev_rho_floor"
                ),
                "model_probe_dev_rho_drop_tolerance": config.get(
                    "best_iter_model_probe_dev_rho_drop_tolerance"
                ),
                "model_probe_dev_l_increase_tolerance": config.get(
                    "best_iter_model_probe_dev_l_increase_tolerance"
                ),
            }
        )
    if bool(config.get("taxonomy_selection_enabled", False)):
        if _sel_mode == "dev_stability_l_align":
            objective_key = "stability_selection_rho_min"
        elif _sel_mode == "dev_l_align":
            objective_key = "dev_selection_score"
        else:
            objective_key = "selection_score"
        selection_cfg.update(
            {
                "objective_key": objective_key,
                "tag_count_min": config.get("taxonomy_selection_min_tags"),
                "tag_count_max": config.get("taxonomy_selection_max_tags"),
            }
        )
    selection_objective_key = _selection_objective_key(selection_cfg)

    # ── Baselines ─────────────────────────────────────────────
    if config.get("run_baseline", True):
        # (a) static I_0
        print(
            f"  [baseline] static I0: generating tags for "
            f"{len(loop_benchmark_names)} benchmarks"
        )
        gen_i0 = _generate_T_for_prompt(
            corpus=loop_corpus,
            benchmark_names=loop_benchmark_names,
            descriptions=descriptions,
            vocab=vocab,
            prompt=base_prompt,
            version=0,
            tag_fn=tag_fn,
            config=config,
            run_dir=run_dir,
            label="iter_000_baseline_static",
            static_from_mapreduce=static_from_mapreduce,
            base_prompt=base_prompt,
            mapreduce_chat_fn=mapreduce_chat_fn,
            mapreduce_reducer_chat_fn=mapreduce_reducer_chat_fn,
        )
        T_i0 = gen_i0.T
        S_i0, m_i0, boot_i0 = _compute_metrics(
            T_i0, loop_benchmark_names, loop_R_raw, loop_R01,
            config["theta_p_q"], config["theta_n_q"], config["bootstrap_B"], config["seed"],
        )
        err_i0 = build_error_report(S_i0, loop_R_raw, loop_R01, top_k=config["error_top_k"],
                                    q_p_s=config["theta_p_q"], q_n_s=config["theta_n_q"])
        res_i0 = build_residual_report(S_i0, loop_R_raw, top_k=config["error_top_k"])
        cal_i0 = _maybe_build_calibration(T_i0, loop_benchmark_names, loop_R_raw, loop_R01, vocab, config)
        _log_metrics("iter_000_baseline_static", m_i0, extra=f"errors={len(err_i0)}")
        save_iteration(
            run_dir, 0, base_prompt, T_i0, S_i0,
            {**m_i0, "bootstrap": boot_i0},
            error_pairs_to_dicts(err_i0),
            None,
            label="iter_000_baseline_static",
            residual_report=res_i0,
            calibrated_tag_vectors=cal_i0["T"] if cal_i0 else None,
            calibrated_similarity=cal_i0["S"] if cal_i0 else None,
            calibrated_metrics=cal_i0["metrics"] if cal_i0 else None,
            tag_weight_metadata=gen_i0.tag_weight_metadata,
        )
        # When v_loop is on, baseline static must expose dev-split L_align so
        # `_is_better` compares baseline and v_loop iters on the same dev
        # signal that drives selection. Train metrics are computed for
        # diagnostic continuity but no longer drive the IterationResult.
        m_i0_train = None
        m_i0_dev = None
        boot_i0_train = None
        boot_i0_dev = None
        model_probe_i0_dev = None
        if v_loop_state is not None:
            bench_split_i0 = v_loop_state["bench_split"]
            _S_i0_tr, m_i0_train, boot_i0_train = _compute_metrics(
                T_i0, bench_split_i0.train,
                v_loop_state["R_raw_train"], v_loop_state["R01_train"],
                config["theta_p_q"], config["theta_n_q"],
                config["bootstrap_B"], config["seed"],
            )
            iter0_dir = os.path.join(run_dir, "iter_000_baseline_static")
            write_json(
                os.path.join(iter0_dir, "metrics_train.json"),
                {**m_i0_train, "bootstrap": boot_i0_train},
            )
            _S_i0_dv, m_i0_dev, boot_i0_dev = _compute_metrics(
                T_i0, bench_split_i0.dev,
                v_loop_state["R_raw_dev"], v_loop_state["R01_dev"],
                config["theta_p_q"], config["theta_n_q"],
                config["bootstrap_B"], config["seed"],
            )
            write_json(
                os.path.join(iter0_dir, "metrics_dev.json"),
                {**m_i0_dev, "bootstrap": boot_i0_dev},
            )
            model_probe_i0_dev = _compute_model_probe_dev_metrics(
                T=T_i0,
                benchmark_names=bench_split_i0.dev,
                model_probe_state=model_probe_state,
                config=config,
            )
            if model_probe_i0_dev is not None:
                write_json(
                    os.path.join(iter0_dir, "metrics_model_probe_dev.json"),
                    model_probe_i0_dev,
                )
        # Compose the IterationResult. When v_loop is on, expose dev-split
        # metrics so selection/gating never mixes train evidence into the
        # held-out dev decision. A non-finite dev Δ_tag must fail the gate.
        if m_i0_dev is not None:
            _gate_delta_i0 = _selection_delta_tag(m_i0_dev)
            ir_static = IterationResult(
                label="iter_000_baseline_static", iter=0, prompt=base_prompt,
                T=T_i0, S=S_i0,
                L_align=m_i0_dev["L_align"],
                L_align_01=m_i0_dev["L_align_01"],
                rho_align_pearson=m_i0_dev["rho_align_pearson"],
                rho_align_spearman=m_i0_dev["rho_align_spearman"],
                delta_tag=_gate_delta_i0, bootstrap=boot_i0_dev,
                error_report_size=len(err_i0),
                tag_weight_metadata=gen_i0.tag_weight_metadata,
                dev_metrics={**m_i0_dev, "bootstrap": boot_i0_dev},
                train_metrics=(
                    {**m_i0_train, "bootstrap": boot_i0_train}
                    if m_i0_train is not None
                    else None
                ),
                model_probe_dev_metrics=model_probe_i0_dev,
            )
        else:
            ir_static = IterationResult(
                label="iter_000_baseline_static", iter=0, prompt=base_prompt,
                T=T_i0, S=S_i0,
                L_align=m_i0["L_align"],
                L_align_01=m_i0["L_align_01"],
                rho_align_pearson=m_i0["rho_align_pearson"],
                rho_align_spearman=m_i0["rho_align_spearman"],
                delta_tag=m_i0["delta_tag"], bootstrap=boot_i0,
                error_report_size=len(err_i0),
                tag_weight_metadata=gen_i0.tag_weight_metadata,
            )
        history.append(ir_static)
        _wandb_log(
            wandb_run,
            {
                "phase": "baseline_static",
                "label": "iter_000_baseline_static",
                **{f"baseline_static/{k}": v for k, v in m_i0.items()},
                **{f"baseline_static/{k}": v for k, v in _flatten_bootstrap(boot_i0).items()},
                **(
                    {f"baseline_static_calibrated/{k}": v for k, v in cal_i0["metrics"].items() if not isinstance(v, dict)}
                    if cal_i0 else {}
                ),
            },
        )

        # (b) random T
        print("  [baseline] random tag baseline")
        T_rand = random_tag_vectors(loop_benchmark_names, vocab, seed=config["seed"])
        S_rand, m_rand, boot_rand = _compute_metrics(
            T_rand, loop_benchmark_names, loop_R_raw, loop_R01,
            config["theta_p_q"], config["theta_n_q"], config["bootstrap_B"], config["seed"],
        )
        res_rand = build_residual_report(S_rand, loop_R_raw, top_k=config["error_top_k"])
        _log_metrics("iter_000_baseline_random", m_rand)
        save_iteration(
            run_dir, 0, "(random tag baseline; no prompt)", T_rand, S_rand,
            {**m_rand, "bootstrap": boot_rand},
            [],
            None,
            label="iter_000_baseline_random",
            residual_report=res_rand,
        )
        history.append(IterationResult(
            label="iter_000_baseline_random", iter=0, prompt="",
            T=T_rand, S=S_rand,
            L_align=m_rand["L_align"],
            L_align_01=m_rand["L_align_01"],
            rho_align_pearson=m_rand["rho_align_pearson"],
            rho_align_spearman=m_rand["rho_align_spearman"],
            delta_tag=m_rand["delta_tag"], bootstrap=boot_rand,
            error_report_size=0,
        ))
        _wandb_log(
            wandb_run,
            {
                "phase": "baseline_random",
                "label": "iter_000_baseline_random",
                **{f"baseline_random/{k}": v for k, v in m_rand.items()},
                **{f"baseline_random/{k}": v for k, v in _flatten_bootstrap(boot_rand).items()},
            },
        )

    # ── Iterative loop ────────────────────────────────────────
    # 2026-05-12 selector fix: seed best with iter_000_baseline_static so the
    # v_loop must EXPLICITLY beat it (on the same train-split denominator).
    # Without this seed, the first gate-passing v_loop iter becomes best
    # regardless of whether static would have been a better choice on the test
    # split — and the post-loop fallback `min(history, key=L_align)` compares
    # static's L (computed on its own pair set) against v_loop iters' (computed
    # on train), which is the pair-set unit-mismatch bug Run C exposed.
    if (
        config.get("run_baseline", True)
        and v_loop_state is not None
        and history
        and history[0].label == "iter_000_baseline_static"
    ):
        ir_static_seed = history[0]
        _seed_cand = _selection_candidate(
            ir_static_seed,
            seed_vocab=vocab,
            config=config,
        )
        _dt_thr_seed = float(config.get("delta_tag_threshold", 0.0))
        seed_is_better, seed_gate_pass, seed_reason = _candidate_improvement_status(
            _seed_cand,
            None,
            selection_cfg=selection_cfg,
            delta_tag_threshold=_dt_thr_seed,
        )
        selection_candidates.append(
            _selection_record(
                _seed_cand,
                objective_key=selection_objective_key,
                gate_pass=seed_gate_pass,
                is_better=seed_is_better,
                reason="seeded_static_best" if seed_is_better else seed_reason,
            )
        )
        if seed_is_better:
            best = _seed_cand
            best_result = ir_static_seed
            print(
                f"  [loop] seeded best with iter_000_baseline_static "
                f"(selection_score={_fmt_metric(_seed_cand.get(selection_objective_key))}, "
                f"raw_L_align={ir_static_seed.L_align:.4f}, "
                f"tag_count={_seed_cand['tag_count']}, "
                f"Δ_tag={ir_static_seed.delta_tag:.4f})"
            )
        else:
            print(
                f"  [loop] static iter does not pass selection gate "
                f"(reason={seed_reason}, Δ_tag={ir_static_seed.delta_tag}); best stays None until "
                f"a v_loop iter passes"
            )

    current_prompt = base_prompt
    prev_L: list[float] = []
    consecutive_small = 0
    consecutive_no_improve = 0
    stalled_delta_tag = False
    stop_status = "ok"
    stop_details: dict | None = None

    main_max_iter = int(config["max_iter"])
    main_pbar = tqdm(
        range(1, main_max_iter + 1),
        desc="[loop] iterations",
        unit="iter",
        total=main_max_iter,
    )
    for i in main_pbar:
        main_pbar.set_postfix_str(f"benchmarks={len(corpus.benchmark_names)}")

        # v3 §2.2.4 — produce V^(i) from (Z_src, I_exec) when V loop is on.
        V_i: list[dict] | None = None
        vocab_hash_i: str | None = None
        active_vocab = vocab
        if v_loop_state is not None:
            target_count_i = _executer_target_count_for_iter(config, i)
            target_msg = (
                f", target_count={target_count_i}"
                if target_count_i is not None
                else ""
            )
            V_i, exec_meta = run_executer(
                source_benchmarks=v_loop_state["source_benchmarks"],
                source_aggregates=v_loop_state["source_aggregates"],
                prompt_i_exec=current_prompt,
                config=config,
                run_dir=run_dir,
                version=i,
                label=f"iter_{i:03d}",
                chat_fn=executer_chat_fn,
                seed=_role_iteration_seed(config, "executer", i),
                target_count=target_count_i,
            )
            if not V_i:
                iter_dir = os.path.join(run_dir, f"iter_{i:03d}")
                os.makedirs(iter_dir, exist_ok=True)
                write_json(
                    os.path.join(iter_dir, "V.json"),
                    {
                        "vocab": [],
                        "vocab_hash": None,
                        "executer_metadata": exec_meta,
                    },
                )
                if not bool(config.get("executer_fallback_to_seed", False)):
                    raise RuntimeError(
                        f"Executer failed to produce a valid vocabulary at iter_{i:03d}: "
                        f"{exec_meta.get('reasons') or ['empty_vocab']}"
                    )
                # Legacy/debug mode only: reuse seed vocab after recording the
                # failed Executer payload in V.json.
                V_i = list(vocab)
            vocab_hash_i = _vocab_hash(V_i)
            active_vocab = V_i
            iter_dir = os.path.join(run_dir, f"iter_{i:03d}")
            os.makedirs(iter_dir, exist_ok=True)
            write_json(
                os.path.join(iter_dir, "V.json"),
                {
                    "vocab": V_i,
                    "vocab_hash": vocab_hash_i,
                    "executer_metadata": exec_meta,
                },
            )
            if target_count_i is not None:
                print(
                    f"  [executer] iter_{i:03d}: count-conditioned candidate"
                    f"{target_msg}, actual_count={len(V_i)}"
                )

        gen_i = _generate_T_for_prompt(
            corpus=loop_corpus,
            benchmark_names=loop_benchmark_names,
            descriptions=descriptions,
            vocab=active_vocab,
            prompt=current_prompt,
            version=i,
            tag_fn=tag_fn,
            config=config,
            run_dir=run_dir,
            label=f"iter_{i:03d}",
            static_from_mapreduce=static_from_mapreduce,
            base_prompt=base_prompt,
            mapreduce_chat_fn=mapreduce_chat_fn,
            mapreduce_reducer_chat_fn=mapreduce_reducer_chat_fn,
        )
        T_i = gen_i.T
        descriptions_i = gen_i.descriptions

        # v3 §2.2.7 — split-aware metrics. With source = train, training-error
        # L_align(train) is no longer useful for selection. Dev drives both
        # selection and Δ_tag/early-stop gates. Train metrics are still
        # computed and logged for diagnostic continuity. Test untouched.
        # codex 2026-05-10 #6: never pass full R into _compute_metrics here.
        m_dev: dict | None = None
        boot_dev: dict | None = None
        m_train: dict | None = None
        boot_train: dict | None = None
        model_probe_dev: dict | None = None
        if v_loop_state is not None:
            bench_split = v_loop_state["bench_split"]
            S_train, m_train, boot_train = _compute_metrics(
                T_i, bench_split.train,
                v_loop_state["R_raw_train"], v_loop_state["R01_train"],
                config["theta_p_q"], config["theta_n_q"],
                config["bootstrap_B"], config["seed"],
            )
            S_dev, m_dev, boot_dev = _compute_metrics(
                T_i, bench_split.dev,
                v_loop_state["R_raw_dev"], v_loop_state["R01_dev"],
                config["theta_p_q"], config["theta_n_q"],
                config["bootstrap_B"], config["seed"],
            )
            model_probe_dev = _compute_model_probe_dev_metrics(
                T=T_i,
                benchmark_names=bench_split.dev,
                model_probe_state=model_probe_state,
                config=config,
            )
            S_i, m_i, boot_i = S_dev, m_dev, boot_dev
            # Improver follows the same non-test signal as selection. Using
            # train residuals here while selection optimizes dev makes the
            # prompt walk chase a different objective from the chosen candidate.
            err_i = build_error_report(
                S_dev,
                v_loop_state["R_raw_dev"], v_loop_state["R01_dev"],
                top_k=config["error_top_k"],
                q_p_s=config["theta_p_q"], q_n_s=config["theta_n_q"],
            )
            res_i = build_residual_report(
                S_dev, v_loop_state["R_raw_dev"], top_k=config["error_top_k"],
            )
            cal_i = _maybe_build_calibration(
                T_i, bench_split.train,
                v_loop_state["R_raw_train"], v_loop_state["R01_train"],
                active_vocab, config,
            )
        else:
            S_i, m_i, boot_i = _compute_metrics(
                T_i, corpus.benchmark_names, R_raw, R01,
                config["theta_p_q"], config["theta_n_q"],
                config["bootstrap_B"], config["seed"],
            )
            err_i = build_error_report(
                S_i, R_raw, R01, top_k=config["error_top_k"],
                q_p_s=config["theta_p_q"], q_n_s=config["theta_n_q"],
            )
            res_i = build_residual_report(S_i, R_raw, top_k=config["error_top_k"])
            cal_i = _maybe_build_calibration(
                T_i, corpus.benchmark_names, R_raw, R01, active_vocab, config,
            )
        _log_metrics(f"iter_{i:03d}", m_i, extra=f"errors={len(err_i)}")
        if m_train is not None:
            _log_metrics(f"iter_{i:03d}/train", m_train)

        improver_result: ImproverResult | None = None
        improver_payload: dict | None = None
        improver_skip_reason: str | None = None

        # When V loop is on, both delta_tag (for the gate) and reported
        # IterationResult fields are dev-derived (m_i := m_dev above). If dev
        # delta_tag is undefined, the candidate is not gate-passable.
        if m_dev is not None:
            gate_delta = _selection_delta_tag(m_dev)
        else:
            gate_delta = m_i["delta_tag"]
        ir = IterationResult(
            label=f"iter_{i:03d}", iter=i, prompt=current_prompt,
            T=T_i, S=S_i,
            L_align=m_i["L_align"],
            L_align_01=m_i["L_align_01"],
            rho_align_pearson=m_i["rho_align_pearson"],
            rho_align_spearman=m_i["rho_align_spearman"],
            delta_tag=gate_delta, bootstrap=boot_i,
            error_report_size=len(err_i),
            improver=improver_payload,
            tag_weight_metadata=gen_i.tag_weight_metadata,
            vocab=V_i,
            vocab_hash=vocab_hash_i,
            dev_metrics=({**m_dev, "bootstrap": boot_dev} if m_dev is not None else None),
            train_metrics=(
                {**m_train, "bootstrap": boot_train}
                if m_train is not None
                else None
            ),
            model_probe_dev_metrics=model_probe_dev,
        )

        candidate = _selection_candidate(
            ir,
            seed_vocab=vocab,
            config=config,
        )
        _dt_thr = float(config.get("delta_tag_threshold", 0.0))
        is_better, gate_pass, no_improve_reason = _candidate_improvement_status(
            candidate,
            best,
            selection_cfg=selection_cfg,
            delta_tag_threshold=_dt_thr,
        )
        selection_candidates.append(
            _selection_record(
                candidate,
                objective_key=selection_objective_key,
                gate_pass=gate_pass,
                is_better=is_better,
                reason=no_improve_reason,
            )
        )
        if is_better:
            best = candidate
            best_result = ir
            consecutive_no_improve = 0
            print(
                f"  [loop] iter {i}: new best taxonomy candidate "
                f"(source={candidate['vocab_source']}, "
                f"tag_count={candidate['tag_count']}, "
                f"selection_score={_fmt_metric(candidate.get(selection_objective_key))})"
            )
        else:
            consecutive_no_improve += 1
            if not gate_pass:
                print(
                    f"  [loop] iter {i}: Δ_tag={ir.delta_tag} fails gate "
                    f"(>{_dt_thr} required); consecutive_no_improve={consecutive_no_improve}"
                )
            else:
                print(
                    f"  [loop] iter {i}: no selection improvement "
                    f"({no_improve_reason}); consecutive_no_improve={consecutive_no_improve}"
                )

        if i < int(config["max_iter"]) and v_loop_state is not None:
            improver_skip_reason = _improver_selection_guard_skip_reason(
                m_dev=m_dev,
                m_train=m_train,
                model_probe_dev=model_probe_dev,
                config=config,
            )
            if improver_skip_reason is None and not is_better:
                improver_skip_reason = no_improve_reason
            if improver_skip_reason:
                print(
                    f"  [loop] iter {i}: skipping prompt improvement "
                    f"({improver_skip_reason})"
                )

        # If not last iteration, ask A_imp for I_{i+1}
        if i < int(config["max_iter"]) and not improver_skip_reason:
            mi = role_cfg(config, "improver_model")
            print(f"  [loop] iter {i}: requesting prompt improvement from {mi['name']}")
            allow_taxonomy_changes = v_loop_state is not None
            if improver_fn is not None:
                improver_kwargs = dict(
                    prev_prompt=current_prompt,
                    base_prompt=base_prompt,
                    error_report=err_i,
                    metrics=m_i,
                    bench_descriptions=descriptions_i,
                    vocab=active_vocab,
                    benchmark_names=corpus.benchmark_names,
                    model=mi["name"],
                    base_url=mi.get("base_url"),
                )
                if _supports_kwarg(improver_fn, "base_url_env"):
                    improver_kwargs["base_url_env"] = mi.get("base_url_env")
                if _supports_kwarg(improver_fn, "api_key_env"):
                    improver_kwargs["api_key_env"] = mi.get("api_key_env")
                if _supports_kwarg(improver_fn, "allow_taxonomy_changes"):
                    improver_kwargs["allow_taxonomy_changes"] = allow_taxonomy_changes
                if _supports_kwarg(improver_fn, "json_contract_strict"):
                    improver_kwargs["json_contract_strict"] = bool(
                        config.get("llm_json_contract_strict", True)
                    )
                if _supports_kwarg(improver_fn, "json_contract_max_attempts"):
                    improver_kwargs["json_contract_max_attempts"] = int(
                        config.get("llm_json_contract_max_attempts", 3)
                    )
                if _supports_kwarg(improver_fn, "empty_content_retries"):
                    improver_kwargs["empty_content_retries"] = llm_empty_content_retries(
                        config
                    )
                if _supports_kwarg(improver_fn, "request_timeout_s"):
                    improver_kwargs["request_timeout_s"] = llm_request_timeout_s(config)
                if _supports_kwarg(improver_fn, "sdk_exception_retries"):
                    improver_kwargs["sdk_exception_retries"] = llm_sdk_exception_retries(
                        config
                    )
                if _supports_kwarg(improver_fn, "debug_dump_dir"):
                    improver_kwargs["debug_dump_dir"] = llm_debug_dump_dir(config)
                if _supports_kwarg(improver_fn, "extra_body"):
                    improver_kwargs["extra_body"] = llm_extra_body(config)
                improver_result = improver_fn(**improver_kwargs)
            else:
                improver_result = improve_prompt(
                    prev_prompt=current_prompt,
                    base_prompt=base_prompt,
                    error_report=err_i,
                    metrics=m_i,
                    bench_descriptions=descriptions_i,
                    vocab=active_vocab,
                    benchmark_names=corpus.benchmark_names,
                    model=mi["name"],
                    base_url=mi.get("base_url"),
                    base_url_env=mi.get("base_url_env"),
                    api_key_env=mi.get("api_key_env"),
                    allow_taxonomy_changes=allow_taxonomy_changes,
                    temperature=float(config.get("improver_temperature", 0.0)),
                    n_samples=int(config.get("improver_n_samples", 1)),
                    json_contract_strict=bool(config.get("llm_json_contract_strict", True)),
                    json_contract_max_attempts=int(
                        config.get("llm_json_contract_max_attempts", 3)
                    ),
                    empty_content_retries=llm_empty_content_retries(config),
                    request_timeout_s=llm_request_timeout_s(config),
                    sdk_exception_retries=llm_sdk_exception_retries(config),
                    debug_dump_dir=llm_debug_dump_dir(config),
                    extra_body=llm_extra_body(config),
                    seed=_role_iteration_seed(config, "improver", i),
                )
            improver_payload = {
                "accepted": improver_result.accepted,
                "reasons": improver_result.reasons,
                "rationale": improver_result.rationale,
                "raw_response": improver_result.raw_response,
            }
            print(
                f"  [loop] iter {i}: improver accepted={improver_result.accepted}, "
                f"reasons={improver_result.reasons or ['ok']}"
            )
            ir.improver = improver_payload

        save_iteration(
            run_dir, i, current_prompt, T_i, S_i,
            {**m_i, "bootstrap": boot_i},
            error_pairs_to_dicts(err_i),
            improver_payload,
            label=f"iter_{i:03d}",
            residual_report=res_i,
            calibrated_tag_vectors=cal_i["T"] if cal_i else None,
            calibrated_similarity=cal_i["S"] if cal_i else None,
            calibrated_metrics=cal_i["metrics"] if cal_i else None,
            tag_weight_metadata=gen_i.tag_weight_metadata,
        )
        if v_loop_state is not None and m_dev is not None:
            iter_dir = os.path.join(run_dir, f"iter_{i:03d}")
            # main metrics.json is dev-derived (selection signal). Also persist
            # the train view for diagnostic continuity and generalization gates.
            write_json(
                os.path.join(iter_dir, "metrics_dev.json"),
                {**m_dev, "bootstrap": boot_dev},
            )
            if m_train is not None:
                write_json(
                    os.path.join(iter_dir, "metrics_train.json"),
                    {**m_train, "bootstrap": boot_train},
                )
            if model_probe_dev is not None:
                write_json(
                    os.path.join(iter_dir, "metrics_model_probe_dev.json"),
                    model_probe_dev,
                )

        history.append(ir)

        _wandb_log(
            wandb_run,
            {
                "phase": "main",
                "phase_iter": i,
                "label": ir.label,
                **{f"main/{k}": v for k, v in m_i.items()},
                **{f"main/{k}": v for k, v in _flatten_bootstrap(boot_i).items()},
                **(
                    {f"main_calibrated/{k}": v for k, v in cal_i["metrics"].items() if not isinstance(v, dict)}
                    if cal_i else {}
                ),
                "main/delta_tag_gate_pass": int(gate_pass),
                "main/consecutive_no_improve": consecutive_no_improve,
            },
        )

        # No-improvement stall guard (v3 §2.2.6 fallback rule)
        if (
            best_result is not None
            and consecutive_no_improve >= int(config.get("early_stop_consecutive", 2))
        ):
            stalled_delta_tag = not gate_pass
            stop_status = (
                "stalled_delta_tag" if stalled_delta_tag else "stalled_no_improvement"
            )
            stop_details = {
                "iter": i,
                "reason": no_improve_reason,
                "gate_pass": bool(gate_pass),
                "candidate": candidate,
                "best": best,
            }
            print(
                f"  [loop] {stop_status} at iter {i}: "
                f"{consecutive_no_improve} consecutive non-improving rounds"
            )
            break

        # Early stop on convergence
        if prev_L:
            if not math.isnan(prev_L[-1]) and not math.isnan(ir.L_align):
                if abs(ir.L_align - prev_L[-1]) < float(config["eps"]):
                    consecutive_small += 1
                else:
                    consecutive_small = 0
        prev_L.append(ir.L_align)
        if consecutive_small >= int(config.get("early_stop_consecutive", 2)):
            print(f"  [loop] early stop at iter {i} (Δ<{config['eps']} for {consecutive_small} iters)")
            break

        # Apply a valid improved prompt only from a selected candidate. Rewriting
        # from rejected candidates turns the loop into a random walk away from
        # the current best prompt.
        if improver_result is not None:
            if v_loop_state is not None and not is_better:
                if best_result is not None:
                    current_prompt = best_result.prompt or base_prompt
                print(
                    f"  [loop] iter {i}: discarded improver prompt because "
                    f"candidate was not selected ({no_improve_reason})"
                )
            else:
                next_prompt, improver_stop = _next_prompt_from_improver(
                    improver_result,
                    current_prompt,
                )
                if next_prompt is not None:
                    current_prompt = next_prompt
                else:
                    improver_stop = improver_stop or "improver_rejected"
                    if _can_continue_with_same_prompt_after_improver_stop(
                        config,
                        v_loop_active=v_loop_state is not None,
                    ):
                        print(
                            f"  [loop] iter {i}: {improver_stop}; "
                            "continuing with unchanged prompt for next candidate count"
                        )
                    else:
                        stop_status = improver_stop
                        stop_details = {
                            "iter": i,
                            "accepted": bool(improver_result.accepted),
                            "reasons": improver_result.reasons,
                        }
                        print(
                            f"  [loop] stop at iter {i}: {stop_status}; "
                            "no valid next prompt"
                        )
                        break
        elif best_result is not None:
            # rollback to best prompt
            current_prompt = best_result.prompt or base_prompt

    if best_result is None:
        if v_loop_state is not None:
            stop_status = "no_gate_passing_candidate"
            stop_details = {
                "delta_tag_threshold": float(config.get("delta_tag_threshold", 0.0)),
                "history_labels": [h.label for h in history],
                "candidate_delta_tags": {
                    h.label: h.delta_tag for h in history
                    if h.label != "iter_000_baseline_random"
                },
            }
            fixed_baseline = next(
                (h for h in history if h.label == "iter_000_baseline_static"),
                None,
            )
            if fixed_baseline is not None:
                best_result = fixed_baseline
                best = _selection_candidate(best_result, seed_vocab=vocab, config=config)
                stop_status = "no_gate_passing_taxonomy_candidate"
                stop_details["selected_fixed_baseline"] = True
                print(
                    "  [loop] no gate-passing taxonomy candidate; "
                    "using iter_000_baseline_static for final/"
                )
            else:
                _write_stop_reason(
                    run_dir,
                    stalled_delta_tag=stalled_delta_tag,
                    consecutive_no_improve=consecutive_no_improve,
                    threshold=int(config.get("early_stop_consecutive", 2)),
                    status=stop_status,
                    details=stop_details,
                )
                if selection_candidates:
                    _write_selection_candidates(
                        run_dir,
                        candidates=selection_candidates,
                        selection_cfg=selection_cfg,
                        selected_label=None,
                    )
                _write_llm_fallbacks(run_dir)
                raise RuntimeError(
                    "v_loop produced no gate-passing candidate; refusing to select "
                    "a best iteration from gate-failed history"
                )
        # only baselines ran; pick the better baseline
        if best_result is None:
            non_random = [h for h in history if h.label != "iter_000_baseline_random"]
            pool = non_random or history
            best_result = min(
                pool,
                key=lambda h: (h.L_align if not math.isnan(h.L_align) else float("inf")),
            )

    # Phase L — schema-symmetric writer. When V loop is on, save the iteration
    # vocab + executer metadata; otherwise the seed vocab is the active vocab.
    final_vocab = best_result.vocab if best_result.vocab is not None else vocab
    final_tag_weight_metadata = best_result.tag_weight_metadata
    T_full_best = best_result.T

    # v_loop keeps D_test completely outside the iterative loop. The selected
    # prompt/V are therefore applied to the full corpus exactly once here, after
    # selection, before final reporting and split metrics are written.
    if v_loop_state is not None:
        print(
            f"  [loop] final full-corpus pass: applying {best_result.label} "
            f"to {len(corpus.benchmark_names)} benchmarks"
        )
        final_descriptions = descriptions
        if (
            not static_from_mapreduce
            and config.get("use_mapreduce_evidence", False)
            and corpus.documents
        ):
            print(
                "  [mapreduce] final: building full-corpus evidence after "
                "v_loop selection"
            )
            final_descriptions, _final_mapreduce_aggregates = build_mapreduce_descriptions(
                corpus=corpus,
                vocab=final_vocab,
                config=config,
                run_dir=run_dir,
                chat_fn=mapreduce_chat_fn,
            )
        gen_final = _generate_T_for_prompt(
            corpus=corpus,
            benchmark_names=corpus.benchmark_names,
            descriptions=final_descriptions,
            vocab=final_vocab,
            prompt=best_result.prompt,
            version=best_result.iter,
            tag_fn=tag_fn,
            config=config,
            run_dir=run_dir,
            label=f"final/{best_result.label}",
            static_from_mapreduce=static_from_mapreduce,
            base_prompt=base_prompt,
            mapreduce_chat_fn=mapreduce_chat_fn,
            mapreduce_reducer_chat_fn=mapreduce_reducer_chat_fn,
        )
        T_full_best = gen_final.T
        final_tag_weight_metadata = gen_final.tag_weight_metadata or final_tag_weight_metadata

    S_full_best, m_full_best, boot_full_best = _compute_metrics(
        T_full_best,
        corpus.benchmark_names,
        R_raw,
        R01,
        config["theta_p_q"],
        config["theta_n_q"],
        config["bootstrap_B"],
        config["seed"],
    )
    final_residuals = build_residual_report(S_full_best, R_raw, top_k=config["error_top_k"])
    final_cal = _maybe_build_calibration(
        T_full_best,
        corpus.benchmark_names,
        R_raw,
        R01,
        final_vocab,
        config,
    )

    best_vocab_source = _result_vocab_source(best_result, config)
    best_tag_count = _result_tag_count(best_result, vocab)
    best_selection_candidate = _selection_candidate(
        best_result,
        seed_vocab=vocab,
        config=config,
    )
    final_vocab_metadata: dict | None = None
    if v_loop_state is not None:
        final_vocab_metadata = {
            "vocab_hash": best_result.vocab_hash,
            "best_iter_label": best_result.label,
            "selected_vocab_source": best_vocab_source,
            "selected_tag_count": best_tag_count,
            "selection_candidate": best_selection_candidate,
        }
        if best_result.vocab is not None:
            final_vocab_metadata["source_benchmarks"] = v_loop_state["source_benchmarks"]
    selection_metrics_with_bootstrap = _metrics_payload(best_result)
    full_metrics_with_bootstrap = {**m_full_best, "bootstrap": boot_full_best}
    final_metrics_payload = {
        **selection_metrics_with_bootstrap,
        "selection": selection_metrics_with_bootstrap,
        "full_corpus": full_metrics_with_bootstrap,
        "selection_scope": "dev" if v_loop_state is not None else "full_corpus",
        "final_eval_scope": "full_corpus",
    }
    save_final(
        run_dir,
        best_iter_label=best_result.label,
        prompt=best_result.prompt,
        tag_vectors=T_full_best,
        metrics_with_bootstrap=final_metrics_payload,
        vocab=final_vocab,
        vocab_metadata=final_vocab_metadata,
        calibrated_tag_vectors=final_cal["T"] if final_cal else None,
        calibrated_metrics_with_bootstrap=final_cal["metrics"] if final_cal else None,
        residual_report=final_residuals,
        tag_weight_metadata=final_tag_weight_metadata,
    )
    save_profile_support(
        run_dir,
        _build_profile_support(
            Y_norm=Y_norm,
            T=T_full_best,
            benchmark_names=corpus.benchmark_names,
            model_names=corpus.model_names,
            subset_sizes=list(config.get("subset_profile_sizes", [])),
            methods=list(config.get("subset_selection_methods", ["greedy"])),
            kmedoids_seed=int(config.get("kmedoids_seed", 0)),
        ),
    )
    _write_stop_reason(
        run_dir,
        stalled_delta_tag=stalled_delta_tag,
        consecutive_no_improve=consecutive_no_improve,
        threshold=int(config.get("early_stop_consecutive", 2)),
        status=stop_status if stop_status != "ok" else None,
        details=stop_details,
    )
    llm_fallback_payload = _write_llm_fallbacks(run_dir)
    fallback_threshold = config.get("llm_fallback_fail_threshold", 0)
    if fallback_threshold is not None and llm_fallback_payload["total"] > int(fallback_threshold):
        raise RuntimeError(
            "LLM fallback threshold exceeded: "
            f"total={llm_fallback_payload['total']} "
            f"threshold={fallback_threshold} counts={llm_fallback_payload['counts']}"
        )
    _write_split_metrics_artifact(
        run_dir=run_dir,
        config=config,
        S=S_full_best,
        R_raw=R_raw,
        R01=R01,
        Y_norm=Y_norm,
        Y_norm_for_held=Y_norm_full,
        benchmark_names=corpus.benchmark_names,
        model_names=corpus.model_names,
        split_required_pair_dicts=split_required_pair_dicts,
    )
    if v_loop_state is not None:
        # codex 2026-05-10 #10 pass criterion #5 — record that test split was
        # evaluated exactly once (after the loop).
        v_loop_state["test_split_eval_count"] += 1
    print(
        f"  [loop] loop best={best_result.label}, "
        f"vocab_source={best_vocab_source}, tag_count={best_tag_count}, "
        f"selection_L_align={_fmt_metric(best_result.L_align)}, "
        f"full_L_align={_fmt_metric(m_full_best.get('L_align'))}, "
        f"saved under final/"
    )
    _wandb_log(
        wandb_run,
        {
            "phase": "final",
            "label": best_result.label,
            "final/L_align": m_full_best.get("L_align"),
            "final/L_align_01": m_full_best.get("L_align_01"),
            "final/rho_align_pearson": m_full_best.get("rho_align_pearson"),
            "final/rho_align_spearman": m_full_best.get("rho_align_spearman"),
            "final/delta_tag": m_full_best.get("delta_tag"),
            "final_selection/L_align": best_result.L_align,
            "final_selection/rho_align_pearson": best_result.rho_align_pearson,
            "final_selection/delta_tag": best_result.delta_tag,
            "final_selection/tag_count": best_tag_count,
            "final_selection/tag_count_penalty": best_selection_candidate.get(
                "selection_penalty_tag_count"
            ),
            **{f"final/{k}": v for k, v in _flatten_bootstrap(boot_full_best).items()},
            **(
                {f"final_calibrated/{k}": v for k, v in final_cal["metrics"].items() if not isinstance(v, dict)}
                if final_cal else {}
            ),
        },
    )
    if wandb_run is not None:
        try:
            wandb_run.summary["best_label"] = best_result.label
            wandb_run.summary["best_L_align"] = m_full_best.get("L_align")
            wandb_run.summary["best_rho_pearson"] = m_full_best.get("rho_align_pearson")
            wandb_run.summary["best_delta_tag"] = m_full_best.get("delta_tag")
        except Exception as exc:
            print(f"  [wandb] summary update failed: {exc}")

    all_final_residuals = [
        abs(float(sv) - float(R_raw[k]))
        for k, sv in S_full_best.items()
        if R_raw.get(k) is not None
    ]
    fixed_metrics = {
        **m_full_best,
        "bootstrap": boot_full_best,
        "residual_mean": (
            float(sum(all_final_residuals) / len(all_final_residuals))
            if all_final_residuals else float("nan")
        ),
        "residual_max": (
            float(max(all_final_residuals))
            if all_final_residuals else float("nan")
        ),
        "n_pairs": int(sum(1 for v in R_raw.values() if v is not None)),
    }
    protected_pairs = _build_protected_pairs(
        S_full_best,
        R_raw,
        top_k=int(config.get("taxonomy_refinement_protected_pairs_top_k", 10)),
        min_r=float(config.get("taxonomy_refinement_protected_pairs_min_r", 0.80)),
    )
    taxonomy_status = _taxonomy_trigger_status(fixed_metrics, config)
    save_taxonomy_status(run_dir, taxonomy_status)
    print(
        "  [taxonomy_refinement] "
        f"enabled={taxonomy_status['enabled']}, triggered={taxonomy_status['triggered']}, "
        f"reasons={taxonomy_status['reasons']}"
    )
    taxonomy_best: IterationResult | None = None
    taxonomy_metrics: dict | None = None
    if taxonomy_status["triggered"]:
        taxonomy_best = _run_taxonomy_unlocked_phase(
            run_dir=run_dir,
            config=config,
            corpus=corpus,
            descriptions=descriptions,
            seed_vocab=vocab,
            base_prompt=base_prompt,
            fixed_best=best_result,
            fixed_metrics=fixed_metrics,
            fixed_residuals=final_residuals,
            protected_pairs=protected_pairs,
            Y_norm=Y_norm,
            R_raw=R_raw,
            R01=R01,
            tag_fn=tag_fn,
            improver_fn=improver_fn,
            taxonomy_refiner_fn=taxonomy_refiner_fn,
            static_from_mapreduce=static_from_mapreduce,
            mapreduce_chat_fn=mapreduce_chat_fn,
            mapreduce_reducer_chat_fn=mapreduce_reducer_chat_fn,
            wandb_run=wandb_run,
        )
        taxonomy_status["completed"] = taxonomy_best is not None
        taxonomy_status["best_iter"] = taxonomy_best.label if taxonomy_best else None
        taxonomy_status["best_L_align"] = taxonomy_best.L_align if taxonomy_best else None
        if taxonomy_best is not None:
            taxonomy_metrics = _taxonomy_metrics_for_selection(taxonomy_best, R_raw)

    adoption = _taxonomy_adoption_decision(fixed_metrics, taxonomy_metrics, config)
    if not taxonomy_status["triggered"] and adoption["reasons"] == ["taxonomy_not_completed"]:
        adoption["reasons"] = ["taxonomy_not_triggered"]
    taxonomy_status["adoption"] = adoption
    save_taxonomy_status(run_dir, taxonomy_status)

    selected_result = taxonomy_best if adoption["adopted"] and taxonomy_best is not None else best_result
    selected_metrics = taxonomy_metrics if adoption["adopted"] and taxonomy_metrics is not None else fixed_metrics
    selected_vocab_source = _result_vocab_source(selected_result, config)
    selected_source = (
        "taxonomy_refinement"
        if adoption["adopted"] and taxonomy_best is not None
        else _selected_source_label(selected_vocab_source)
    )
    selected_tag_count = _result_tag_count(selected_result, vocab)
    selected_vocab_hash = selected_result.vocab_hash or _vocab_hash(
        selected_result.vocab if selected_result.vocab is not None else vocab
    )
    if selection_candidates:
        _write_selection_candidates(
            run_dir,
            candidates=selection_candidates,
            selection_cfg=selection_cfg,
            selected_label=(
                selected_result.label if selected_source != "taxonomy_refinement" else None
            ),
        )
    # Phase L — `mode` / `chosen_iter_label` / `chosen_metrics` are the canonical
    # keys per the unified schema; `selected_*` are kept as aliases for older
    # consumers (deletion would break tests / external readers).
    mode = "taxonomy" if selected_source != "fixed" else "fixed"
    selection_payload = {
        "mode": mode,
        "chosen_iter_label": selected_result.label,
        "chosen_metrics": selected_metrics,
        "selected_label": selected_result.label,
        "selected_source": selected_source,
        "selected_metrics": selected_metrics,
        "selected_vocab_source": selected_vocab_source,
        "selected_tag_count": selected_tag_count,
        "selected_vocab_hash": selected_vocab_hash,
        "selected_vocab_path": (
            "taxonomy_refinement/final/vocab_star.json"
            if selected_source == "taxonomy_refinement"
            else "final/vocab_star.json"
        ),
        "selected_tag_vectors_path": (
            "taxonomy_refinement/final/T_star.json"
            if selected_source == "taxonomy_refinement"
            else "final/T_star.json"
        ),
        "selected_candidate": _selection_candidate(
            selected_result,
            seed_vocab=vocab,
            config=config,
        ),
        "candidate_history_path": (
            "selection_candidates.json" if selection_candidates else None
        ),
        "fixed_path": "final",
        "taxonomy_path": "taxonomy_refinement/final" if taxonomy_best is not None else None,
        "adoption": adoption,
    }
    save_selection(run_dir, selection_payload)
    _wandb_log(
        wandb_run,
        {
            "phase": "selection",
            "selected/source": selected_source,
            "selected/vocab_source": selected_vocab_source,
            "selected/tag_count": selected_tag_count,
            "selected/L_align": selected_metrics.get("L_align"),
            "selected/L_align_01": selected_metrics.get("L_align_01"),
            "selected/rho_align_pearson": selected_metrics.get("rho_align_pearson"),
            "selected/rho_align_spearman": selected_metrics.get("rho_align_spearman"),
            "selected/delta_tag": selected_metrics.get("delta_tag"),
            "selected/residual_mean": selected_metrics.get("residual_mean"),
            "selected/residual_max": selected_metrics.get("residual_max"),
            "taxonomy/adoption_adopted": 1 if adoption["adopted"] else 0,
            "taxonomy/adoption_reason_count": len(adoption["reasons"]),
        },
    )
    if wandb_run is not None:
        try:
            wandb_run.summary["selected_label"] = selected_result.label
            wandb_run.summary["selected_source"] = selected_source
            wandb_run.summary["selected_vocab_source"] = selected_vocab_source
            wandb_run.summary["selected_tag_count"] = selected_tag_count
            wandb_run.summary["selected_L_align"] = selected_metrics.get("L_align")
            wandb_run.summary["selected_rho_pearson"] = selected_metrics.get("rho_align_pearson")
            wandb_run.summary["selected_delta_tag"] = selected_metrics.get("delta_tag")
        except Exception as exc:
            print(f"  [wandb] selected summary update failed: {exc}")
    print(
        f"  [selection] selected={selected_source}, label={selected_result.label}, "
        f"tag_count={selected_tag_count}, "
        f"L_align={_fmt_metric(selected_metrics.get('L_align'))}, "
        f"adoption_reasons={adoption['reasons'] or ['ok']}"
    )

    return history, selected_result
