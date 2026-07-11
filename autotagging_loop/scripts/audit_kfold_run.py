"""Audit a v3 K-fold run directory without making LLM calls."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from autotagging_loop.experiment.maker import _maker_evidence_quality_reasons
from autotagging_loop.experiment.mapreduce_evidence import _mapper_evidence_quality_reasons
from autotagging_loop.experiment.taxonomy_refiner import vocab_quality_reasons


METRIC_KEYS = (
    "L_align",
    "L_align_01",
    "rho_align_pearson",
    "rho_align_spearman",
    "delta_tag",
)

STRICT_MIN_COMMON_MODELS = 6
STRICT_MIN_SPLIT_VALID_PAIRS = 10
STRICT_MIN_EFFECTIVE_BENCHMARKS = 6
DEFAULT_ALPHA = 0.05
DEFAULT_MIN_RHO_S = 0.20
DEFAULT_MIN_FOLD_RHO_S = 0.0


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _finite(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _fmt(value: Any) -> str:
    v = _finite(value)
    return "nan" if v is None else f"{v:+.4f}"


def _int_value(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def strict_config_failures(config: dict[str, Any]) -> list[str]:
    """Return failures that distinguish strict runs from relaxed smoke runs."""

    exp = config.get("experiment") if isinstance(config.get("experiment"), dict) else config
    failures: list[str] = []

    min_common = _int_value(exp.get("min_common_models"))
    if min_common is None:
        failures.append("strict_config_missing:min_common_models")
    elif min_common < STRICT_MIN_COMMON_MODELS:
        failures.append(
            "strict_config_min_common_models:"
            f"{min_common}<{STRICT_MIN_COMMON_MODELS}"
        )

    for key in (
        "v_loop_min_train_valid_pairs",
        "v_loop_min_dev_valid_pairs",
        "v_loop_min_test_valid_pairs",
    ):
        value = _int_value(exp.get(key))
        if value is None:
            failures.append(f"strict_config_missing:{key}")
        elif value < STRICT_MIN_SPLIT_VALID_PAIRS:
            failures.append(
                f"strict_config_{key}:{value}<"
                f"{STRICT_MIN_SPLIT_VALID_PAIRS}"
            )

    for key in (
        "v_loop_min_train_effective_benchmarks",
        "v_loop_min_dev_effective_benchmarks",
        "v_loop_min_test_effective_benchmarks",
    ):
        value = _int_value(exp.get(key))
        if value is None:
            failures.append(f"strict_config_missing:{key}")
        elif value < STRICT_MIN_EFFECTIVE_BENCHMARKS:
            failures.append(
                f"strict_config_{key}:{value}<"
                f"{STRICT_MIN_EFFECTIVE_BENCHMARKS}"
            )

    if exp.get("v_loop_require_held_model_test") is not True:
        failures.append("strict_config_v_loop_require_held_model_test_not_true")

    scope = str(exp.get("v_loop_score_model_scope", "")).strip().lower()
    if scope != "seen":
        failures.append(
            f"strict_config_v_loop_score_model_scope:{scope or 'missing'}!=seen"
        )

    if exp.get("executer_fallback_to_seed") is not False:
        failures.append("strict_config_executer_fallback_to_seed_not_false")

    if exp.get("tag_generator_allow_uniform_fallback") is not False:
        failures.append("strict_config_tag_generator_allow_uniform_fallback_not_false")

    if exp.get("llm_json_contract_strict") is not True:
        failures.append("strict_config_llm_json_contract_strict_not_true")

    return failures


def pooled_quality_failures(
    agg: dict[str, Any],
    *,
    alpha: float = DEFAULT_ALPHA,
    min_rho_s: float = DEFAULT_MIN_RHO_S,
) -> list[str]:
    pooled = (agg or {}).get("pooled") or {}
    rho = pooled.get("rho_spearman") or {}
    failures: list[str] = []
    observed = _finite(rho.get("observed"))
    p_two = _finite(rho.get("p_two_sided"))
    n_pairs = _int_value(pooled.get("n_pairs"))

    if observed is None:
        failures.append("quality_pooled_rho_s_not_finite")
    elif observed < float(min_rho_s):
        failures.append(
            f"quality_pooled_rho_s_below_floor:"
            f"{observed:.4f}<{float(min_rho_s):.4f}"
        )

    if p_two is None:
        failures.append("quality_pooled_p_two_not_finite")
    elif p_two > float(alpha):
        failures.append(
            f"quality_pooled_p_two_above_alpha:{p_two:.4f}>{float(alpha):.4f}"
        )

    if n_pairs is None:
        failures.append("quality_pooled_n_pairs_missing")
    elif n_pairs <= 0:
        failures.append("quality_pooled_n_pairs_empty")
    return failures


def _block_quality_failures(
    label: str,
    block: Any,
    *,
    min_pairs: int = STRICT_MIN_SPLIT_VALID_PAIRS,
    min_effective_benchmarks: int = STRICT_MIN_EFFECTIVE_BENCHMARKS,
    min_rho_s: float | None = DEFAULT_MIN_FOLD_RHO_S,
) -> list[str]:
    failures: list[str] = []
    if not isinstance(block, dict) or not block:
        return [f"quality_{label}_missing"]

    n_pairs = _int_value(block.get("n_pairs"))
    if n_pairs is None:
        failures.append(f"quality_{label}_n_pairs_missing")
    elif n_pairs < int(min_pairs):
        failures.append(f"quality_{label}_n_pairs:{n_pairs}<{int(min_pairs)}")

    n_effective = _int_value(block.get("n_effective_benchmarks"))
    if n_effective is None:
        failures.append(f"quality_{label}_n_effective_benchmarks_missing")
    elif n_effective < int(min_effective_benchmarks):
        failures.append(
            f"quality_{label}_n_effective_benchmarks:"
            f"{n_effective}<{int(min_effective_benchmarks)}"
        )

    if _finite(block.get("L_align")) is None:
        failures.append(f"quality_{label}_L_align_not_finite")
    rho_s = _finite(block.get("rho_align_spearman"))
    if rho_s is None:
        failures.append(f"quality_{label}_rho_s_not_finite")
    elif min_rho_s is not None and rho_s < float(min_rho_s):
        failures.append(
            f"quality_{label}_rho_s_below_floor:"
            f"{rho_s:.4f}<{float(min_rho_s):.4f}"
        )
    return failures


def fold_quality_failures(
    split_metrics: dict[str, Any],
    *,
    selection_scope: str | None,
    min_pairs: int = STRICT_MIN_SPLIT_VALID_PAIRS,
    min_effective_benchmarks: int = STRICT_MIN_EFFECTIVE_BENCHMARKS,
    min_rho_s: float | None = DEFAULT_MIN_FOLD_RHO_S,
) -> list[str]:
    """Return failures for fold-level evidence quality.

    The pooled permutation test can look directionally good even when each fold
    is backed by too few comparable benchmark pairs. Research-grade runs need
    finite selection/test signals and enough held-out pairs per fold.
    """

    failures: list[str] = []
    if not isinstance(split_metrics, dict) or not split_metrics:
        return ["quality_fold_split_metrics_missing"]

    if not selection_scope:
        failures.append("quality_fold_selection_scope_missing")
    else:
        failures.extend(
            _block_quality_failures(
                f"selection_{selection_scope}",
                split_metrics.get(selection_scope),
                min_pairs=min_pairs,
                min_effective_benchmarks=min_effective_benchmarks,
                min_rho_s=min_rho_s,
            )
        )

    failures.extend(
        _block_quality_failures(
            "test",
            split_metrics.get("test"),
            min_pairs=min_pairs,
            min_effective_benchmarks=min_effective_benchmarks,
            min_rho_s=min_rho_s,
        )
    )

    held = split_metrics.get("held_model_test")
    if isinstance(held, dict) and held.get("skipped"):
        failures.append("quality_held_model_test_skipped")
    else:
        failures.extend(
            _block_quality_failures(
                "held_model_test",
                held,
                min_pairs=min_pairs,
                min_effective_benchmarks=min_effective_benchmarks,
                min_rho_s=min_rho_s,
            )
        )
    return failures


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _safe_load_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _fold_benchmark_names(fold_dir: Path) -> list[str]:
    corpus = _safe_load_json(fold_dir / "corpus.json")
    if not isinstance(corpus, dict):
        return []
    names = corpus.get("benchmark_names")
    if not isinstance(names, list):
        return []
    return [str(name) for name in names if str(name).strip()]


def _vocab_from_payload(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("vocab", "refined_vocab", "vocab_star"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def role_output_quality_failures(fold_dir: Path) -> list[str]:
    """Return saved-role-output quality failures for one fold directory.

    Runtime contracts reject new bad outputs, but research acceptance should also
    reject saved artifacts whose Mapper evidence, generated vocabularies, or
    Maker rationales leak benchmark identity, surface answer format,
    model-performance shortcuts, or pure difficulty axes.
    """

    failures: list[str] = []
    benchmark_names = _fold_benchmark_names(fold_dir)

    for path in sorted((fold_dir / "map_evidence").rglob("aggregate.json")):
        payload = _safe_load_json(path)
        if not isinstance(payload, dict):
            failures.append(f"role_quality_mapper_unreadable:{_rel(path, fold_dir)}")
            continue
        benchmark = str(payload.get("benchmark") or path.parent.name)
        chunks = payload.get("chunk_evidence")
        if not isinstance(chunks, list):
            continue
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            chunk_idx = chunk.get("chunk_index", "?")
            drift = chunk.get("_banned_drift") or []
            if drift:
                failures.append(
                    "role_quality_mapper_banned_drift:"
                    f"{_rel(path, fold_dir)}:chunk_{chunk_idx}:{','.join(map(str, drift))}"
                )
            reasons = _mapper_evidence_quality_reasons(
                {
                    "chunk_summary": chunk.get("summary", ""),
                    "task_patterns": chunk.get("task_patterns", []),
                    "reasoning_patterns": chunk.get("reasoning_patterns", []),
                    "justifications": chunk.get("justifications", []),
                },
                benchmark=benchmark,
            )
            failures.extend(
                f"role_quality_mapper:{_rel(path, fold_dir)}:chunk_{chunk_idx}:{reason}"
                for reason in reasons
            )

    vocab_paths = list(sorted(fold_dir.rglob("V.json")))
    for extra in (
        fold_dir / "final" / "vocab_star.json",
        fold_dir / "taxonomy_refinement" / "refinement_result.json",
        fold_dir / "no_seed_taxonomy" / "proposal.json",
    ):
        if extra.exists():
            vocab_paths.append(extra)
    seen_vocab_paths: set[Path] = set()
    for path in vocab_paths:
        if path in seen_vocab_paths:
            continue
        seen_vocab_paths.add(path)
        payload = _safe_load_json(path)
        vocab = _vocab_from_payload(payload)
        if not vocab:
            continue
        reasons = vocab_quality_reasons(vocab, benchmark_names)
        failures.extend(
            f"role_quality_vocab:{_rel(path, fold_dir)}:{reason}"
            for reason in reasons
        )

    for path in sorted((fold_dir / "mapreduce_reducer").rglob("*.json")):
        payload = _safe_load_json(path)
        if not isinstance(payload, dict):
            failures.append(f"role_quality_maker_unreadable:{_rel(path, fold_dir)}")
            continue
        if "ability_levels" not in payload and "ability_rationale" not in payload:
            continue
        vocab_ids = sorted(
            {
                *[str(k) for k in (payload.get("ability_levels") or {})],
                *[str(k) for k in (payload.get("ability_rationale") or {})],
            }
        )
        if not vocab_ids:
            continue
        benchmark = str(payload.get("benchmark") or path.parent.name)
        reasons = _maker_evidence_quality_reasons(
            payload,
            vocab_ids,
            benchmark=benchmark,
        )
        failures.extend(
            f"role_quality_maker:{_rel(path, fold_dir)}:{reason}"
            for reason in reasons
        )

    return failures


def _metric_mismatches(selection: dict, split_block: dict) -> list[str]:
    mismatches: list[str] = []
    for key in METRIC_KEYS:
        sel_v = _finite(selection.get(key))
        split_v = _finite(split_block.get(key))
        if sel_v is None and split_v is None:
            continue
        if sel_v is None or split_v is None:
            mismatches.append(f"{key}:finite_mismatch")
        elif abs(sel_v - split_v) > 1e-9:
            mismatches.append(f"{key}:{sel_v:.6g}!={split_v:.6g}")
    return mismatches


def _metrics_path_for_scope(iter_dir: Path, scope: str | None) -> Path:
    if scope in {"train", "dev", "test"}:
        scoped = iter_dir / f"metrics_{scope}.json"
        if scoped.exists():
            return scoped
    return iter_dir / "metrics.json"


def _iteration_rows(
    fold_dir: Path,
    selected_label: str,
    threshold: float,
    selection_scope: str | None,
) -> list[dict]:
    rows: list[dict] = []
    for iter_dir in sorted(p for p in fold_dir.iterdir() if p.is_dir() and p.name.startswith("iter_")):
        metrics_path = _metrics_path_for_scope(iter_dir, selection_scope)
        if not metrics_path.exists():
            continue
        metrics = _load_json(metrics_path)
        delta = _finite(metrics.get("delta_tag"))
        l_align = _finite(metrics.get("L_align"))
        gate = delta is not None and delta > threshold
        if iter_dir.name == selected_label:
            decision = "selected"
        elif not gate:
            decision = "rejected_delta"
        else:
            decision = "not_best"
        rows.append(
            {
                "label": iter_dir.name,
                "L_align": l_align,
                "rho_s": _finite(metrics.get("rho_align_spearman")),
                "delta_tag": delta,
                "n_pos": metrics.get("n_pos"),
                "n_neg": metrics.get("n_neg"),
                "gate": gate,
                "decision": decision,
            }
        )
    return rows


def audit(
    parent_dir: Path,
    *,
    require_strict_config: bool = False,
    require_significant: bool = False,
    require_fold_quality: bool = False,
    require_role_quality: bool = False,
    alpha: float = DEFAULT_ALPHA,
    min_rho_s: float = DEFAULT_MIN_RHO_S,
    min_fold_pairs: int = STRICT_MIN_SPLIT_VALID_PAIRS,
    min_fold_rho_s: float | None = DEFAULT_MIN_FOLD_RHO_S,
) -> int:
    config_path = parent_dir / "config.json"
    config: dict[str, Any] = {}
    agg: dict[str, Any] = {}
    threshold = -0.10
    selection_mode = "unknown"
    if config_path.exists():
        config = _load_json(config_path)
        exp = config.get("experiment", {})
        threshold = float(exp.get("delta_tag_threshold", threshold))
        selection_mode = str(exp.get("best_iter_selection", selection_mode))

    agg_path = parent_dir / "agg" / "permutation_test.json"
    if agg_path.exists():
        agg = _load_json(agg_path)
        pooled = agg.get("pooled", {})
        rho = pooled.get("rho_spearman", {})
        print(
            "[audit] pooled "
            f"n_pairs={pooled.get('n_pairs')} "
            f"rho_s={_fmt(rho.get('observed'))} "
            f"p_two={_fmt(rho.get('p_two_sided'))}"
        )
    print(f"[audit] selection_mode={selection_mode}, delta_tag_threshold={threshold:+.4f}")

    failures: list[str] = []
    if require_strict_config:
        if not config_path.exists():
            failures.append("strict_config_missing:config.json")
        else:
            failures.extend(strict_config_failures(config))
    if require_significant:
        if not agg_path.exists():
            failures.append("quality_missing:agg/permutation_test.json")
        else:
            failures.extend(
                pooled_quality_failures(
                    agg,
                    alpha=float(alpha),
                    min_rho_s=float(min_rho_s),
                )
            )
    fold_dirs = sorted(p for p in parent_dir.glob("fold*") if p.is_dir())
    if not fold_dirs:
        print(f"[audit] FAIL no fold dirs under {parent_dir}")
        return 2

    for fold_dir in fold_dirs:
        final_dir = fold_dir / "final"
        metrics_path = final_dir / "metrics_with_bootstrap.json"
        split_path = final_dir / "split_metrics.json"
        stop_path = final_dir / "stop_reason.json"
        fallback_path = final_dir / "llm_fallbacks.json"
        best_path = final_dir / "best_iter.txt"

        selected_label = best_path.read_text(encoding="utf-8").strip() if best_path.exists() else ""
        metrics = _load_json(metrics_path) if metrics_path.exists() else {}
        selection = metrics.get("selection") or metrics
        selection_scope = metrics.get("selection_scope")
        split_metrics = _load_json(split_path) if split_path.exists() else {}
        split_block = split_metrics.get(selection_scope, {}) if selection_scope else {}
        test_block = split_metrics.get("test", {}) if split_metrics else {}
        held_block = split_metrics.get("held_model_test", {}) if split_metrics else {}
        mismatches = _metric_mismatches(selection, split_block) if split_block else []
        if mismatches:
            failures.append(f"{fold_dir.name}:selection_split_mismatch")
        if require_fold_quality:
            failures.extend(
                f"{fold_dir.name}:{failure}"
                for failure in fold_quality_failures(
                    split_metrics,
                    selection_scope=selection_scope,
                    min_pairs=int(min_fold_pairs),
                    min_rho_s=min_fold_rho_s,
                )
            )
        if require_role_quality:
            failures.extend(
                f"{fold_dir.name}:{failure}"
                for failure in role_output_quality_failures(fold_dir)
            )

        stop = _load_json(stop_path) if stop_path.exists() else {}
        fallbacks = _load_json(fallback_path) if fallback_path.exists() else {"total": None}
        if fallbacks.get("total") not in (0, None):
            failures.append(f"{fold_dir.name}:llm_fallbacks={fallbacks.get('total')}")

        print(
            f"[audit] {fold_dir.name} best={selected_label or 'missing'} "
            f"scope={selection_scope or 'unknown'} "
            f"L={_fmt(selection.get('L_align'))} "
            f"rho_s={_fmt(selection.get('rho_align_spearman'))} "
            f"delta={_fmt(selection.get('delta_tag'))} "
            f"stop={stop.get('status', 'missing')} "
            f"fallbacks={fallbacks.get('total')}"
        )
        if test_block:
            print(
                "  test: "
                f"n={test_block.get('n_pairs')} "
                f"L={_fmt(test_block.get('L_align'))} "
                f"rho_s={_fmt(test_block.get('rho_align_spearman'))} "
                f"delta={_fmt(test_block.get('delta_tag'))}"
            )
        if held_block:
            held_suffix = (
                f" skipped={held_block.get('skipped')}"
                if held_block.get("skipped") else ""
            )
            print(
                "  held_model_test: "
                f"n={held_block.get('n_pairs')} "
                f"held_models={held_block.get('n_held_models')} "
                f"min_common={held_block.get('min_common')} "
                f"L={_fmt(held_block.get('L_align'))} "
                f"rho_s={_fmt(held_block.get('rho_align_spearman'))}"
                f"{held_suffix}"
            )
        if mismatches:
            print("  mismatches: " + ", ".join(mismatches))
        rows = _iteration_rows(fold_dir, selected_label, threshold, selection_scope)
        for row in rows:
            if row["decision"] == "selected" and not row["gate"]:
                failures.append(f"{fold_dir.name}:selected_candidate_failed_gate")
            print(
                "  "
                f"{row['label']:<25} "
                f"L={_fmt(row['L_align'])} "
                f"rho_s={_fmt(row['rho_s'])} "
                f"delta={_fmt(row['delta_tag'])} "
                f"bins={row['n_pos']}/{row['n_neg']} "
                f"gate={str(row['gate']).lower():<5} "
                f"{row['decision']}"
            )

    if failures:
        print("[audit] FAIL " + ", ".join(failures))
        return 2
    print("[audit] PASS artifact consistency checks")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_run_dir")
    parser.add_argument(
        "--require-strict-config",
        action="store_true",
        help=(
            "Fail relaxed smoke runs by requiring strict score-coverage gates "
            "from the saved parent config."
        ),
    )
    parser.add_argument(
        "--require-significant",
        action="store_true",
        help=(
            "Fail unless the pooled permutation result has positive enough "
            "Spearman rho and p_two <= alpha."
        ),
    )
    parser.add_argument(
        "--require-research-grade",
        action="store_true",
        help=(
            "Equivalent to --require-strict-config --require-significant "
            "--require-fold-quality --require-role-quality. Use this for "
            "final research-result acceptance."
        ),
    )
    parser.add_argument(
        "--require-fold-quality",
        action="store_true",
        help=(
            "Fail unless each fold has finite selection/test/held-model "
            "signals with enough comparable pairs."
        ),
    )
    parser.add_argument(
        "--require-role-quality",
        action="store_true",
        help=(
            "Fail if saved Mapper, Executer/taxonomy vocabulary, or Maker "
            "artifacts violate the role-output quality contracts."
        ),
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--min-rho-s", type=float, default=DEFAULT_MIN_RHO_S)
    parser.add_argument(
        "--min-fold-rho-s",
        type=float,
        default=DEFAULT_MIN_FOLD_RHO_S,
        help=(
            "Minimum rho_align_spearman required for each selection/test/"
            "held-model fold block when --require-fold-quality is active."
        ),
    )
    parser.add_argument(
        "--min-fold-pairs",
        type=int,
        default=STRICT_MIN_SPLIT_VALID_PAIRS,
        help="Minimum n_pairs required in each fold signal block.",
    )
    args = parser.parse_args()
    return audit(
        Path(args.parent_run_dir),
        require_strict_config=bool(
            args.require_strict_config or args.require_research_grade
        ),
        require_significant=bool(
            args.require_significant or args.require_research_grade
        ),
        require_fold_quality=bool(
            args.require_fold_quality or args.require_research_grade
        ),
        require_role_quality=bool(
            args.require_role_quality or args.require_research_grade
        ),
        alpha=float(args.alpha),
        min_rho_s=float(args.min_rho_s),
        min_fold_pairs=int(args.min_fold_pairs),
        min_fold_rho_s=float(args.min_fold_rho_s),
    )


if __name__ == "__main__":
    raise SystemExit(main())
