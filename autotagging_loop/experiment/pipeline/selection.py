"""Selection gates and taxonomy-adoption decisions for the v3 pipeline."""

from __future__ import annotations

import math


def passes_delta_tag_gate(curr: dict, *, threshold: float = 0.0) -> bool:
    """Return whether a candidate passes the strict delta-tag gate."""
    cd = curr.get("delta_tag", float("nan"))
    if cd is None or math.isnan(cd):
        return False
    delta = float(cd)
    threshold = float(threshold)
    if delta > threshold:
        return True
    if threshold < 0.0 and math.isclose(delta, threshold, abs_tol=1e-12):
        return True
    return False


def is_better(
    curr: dict,
    best: dict | None,
    *,
    selection_cfg: dict | None = None,
    delta_tag_threshold: float = 0.0,
) -> bool:
    """Argmin selection among candidates that pass the delta-tag gate."""
    if not passes_delta_tag_gate(curr, threshold=delta_tag_threshold):
        return False
    if not passes_tag_count_gate(curr, selection_cfg or {}):
        return False

    cfg = selection_cfg or {}
    mode = cfg.get("mode", "train_l_align")
    dev_mode = mode in ("dev_l_align", "dev_stability_l_align")
    if dev_mode:
        if finite_float(curr.get("dev_L_align")) is None:
            return False
        if best is not None and finite_float(best.get("dev_L_align")) is None:
            return True

    if dev_mode:
        L_key = cfg.get("objective_key") or "dev_L_align"
        rho_key = "dev_rho_pearson"
        floor = cfg.get("dev_rho_floor")
        if floor is not None:
            curr_dev_rho = finite_float(curr.get("dev_rho_spearman"))
            if curr_dev_rho is None or curr_dev_rho < float(floor):
                return False
            if best is not None:
                best_dev_rho = finite_float(best.get("dev_rho_spearman"))
                if best_dev_rho is None or best_dev_rho < float(floor):
                    return True
        drop_tolerance = cfg.get("dev_rho_drop_tolerance")
        if drop_tolerance is not None and best is not None:
            curr_dev_rho = finite_float(curr.get("dev_rho_spearman"))
            best_dev_rho = finite_float(best.get("dev_rho_spearman"))
            if curr_dev_rho is None:
                return False
            if (
                best_dev_rho is not None
                and curr_dev_rho < best_dev_rho - float(drop_tolerance)
            ):
                return False
        train_l_increase_tolerance = cfg.get("train_l_increase_tolerance")
        if train_l_increase_tolerance is not None and best is not None:
            curr_train_l = finite_float(curr.get("train_L_align"))
            best_train_l = finite_float(best.get("train_L_align"))
            if curr_train_l is None or best_train_l is None:
                return False
            if curr_train_l > best_train_l + float(train_l_increase_tolerance):
                return False
        train_rho_floor = cfg.get("train_rho_floor")
        if train_rho_floor is not None:
            curr_train_rho = finite_float(curr.get("train_rho_spearman"))
            if curr_train_rho is None or curr_train_rho < float(train_rho_floor):
                return False
        train_rho_drop_tolerance = cfg.get("train_rho_drop_tolerance")
        if train_rho_drop_tolerance is not None and best is not None:
            curr_train_rho = finite_float(curr.get("train_rho_spearman"))
            best_train_rho = finite_float(best.get("train_rho_spearman"))
            if curr_train_rho is None or best_train_rho is None:
                return False
            if curr_train_rho < best_train_rho - float(train_rho_drop_tolerance):
                return False
        probe_rho_floor = cfg.get("model_probe_dev_rho_floor")
        if probe_rho_floor is not None:
            curr_probe_rho = finite_float(curr.get("model_probe_dev_rho_spearman_min"))
            if curr_probe_rho is None or curr_probe_rho < float(probe_rho_floor):
                return False
        probe_rho_drop_tolerance = cfg.get("model_probe_dev_rho_drop_tolerance")
        if probe_rho_drop_tolerance is not None and best is not None:
            curr_probe_rho = finite_float(curr.get("model_probe_dev_rho_spearman_min"))
            best_probe_rho = finite_float(best.get("model_probe_dev_rho_spearman_min"))
            if curr_probe_rho is None or best_probe_rho is None:
                return False
            if curr_probe_rho < best_probe_rho - float(probe_rho_drop_tolerance):
                return False
        probe_l_increase_tolerance = cfg.get("model_probe_dev_l_increase_tolerance")
        if probe_l_increase_tolerance is not None and best is not None:
            curr_probe_l = probe_l_for_selection(curr)
            best_probe_l = probe_l_for_selection(best)
            if curr_probe_l is None or best_probe_l is None:
                return False
            if curr_probe_l > best_probe_l + float(probe_l_increase_tolerance):
                return False
    else:
        L_key = cfg.get("objective_key") or "L_align"
        rho_key = "rho_align_pearson"

    if mode == "dev_stability_l_align":
        curr_rho = finite_float(curr.get("stability_selection_rho_min"))
        curr_l = finite_float(curr.get("stability_selection_l_max"))
        if curr_rho is not None and curr_l is not None:
            if best is None:
                return True
            best_rho = finite_float(best.get("stability_selection_rho_min"))
            best_l = finite_float(best.get("stability_selection_l_max"))
            if best_rho is None or best_l is None:
                return True
            if curr_rho > best_rho + 1e-12:
                return True
            if curr_rho < best_rho - 1e-12:
                return False
            if curr_l < best_l - 1e-12:
                return True
            if curr_l > best_l + 1e-12:
                return False

    if best is None:
        return True

    curr_L = finite_float(curr.get(L_key))
    best_L = finite_float(best.get(L_key))
    if curr_L is None:
        return False
    if best_L is None:
        return True
    if curr_L < best_L - 1e-12:
        return True
    if curr_L > best_L + 1e-12:
        return False

    cb = finite_float(curr.get(rho_key))
    bb = finite_float(best.get(rho_key))
    if cb is not None and (bb is None or cb > bb):
        return True
    if bb is not None and (cb is None or bb > cb):
        return False

    cd = finite_float(curr.get("delta_tag"))
    bd = finite_float(best.get("delta_tag"))
    if cd is not None and (bd is None or cd > bd):
        return True
    return False


def passes_tag_count_gate(curr: dict, cfg: dict | None = None) -> bool:
    """Reject taxonomy candidates outside a configured tag-count range."""

    cfg = cfg or {}
    min_tags = cfg.get("tag_count_min")
    max_tags = cfg.get("tag_count_max")
    if min_tags is None and max_tags is None:
        return True

    count = finite_float(curr.get("tag_count"))
    if count is None:
        return False
    if min_tags is not None and count < float(min_tags):
        return False
    if max_tags is not None and count > float(max_tags):
        return False
    return True


def finite_ge(value: float | None, threshold: float | None) -> bool:
    if threshold is None or value is None:
        return False
    try:
        v = float(value)
        t = float(threshold)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v >= t


def finite_float(value: float | None) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def probe_l_for_selection(candidate: dict) -> float | None:
    """Prefer worst probe loss; old artifacts may only have the mean."""
    value = finite_float(candidate.get("model_probe_dev_L_align_max"))
    if value is not None:
        return value
    return finite_float(candidate.get("model_probe_dev_L_align_mean"))


def metric_delta(candidate_metrics: dict, fixed_metrics: dict, key: str) -> float | None:
    candidate = finite_float(candidate_metrics.get(key))
    fixed = finite_float(fixed_metrics.get(key))
    if candidate is None or fixed is None:
        return None
    return float(candidate - fixed)


def build_protected_pairs(
    S: dict[tuple[str, str], float],
    R_raw: dict[tuple[str, str], float],
    top_k: int,
    min_r: float,
) -> list[dict]:
    """High score-pattern-similarity pairs the taxonomy refiner should preserve."""
    rows: list[dict] = []
    for pair, r_value in R_raw.items():
        r = finite_float(r_value)
        if r is None or r < float(min_r):
            continue
        s = finite_float(S.get(pair))
        residual = abs(float(s) - r) if s is not None else None
        rows.append(
            {
                "benchmark_pair": [pair[0], pair[1]],
                "score_similarity": r,
                "tag_similarity": s,
                "residual_abs": residual,
                "post_part1_use": "protected_high_similarity_pair_for_taxonomy_refinement",
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["score_similarity"]),
            float(row["residual_abs"]) if row["residual_abs"] is not None else float("inf"),
            row["benchmark_pair"][0],
            row["benchmark_pair"][1],
        )
    )
    return rows[: max(0, int(top_k))]


def taxonomy_adoption_decision(
    fixed_metrics: dict,
    taxonomy_metrics: dict | None,
    config: dict,
) -> dict:
    thresholds = {
        "min_l_align_improvement": config.get("taxonomy_refinement_min_l_align_improvement", 0.0),
        "max_rho_pearson_drop": config.get("taxonomy_refinement_max_rho_pearson_drop", 0.02),
        "max_delta_tag_drop": config.get("taxonomy_refinement_max_delta_tag_drop", 0.05),
        "max_residual_increase": config.get("taxonomy_refinement_max_residual_increase", 0.0),
    }
    if not taxonomy_metrics:
        return {
            "adopted": False,
            "selected": "fixed",
            "reasons": ["taxonomy_not_completed"],
            "thresholds": thresholds,
            "fixed_metrics": fixed_metrics,
            "taxonomy_metrics": taxonomy_metrics,
            "deltas": {},
        }

    deltas = {
        "L_align": metric_delta(taxonomy_metrics, fixed_metrics, "L_align"),
        "rho_align_pearson": metric_delta(taxonomy_metrics, fixed_metrics, "rho_align_pearson"),
        "delta_tag": metric_delta(taxonomy_metrics, fixed_metrics, "delta_tag"),
        "residual_max": metric_delta(taxonomy_metrics, fixed_metrics, "residual_max"),
    }

    if not bool(config.get("taxonomy_refinement_adoption_enabled", True)):
        return {
            "adopted": True,
            "selected": "taxonomy_refinement",
            "reasons": ["adoption_gate_disabled"],
            "thresholds": thresholds,
            "fixed_metrics": fixed_metrics,
            "taxonomy_metrics": taxonomy_metrics,
            "deltas": deltas,
        }

    reasons: list[str] = []
    fixed_l = finite_float(fixed_metrics.get("L_align"))
    tax_l = finite_float(taxonomy_metrics.get("L_align"))
    min_l = float(thresholds["min_l_align_improvement"] or 0.0)
    if fixed_l is None or tax_l is None or not (tax_l < fixed_l - min_l):
        reasons.append("L_align_not_improved")

    fixed_rho = finite_float(fixed_metrics.get("rho_align_pearson"))
    tax_rho = finite_float(taxonomy_metrics.get("rho_align_pearson"))
    max_rho_drop = float(thresholds["max_rho_pearson_drop"] or 0.0)
    if fixed_rho is None or tax_rho is None or tax_rho < fixed_rho - max_rho_drop:
        reasons.append("rho_align_pearson_drop_exceeded")

    fixed_delta = finite_float(fixed_metrics.get("delta_tag"))
    tax_delta = finite_float(taxonomy_metrics.get("delta_tag"))
    max_delta_drop = float(thresholds["max_delta_tag_drop"] or 0.0)
    if fixed_delta is None or tax_delta is None or tax_delta < fixed_delta - max_delta_drop:
        reasons.append("delta_tag_drop_exceeded")

    fixed_residual = finite_float(fixed_metrics.get("residual_max"))
    tax_residual = finite_float(taxonomy_metrics.get("residual_max"))
    max_residual_increase = float(thresholds["max_residual_increase"] or 0.0)
    if (
        fixed_residual is None
        or tax_residual is None
        or tax_residual > fixed_residual + max_residual_increase
    ):
        reasons.append("residual_max_increase_exceeded")

    adopted = not reasons
    return {
        "adopted": adopted,
        "selected": "taxonomy_refinement" if adopted else "fixed",
        "reasons": reasons,
        "thresholds": thresholds,
        "fixed_metrics": fixed_metrics,
        "taxonomy_metrics": taxonomy_metrics,
        "deltas": deltas,
    }


def taxonomy_trigger_status(metrics: dict, config: dict) -> dict:
    enabled = bool(config.get("taxonomy_refinement_enabled", False))
    status = {
        "enabled": enabled,
        "triggered": False,
        "reasons": [],
        "metrics": {
            "L_align": metrics.get("L_align"),
            "residual_mean": metrics.get("residual_mean"),
            "residual_max": metrics.get("residual_max"),
            "n_pairs": metrics.get("n_pairs"),
        },
        "thresholds": {
            "min_pairs": config.get("taxonomy_refinement_min_pairs", 3),
            "residual_max": config.get("taxonomy_refinement_residual_max_threshold"),
            "residual_mean": config.get("taxonomy_refinement_residual_mean_threshold"),
            "L_align": config.get("taxonomy_refinement_l_align_threshold"),
        },
    }
    if not enabled:
        status["reasons"].append("disabled")
        return status

    n_pairs = int(metrics.get("n_pairs") or 0)
    min_pairs = int(config.get("taxonomy_refinement_min_pairs", 3))
    if n_pairs < min_pairs:
        status["reasons"].append(f"insufficient_pairs:{n_pairs}<{min_pairs}")
        return status

    if finite_ge(metrics.get("residual_max"), config.get("taxonomy_refinement_residual_max_threshold")):
        status["reasons"].append("residual_max_threshold_met")
    if finite_ge(metrics.get("residual_mean"), config.get("taxonomy_refinement_residual_mean_threshold")):
        status["reasons"].append("residual_mean_threshold_met")
    if finite_ge(metrics.get("L_align"), config.get("taxonomy_refinement_l_align_threshold")):
        status["reasons"].append("L_align_threshold_met")

    status["triggered"] = bool(status["reasons"])
    if not status["triggered"]:
        status["reasons"].append("thresholds_not_met")
    return status
