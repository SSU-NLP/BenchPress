"""experiment/storage.py — JSON 저장 헬퍼 (results/experiment/run_<ts>/)."""

from __future__ import annotations

import json
import os
import time
from typing import Any


def _serialize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k) if not isinstance(k, str) else k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def _key_to_str(pair: tuple) -> str:
    return f"{pair[0]}||{pair[1]}"


def _stringify_pair_keys(d: dict) -> dict:
    out: dict = {}
    for k, v in d.items():
        if isinstance(k, tuple) and len(k) == 2:
            out[_key_to_str(k)] = _serialize(v)
        else:
            out[str(k)] = _serialize(v)
    return out


def make_run_dir(results_dir: str, run_id: str | None = None) -> str:
    rid = run_id or time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"run_{rid}")
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_serialize(data), f, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_iteration(
    run_dir: str,
    iter_index: int,
    prompt: str,
    tag_vectors: dict,
    similarity: dict,
    metrics: dict,
    error_report: list,
    improver_response: dict | None,
    label: str | None = None,
    residual_report: list | None = None,
    calibrated_tag_vectors: dict | None = None,
    calibrated_similarity: dict | None = None,
    calibrated_metrics: dict | None = None,
    tag_weight_metadata: dict | None = None,
) -> str:
    name = label if label else f"iter_{iter_index:03d}"
    iter_dir = os.path.join(run_dir, name)
    os.makedirs(iter_dir, exist_ok=True)
    write_text(os.path.join(iter_dir, "prompt.txt"), prompt)
    write_json(os.path.join(iter_dir, "tag_vectors.json"), tag_vectors)
    write_json(os.path.join(iter_dir, "similarity_matrix.json"), _stringify_pair_keys(similarity))
    write_json(os.path.join(iter_dir, "metrics.json"), metrics)
    write_json(os.path.join(iter_dir, "error_report.json"), error_report)
    if residual_report is not None:
        write_json(os.path.join(iter_dir, "residual_report.json"), residual_report)
    if calibrated_tag_vectors is not None:
        write_json(os.path.join(iter_dir, "T_calibrated.json"), calibrated_tag_vectors)
    if calibrated_similarity is not None:
        write_json(
            os.path.join(iter_dir, "similarity_matrix_calibrated.json"),
            _stringify_pair_keys(calibrated_similarity),
        )
    if calibrated_metrics is not None:
        write_json(os.path.join(iter_dir, "metrics_calibrated.json"), calibrated_metrics)
    if tag_weight_metadata is not None:
        write_json(os.path.join(iter_dir, "tag_weight_metadata.json"), tag_weight_metadata)
    if improver_response is not None:
        write_json(os.path.join(iter_dir, "improver_response.json"), improver_response)
    return iter_dir


def save_score_matrix(run_dir: str, payload: dict) -> None:
    serialized = dict(payload)
    if "R_raw" in serialized and isinstance(serialized["R_raw"], dict):
        serialized["R_raw"] = _stringify_pair_keys(serialized["R_raw"])
    if "R01" in serialized and isinstance(serialized["R01"], dict):
        serialized["R01"] = _stringify_pair_keys(serialized["R01"])
    if "common_count" in serialized and isinstance(serialized["common_count"], dict):
        serialized["common_count"] = _stringify_pair_keys(serialized["common_count"])
    write_json(os.path.join(run_dir, "score_matrix.json"), serialized)


def save_corpus(run_dir: str, payload: dict) -> None:
    write_json(os.path.join(run_dir, "corpus.json"), payload)


def save_config(run_dir: str, config: dict) -> None:
    write_json(os.path.join(run_dir, "config.json"), config)


def save_no_seed_taxonomy(run_dir: str, payload: dict) -> None:
    write_json(os.path.join(run_dir, "no_seed_taxonomy", "proposal.json"), payload)


def _write_final_artifacts(
    final_dir: str,
    *,
    best_iter_label: str,
    prompt: str,
    tag_vectors: dict,
    metrics_with_bootstrap: dict,
    vocab: list[dict] | None = None,
    vocab_metadata: dict | None = None,
    calibrated_tag_vectors: dict | None = None,
    calibrated_metrics_with_bootstrap: dict | None = None,
    residual_report: list | None = None,
    profile_support: dict | None = None,
    tag_weight_metadata: dict | None = None,
) -> None:
    """Phase L — single writer used by both fixed and taxonomy paths.

    Both paths emit the same core file set so downstream consumers can read
    `<base>/final/{best_iter.txt, I_star.txt, T_star.json, T_star_raw.json,
    metrics_with_bootstrap.json, metrics_raw.json}` regardless of which path
    produced them. Optional artifacts (vocab, calibration, residuals,
    profile_support, tag_weight_metadata) elide cleanly when not provided.
    """
    os.makedirs(final_dir, exist_ok=True)
    write_text(os.path.join(final_dir, "best_iter.txt"), best_iter_label)
    write_text(os.path.join(final_dir, "I_star.txt"), prompt)
    write_json(os.path.join(final_dir, "T_star.json"), tag_vectors)
    write_json(os.path.join(final_dir, "T_star_raw.json"), tag_vectors)
    write_json(os.path.join(final_dir, "selected_tag_vectors.json"), tag_vectors)
    write_json(os.path.join(final_dir, "metrics_with_bootstrap.json"), metrics_with_bootstrap)
    write_json(os.path.join(final_dir, "metrics_raw.json"), metrics_with_bootstrap)
    if vocab is not None:
        write_json(os.path.join(final_dir, "vocab_star.json"), vocab)
        write_json(os.path.join(final_dir, "selected_vocab.json"), vocab)
    if vocab_metadata is not None:
        write_json(os.path.join(final_dir, "vocab_star_metadata.json"), vocab_metadata)
    if calibrated_tag_vectors is not None:
        write_json(os.path.join(final_dir, "T_star_calibrated.json"), calibrated_tag_vectors)
    if calibrated_metrics_with_bootstrap is not None:
        write_json(os.path.join(final_dir, "metrics_calibrated.json"), calibrated_metrics_with_bootstrap)
    if residual_report is not None:
        write_json(os.path.join(final_dir, "residual_report.json"), residual_report)
    if profile_support is not None:
        write_json(os.path.join(final_dir, "profile_support.json"), profile_support)
    if tag_weight_metadata is not None:
        write_json(os.path.join(final_dir, "tag_weight_metadata.json"), tag_weight_metadata)


def save_final(
    run_dir: str,
    best_iter_label: str,
    prompt: str,
    tag_vectors: dict,
    metrics_with_bootstrap: dict,
    vocab: list[dict] | None = None,
    vocab_metadata: dict | None = None,
    calibrated_tag_vectors: dict | None = None,
    calibrated_metrics_with_bootstrap: dict | None = None,
    residual_report: list | None = None,
    tag_weight_metadata: dict | None = None,
) -> None:
    _write_final_artifacts(
        os.path.join(run_dir, "final"),
        best_iter_label=best_iter_label,
        prompt=prompt,
        tag_vectors=tag_vectors,
        metrics_with_bootstrap=metrics_with_bootstrap,
        vocab=vocab,
        vocab_metadata=vocab_metadata,
        calibrated_tag_vectors=calibrated_tag_vectors,
        calibrated_metrics_with_bootstrap=calibrated_metrics_with_bootstrap,
        residual_report=residual_report,
        tag_weight_metadata=tag_weight_metadata,
    )


def save_profile_support(run_dir: str, payload: dict) -> None:
    write_json(os.path.join(run_dir, "final", "profile_support.json"), payload)


def save_taxonomy_refinement(run_dir: str, payload: dict) -> None:
    write_json(os.path.join(run_dir, "taxonomy_refinement", "refinement_result.json"), payload)


def save_taxonomy_status(run_dir: str, payload: dict) -> None:
    write_json(os.path.join(run_dir, "taxonomy_refinement", "status.json"), payload)


def save_selection(run_dir: str, payload: dict) -> None:
    write_json(os.path.join(run_dir, "selection.json"), payload)


def save_taxonomy_final(
    run_dir: str,
    best_iter_label: str,
    prompt: str,
    vocab: list[dict],
    tag_vectors: dict,
    metrics_with_bootstrap: dict,
    residual_report: list | None = None,
    profile_support: dict | None = None,
    tag_weight_metadata: dict | None = None,
) -> None:
    _write_final_artifacts(
        os.path.join(run_dir, "taxonomy_refinement", "final"),
        best_iter_label=best_iter_label,
        prompt=prompt,
        tag_vectors=tag_vectors,
        metrics_with_bootstrap=metrics_with_bootstrap,
        vocab=vocab,
        residual_report=residual_report,
        profile_support=profile_support,
        tag_weight_metadata=tag_weight_metadata,
    )
