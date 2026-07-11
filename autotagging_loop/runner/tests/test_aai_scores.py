from __future__ import annotations

import json

from autotagging_loop.runner.aai_scores import extract_aai_scores, read_aai_scores
from autotagging_loop.runner.corpus import (
    filter_score_matrix,
    load_curated_score_backfill,
    load_leaderboard_scores,
    load_score_sources,
    merge_score_sources,
)


def test_filter_score_matrix_applies_benchmark_and_model_allowlists():
    filtered = filter_score_matrix(
        {
            "MMLU-Pro": {"GPT 5": 0.9, "DeepSeek-v3": 0.8, "Other": 0.1},
            "GPQA": {"GPT-5": 0.7, "DeepSeek v3": 0.6},
            "Drop": {"GPT-5": 0.4, "DeepSeek-v3": 0.3},
        },
        include_benchmarks=["mmlu pro", "GPQA"],
        include_models=["GPT-5", "DeepSeek V3"],
    )

    assert filtered == {
        "MMLU-Pro": {"GPT-5": 0.9, "DeepSeek-v3": 0.8},
        "GPQA": {"GPT-5": 0.7, "DeepSeek-v3": 0.6},
    }


def test_extract_aai_scores_uses_only_matching_individual_evaluations():
    payload = {
        "data": [
            {
                "name": "Model A",
                "evaluations": {
                    "artificial_analysis_intelligence_index": 62.9,
                    "mmlu_pro": 0.791,
                    "math_500": 97.3,
                    "aime": 0.77,
                },
            },
            {
                "name": "Model B",
                "evaluations": {"mmlu_pro": 0.5, "gpqa": 0.25},
            },
        ]
    }

    scores = extract_aai_scores(payload, benchmarks=["MMLU-Pro", "MATH-500", "AIME 2025", "GPQA"])

    assert scores["MMLU-Pro"] == {"Model A": 0.791, "Model B": 0.5}
    assert scores["MATH-500"] == {"Model A": 0.973}
    assert scores["AIME 2025"] == {"Model A": 0.77}
    assert scores["GPQA"] == {"Model B": 0.25}
    assert "Artificial Analysis Intelligence Index" not in scores


def test_read_aai_scores_accepts_wrapped_cache(tmp_path):
    path = tmp_path / "aai_scores.json"
    path.write_text(json.dumps({"scores": {"HLE": {"Model A": 0.1}}}), encoding="utf-8")

    assert read_aai_scores(path) == {"HLE": {"Model A": 0.1}}


def test_merge_score_sources_preserves_primary_values():
    merged = merge_score_sources(
        {"MMLU-Pro": {"Model A": 0.7}},
        {"MMLU Pro": {"Model A": 0.8, "Model B": 0.6}},
    )

    assert merged == {"MMLU-Pro": {"Model A": 0.7, "Model B": 0.6}}


def test_merge_score_sources_canonicalizes_model_aliases_without_duplicates():
    merged = merge_score_sources(
        {"MMLU-Pro": {"GPT-5": 0.9, "DeepSeek-v3": 0.8}},
        {"MMLU Pro": {"GPT 5": 0.1, "DeepSeek v3": 0.2, "Qwen 2.5": 0.7}},
    )

    assert merged == {
        "MMLU-Pro": {
            "GPT-5": 0.9,
            "DeepSeek-v3": 0.8,
            "Qwen2.5-72B": 0.7,
        }
    }


def test_load_leaderboard_scores_preserves_first_canonical_alias_value(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(
        json.dumps({"Bench": {"GPT-5": 0.9, "GPT 5": 0.1}}),
        encoding="utf-8",
    )

    scores = load_leaderboard_scores(str(path))

    assert scores == {"Bench": {"GPT-5": 0.9}}


def test_load_curated_score_backfill_requires_cell_provenance(tmp_path):
    path = tmp_path / "curated.json"
    path.write_text(
        json.dumps({
            "scores": [{
                "benchmark": "MMLU-Pro",
                "model": "GPT 5",
                "score": 0.86,
                "metric": "accuracy",
                "scale": "0-1",
                "source": {
                    "title": "Official model report",
                    "url": "https://provider.ai/report",
                    "date": "2026-06-01",
                },
            }]
        }),
        encoding="utf-8",
    )

    scores = load_curated_score_backfill(str(path))

    assert scores == {"MMLU-Pro": {"GPT-5": 0.86}}


def test_load_curated_score_backfill_rejects_ambiguous_scale(tmp_path):
    path = tmp_path / "curated.json"
    path.write_text(
        json.dumps({
            "scores": [{
                "benchmark": "MMLU-Pro",
                "model": "GPT-5",
                "score": 86.0,
                "metric": "accuracy",
                "scale": "percent",
                "source": {
                    "title": "Official model report",
                    "url": "https://provider.ai/report",
                    "date": "2026-06-01",
                },
            }]
        }),
        encoding="utf-8",
    )

    try:
        load_curated_score_backfill(str(path))
    except ValueError as exc:
        assert "scale='0-1'" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_curated_score_backfill_rejects_placeholder_metric(tmp_path):
    path = tmp_path / "curated.json"
    path.write_text(
        json.dumps({
            "scores": [{
                "benchmark": "MMLU-Pro",
                "model": "GPT-5",
                "score": 0.86,
                "metric": "TODO exact benchmark metric",
                "scale": "0-1",
                "source": {
                    "title": "Official model report",
                    "url": "https://provider.ai/report",
                    "date": "2026-06-01",
                },
            }]
        }),
        encoding="utf-8",
    )

    try:
        load_curated_score_backfill(str(path))
    except ValueError as exc:
        assert "metric appears to be a placeholder" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_curated_score_backfill_rejects_non_public_source_url(tmp_path):
    path = tmp_path / "curated.json"
    path.write_text(
        json.dumps({
            "scores": [{
                "benchmark": "MMLU-Pro",
                "model": "GPT-5",
                "score": 0.86,
                "metric": "accuracy",
                "scale": "0-1",
                "source": {
                    "title": "Official model report",
                    "url": "http://localhost/report",
                    "date": "2026-06-01",
                },
            }]
        }),
        encoding="utf-8",
    )

    try:
        load_curated_score_backfill(str(path))
    except ValueError as exc:
        assert "local host" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_curated_score_backfill_rejects_bad_source_date(tmp_path):
    path = tmp_path / "curated.json"
    path.write_text(
        json.dumps({
            "scores": [{
                "benchmark": "MMLU-Pro",
                "model": "GPT-5",
                "score": 0.86,
                "metric": "accuracy",
                "scale": "0-1",
                "source": {
                    "title": "Official model report",
                    "url": "https://provider.ai/report",
                    "date": "June 1 2026",
                },
            }]
        }),
        encoding="utf-8",
    )

    try:
        load_curated_score_backfill(str(path))
    except ValueError as exc:
        assert "source.date must be ISO formatted" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_curated_score_backfill_rejects_duplicate_alias_cells(tmp_path):
    path = tmp_path / "curated.json"
    source = {
        "title": "Official model report",
        "url": "https://provider.ai/report",
        "date": "2026-06-01",
    }
    path.write_text(
        json.dumps({
            "scores": [
                {
                    "benchmark": "MMLU-Pro",
                    "model": "GPT-5",
                    "score": 0.86,
                    "metric": "accuracy",
                    "scale": "0-1",
                    "source": source,
                },
                {
                    "benchmark": "MMLU Pro",
                    "model": "GPT 5",
                    "score": 0.87,
                    "metric": "accuracy",
                    "scale": "0-1",
                    "source": source,
                },
            ]
        }),
        encoding="utf-8",
    )

    try:
        load_curated_score_backfill(str(path))
    except ValueError as exc:
        assert "duplicate curated score cell" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_score_sources_merges_curated_backfill_without_overriding_primary(tmp_path):
    leaderboard_path = tmp_path / "leaderboard.json"
    curated_path = tmp_path / "curated.json"
    leaderboard_path.write_text(
        json.dumps({"MMLU-Pro": {"GPT-5": 0.9}}),
        encoding="utf-8",
    )
    curated_path.write_text(
        json.dumps({
            "scores": [
                {
                    "benchmark": "MMLU Pro",
                    "model": "GPT 5",
                    "score": 0.1,
                    "metric": "accuracy",
                    "scale": "0-1",
                    "source": {
                        "title": "Official model report",
                        "url": "https://provider.ai/report",
                        "date": "2026-06-01",
                    },
                },
                {
                    "benchmark": "MMLU Pro",
                    "model": "DeepSeek v3",
                    "score": 0.8,
                    "metric": "accuracy",
                    "scale": "0-1",
                    "source": {
                        "title": "Official model report",
                        "url": "https://provider.ai/report",
                        "date": "2026-06-01",
                    },
                },
            ]
        }),
        encoding="utf-8",
    )

    scores = load_score_sources({
        "leaderboard_path": str(leaderboard_path),
        "curated_score_backfill_path": str(curated_path),
        "use_aai_scores": False,
        "use_curated_score_backfill": True,
    })

    assert scores == {"MMLU-Pro": {"GPT-5": 0.9, "DeepSeek-v3": 0.8}}
