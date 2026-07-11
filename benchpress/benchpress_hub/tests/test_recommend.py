"""Offline tests for benchpress_hub.recommend and scripts/export_tag_map.py."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from benchpress_hub import rank_models, relevance_ranking

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _load_export_module():
    path = _REPO_ROOT / "autotagging_loop" / "scripts" / "export_tag_map.py"
    spec = importlib.util.spec_from_file_location("export_tag_map", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export_tag_map = _load_export_module()

TAG_SCORES = {
    "Bench A": {"t1": 1.0, "t2": 0.0},
    "Bench B": {"t1": 0.5, "t2": 0.5},
}


def test_relevance_ranking_cosine() -> None:
    ranking = relevance_ranking(TAG_SCORES, ["t1"])
    assert [bench for bench, _ in ranking] == ["Bench A", "Bench B"]
    rel = dict(ranking)
    assert rel["Bench A"] == pytest.approx(1.0)
    assert rel["Bench B"] == pytest.approx(1.0 / math.sqrt(2.0))


def test_relevance_ranking_list_equals_unit_weight_dict() -> None:
    assert relevance_ranking(TAG_SCORES, ["t1", "t2"]) == relevance_ranking(
        TAG_SCORES, {"t1": 1.0, "t2": 1.0}
    )


def test_relevance_ranking_unknown_only_target_raises() -> None:
    with pytest.raises(ValueError, match="no target tag"):
        relevance_ranking(TAG_SCORES, ["nope"])


def test_relevance_ranking_empty_target_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        relevance_ranking(TAG_SCORES, [])
    with pytest.raises(ValueError, match="empty"):
        relevance_ranking(TAG_SCORES, {})


def test_relevance_ranking_zero_vector_scores_zero_and_sorts_last() -> None:
    scores = {**TAG_SCORES, "Bench Z": {"t1": 0.0, "t2": 0.0}}
    ranking = relevance_ranking(scores, ["t1"])
    assert ranking[-1] == ("Bench Z", 0.0)


def test_relevance_ranking_tie_breaks_by_name() -> None:
    # Parallel vectors → identical cosine; order must fall back to the name.
    scores = {"B bench": {"t1": 2.0}, "A bench": {"t1": 1.0}}
    ranking = relevance_ranking(scores, ["t1"])
    assert [bench for bench, _ in ranking] == ["A bench", "B bench"]


def test_rank_models_normalizes_and_requires_full_coverage() -> None:
    leaderboard = {
        "_source": "junk",
        "_junk_scores": {"M9": 1.0},
        "Bench A": {"M1": 0.9, "M2": 0.1, "M3": 0.5, "Broken": "n/a"},
        "Bench B": {"M1": 0.8, "M2": 0.2},
    }
    ranked = rank_models(leaderboard, ["_junk_scores", "Bench A", "Bench B"])
    # M3 lacks Bench B; "_junk_scores" is ignored so M9 never counts.
    assert ranked == [("M1", 1.0), ("M2", 0.0)]


def test_rank_models_constant_bench_gives_half() -> None:
    leaderboard = {"Bench A": {"M1": 0.7, "M2": 0.7}}
    assert rank_models(leaderboard, ["Bench A"]) == [("M1", 0.5), ("M2", 0.5)]


def test_rank_models_no_overlap_returns_empty() -> None:
    assert rank_models({"Bench A": {"M1": 1.0}}, ["Nope"]) == []
    assert rank_models({}, ["Bench A"]) == []


def _write_fold(tmp_path: Path, tag_scores: dict, vocab: list) -> Path:
    final = tmp_path / "final"
    final.mkdir()
    (final / "T_star.json").write_text(json.dumps(tag_scores), encoding="utf-8")
    (final / "vocab_star.json").write_text(json.dumps(vocab), encoding="utf-8")
    return tmp_path


def test_build_tag_map_round_trip(tmp_path: Path) -> None:
    vocab = [{"id": "t1", "name": "T1", "definition": "d"}]
    scores = {"Bench A": {"t1": 0.5}}
    fold = _write_fold(tmp_path, scores, vocab)
    tag_map = export_tag_map.build_tag_map(fold)
    assert tag_map["meta"]["n_benchmarks"] == 1
    assert tag_map["meta"]["n_tags"] == 1
    assert tag_map["vocab"] == vocab
    assert tag_map["tag_scores"] == scores


def test_build_tag_map_rejects_tag_missing_from_vocab(tmp_path: Path) -> None:
    vocab = [{"id": "t1", "name": "T1", "definition": "d"}]
    scores = {"Bench A": {"t1": 0.5, "t_missing": 0.2}}
    fold = _write_fold(tmp_path, scores, vocab)
    with pytest.raises(ValueError, match="t_missing"):
        export_tag_map.build_tag_map(fold)
