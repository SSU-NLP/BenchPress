"""Offline tests for scripts/export_model_profiles.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _load_export_module():
    path = _REPO_ROOT / "autotagging_loop" / "scripts" / "export_model_profiles.py"
    spec = importlib.util.spec_from_file_location("export_model_profiles", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export_model_profiles = _load_export_module()

VOCAB = [
    {"id": "t1", "name": "T1", "definition": "d1"},
    {"id": "t2", "name": "T2", "definition": "d2"},
]
T_STAR = {
    "B1": {"t1": 1.0, "t2": 0.0},
    "B2": {"t1": 0.5, "t2": 1.0},
}
LEADERBOARD = {
    "_junk": {"X": 1.0},
    "B1": {"M1": 0.9, "M2": 0.5, "M3": 0.1, "Broken": "n/a"},
    "B2": {"M1": 0.8, "M2": 0.2},  # M3 has no B2 score -> coverage case
}
SHORTLIST = {
    "rankings": [
        {"model_name": "M1", "model_id": "m-one", "vendor": "VendorA"},
        {"model_name": "M2", "model_id": "m-two", "vendor": "VendorB"},
        # M3 intentionally absent -> exercises the slug fallback.
    ]
}


def _write_fold(tmp_path: Path) -> Path:
    final = tmp_path / "final"
    final.mkdir()
    (final / "T_star.json").write_text(json.dumps(T_STAR), encoding="utf-8")
    (final / "vocab_star.json").write_text(json.dumps(VOCAB), encoding="utf-8")
    return tmp_path


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _build(tmp_path: Path) -> dict:
    fold = _write_fold(tmp_path)
    scores_path = _write_json(tmp_path / "leaderboard.json", LEADERBOARD)
    shortlist_path = _write_json(tmp_path / "shortlist.json", SHORTLIST)
    return export_model_profiles.build_model_profiles(fold, scores_path, shortlist_path)


def test_tags_follow_vocab_order(tmp_path: Path) -> None:
    payload = _build(tmp_path)
    assert payload["tags"] == ["t1", "t2"]


def test_junk_key_and_broken_score_excluded(tmp_path: Path) -> None:
    payload = _build(tmp_path)
    names = {m["name"] for m in payload["models"]}
    assert "X" not in names
    assert "Broken" not in names
    assert names == {"M1", "M2", "M3"}


def test_models_sorted_by_name(tmp_path: Path) -> None:
    payload = _build(tmp_path)
    assert [m["name"] for m in payload["models"]] == ["M1", "M2", "M3"]


def test_id_mapping_and_slug_fallback(tmp_path: Path) -> None:
    payload = _build(tmp_path)
    by_name = {m["name"]: m for m in payload["models"]}
    assert by_name["M1"]["id"] == "m-one"
    assert by_name["M1"]["vendor"] == "VendorA"
    assert by_name["M2"]["id"] == "m-two"
    assert by_name["M2"]["vendor"] == "VendorB"
    # M3 is missing from the shortlist -> slug fallback, empty vendor.
    assert by_name["M3"]["id"] == "m3"
    assert by_name["M3"]["vendor"] == ""


def test_coverage_normalized_profile_values(tmp_path: Path) -> None:
    payload = _build(tmp_path)
    by_name = {m["name"]: m["profile"] for m in payload["models"]}

    # M1: full coverage on both benchmarks -> top percentile everywhere.
    assert by_name["M1"]["t1"] == pytest.approx(1.0)
    assert by_name["M1"]["t2"] == pytest.approx(1.0)

    # M2: mid/bottom percentile on both benchmarks.
    assert by_name["M2"]["t1"] == pytest.approx(1.0 / 3.0, abs=1e-4)
    assert by_name["M2"]["t2"] == pytest.approx(0.0)

    # M3: no B2 score -> its t1 denominator only includes B1's weight (1.0),
    # and its t2 denominator is 0 (only B2 carries t2 weight) -> 0.0.
    assert by_name["M3"]["t1"] == pytest.approx(0.0)
    assert by_name["M3"]["t2"] == pytest.approx(0.0)


def test_all_profile_values_in_unit_range(tmp_path: Path) -> None:
    payload = _build(tmp_path)
    for model in payload["models"]:
        for value in model["profile"].values():
            assert 0.0 <= value <= 1.0


def test_meta_counts(tmp_path: Path) -> None:
    payload = _build(tmp_path)
    assert payload["meta"]["n_models"] == 3
    assert payload["meta"]["n_tags"] == 2
    assert payload["meta"]["mode"] == "percentile-coverage-normalized"
