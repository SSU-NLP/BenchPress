from __future__ import annotations

import json

import autotagging_loop.runner.build_corpus as build_corpus
from autotagging_loop.runner.hf_sampling import DatasetSpec


def test_build_hf_corpus_fetches_all_hf_map_entries(tmp_path, monkeypatch):
    specs = {
        "bencha": DatasetSpec("BenchA", "org/a", "default", "test"),
        "benchb": DatasetSpec("BenchB", "org/b", "default", "test"),
    }
    config = {
        "hf_sample_n": 2,
        "labels_dir": str(tmp_path / "labels"),
        "hf_dataset_map_path": "unused-map.json",
        "leaderboard_path": "unused-lb.json",
        "exclude": [],
    }

    monkeypatch.setattr(build_corpus, "load_dataset_map", lambda _path: specs)
    monkeypatch.setattr(build_corpus, "load_leaderboard_scores", lambda *_args, **_kwargs: {
        "BenchA": {"m1": 1.0},
    })
    monkeypatch.setattr(build_corpus, "load_score_sources", lambda _config: {
        "BenchA": {"m1": 1.0},
        "BenchB": {"m2": 0.5},
    })
    monkeypatch.setattr(build_corpus, "fetch_rows", lambda spec, n, token=None: [
        {"question": f"{spec.benchmark} q{i}", "answer": "a"}
        for i in range(n)
    ])

    manifest = build_corpus.build_hf_corpus(config)

    assert set(manifest["benchmarks"]) == {"BenchA", "BenchB"}
    assert manifest["benchmarks"]["BenchA"]["in_leaderboard"] is True
    assert manifest["benchmarks"]["BenchA"]["has_score"] is True
    assert manifest["benchmarks"]["BenchB"]["in_leaderboard"] is False
    assert manifest["benchmarks"]["BenchB"]["has_score"] is True
    assert (tmp_path / "labels" / "bencha" / "tasks.jsonl").exists()
    with open(tmp_path / "labels" / "benchb" / "tasks.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 2
    assert rows[0]["benchmark"] == "BenchB"


def test_build_hf_corpus_all_passes_none_to_fetch_rows(tmp_path, monkeypatch):
    specs = {"bencha": DatasetSpec("BenchA", "org/a", "default", "test")}
    config = {
        "hf_sample_n": 2,
        "labels_dir": str(tmp_path / "labels"),
        "hf_dataset_map_path": "unused-map.json",
        "leaderboard_path": "unused-lb.json",
        "exclude": [],
    }
    seen = []

    monkeypatch.setattr(build_corpus, "load_dataset_map", lambda _path: specs)
    monkeypatch.setattr(build_corpus, "load_leaderboard_scores", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(build_corpus, "load_score_sources", lambda _config: {"BenchA": {"m1": 1.0}})

    def fake_fetch_rows(spec, n, token=None):
        seen.append(n)
        return [{"question": "q", "answer": "a"}]

    monkeypatch.setattr(build_corpus, "fetch_rows", fake_fetch_rows)

    manifest = build_corpus.build_hf_corpus(config, sample_n="all")

    assert seen == [None]
    assert manifest["sample_n"] == "all"
