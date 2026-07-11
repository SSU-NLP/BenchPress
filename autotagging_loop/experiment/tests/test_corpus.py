"""Tests for experiment/corpus.py — alias dedupe, _meta drop, composite reject, min_models."""

from __future__ import annotations

import json

from autotagging_loop.experiment.corpus import load_corpus, load_label_documents


def _write_leaderboard(tmp_path, payload):
    path = tmp_path / "lb.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_alias_dedupe(tmp_path):
    payload = {
        "_meta": {"description": "x"},
        "ARC Challenge": {"m1": 0.9, "m2": 0.8, "m3": 0.7, "m4": 0.6, "m5": 0.5, "m6": 0.4},
        "ARC-Challenge": {"_alias": "ARC Challenge"},
        "ARC": {"_alias": "ARC Challenge"},
    }
    corpus = load_corpus(_write_leaderboard(tmp_path, payload), min_models_per_bench=6)
    assert "ARC Challenge" in corpus.benchmark_names
    assert "ARC-Challenge" not in corpus.benchmark_names
    assert "ARC" not in corpus.benchmark_names
    assert "_meta" not in corpus.benchmark_names


def test_min_models_drop(tmp_path):
    payload = {
        "Big": {f"m{i}": 0.5 for i in range(7)},
        "Small": {"m1": 0.3, "m2": 0.4},
    }
    corpus = load_corpus(_write_leaderboard(tmp_path, payload), min_models_per_bench=6)
    assert "Big" in corpus.benchmark_names
    assert "Small" not in corpus.benchmark_names
    assert "Small" in corpus.drop_log


def test_composite_reject(tmp_path):
    payload = {
        "Intelligence Index": {f"m{i}": 0.5 for i in range(7)},
        "Real Bench": {f"m{i}": 0.6 for i in range(7)},
    }
    corpus = load_corpus(_write_leaderboard(tmp_path, payload), min_models_per_bench=6)
    assert "Intelligence Index" not in corpus.benchmark_names
    assert "Intelligence Index" in corpus.drop_log
    assert corpus.drop_log["Intelligence Index"] == "composite"
    assert "Real Bench" in corpus.benchmark_names


def test_exclude_list(tmp_path):
    payload = {
        "MMLU": {f"m{i}": 0.5 for i in range(7)},
        "arena_hard": {f"m{i}": 0.6 for i in range(7)},
    }
    corpus = load_corpus(
        _write_leaderboard(tmp_path, payload),
        min_models_per_bench=6,
        exclude=["arena_hard"],
    )
    assert "MMLU" in corpus.benchmark_names
    assert "arena_hard" not in corpus.benchmark_names
    assert corpus.drop_log["arena_hard"] == "excluded"


def test_load_label_documents_as_benchmark_evidence(tmp_path):
    payload = {
        "ARC Challenge": {f"m{i}": 0.5 for i in range(7)},
        "Other Bench": {f"m{i}": 0.6 for i in range(7)},
    }
    labels_dir = tmp_path / "labels"
    arc_dir = labels_dir / "arc-challenge"
    arc_dir.mkdir(parents=True)
    rows = [
        {
            "item_id": "arc-challenge_00000",
            "benchmark": "ARC Challenge",
            "question": "Which event most likely causes shorter days?",
            "answer": "C",
            "choices": ["A", "B", "C", "D"],
            "gt_topic": "physics",
            "gt_reasoning_depth": "single_step",
            "gt_answer_format": "multiple_choice",
            "reviewer_status": "reviewed",
        },
        {
            "item_id": "arc-challenge_00001",
            "benchmark": "ARC Challenge",
            "question": "Which property is measured in grams?",
            "answer": "mass",
            "choices": ["mass", "volume"],
            "gt_topic": "measurement",
            "gt_reasoning_depth": "recall",
            "gt_answer_format": "multiple_choice",
            "reviewer_status": "reviewed",
        },
    ]
    tasks = arc_dir / "tasks.jsonl"
    tasks.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    corpus = load_corpus(
        _write_leaderboard(tmp_path, payload),
        min_models_per_bench=6,
        labels_dir=str(labels_dir),
        examples_per_benchmark=1,
    )

    assert "ARC Challenge" in corpus.documents
    doc = corpus.documents["ARC Challenge"]
    assert doc["reviewed_rows"] == 2
    assert doc["topic_counts"] == {"physics": 1, "measurement": 1}
    assert "Representative examples" in corpus.descriptions["ARC Challenge"]
    assert "gt_reasoning_depth distribution" in corpus.descriptions["ARC Challenge"]
    assert corpus.descriptions["Other Bench"] == ""


def test_load_label_documents_all_examples(tmp_path):
    labels_dir = tmp_path / "labels"
    bench_dir = labels_dir / "toybench"
    bench_dir.mkdir(parents=True)
    with open(bench_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        for idx in range(3):
            f.write(json.dumps({
                "benchmark": "ToyBench",
                "question": f"q{idx}",
                "answer": "a",
                "reviewer_status": "reviewed",
            }) + "\n")

    docs = load_label_documents(str(labels_dir), ["ToyBench"], examples_per_benchmark="all")

    assert docs["ToyBench"]["reviewed_rows"] == 3
    assert len(docs["ToyBench"]["examples"]) == 3


def test_load_corpus_caps_prompt_examples_while_storing_all(tmp_path):
    payload = {"ToyBench": {f"m{i}": 0.5 for i in range(7)}}
    labels_dir = tmp_path / "labels"
    bench_dir = labels_dir / "toybench"
    bench_dir.mkdir(parents=True)
    with open(bench_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        for idx in range(5):
            f.write(json.dumps({
                "benchmark": "ToyBench",
                "question": f"question {idx}",
                "answer": "a",
                "reviewer_status": "reviewed",
            }) + "\n")

    corpus = load_corpus(
        _write_leaderboard(tmp_path, payload),
        min_models_per_bench=6,
        labels_dir=str(labels_dir),
        examples_per_benchmark="all",
        prompt_examples_per_benchmark=2,
        max_prompt_chars_per_benchmark=1000,
    )

    doc = corpus.documents["ToyBench"]
    assert doc["reviewed_rows"] == 5
    assert len(doc["examples"]) == 5
    assert doc["prompt_example_count"] == 2
    assert "stored_examples: 5" in corpus.descriptions["ToyBench"]
    assert "prompt_examples: 2" in corpus.descriptions["ToyBench"]
    assert "[example 3]" not in corpus.descriptions["ToyBench"]
