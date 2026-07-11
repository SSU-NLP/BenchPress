"""Offline tests for benchpress_hub.composition — HF API is faked throughout."""

from __future__ import annotations

import json
from typing import Any

import pytest

from benchpress_hub import composition as comp


class FakeInfo:
    def __init__(self, sha: str = "abc123", gated: Any = False, license_id: str | None = "mit"):
        self.sha = sha
        self.gated = gated
        self.card_data = {"license": license_id} if license_id else None


class FakeApi:
    def __init__(self, infos: dict[str, FakeInfo]):
        self.infos = infos
        self.repos: list[tuple[str, dict]] = []
        self.commits: list[dict] = []

    def dataset_info(self, repo_id: str) -> FakeInfo:
        if repo_id not in self.infos:
            raise RuntimeError(f"404: {repo_id}")
        return self.infos[repo_id]

    def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        self.repos.append((repo_id, kwargs))

    def create_commit(self, **kwargs: Any) -> None:
        self.commits.append(kwargs)


def _valid_manifest() -> dict[str, Any]:
    return {
        "type": comp.MANIFEST_TYPE,
        "schema_version": comp.SCHEMA_VERSION,
        "name": "mix",
        "abilities": [],
        "sources": [
            {
                "benchmark": "GSM8K",
                "repo_id": "openai/gsm8k",
                "config": "main",
                "split": "test",
                "revision": "sha1",
                "gated": False,
                "license": "mit",
                "n_samples": 2,
            },
            {
                "benchmark": "MMLU",
                "repo_id": "tinyBenchmarks/tinyMMLU",
                "config": "all",
                "split": "test",
                "revision": "sha2",
                "gated": False,
                "license": None,
                "n_samples": 3,
            },
        ],
        "combine": {"method": "concat", "seed": 7, "shuffle": True},
        "references": {"models": {}, "per_benchmark": {}},
    }


def test_validate_manifest_accepts_valid() -> None:
    assert comp.validate_manifest(_valid_manifest()) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update(type="other"),
        lambda m: m.update(schema_version=2),
        lambda m: m.update(sources=[]),
        lambda m: m["combine"].update(method="interleave"),
        lambda m: m["sources"][0].update(n_samples=0),
        lambda m: m["sources"][0].pop("revision"),
    ],
)
def test_validate_manifest_rejects(mutate) -> None:
    manifest = _valid_manifest()
    mutate(manifest)
    assert comp.validate_manifest(manifest)


def test_build_manifest_pins_revision_and_flags_gated() -> None:
    api = FakeApi(
        {
            "openai/gsm8k": FakeInfo(sha="sha-gsm"),
            "Idavidrein/gpqa": FakeInfo(sha="sha-gpqa", gated="manual", license_id=None),
        }
    )
    manifest = comp.build_manifest({"GSM8K": 100, "GPQA": 50}, name="mix", api=api)
    by_bench = {src["benchmark"]: src for src in manifest["sources"]}
    assert by_bench["GSM8K"]["revision"] == "sha-gsm"
    assert by_bench["GSM8K"]["gated"] is False
    assert by_bench["GSM8K"]["license"] == "mit"
    assert by_bench["GPQA"]["revision"] == "sha-gpqa"
    assert by_bench["GPQA"]["gated"] is True
    assert manifest["combine"] == {"method": "concat", "seed": 42, "shuffle": True}
    assert comp.validate_manifest(manifest) == []


def test_build_manifest_rejects_unknown_and_non_hf_benchmarks() -> None:
    api = FakeApi({})
    with pytest.raises(ValueError, match="unknown or non-HF"):
        comp.build_manifest({"NopeBench": 10}, name="mix", api=api)
    # Aider-Polyglot is in hf_dataset_map.json but has no HF repo id.
    with pytest.raises(ValueError, match="unknown or non-HF"):
        comp.build_manifest({"Aider-Polyglot": 10}, name="mix", api=api)


def test_build_references_weighted_mean(tmp_path) -> None:
    leaderboard = {
        "_source": "junk",
        "_score_eval_x": {"note": "junk"},
        "Bench A": {"GPT": 0.9, "Other": 0.5, "Broken": "n/a"},
        "Bench B": {"GPT": 0.5},
    }
    path = tmp_path / "lb.json"
    path.write_text(json.dumps(leaderboard), encoding="utf-8")
    sources = [
        {"benchmark": "Bench A", "n_samples": 100},
        {"benchmark": "Bench B", "n_samples": 300},
    ]
    refs = comp.build_references(sources, leaderboard_path=path)
    # Only GPT covers both sources: (0.9*100 + 0.5*300) / 400 = 0.6
    assert refs["models"] == {"GPT": 0.6}
    assert refs["per_benchmark"]["Bench A"]["Other"] == 0.5
    assert "Broken" not in refs["per_benchmark"]["Bench A"]


def test_normalize_row_uniform_schema() -> None:
    gsm8k = comp.normalize_row("GSM8K", 0, {"question": "1+1?", "answer": "2"})
    assert gsm8k == {
        "item_id": "gsm8k_00000",
        "benchmark": "GSM8K",
        "question": "1+1?",
        "answer": "2",
        "choices": [],
    }
    mmlu = comp.normalize_row("MMLU", 3, {"question": "Q?", "choices": ["a", "b"], "answer": 1})
    assert mmlu["item_id"] == "mmlu_00003"
    assert mmlu["choices"] == ["a", "b"]
    tqa = comp.normalize_row("TruthfulQA", 0, {"question": "Q?", "mc1_targets": {"choices": ["x", "y"]}})
    assert tqa["choices"] == ["x", "y"]
    # answerKey must win over the choice list (ARC-style rows).
    arc = comp.normalize_row("ARC Challenge", 0, {"question": "Q?", "choices": ["a", "b"], "answerKey": "B"})
    assert arc["answer"] == "B"
    assert arc["choices"] == ["a", "b"]


def _fake_rows(source: dict, **_: Any):
    for i in range(source["n_samples"]):
        yield {"question": f"{source['benchmark']}-q{i}", "answer": str(i)}


def test_compose_rows_deterministic(monkeypatch) -> None:
    monkeypatch.setattr(comp, "_iter_source_rows", _fake_rows)
    manifest = _valid_manifest()
    first = comp.compose_rows(manifest)
    second = comp.compose_rows(manifest)
    assert first == second
    assert len(first) == 5
    unshuffled_manifest = _valid_manifest()
    unshuffled_manifest["combine"]["shuffle"] = False
    unshuffled = comp.compose_rows(unshuffled_manifest)
    assert unshuffled != first  # seed-7 shuffle reorders
    assert sorted(r["item_id"] for r in unshuffled) == sorted(r["item_id"] for r in first)


def test_compose_rows_short_source_raises(monkeypatch) -> None:
    def short_rows(source: dict, **_: Any):
        yield {"question": "only-one", "answer": ""}

    monkeypatch.setattr(comp, "_iter_source_rows", short_rows)
    with pytest.raises(RuntimeError, match="not reproducible"):
        comp.compose_rows(_valid_manifest())


def test_load_composition_returns_dataset(monkeypatch) -> None:
    monkeypatch.setattr(comp, "load_manifest", lambda repo_id, **_: _valid_manifest())
    monkeypatch.setattr(comp, "_iter_source_rows", _fake_rows)
    ds = comp.load_composition("user/mix")
    assert len(ds) == 5
    assert set(ds.column_names) == {"item_id", "benchmark", "question", "answer", "choices"}
    raw = comp.load_composition("user/mix", normalize=False)
    assert set(raw) == {"GSM8K", "MMLU"}
    assert len(raw["MMLU"]) == 3


def test_push_composition_single_atomic_commit() -> None:
    api = FakeApi({})
    manifest = _valid_manifest()
    url = comp.push_composition("user/mix", manifest, api=api)
    assert url == "https://huggingface.co/datasets/user/mix"
    assert api.repos[0][0] == "user/mix"
    assert api.repos[0][1]["exist_ok"] is True
    assert len(api.commits) == 1
    operations = api.commits[0]["operations"]
    assert [op.path_in_repo for op in operations] == [comp.MANIFEST_FILENAME, "README.md"]
    round_trip = json.loads(operations[0].path_or_fileobj.decode("utf-8"))
    assert round_trip == manifest


def test_push_composition_rejects_invalid() -> None:
    manifest = _valid_manifest()
    manifest["combine"]["method"] = "interleave"
    with pytest.raises(ValueError, match="refusing to publish"):
        comp.push_composition("user/mix", manifest, api=FakeApi({}))


def test_render_readme_contents() -> None:
    manifest = _valid_manifest()
    manifest["abilities"] = ["quantitative_reasoning"]
    manifest["references"] = {"models": {"GPT-4o": 0.808}, "per_benchmark": {}}
    readme = comp.render_readme(manifest, "user/mix")
    assert readme.startswith("---\nviewer: false")
    assert "[openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)" in readme
    assert "| GPT-4o | 0.808 |" in readme
    assert 'load_composition("user/mix")' in readme
    assert "quantitative_reasoning" in readme
