"""Dataset composition: publish recipe manifests to the HF Hub, load them lazily.

A composition is a fixed evaluation set: per-source sample counts, deterministic
concat, seeded shuffle. The HF repo stores only ``manifest.json`` + ``README.md``
(a reproducible recipe) — source data is streamed at load time.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from autotagging_loop.runner.hf_sampling import (
    A_FIELD_CANDIDATES,
    Q_FIELD_CANDIDATES,
    extract_choices,
    find_field,
    load_dataset_map,
    name_key,
)

MANIFEST_TYPE = "benchpress-composition"
SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_MAP_PATH = _REPO_ROOT / "data" / "hf_dataset_map.json"
LEADERBOARD_PATH = _REPO_ROOT / "data" / "leaderboard_scores.json"
ABILITIES_PATH = _REPO_ROOT / "data" / "cognitive_abilities.json"

_SOURCE_REQUIRED = ("benchmark", "repo_id", "split", "revision", "n_samples")

# The choice list is its own column; letting "choices"/"options" satisfy the
# answer field would store the options as the answer (e.g. ARC's answerKey).
_ANSWER_FIELDS = [c for c in A_FIELD_CANDIDATES if c not in ("choices", "options")]


def validate_manifest(manifest: Any) -> list[str]:
    """Return a list of schema errors; empty list means valid."""
    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]
    errors: list[str] = []
    if manifest.get("type") != MANIFEST_TYPE:
        errors.append(f"type must be '{MANIFEST_TYPE}'")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"unsupported schema_version {manifest.get('schema_version')!r} "
            f"(this loader supports {SCHEMA_VERSION})"
        )
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        errors.append("sources must be a non-empty list")
        sources = []
    for i, src in enumerate(sources):
        if not isinstance(src, dict):
            errors.append(f"sources[{i}] must be an object")
            continue
        missing = [key for key in _SOURCE_REQUIRED if not src.get(key)]
        if missing:
            errors.append(f"sources[{i}] missing {missing}")
        n_samples = src.get("n_samples")
        if not isinstance(n_samples, int) or n_samples <= 0:
            errors.append(f"sources[{i}].n_samples must be a positive int")
    if not isinstance(manifest.get("combine"), dict) or manifest["combine"].get("method") != "concat":
        errors.append("combine.method must be 'concat' (schema v1 is deterministic concat only)")
    return errors


def _card_license(card: Any) -> str | None:
    if card is None:
        return None
    value = card.get("license") if isinstance(card, dict) else getattr(card, "license", None)
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    return str(value) if value else None


def _resolve_repo_meta(api: Any, repo_id: str) -> tuple[str, bool, str | None]:
    """Pin the source to its current commit SHA; detect gating and license."""
    try:
        info = api.dataset_info(repo_id)
    except Exception as exc:
        raise RuntimeError(f"failed to resolve '{repo_id}' on the HF Hub: {exc}") from exc
    sha = getattr(info, "sha", None)
    if not sha:
        raise RuntimeError(f"no commit sha returned for '{repo_id}'")
    gated = bool(getattr(info, "gated", False))
    return str(sha), gated, _card_license(getattr(info, "card_data", None))


def build_references(
    sources: list[dict[str, Any]],
    *,
    leaderboard_path: str | Path = LEADERBOARD_PATH,
) -> dict[str, Any]:
    """Reference model scores: per-benchmark, plus n_samples-weighted composites.

    Composite scores are computed only for models with a score on every source
    benchmark — a partial average would misstate the expected score.
    """
    try:
        with open(leaderboard_path, encoding="utf-8") as fh:
            leaderboard = json.load(fh)
    except FileNotFoundError:
        leaderboard = {}
    by_key = {
        name_key(bench): scores
        for bench, scores in leaderboard.items()
        if not bench.startswith("_") and isinstance(scores, dict)
    }
    per_benchmark: dict[str, dict[str, float]] = {}
    for src in sources:
        scores = by_key.get(name_key(src["benchmark"]), {})
        numeric = {m: float(s) for m, s in scores.items() if isinstance(s, (int, float))}
        if numeric:
            per_benchmark[src["benchmark"]] = numeric
    total = sum(src["n_samples"] for src in sources)
    covered = [set(per_benchmark.get(src["benchmark"], {})) for src in sources]
    full_coverage = set.intersection(*covered) if covered and all(covered) else set()
    models = {
        model: round(
            sum(per_benchmark[src["benchmark"]][model] * src["n_samples"] for src in sources) / total,
            4,
        )
        for model in sorted(full_coverage)
    }
    return {
        "models": models,
        "per_benchmark": per_benchmark,
        "note": "models = n_samples-weighted mean; only models scored on every source are included.",
    }


def build_manifest(
    selections: dict[str, int],
    *,
    name: str,
    abilities: list[str] | None = None,
    seed: int = 42,
    shuffle: bool = True,
    api: Any | None = None,
    token: str | None = None,
    dataset_map_path: str | Path = DATASET_MAP_PATH,
    leaderboard_path: str | Path = LEADERBOARD_PATH,
) -> dict[str, Any]:
    """Build a schema-v1 manifest from ``{benchmark display name: n_samples}``."""
    if not selections:
        raise ValueError("selections is empty")
    dataset_map = load_dataset_map(dataset_map_path)
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    sources: list[dict[str, Any]] = []
    for bench, n_samples in selections.items():
        spec = dataset_map.get(name_key(bench))
        if spec is None:
            known = ", ".join(sorted(s.benchmark for s in dataset_map.values()))
            raise ValueError(f"unknown or non-HF benchmark '{bench}' (available: {known})")
        if int(n_samples) <= 0:
            raise ValueError(f"n_samples for '{bench}' must be positive")
        revision, gated, license_id = _resolve_repo_meta(api, spec.dataset_id)
        sources.append(
            {
                "benchmark": spec.benchmark,
                "repo_id": spec.dataset_id,
                "config": spec.config,
                "split": spec.split or "test",
                "revision": revision,
                "gated": gated,
                "license": license_id,
                "n_samples": int(n_samples),
            }
        )
    manifest = {
        "type": MANIFEST_TYPE,
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "abilities": list(abilities or []),
        "sources": sources,
        "combine": {"method": "concat", "seed": seed, "shuffle": bool(shuffle)},
        "references": build_references(sources, leaderboard_path=leaderboard_path),
    }
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError(f"built an invalid manifest (bug): {'; '.join(errors)}")
    return manifest


def render_readme(manifest: dict[str, Any], repo_id: str) -> str:
    """Render the dataset card for a composition repo."""
    lines = [
        "---",
        "viewer: false",
        "tags:",
        "- benchpress",
        "- dataset-composition",
        "---",
        "",
        f"# {manifest['name']}",
        "",
        "Fixed evaluation-set composition built with BenchPress. This repo stores",
        "only the recipe (`manifest.json`) — source data is streamed at load time.",
        "",
    ]
    if manifest.get("abilities"):
        lines += ["**Abilities:** " + ", ".join(manifest["abilities"]), ""]
    lines += [
        "## Sources",
        "",
        "| benchmark | source repo | config | split | n_samples | license | gated |",
        "|---|---|---|---|---|---|---|",
    ]
    for src in manifest["sources"]:
        lines.append(
            f"| {src['benchmark']} "
            f"| [{src['repo_id']}](https://huggingface.co/datasets/{src['repo_id']}) "
            f"| {src.get('config') or '-'} | {src['split']} | {src['n_samples']} "
            f"| {src.get('license') or 'unknown'} | {'yes' if src.get('gated') else 'no'} |"
        )
    models = manifest.get("references", {}).get("models", {})
    if models:
        lines += [
            "",
            "## Reference scores (expected on this composition)",
            "",
            "| model | expected score |",
            "|---|---|",
        ]
        lines += [f"| {model} | {score} |" for model, score in models.items()]
    lines += [
        "",
        "## Usage",
        "",
        "```python",
        "from benchpress_hub import load_composition",
        "",
        f'ds = load_composition("{repo_id}")',
        "```",
        "",
        "Sources pinned to commit SHAs; loading is deterministic (fixed items, seeded order).",
        "",
    ]
    return "\n".join(lines)


def push_composition(
    repo_id: str,
    manifest: dict[str, Any],
    *,
    token: str | None = None,
    private: bool = False,
    api: Any | None = None,
) -> str:
    """Create/update the composition repo atomically; return its URL."""
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError(f"refusing to publish invalid manifest: {'; '.join(errors)}")
    from huggingface_hub import CommitOperationAdd

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    readme_bytes = render_readme(manifest, repo_id).encode("utf-8")
    try:
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(path_in_repo=MANIFEST_FILENAME, path_or_fileobj=manifest_bytes),
                CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme_bytes),
            ],
            commit_message=f"Publish composition '{manifest['name']}'",
        )
    except Exception as exc:
        raise RuntimeError(f"failed to publish composition to '{repo_id}': {exc}") from exc
    return f"https://huggingface.co/datasets/{repo_id}"


def load_manifest(
    repo_id: str,
    *,
    token: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Download and validate only the manifest — no source data is touched."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=MANIFEST_FILENAME,
            token=token,
            revision=revision,
        )
    except Exception as exc:
        raise RuntimeError(f"failed to fetch {MANIFEST_FILENAME} from '{repo_id}': {exc}") from exc
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError(f"invalid manifest in '{repo_id}': {'; '.join(errors)}")
    return manifest


def _iter_source_rows(
    source: dict[str, Any],
    *,
    streaming: bool,
    token: str | None,
) -> Iterator[dict[str, Any]]:
    """Yield up to n_samples raw rows from one pinned source."""
    from datasets import load_dataset

    try:
        ds = load_dataset(
            source["repo_id"],
            source.get("config"),
            split=source["split"],
            revision=source["revision"],
            streaming=streaming,
            token=token,
        )
    except Exception as exc:
        hint = ""
        if source.get("gated"):
            hint = (
                " This dataset is gated: accept its terms at "
                f"https://huggingface.co/datasets/{source['repo_id']} and pass a token with access."
            )
        raise RuntimeError(
            f"failed to load source '{source['benchmark']}' ({source['repo_id']}): {exc}.{hint}"
        ) from exc
    n_samples = source["n_samples"]
    rows = ds.take(n_samples) if streaming else ds.select(range(min(n_samples, len(ds))))
    for row in rows:
        yield dict(row)


def normalize_row(benchmark: str, index: int, row: dict[str, Any]) -> dict[str, Any]:
    """Map a heterogeneous source row onto the uniform composition schema."""
    question = find_field(row, Q_FIELD_CANDIDATES)
    if not question:
        question = "[raw] " + json.dumps(row, ensure_ascii=False, default=str)[:800]
    return {
        "item_id": f"{name_key(benchmark)}_{index:05d}",
        "benchmark": benchmark,
        "question": question,
        "answer": find_field(row, _ANSWER_FIELDS) or "",
        "choices": extract_choices(row),
    }


def compose_rows(
    manifest: dict[str, Any],
    *,
    streaming: bool = True,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Materialize the fixed item list a manifest describes, in seeded order."""
    rows: list[dict[str, Any]] = []
    for source in manifest["sources"]:
        count = 0
        for row in _iter_source_rows(source, streaming=streaming, token=token):
            rows.append(normalize_row(source["benchmark"], count, row))
            count += 1
        if count < source["n_samples"]:
            raise RuntimeError(
                f"source '{source['benchmark']}' yielded {count} rows "
                f"< n_samples={source['n_samples']}; the composition is not reproducible as published"
            )
    combine = manifest.get("combine", {})
    if combine.get("shuffle", True):
        random.Random(combine.get("seed", 42)).shuffle(rows)
    return rows


def load_composition(
    repo_id: str,
    *,
    token: str | None = None,
    streaming: bool = True,
    normalize: bool = True,
    revision: str | None = None,
) -> Any:
    """Load a published composition as a fixed, uniform-schema eval set.

    Only ``n_samples`` rows per source are pulled (streamed by default); the
    result is materialized as a ``datasets.Dataset`` — eval sets are small by
    design. With ``normalize=False``, returns ``{benchmark: [raw rows]}`` and
    performs no concat (source schemas are heterogeneous).
    """
    manifest = load_manifest(repo_id, token=token, revision=revision)
    if not normalize:
        return {
            src["benchmark"]: list(_iter_source_rows(src, streaming=streaming, token=token))
            for src in manifest["sources"]
        }
    rows = compose_rows(manifest, streaming=streaming, token=token)
    from datasets import Dataset, Features, Sequence, Value

    features = Features(
        {
            "item_id": Value("string"),
            "benchmark": Value("string"),
            "question": Value("string"),
            "answer": Value("string"),
            "choices": Sequence(Value("string")),
        }
    )
    return Dataset.from_list(rows, features=features)
