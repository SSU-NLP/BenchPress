"""Publish the 3 canonical example compositions to the service org (P0).

Fixed names, no ``demo-`` prefix, no random suffix — these are the protected
examples the Space links to. After all pushes, every repo is round-trip
verified with ``load_composition``; only then are ``space/examples.json`` and
``benchboard/data/examples.json`` written (atomically, same content), so
neither the Space nor the demo site ever advertises a broken example.

Usage:
    BENCHPRESS_ORG=<org> [HF_TOKEN=<token>] uv run python scripts/publish_examples.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from benchpress_hub import build_manifest, load_composition, push_composition, resolve_publisher
from benchpress_hub.publishing import scrub_secrets

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES_JSON_PATH = _REPO_ROOT / "space" / "examples.json"
BENCHBOARD_EXAMPLES_JSON_PATH = _REPO_ROOT / "benchboard" / "data" / "examples.json"

# Ability ids must exist in the space/tag_map.json vocab.
EXAMPLES: dict[str, dict[str, Any]] = {
    "math-reasoning-mix": {
        "selections": {"GSM8K": 200, "MATH-500": 150, "AIME 2024": 30},
        "abilities": ["quantitative_manipulation", "algorithmic_decomposition"],
    },
    "coding-mix": {
        "selections": {"HumanEval": 100, "MBPP": 150, "LiveCodeBench": 100},
        "abilities": ["algorithmic_decomposition", "rule_constrained_reasoning"],
    },
    # MMLU/HellaSwag map to tinyBenchmarks sources (~100 rows) — 100 is the safe max.
    "knowledge-commonsense-mix": {
        "selections": {"MMLU": 100, "ARC Challenge": 150, "HellaSwag": 100},
        "abilities": ["evidence_retrieval", "abstract_pattern_induction"],
    },
}


def _describe(selections: dict[str, int]) -> str:
    return " + ".join(f"{bench} {n}" for bench, n in selections.items())


def _snippet(repo_id: str) -> str:
    return (
        "from benchpress_hub import load_composition\n"
        f'c = load_composition("{repo_id}")\n'
        "print(len(c))"
    )


def _write_examples_json(entries: list[dict[str, str]]) -> None:
    payload = json.dumps(entries, ensure_ascii=False, indent=2) + "\n"
    for path in (EXAMPLES_JSON_PATH, BENCHBOARD_EXAMPLES_JSON_PATH):
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, path)


def main() -> int:
    org = os.environ.get("BENCHPRESS_ORG")
    if not org:
        print("BENCHPRESS_ORG 환경변수가 필요합니다 (예: BENCHPRESS_ORG=my-org).", file=sys.stderr)
        return 1
    token = os.environ.get("HF_TOKEN")
    published: list[tuple[str, str, dict[str, int]]] = []
    try:
        api, namespace = resolve_publisher(token=token, org=org)
        for name, spec in EXAMPLES.items():
            manifest = build_manifest(
                spec["selections"], name=name, abilities=spec["abilities"], api=api
            )
            repo_id = f"{namespace}/{name}"
            url = push_composition(repo_id, manifest, api=api)
            published.append((repo_id, url, spec["selections"]))
        for repo_id, _, selections in published:
            expected = sum(selections.values())
            ds = load_composition(repo_id, token=token)
            if len(ds) != expected:
                raise RuntimeError(
                    f"round-trip mismatch for '{repo_id}': {len(ds)} rows != {expected}"
                )
    except Exception as exc:
        print(scrub_secrets(f"게시 실패: {exc!r}", [token]), file=sys.stderr)
        return 1
    _write_examples_json(
        [
            {
                "title": repo_id.split("/", 1)[1],
                "repo_id": repo_id,
                "description": _describe(selections),
                "snippet": _snippet(repo_id),
            }
            for repo_id, _, selections in published
        ]
    )
    for repo_id, url, _ in published:
        print(f"published: {repo_id} -> {url}")
    print(f"examples.json written: {EXAMPLES_JSON_PATH} and {BENCHBOARD_EXAMPLES_JSON_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
