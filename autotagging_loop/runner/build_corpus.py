"""Build Part 2 local label corpus from HuggingFace datasets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from autotagging_loop.runner.config import load_config
from autotagging_loop.runner.corpus import load_leaderboard_scores, load_score_sources
from autotagging_loop.runner.hf_sampling import load_dataset_map, name_key, fetch_rows, rows_to_tasks


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_sample_n(value: int | str | None, default: int) -> int | None:
    if isinstance(value, str) and value.strip().lower() == "all":
        return None
    return int(value or default)


def build_hf_corpus(config: dict, *, sample_n: int | str | None = None, out_dir: str | None = None) -> dict:
    n = parse_sample_n(sample_n, int(config.get("hf_sample_n", 100)))
    labels_dir = Path(out_dir or config["labels_dir"])
    dataset_map = load_dataset_map(config["hf_dataset_map_path"])
    leaderboard_Y = load_leaderboard_scores(config["leaderboard_path"], exclude=config.get("exclude", []))
    Y = load_score_sources(config)
    leaderboard_keys = {name_key(benchmark) for benchmark in leaderboard_Y}
    scored_keys = {name_key(benchmark) for benchmark in Y}
    full_scored_only = n is None and bool(config.get("hf_full_scored_only", True))
    specs = list(dataset_map.values())
    if full_scored_only:
        specs = [spec for spec in specs if name_key(spec.benchmark) in scored_keys]
    token = os.getenv("HF_TOKEN") or os.getenv("huggingface_token")

    manifest = {
        "sample_n": "all" if n is None else n,
        "labels_dir": str(labels_dir),
        "source": "hf_dataset_map",
        "hf_dataset_map_count": len(dataset_map),
        "hf_build_count": len(specs),
        "hf_full_scored_only": full_scored_only,
        "scored_benchmark_count": len(Y),
        "score_sources": ["leaderboard_scores", "aai_scores"],
        "benchmarks": {},
    }
    for spec in sorted(specs, key=lambda item: item.benchmark.lower()):
        benchmark = spec.benchmark
        rows = fetch_rows(spec, n, token=token)
        tasks = rows_to_tasks(benchmark, rows)
        slug = name_key(benchmark)
        path = labels_dir / slug / "tasks.jsonl"
        write_jsonl(path, tasks)
        manifest["benchmarks"][benchmark] = {
            "status": "ok" if tasks else "empty",
            "dataset_id": spec.dataset_id,
            "config": spec.config,
            "split": spec.split,
            "rows_fetched": len(rows),
            "tasks_written": len(tasks),
            "has_score": name_key(benchmark) in scored_keys,
            "in_leaderboard": name_key(benchmark) in leaderboard_keys,
            "path": str(path),
        }

    labels_dir.mkdir(parents=True, exist_ok=True)
    with open(labels_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Part 2 HF sample corpus")
    parser.add_argument("--config", default=None, help="optional JSON config path")
    parser.add_argument("--n", default=None, help="samples per benchmark, or 'all'")
    parser.add_argument("--out", default=None, help="output labels directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(config_path=args.config)
    manifest = build_hf_corpus(config, sample_n=args.n, out_dir=args.out)
    ok = sum(1 for item in manifest["benchmarks"].values() if item.get("status") == "ok")
    print(f"built HF corpus: ok={ok}, total={len(manifest['benchmarks'])}, dir={manifest['labels_dir']}")


if __name__ == "__main__":
    main()
