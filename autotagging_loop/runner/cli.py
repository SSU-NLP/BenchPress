"""CLI for the Part 2 experiment package."""

from __future__ import annotations

import argparse

from autotagging_loop.runner.aai_scores import refresh_aai_scores
from autotagging_loop.runner.build_corpus import build_hf_corpus
from autotagging_loop.runner.config import load_config
from autotagging_loop.runner.hf_sampling import load_dataset_map
from autotagging_loop.runner.run import run_part2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BenchPress Part 2 experiment")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-corpus", help="fetch 100 HF samples per benchmark")
    build.add_argument("--config", default=None)
    build.add_argument("--n", default=None)
    build.add_argument("--out", default=None)

    run = sub.add_parser("run", help="run the Part 2 main experiment")
    run.add_argument("--config", default=None)
    run.add_argument("--labels-dir", default=None)
    run.add_argument("--bootstrap-B", type=int, default=None)
    run.add_argument("--resume-run-dir", default=None)
    run.add_argument("--wandb", action="store_true", help="enable W&B logging")
    run.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    run.add_argument(
        "--require-research-grade",
        action="store_true",
        help=(
            "Fail before model calls unless strict score/split preflight passes, "
            "and fail after the run unless agg/quality_gate.json is research-grade."
        ),
    )

    aai = sub.add_parser("refresh-aai-scores", help="fetch score columns from Artificial Analysis")
    aai.add_argument("--config", default=None)
    aai.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build-corpus":
        config = load_config(config_path=args.config)
        manifest = build_hf_corpus(config, sample_n=args.n, out_dir=args.out)
        ok = sum(1 for item in manifest["benchmarks"].values() if item.get("status") == "ok")
        print(f"built HF corpus: ok={ok}, total={len(manifest['benchmarks'])}, dir={manifest['labels_dir']}")
    elif args.command == "run":
        overrides = {}
        if args.labels_dir:
            overrides["labels_dir"] = args.labels_dir
        if args.bootstrap_B is not None:
            overrides["bootstrap_B"] = args.bootstrap_B
        if args.resume_run_dir:
            overrides["resume_run_dir"] = args.resume_run_dir
        if args.wandb:
            overrides["wandb"] = True
        if args.wandb_mode:
            overrides["wandb_mode"] = args.wandb_mode
        config = load_config(overrides, config_path=args.config)
        try:
            run_part2(config, require_research_grade=args.require_research_grade)
        except (RuntimeError, ValueError) as exc:
            if args.require_research_grade and str(exc).startswith("research-grade "):
                raise SystemExit(str(exc)) from exc
            raise
    elif args.command == "refresh-aai-scores":
        config = load_config(config_path=args.config)
        dataset_map = load_dataset_map(config["hf_dataset_map_path"])
        scores = refresh_aai_scores(
            args.out or config["aai_scores_path"],
            benchmarks=[spec.benchmark for spec in dataset_map.values()],
            api_url=config["aai_api_url"],
        )
        print(f"fetched AAI scores: benchmarks={len(scores)}, path={args.out or config['aai_scores_path']}")


if __name__ == "__main__":
    main()
