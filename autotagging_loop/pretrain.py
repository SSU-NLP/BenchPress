"""Pre-experiment (Part 1, 사전 실험) entrypoint.

Produces `final/I_star.txt` + `data/cognitive_abilities.json` (V_seed) used
to seed the main experiment's first iteration. The main experiment (Part 2)
is implemented separately in `autotagging_loop/main.py` + `autotagging_loop/
runner/` and reuses these seed artifacts; running `main.py` does not invoke
this file.
"""

from __future__ import annotations

import argparse
import json

from autotagging_loop.experiment.config import load_experiment_config
from autotagging_loop.experiment.loop import run_part1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BenchPress Part 1 — single global tag-score alignment loop")
    parser.add_argument("--max-iter", type=int, default=None, help="override config.max_iter")
    parser.add_argument("--no-baseline", action="store_true", help="skip I_0/random baselines")
    parser.add_argument("--wandb", action="store_true", help="enable wandb logging")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="optional JSON file to override DEFAULT_EXPERIMENT_CONFIG",
    )
    return parser.parse_args()


def load_overrides_from_file(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    overrides = load_overrides_from_file(args.config)
    if args.max_iter is not None:
        overrides["max_iter"] = args.max_iter
    if args.no_baseline:
        overrides["run_baseline"] = False
    if args.wandb:
        overrides["wandb"] = True

    config = load_experiment_config(overrides)

    wandb_run = None
    if config.get("wandb"):
        try:
            import wandb
            from datetime import datetime

            run_name = f"part1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            tags = ["part1"]
            if config.get("use_mapreduce_evidence"):
                tags.append("mapreduce")
            if config.get("optimize_tag_weights"):
                tags.append("calibrated")
            wandb_run = wandb.init(
                project="bench experiment",
                name=run_name,
                tags=tags,
                config=config,
            )
        except Exception as exc:
            print(f"  [main_experiment] wandb init failed: {exc}")

    history, best = run_part1(config, wandb_run=wandb_run)
    print()
    print(f"  [main_experiment] iterations: {len(history)}, best={best.label}")
    print(
        f"  [main_experiment] best L_align={best.L_align:.4f}, "
        f"rho_p={best.rho_align_pearson:.4f}, delta_tag={best.delta_tag:.4f}"
    )

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
