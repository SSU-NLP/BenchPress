"""Export a model x tag capability profile to benchboard/data/model_tag_profiles.json.

Combines a CV fold's tag data (T_star.json + vocab_star.json) with raw
leaderboard scores (data/leaderboard_scores.json) to produce, per model, a
coverage-normalized [0, 1] profile over the run's learned tag vocabulary.

Usage:
    uv run python scripts/export_model_profiles.py --run results/part2_experiment/<run>/fold2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from autotagging_loop.experiment.profiling import y_to_percentile
from autotagging_loop.runner.hf_sampling import name_key

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RUN = _REPO_ROOT / "results" / "part2_experiment" / "run_cv_20260708_183338" / "fold2"
DEFAULT_SCORES = _REPO_ROOT / "data" / "leaderboard_scores.json"
DEFAULT_SHORTLIST = _REPO_ROOT / "benchboard" / "data" / "model_shortlist.json"
DEFAULT_OUT = _REPO_ROOT / "benchboard" / "data" / "model_tag_profiles.json"


def _slug(name: str) -> str:
    return "-".join(name.lower().split())


def _load_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read {path}: {exc}") from exc


def _load_run(run_dir: Path) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Read T_star.json and vocab_star.json from a fold dir (or its final/)."""
    if (run_dir / "final").is_dir():
        run_dir = run_dir / "final"
    t_star = _load_json(run_dir / "T_star.json")
    vocab = _load_json(run_dir / "vocab_star.json")
    if not isinstance(vocab, list) or not vocab:
        raise ValueError("vocab_star.json must be a non-empty list")
    for entry in vocab:
        if not isinstance(entry, dict) or not all(entry.get(k) for k in ("id", "name", "definition")):
            raise ValueError(f"vocab entry missing id/name/definition: {entry!r}")
    if not isinstance(t_star, dict) or not t_star:
        raise ValueError("T_star.json must be a non-empty object")
    return t_star, vocab


def _load_leaderboard(scores_path: Path) -> dict[str, dict[str, float]]:
    """Read leaderboard scores, skipping '_'-prefixed keys and non-numeric entries."""
    raw = _load_json(scores_path)
    if not isinstance(raw, dict):
        raise ValueError("leaderboard scores must be a JSON object")
    out: dict[str, dict[str, float]] = {}
    for bench, models in raw.items():
        if bench.startswith("_") or not isinstance(models, dict):
            continue
        row: dict[str, float] = {}
        for model, score in models.items():
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                row[model] = float(score)
        out[bench] = row
    return out


def _match_benchmarks(t_star_benches: list[str], leaderboard_benches: list[str]) -> dict[str, str]:
    """Map T_star bench name -> leaderboard bench name, exact match first, name_key fallback."""
    lb_set = set(leaderboard_benches)
    lb_by_key = {name_key(b): b for b in leaderboard_benches}
    matched: dict[str, str] = {}
    for bench in t_star_benches:
        if bench in lb_set:
            matched[bench] = bench
            continue
        key = name_key(bench)
        if key in lb_by_key:
            matched[bench] = lb_by_key[key]
    return matched


def _coverage_normalized_profile(
    model: str,
    tag_ids: list[str],
    kept_benches: list[str],
    t_star: dict[str, dict[str, float]],
    pct: dict[str, dict[str, float]],
) -> dict[str, float]:
    """profile[tag] = (sum pct*weight over covered benches) / (sum weight over covered benches)."""
    profile: dict[str, float] = {}
    for tag_id in tag_ids:
        numerator = 0.0
        denominator = 0.0
        for bench in kept_benches:
            weight = t_star[bench].get(tag_id, 0.0)
            if weight == 0.0:
                continue
            p = pct.get(bench, {}).get(model)
            if p is None:
                continue
            numerator += p * weight
            denominator += weight
        profile[tag_id] = round(numerator / denominator, 4) if denominator > 0 else 0.0
    return profile


def _relative_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _load_shortlist(shortlist_path: Path) -> dict[str, tuple[str, str]]:
    """Map model display name -> (model_id, vendor) from a model_shortlist.json's rankings[]."""
    try:
        raw = _load_json(shortlist_path)
    except ValueError:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, tuple[str, str]] = {}
    for entry in raw.get("rankings", []):
        if not isinstance(entry, dict):
            continue
        name = entry.get("model_name")
        model_id = entry.get("model_id")
        if not name or not model_id:
            continue
        out[name] = (str(model_id), str(entry.get("vendor", "")))
    return out


def build_model_profiles(run_dir: Path, scores_path: Path, shortlist_path: Path) -> dict[str, Any]:
    """Build the model x tag capability-profile payload.

    Coverage normalization: `profiling.build_profile_percentile` sums
    `pct[bench][model] * T[bench][tag]` across all benchmarks a model
    appears in, with no denominator. That is fine when every model is
    scored on the same fixed benchmark suite, but the public leaderboard
    used here has uneven per-model coverage (some models are missing
    scores on some benchmarks). Without normalizing, a model that
    happens to be scored on more (or higher-weight) benchmarks for a
    tag would get an inflated profile value purely from coverage, not
    from being stronger on that tag. Dividing by the covered weight
    mass (`sum of T[bench][tag]` over only the benchmarks the model
    actually has a score for) makes profiles comparable across models
    with different coverage and keeps every value in [0, 1], which is
    required for the radar-chart consumer in benchboard.
    """
    t_star, vocab = _load_run(run_dir)
    leaderboard = _load_leaderboard(scores_path)

    bench_map = _match_benchmarks(list(t_star.keys()), list(leaderboard.keys()))
    if not bench_map:
        raise ValueError("no overlapping benchmarks between T_star and leaderboard scores")

    tag_ids = [entry["id"] for entry in vocab]

    model_names: set[str] = set()
    for t_bench, lb_bench in bench_map.items():
        model_names.update(leaderboard[lb_bench].keys())
    model_names_sorted = sorted(model_names)

    Y_raw = {t_bench: leaderboard[lb_bench] for t_bench, lb_bench in bench_map.items()}
    kept_benches = list(bench_map.keys())
    pct = y_to_percentile(Y_raw, kept_benches, model_names_sorted)

    shortlist = _load_shortlist(shortlist_path)

    models_out: list[dict[str, Any]] = []
    for model in model_names_sorted:
        profile = _coverage_normalized_profile(model, tag_ids, kept_benches, t_star, pct)
        model_id, vendor = shortlist.get(model, (_slug(model), ""))
        models_out.append({"id": model_id, "name": model, "vendor": vendor, "profile": profile})

    for entry in models_out:
        for value in entry["profile"].values():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"profile value out of [0,1] for {entry['name']}: {value}")

    models_out.sort(key=lambda e: e["name"])

    return {
        "meta": {
            "run": _relative_label(run_dir),
            "scores": _relative_label(scores_path),
            "mode": "percentile-coverage-normalized",
            "n_models": len(models_out),
            "n_tags": len(tag_ids),
        },
        "tags": tag_ids,
        "models": models_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export model x tag capability profiles")
    parser.add_argument("--run", default=str(DEFAULT_RUN), help="fold dir or its final/ dir")
    parser.add_argument("--scores", default=str(DEFAULT_SCORES), help="leaderboard scores json")
    parser.add_argument("--shortlist", default=str(DEFAULT_SHORTLIST), help="model_shortlist.json")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output path")
    args = parser.parse_args()

    try:
        payload = build_model_profiles(Path(args.run), Path(args.scores), Path(args.shortlist))
    except ValueError as exc:
        print(f"export_model_profiles: {exc}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_path, out_path)
    except OSError as exc:
        print(f"export_model_profiles: failed to write {out_path}: {exc}", file=sys.stderr)
        return 1

    meta = payload["meta"]
    print(f"wrote {out_path} ({meta['n_models']} models, {meta['n_tags']} tags)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
