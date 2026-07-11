"""Compatibility facade for the v3 experiment pipeline.

The implementation lives under :mod:`experiment.pipeline`. Keep this module so
existing imports such as ``from experiment.loop import run_part1`` continue to
work while the pipeline internals are split into smaller modules.
"""

from __future__ import annotations

from autotagging_loop.experiment.pipeline import run as _impl
from autotagging_loop.experiment.pipeline.run import *  # noqa: F401,F403

IterationResult = _impl.IterationResult
TagGenerationResult = _impl.TagGenerationResult

_build_profile_support = _impl._build_profile_support
_compute_metrics = _impl._compute_metrics
_generate_T_via_prompt = _impl._generate_T_via_prompt
_candidate_improvement_status = _impl._candidate_improvement_status
_is_better = _impl._is_better
_passes_delta_tag_gate = _impl._passes_delta_tag_gate
_select_kmedoids_subset = _impl._select_kmedoids_subset
_select_tag_cover_subset = _impl._select_tag_cover_subset
_write_stop_reason = _impl._write_stop_reason


def run_part1(*args, **kwargs):
    """Delegate to the implementation module while preserving monkeypatch hooks."""
    if globals().get("_compute_metrics") is not _impl._compute_metrics:
        _impl._compute_metrics = globals()["_compute_metrics"]
    return _impl.run_part1(*args, **kwargs)
