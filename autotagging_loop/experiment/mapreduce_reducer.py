"""Deprecated re-export of the v3 Maker.

The benchmark-level LLM call previously named "Reducer" is now
`experiment.maker.run_maker` (v3 §2.2.5). This module is kept as a
thin shim so external callers and pre-Phase-2 imports keep working.
New code should import from `experiment.maker` directly.
"""

from __future__ import annotations

from autotagging_loop.experiment.maker import (
    MakerChatFn as ReducerChatFn,
    _apply_one as _reduce_one,  # noqa: F401  (legacy private name)
    _maker_cache_key as _reducer_cache_key,  # noqa: F401
    _maker_root as _reducer_root,  # noqa: F401
    run_maker as build_mapreduce_reducer_outputs,
)

__all__ = ["ReducerChatFn", "build_mapreduce_reducer_outputs"]
