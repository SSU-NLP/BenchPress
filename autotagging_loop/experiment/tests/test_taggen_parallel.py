"""Phase C — _generate_T_via_prompt parallel correctness."""

from __future__ import annotations

import threading
import time

from autotagging_loop.experiment.loop import _generate_T_via_prompt
from autotagging_loop.experiment.tag_generator import TagVector


def test_generate_T_via_prompt_parallel_outputs_all_benchmarks():
    bench_names = [f"B{i:02d}" for i in range(8)]
    descriptions = {b: f"description-{b}" for b in bench_names}
    vocab = [{"id": "tag_x"}, {"id": "tag_y"}]
    call_lock = threading.Lock()
    call_log: list[str] = []

    def tag_fn(benchmark, description, vocab_arg, prompt, version):
        with call_lock:
            call_log.append(benchmark)
        time.sleep(0.005)  # encourage real overlap
        return TagVector(benchmark=benchmark, weights={"tag_x": 1.0, "tag_y": 0.0})

    T = _generate_T_via_prompt(
        bench_names, descriptions, vocab,
        prompt="prompt", version=1, tag_fn=tag_fn,
        desc="test", max_workers=4,
    )

    # All benchmarks tagged exactly once.
    assert sorted(call_log) == bench_names
    assert set(T) == set(bench_names)
    # Sorted dict on return.
    assert list(T.keys()) == sorted(bench_names)
    # Per-benchmark weights correctly assigned (no race).
    for b in bench_names:
        assert T[b] == {"tag_x": 1.0, "tag_y": 0.0}


def test_generate_T_via_prompt_empty_input_short_circuits():
    T = _generate_T_via_prompt(
        [], {}, [], prompt="", version=1, tag_fn=lambda *a: None,
    )
    assert T == {}


def test_generate_T_via_prompt_max_workers_clamped_to_input_size():
    """max_workers > len(benchmark_names) must not raise."""
    descriptions = {"only": "only"}

    def tag_fn(b, d, v, p, ver):
        return TagVector(benchmark=b, weights={"x": 1.0})

    T = _generate_T_via_prompt(
        ["only"], descriptions, [{"id": "x"}],
        prompt="p", version=1, tag_fn=tag_fn,
        desc="t", max_workers=64,
    )
    assert T == {"only": {"x": 1.0}}
