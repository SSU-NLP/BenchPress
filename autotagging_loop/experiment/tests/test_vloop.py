"""Stage 1.5 v_loop — pure-math checks + the offline gated-loop smoke.

The gated hill-climb invariants (seed always adopted, best = max accepted score,
every non-seed acceptance carries p_gt >= GATE_CONFIDENCE) are already asserted
inside vloop_pilot._smoke(); this wraps it for pytest and adds the load-bearing
similarity/objective math the smoke does not exercise directly.
"""
from __future__ import annotations

from autotagging_loop.experiment import vloop, vloop_pilot


def test_spearman_monotonic_is_one():
    assert vloop.spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == 1.0
    assert vloop.spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == -1.0


def test_cosine_identical_and_orthogonal():
    a = {"x": 1.0, "y": 0.0}
    assert vloop.cosine(a, a) == 1.0
    assert vloop.cosine({"x": 1.0, "y": 0.0}, {"x": 0.0, "y": 1.0}) == 0.0


def test_objective_selects_top_q_by_tag_similarity():
    """3 benchmarks -> 3 valid pairs -> top-20% = 1 pair. It must be the
    highest-tag-cos pair (a,b, cos=1), and score = that pair's rank_similarity."""
    def bench(name: str, vec: dict[str, float], scores: dict[str, float]):
        return vloop.BenchmarkProfile(
            vloop.Benchmark(name=name, items=[], scores=scores), vec
        )

    models = {f"m{i}": float(i) for i in range(5)}          # a,b share this ranking
    shuffled = {f"m{i}": float((i * 3) % 5) for i in range(5)}
    profiles = [
        bench("a", {"t": 1.0, "u": 0.0}, models),           # cos(a,b)=1
        bench("b", {"t": 1.0, "u": 0.0}, models),
        bench("c", {"t": 0.0, "u": 1.0}, shuffled),         # cos(a,c)=cos(b,c)=0
    ]
    report = vloop.PositivePairObjective(top_q=0.2, min_common_models=4).evaluate(profiles)
    assert len(report.selected_pairs) == 1
    assert (report.selected_pairs[0].a, report.selected_pairs[0].b) == ("a", "b")
    assert report.score == 1.0


def test_pilot_gated_loop_smoke():
    vloop_pilot._smoke()


def test_main_bench_split_smoke():
    from autotagging_loop.experiment import vloop_main
    vloop_main._smoke()
