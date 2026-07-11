from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Protocol

LEVEL_VALUE = {
    "absent": 0.0,
    "weak": 0.25,
    "medium": 0.5,
    "strong": 0.75,
    "dominant": 1.0,
}
LEVELS = set(LEVEL_VALUE)

TagVector = dict[str, float]
ScoreVector = dict[str, float]


class JsonLLM(Protocol):
    def complete_json(self, prompt: str) -> dict:
        ...


@dataclass(frozen=True)
class TagAxis:
    id: str
    name: str
    definition: str


@dataclass(frozen=True)
class PromptState:
    iteration: int
    tagger_instructions: str
    vocab: list[TagAxis]


@dataclass(frozen=True)
class Item:
    id: str
    prompt: str


@dataclass(frozen=True)
class Benchmark:
    name: str
    items: list[Item]
    scores: ScoreVector  # model_id -> benchmark score


@dataclass(frozen=True)
class ItemTag:
    item_id: str
    levels: dict[str, str]  # tag_id -> absent|weak|medium|strong|dominant

    def vector(self) -> TagVector:
        return {k: LEVEL_VALUE[v] for k, v in self.levels.items()}


@dataclass(frozen=True)
class BenchmarkProfile:
    benchmark: Benchmark
    vector: TagVector


@dataclass(frozen=True)
class PairRow:
    a: str
    b: str
    tag_similarity: float
    rank_similarity: float
    n_common_models: int


@dataclass(frozen=True)
class ObjectiveReport:
    score: float
    loss: float
    selected_pairs: list[PairRow]
    all_pairs: list[PairRow]


class ItemTagger:
    def __init__(self, llm: JsonLLM):
        self.llm = llm

    def tag(self, item: Item, state: PromptState) -> ItemTag:
        data = self.llm.complete_json(render_item_tagger_prompt(item, state))
        levels = data.get("tags")
        if not isinstance(levels, dict):
            raise ValueError("missing tags object")

        expected = {axis.id for axis in state.vocab}
        got = set(levels)
        if got != expected:
            raise ValueError(f"tag ids mismatch: missing={expected - got}, extra={got - expected}")

        bad = {k: v for k, v in levels.items() if v not in LEVELS}
        if bad:
            raise ValueError(f"invalid tag levels: {bad}")

        return ItemTag(item_id=item.id, levels=dict(levels))


def render_item_tagger_prompt(item: Item, state: PromptState) -> str:
    axes = "\n".join(
        f"- {axis.id}: {axis.name}\n  Definition: {axis.definition}"
        for axis in state.vocab
    )
    return f"""You classify which abilities an item requires.

Rules:
- Do not solve the item.
- Do not predict model score, correctness, difficulty, pass rate, or leaderboard performance.
- Output ordinal labels only: absent, weak, medium, strong, dominant.
- Return JSON only.

Current tagging instructions:
{state.tagger_instructions}

Ability axes:
{axes}

Item:
{item.prompt}

Return exactly:
{{"tags": {{"<tag_id>": "<level>"}}}}
"""


def build_profile(benchmark: Benchmark, state: PromptState, tagger: ItemTagger) -> BenchmarkProfile:
    item_tags = [tagger.tag(item, state) for item in benchmark.items]
    return BenchmarkProfile(
        benchmark=benchmark,
        vector=aggregate_item_tags(item_tags, state.vocab),
    )


def aggregate_item_tags(item_tags: list[ItemTag], vocab: list[TagAxis]) -> TagVector:
    if not item_tags:
        return {axis.id: 0.0 for axis in vocab}

    return {
        axis.id: sum(tag.vector().get(axis.id, 0.0) for tag in item_tags) / len(item_tags)
        for axis in vocab
    }


class PositivePairObjective:
    def __init__(self, top_q: float = 0.2, min_common_models: int = 4):
        self.top_q = top_q
        self.min_common_models = min_common_models

    def evaluate(self, profiles: list[BenchmarkProfile]) -> ObjectiveReport:
        pairs: list[PairRow] = []

        for x, y in combinations(sorted(profiles, key=lambda p: p.benchmark.name), 2):
            rank_sim, n_common = rank_similarity(
                x.benchmark.scores,
                y.benchmark.scores,
                self.min_common_models,
            )
            if rank_sim is None:
                continue

            pairs.append(PairRow(
                a=x.benchmark.name,
                b=y.benchmark.name,
                tag_similarity=cosine(x.vector, y.vector),
                rank_similarity=rank_sim,
                n_common_models=n_common,
            ))

        if not pairs:
            return ObjectiveReport(float("-inf"), float("inf"), [], [])

        ranked = sorted(pairs, key=lambda p: (-p.tag_similarity, p.a, p.b))
        k = max(1, math.ceil(len(ranked) * self.top_q))
        selected = ranked[:k]
        score = mean([p.rank_similarity for p in selected])

        return ObjectiveReport(
            score=score,
            loss=1.0 - ((score + 1.0) / 2.0),
            selected_pairs=selected,
            all_pairs=ranked,
        )


class PromptImprover:
    def __init__(self, llm: JsonLLM):
        self.llm = llm

    def improve(self, state: PromptState, report: ObjectiveReport) -> PromptState:
        data = self.llm.complete_json(render_improver_prompt(state, report))

        axes = [
            TagAxis(
                id=str(x["id"]),
                name=str(x["name"]),
                definition=str(x["definition"]),
            )
            for x in data["vocab"]
        ]

        return PromptState(
            iteration=state.iteration + 1,
            tagger_instructions=str(data["tagger_instructions"]),
            vocab=axes,
        )


def render_improver_prompt(state: PromptState, report: ObjectiveReport) -> str:
    selected = [
        {
            "a": p.a,
            "b": p.b,
            "tag_similarity": p.tag_similarity,
            "rank_similarity": p.rank_similarity,
        }
        for p in report.selected_pairs
    ]

    return f"""Improve the item-tagger prompt and ability vocabulary.

Goal:
Benchmark pairs that receive similar tag profiles should have similar model-rank patterns.

Do not:
- Add model names.
- Mention leaderboard scores.
- Optimize for known benchmark scores.
- Add difficulty or expected-correctness tags.
- Turn ordinal classification into scoring.

You may:
- rewrite tagging instructions,
- merge redundant axes,
- split ambiguous axes,
- add missing ability axes,
- remove unused axes.

Current tagger instructions:
{state.tagger_instructions}

Current vocab:
{json.dumps([axis.__dict__ for axis in state.vocab], ensure_ascii=False, indent=2)}

Positive-pair report:
score={report.score}
loss={report.loss}
selected_pairs={json.dumps(selected, ensure_ascii=False, indent=2)}

Return JSON only:
{{
  "tagger_instructions": "...",
  "vocab": [
    {{"id": "...", "name": "...", "definition": "..."}}
  ]
}}
"""


class VLoop:
    def __init__(self, tagger: ItemTagger, objective: PositivePairObjective, improver: PromptImprover):
        self.tagger = tagger
        self.objective = objective
        self.improver = improver

    def run(self, benchmarks: list[Benchmark], initial_state: PromptState, max_iter: int):
        state = initial_state
        best = None

        for _ in range(max_iter):
            profiles = [build_profile(b, state, self.tagger) for b in benchmarks]
            report = self.objective.evaluate(profiles)

            if best is None or report.score > best["report"].score:
                best = {"state": state, "profiles": profiles, "report": report}

            state = self.improver.improve(state, report)

        return best


def rank_similarity(a: ScoreVector, b: ScoreVector, min_common: int) -> tuple[float | None, int]:
    shared = sorted(set(a) & set(b))
    if len(shared) < min_common:
        return None, len(shared)

    va = [a[m] for m in shared]
    vb = [b[m] for m in shared]

    if len(set(va)) < 2 or len(set(vb)) < 2:
        return None, len(shared)

    return spearman(va, vb), len(shared)


def spearman(a: list[float], b: list[float]) -> float:
    return pearson(ranks(a), ranks(b))


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        for k in range(i, j):
            out[order[k]] = avg
        i = j
    return out


def pearson(a: list[float], b: list[float]) -> float:
    ma, mb = mean(a), mean(b)
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    denom = math.sqrt(sum(x * x for x in da) * sum(x * x for x in db))
    return 0.0 if denom == 0.0 else sum(x * y for x, y in zip(da, db)) / denom


def cosine(a: TagVector, b: TagVector) -> float:
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(a.get(k, 0.0) ** 2 for k in keys))
    nb = math.sqrt(sum(b.get(k, 0.0) ** 2 for k in keys))
    return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)
