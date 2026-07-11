<h1 align="center">BenchPress</h1>

<p align="center">
  <em>Evaluation Planning with Cognitive-Ability Tags Aligned to LLM Benchmark Score Patterns</em>
</p>

<p align="center">
  <b>https://ssu-nlp.github.io/BenchPress/</b>
</p>

<p align="center">
  <a href="https://youtu.be/zF_kGEbVKsI">Demo video</a>
</p>

## What it does

Picking benchmarks is a decision made *before* evaluation, but the ecosystem only helps *after*.
Leaderboards and harnesses assume you already know what to run, and the labels they organize
around — `math`, `coding`, `knowledge` — describe what a benchmark is *about*, not what it
*demands*. GPQA and SimpleQA are both "knowledge," yet one wants multi-step scientific reasoning
and the other simple fact recall; their model rankings are nearly uncorrelated. GPQA and SciCode
sit in different domains, share scientific reasoning, and rank models almost identically.

BenchPress reorganizes benchmarks by the **cognitive abilities their items require**. An LLM
tags every benchmark item on each ability axis, the tags aggregate into a per-benchmark ability
profile, and a closed loop refines the ability vocabulary against real model-score patterns —
keeping a revision only when it measurably improves the alignment between tag similarity and
ranking similarity. No human ground-truth labels are involved.

You pick a target ability and get three things back: **ability tags** that characterize each
benchmark, a **compact evaluation set** published as a manifest with fixed revisions and
sampling seeds — reproducible from the Hugging Face Hub without redistributing raw data — and
**neighbor benchmarks** whose historical model scores give you a reference point for reading
your own results.

## Quickstart

Requires **Python ≥ 3.10** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/SSU-NLP/BenchPress.git
cd BenchPress
uv sync
cp .env.example .env    # fill in your API keys
```

### Run the Autotagging Loop

```bash
uv run python autotagging_loop/main.py run
```

Subcommands: `build-corpus` (assemble the benchmark corpus), `run` (the closed loop),
`refresh-aai-scores`. To regenerate the seed taxonomy first:

```bash
uv run python autotagging_loop/pretrain.py
```

Results land in `results/`. Models, concurrency, and loop hyperparameters live in
`benchpress_config.json`; each model entry names its own `base_url_env` / `api_key_env`, so roles
can point at different providers without touching code.

The shipped `mapper_model` is a lightweight default. To reproduce the tagging reported in the
paper, set it to **Qwen3.5-27B with reasoning disabled** — the tagger selected for passing the
consensus-fidelity gate against teachers from four different model families.

### Run the demo locally

```bash
# Composer — publishing backend → http://127.0.0.1:7860
uv run python benchpress/space/app.py

# Builder — frontend → http://localhost:5173/BenchPress/
cd benchpress/benchboard && npm install && npm run dev
```
