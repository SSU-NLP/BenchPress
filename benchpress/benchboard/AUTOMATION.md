# Automation

Three repeating jobs keep benchboard's data fresh. All three are designed to
be re-run as often as you want without breaking state.

## 1. Non-deterministic refresh (Arena, AlpacaEval, Arena-Hard, WildBench)

```bash
npm run fetch:all
```

Pulls public CSVs / JSONs, normalises model names against
`data/models.json`, and merges into `data/scores/<source>.json`. Fully
deterministic: same source contents → same output. Unresolved model names
are printed at the end of each run; add them to
`data/_well-known-models.json` to capture more rows next time.

A GitHub Action at [.github/workflows/fetch-daily.yml](.github/workflows/fetch-daily.yml)
runs this every day at 08:00 UTC and opens a PR with the diff.

## 2. Tech-report ingest (deterministic scores from PDFs)

The drop-zone workflow — described in [docs/README.md](docs/README.md). One
sentence: drop a PDF into `docs/`, run `npm run ingest:watch`, review the
draft, run `npm run publish:drafts -- <model-id>`.

### Required environment

Both ingest commands read these from `process.env`. The endpoint is
OpenAI-compatible — point it at DashScope (Qwen), local Ollama, OpenRouter,
or anything else that speaks the OpenAI chat-completions protocol.

| Variable | Required | Notes |
|---|---|---|
| `OPENAI_API_KEY` | yes | Provider key; for Ollama set any non-empty string |
| `OPENAI_BASE_URL` | yes | e.g. `https://dashscope.aliyuncs.com/compatible-mode/v1`, `http://localhost:11434/v1`, `https://openrouter.ai/api/v1` |
| `OPENAI_MODEL` | yes | e.g. `qwen3-8b`, `qwen3:8b`, `qwen/qwen3-8b` |

Use whatever you like to populate them — direnv, a `.env` loaded by your
shell, `$env:OPENAI_API_KEY=…` for the session, etc. benchboard treats
`.env` as your private file and never reads or writes it.

PDF text is extracted locally via `pdf-parse` (no vision call). For very
clean tables this works fine; for multi-column layouts a 9B model can miss
rows — always review the draft before publishing.

A copy of the expected variable names is at [.env.example](.env.example).

### Cascade behaviour

When `ingest-tech-report.ts` processes a report, every row in every table is
emitted as a `ScoreRecord` — *including the comparison models cited in the
report*. Those rows get `source.reported_by = <target-model-id>` so the
leaderboard can later trace where a given score came from. This is the
"새 모델이 나올 때마다 그 안에서 비교했던 다른 모델과 벤치마크 점수도 가져온다"
requirement from the original spec.

### Auto-registration of new benchmarks

If the report mentions a benchmark not yet in `data/benchmarks.json`, the
ingest CLI registers it on the fly with a heuristic category and
`type: "deterministic"`. The score is never dropped. The new entries are
listed at the end of each ingest run so you can refine categories later.
This is the "새로운 벤치마크가 있으면 동기화" requirement.

## 3. Catalog upkeep

```bash
npm run seed
```

Idempotent: re-folds `Desktop/benchmark.json` and
`data/_well-known-{models,benchmarks}.json` into the canonical catalogs.
Run after editing either pack. Aliases are unioned (never dropped), so your
hand-curated entries survive re-seeding.

## Quick reference

```bash
# data refresh
npm run fetch:all              # non-deterministic
npm run ingest:watch           # all new docs/*.pdf
npm run ingest -- <id> <pdf>   # one-off PDF or URL

# catalog hygiene
npm run seed                   # re-fold catalogs from packs

# publishing
npm run publish:drafts             # list reviewed drafts
npm run publish:drafts -- <id>     # promote one
npm run publish:drafts -- --all    # promote everything

# site
npm run dev                    # http://localhost:4321
npm run build                  # static output → dist/
```
