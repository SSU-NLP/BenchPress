# AGENTS.md

Project-wide instructions for any agent (or human) working on benchboard.
Loaded on every session — read this first.

## What this project is

Astro static site that aggregates LLM benchmark scores from three places:
- non-deterministic public leaderboards (Arena, AlpacaEval, Arena-Hard, WildBench)
- deterministic scores ingested from tech-report / system-card PDFs via an
  OpenAI-compatible LLM (Qwen by default)
- one-off manual edits to `data/scores/**.json`

Source of truth is JSON files in `data/`. There is no DB.

## The four scripts you'll run

```bash
npm run fetch:all              # pull non-deterministic public sources
npm run ingest:watch           # batch-ingest every PDF in docs/
npm run publish:drafts -- --all  # promote reviewed drafts + canonicalize
npm run dev                    # http://localhost:4321
```

`publish:drafts` is chained with `canonicalize-benchmarks.ts`. Anything you
publish goes through canonical collapse automatically — see "Canonicalization"
below.

## Canonicalization (read this before adding extraction logic)

PDF extraction emits benchmark names verbatim, which means the same benchmark
arrives under many spellings: `AIME 2024`, `AIME'24`, `AIME 2024 (no tools)`,
`AIME 2024 (Pass@1)`, `AIME 2024 (Competition Math) (With Tools)`. If left as
distinct catalog entries the deterministic tab fills with duplicated columns.

`scripts/canonicalize-benchmarks.ts` collapses these:
- the part inside parentheses moves into `score.config`
- alias map in the script handles known spellings
  (`aime'24 → aime-2024`, `math500 → math-500`, `simpleqa-evaluations → simpleqa`)
- the existing `mergeScores` rule then folds the duplicates into one record
  with `variants[]` per distinct config

This runs automatically every time `publish:drafts` runs. **You generally do
not need to call it directly.** Only call `npm run canonicalize` if you
hand-edited JSON in `data/` and want to re-collapse.

## Rules for new ingests

- **Filename convention** in `docs/`: `<model-id>.pdf` or
  `<model-id>__<anything>.pdf`. The `model-id` must already exist in
  `data/models.json`. Add aliases via `data/_well-known-models.json` and
  `npm run seed` if introducing a new model family.
- **Pick the right Qwen size**:
  - Academic tech reports (Kimi K2, DeepSeek v3, Qwen3, Llama 3.1):
    `qwen3.5-35b-a3b` (MoE, 3B active) is enough.
  - Safety-heavy system cards (GPT-5, GPT-4o, Codex Opus 4.6, Gemini 3 Pro):
    `qwen3.5-122b-a10b` (MoE, 10B active). The smaller model returns empty
    on red-team/jailbreak content, even with strict JSON prompts.
  - Override per run: `OPENAI_MODEL=openrouter/qwen/qwen3.5-122b-a10b npm run ingest:watch`
- **Cascade is the point**. Every PDF picks up comparison-model rows too.
  When the extractor emits a row for a model not in `data/models.json`, it's
  reported at end of run as "unresolved" — add the alias to
  `data/_well-known-models.json` and re-seed so the next ingest captures
  those rows.
- **Drafts**: ingest produces `data/scores/tech-reports/<model-id>.draft.json`.
  Open the file, sanity-check rows, then `npm run publish:drafts -- <model-id>`.

## Env and secrets

The scripts only read `process.env`. They never open `.env` directly.

Required for any ingest command:
```
OPENAI_API_KEY=<your-key>
OPENAI_BASE_URL=<openai-compatible-endpoint>
OPENAI_MODEL=<model-name-e.g. openrouter/qwen/qwen3.5-35b-a3b>
```

Load it however you want — direnv, shell export, `uv run --env-file=.env`,
dotenv-cli, etc.

## Files NOT to touch

- `data/scores/tech-reports/<x>.json` — published, regenerated only on the
  next ingest of `<x>.pdf` (or by a hand-edit you explicitly want)
- `data/benchmarks.json`, `data/models.json` — canonical catalogs. Edit
  through the seed/alias-pack files (`data/_well-known-*.json`) and re-run
  `npm run seed`.
- `.env` — user owns it.

## Files that ARE meant to be edited

- `data/_well-known-models.json` — add canonical models + aliases here, then
  `npm run seed`.
- `data/_well-known-benchmarks.json` — same for benchmarks.
- `scripts/canonicalize-benchmarks.ts` `ALIAS_MAP` — add benchmark spelling
  variants the extractor keeps producing for the same canonical bench.

## Where things live

| Path | Role |
|---|---|
| `data/benchmarks.json` | canonical benchmark catalog |
| `data/models.json` | canonical model catalog |
| `data/scores/<source>.json` | non-deterministic fetchers' output |
| `data/scores/tech-reports/<model>.json` | published deterministic ingests |
| `docs/<model-id>.pdf` | drop-zone for ingest |
| `docs/_index.json` | sha-indexed processed list (auto) |
| `src/` | Astro pages + React leaderboard component |
| `scripts/` | fetchers, ingest, canonicalize, publish helpers |

See also: [AUTOMATION.md](AUTOMATION.md) for command reference,
[SKILLS.md](SKILLS.md) for reusable patterns, [docs/README.md](docs/README.md)
for the drop-zone workflow.
