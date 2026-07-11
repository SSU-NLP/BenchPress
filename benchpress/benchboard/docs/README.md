# docs/ — drop zone for tech reports

This folder is the inbox for LLM technical reports and system cards.
`npm run ingest:watch` scans it, extracts benchmark scores via the Anthropic
API, and produces a draft JSON you review before publishing.

## Drop-in convention

Filename encodes which `model_id` (from `data/models.json`) the report is
about. Two forms work:

```
<model-id>.pdf
<model-id>__<anything>.pdf
```

Examples:

```
docs/claude-opus-4.7.pdf
docs/claude-opus-4.7__system-card.pdf
docs/gpt-5.4__technical-report-v2.pdf
docs/gemini-3-pro__paper.pdf
```

The `model_id` must already exist in `data/models.json`. If you're ingesting a
brand-new model, add it (and its aliases) to `data/_well-known-models.json`
and run `npm run seed` first.

If a filename doesn't follow the convention, hand-edit `docs/_index.json` to
record the mapping explicitly:

```json
[
  { "pdf": "weird-filename.pdf", "model_id": "claude-opus-4.7" }
]
```

## Flow

```
1. Drop PDF into docs/
2. npm run ingest:watch
     → data/scores/tech-reports/<model-id>.draft.json
3. Open the draft, sanity-check the rows
4. npm run publish:drafts -- <model-id>   (or --all)
5. Commit data/scores/tech-reports/<model-id>.json
6. npm run dev (or rebuild) — new scores appear in the leaderboard
```

`ingest:watch` is idempotent: each processed PDF gets its sha recorded in
`docs/_index.json`. Re-running won't re-spend API tokens on unchanged files.
Replace the PDF (different sha) to force re-ingest.

## Auto-flexibility for new benchmarks

When the extracted report mentions a benchmark not yet in
`data/benchmarks.json`, the ingest CLI **auto-registers** it with a guessed
category (based on keyword heuristics) and `type: "deterministic"`. The score
is preserved. You can fix the category later by editing
`data/benchmarks.json` — nothing breaks.

The auto-registered list is printed at the end of each ingest run, so you can
spot-check.

## What's NOT in this folder

- `.env` — kept at the project root, never read by these scripts directly.
  Export the variables in your shell (or use a tool like `dotenv-cli`) before
  running `npm run ingest:watch`.
- The PDFs themselves are gitignored by default to keep the repo small — see
  the entry in `.gitignore`. If you want to track them, delete that line.
