# benchboard

Static leaderboard for LLM benchmarks. Pulls non-deterministic benchmarks (Chatbot Arena, AlpacaEval, Arena-Hard, WildBench, LiveBench) from public sources on a daily cron; supports manual + LLM-assisted ingest of deterministic scores from technical reports, with cascade to comparison models cited in the report.

When the same `(model, benchmark)` pair has scores under different configurations, the leaderboard renders the primary as the main number and the alternates in parentheses: e.g. `87.2 (85.1, 88.0)`. Hover for config + source per variant.

## Quickstart

```bash
npm install
npm run seed             # one-time: build benchmark/model catalogs from ../benchmark.json
npm run fetch:all        # pull all non-deterministic sources
npm run dev              # http://localhost:4321
```

## Structure

- `data/` — JSON source of truth (catalogs + scores per source)
- `scripts/` — fetchers, ingest CLI, merge logic
- `src/` — Astro pages + React leaderboard components
- `.github/workflows/fetch-daily.yml` — daily refresh

See `../  .claude/plans/unified-beaming-wreath.md` for full design notes.
