# Skills — recurring patterns worth reusing

Patterns we hit while building benchboard that pay off again across this and
future data-pipeline projects. Each entry is a few lines of intent + the
canonical file(s) where the pattern lives, so you can copy or import.

---

## 1. Variant-merge with rank-tiered sources

**When**: you have multiple sources reporting on the same `(entity, metric)`
pair under possibly-different configurations, and you want one canonical
record plus the alternates kept on the side (not lost, not duplicated).

**The rule**:
- Same normalised config → newer source wins; old becomes a variant only if
  the score actually differs.
- Different config → higher source-rank wins; loser is demoted to variants.
- Ties broken by recency.
- Config strings are normalised (lowercase, collapsed whitespace) before
  comparison so `"0-shot"` and `"0-Shot "` don't fork.

**Where**: [scripts/lib/merge.ts](scripts/lib/merge.ts) — the `mergeOne` and
`mergeScores` functions.

**Re-using it**: a `Record` type, a `SOURCE_RANK` map, and a function that
returns a stable key. That's the whole interface. Copy the file, retype the
record, swap the rank table.

---

## 2. Alias-resolved name normalisation

**When**: external sources spell the same entity inconsistently
(`gpt-4o-2024-05-13` vs `gpt-4o` vs `GPT-4o`). You want one lookup function
that accepts any of them and returns your canonical id — and you want
unresolved names surfaced for review instead of silently dropped.

**The pattern**:
1. Each canonical entity has `id`, `name`, and `aliases: string[]`.
2. `flatten()` strips spacing, casing, punctuation before comparison so you
   don't have to enumerate every spelling.
3. A module-level `unresolved` Set + `flushUnresolved(label)` prints
   diagnostics at the end of each script run, so missing aliases are visible
   without crashing the run.

**Where**: [scripts/lib/normalize.ts](scripts/lib/normalize.ts).

---

## 3. Source-of-truth as JSON files + idempotent re-folding

**When**: you want git-tracked, human-editable data with no DB overhead, but
still need scripts to merge in new data without clobbering manual edits.

**The pattern**:
- Hand-curated "packs" (`data/_well-known-models.json`,
  `data/_well-known-benchmarks.json`) and the auto-derived seed
  (`Desktop/benchmark.json`) are *inputs*.
- `npm run seed` re-folds inputs into the canonical files
  (`data/models.json`, `data/benchmarks.json`).
- Folding is a per-id union: existing fields kept, new aliases unioned.
- Re-running seed never destroys hand-edits — it only adds.

**Where**: [scripts/seed-from-benchmark-json.ts](scripts/seed-from-benchmark-json.ts),
[scripts/apply-alias-pack.ts](scripts/apply-alias-pack.ts).

---

## 4. Drop-zone + sha-indexed batch

**When**: a human keeps dropping files into a folder and you want a script
to process the new ones without redoing work.

**The pattern**:
- A drop folder (`docs/`) with a known filename convention.
- An `_index.json` next to the files: one row per processed file with sha
  and `processed_at`.
- The batch script computes sha on each run; skips when (path, sha) is
  unchanged; re-runs when the user replaces the file.

**Where**: [scripts/ingest-watch.ts](scripts/ingest-watch.ts).

**Why sha not mtime**: mtime is touched by editors, cloud-sync, and `cp`.
Sha is the only honest "did the content change" signal.

---

## 5. Cascade extraction from documents

**When**: a single source document (e.g. a tech report) reports scores for
several entities at once — the target plus its comparisons — and you want
*all* of them captured, tagged by source.

**The pattern**:
- Prompt the extractor (Claude) to emit one row per cell, including comparison
  rows, with an `is_target` boolean.
- Target rows → `source.kind = "tech_report"`, no `reported_by`.
- Comparison rows → `source.reported_by = <target_id>` so the leaderboard can
  later distinguish "Anthropic said X about GPT-Y" from "OpenAI said X about
  GPT-Y in their own report".
- New benchmark names encountered in the extraction are auto-registered to
  the catalog (with heuristic categories) so the data is never lost.

**Where**: [scripts/ingest-tech-report.ts](scripts/ingest-tech-report.ts) —
see the `PROMPT` constant and `registerNewBenchmarks`.

---

## 6. Static-site islands for sortable tables

**When**: you want a fully static site (no SSR, no JS runtime) but tables
need client-side sort/filter interactivity.

**The pattern**:
- Astro pages render at build time, embed the data as props.
- A single React component (`LeaderboardTable.tsx`) is the only client
  island. Astro hydrates only that subtree with `client:load`.
- Build-time data fetch reads JSON directly from disk via Node fs — no API
  needed.

**Where**: [src/components/LeaderboardTable.tsx](src/components/LeaderboardTable.tsx),
mounted in any tab page e.g. [src/pages/non-deterministic.astro](src/pages/non-deterministic.astro).

---

## 7. Pluggable env that the project never opens directly

**When**: you want users to manage secrets in `.env`, but your scripts
should treat `.env` as private and only read `process.env`.

**The pattern**:
- Ship a `.env.example` that documents the variables.
- Gitignore `.env`.
- Every script reads `process.env.X` directly — no `dotenv` import, no
  config file traversal. Whoever runs the script is responsible for
  exporting variables (their shell, direnv, or a Make wrapper).
- Document the contract in [AUTOMATION.md](AUTOMATION.md) so users know
  what's expected.

**Why**: keeps secrets out of the dependency surface and out of the
script's responsibilities. Users with proxies, KMS-backed secrets, or
unusual shells aren't fighting the project's opinion about how to load env.

---

## How to add a new skill

When you spot a pattern in this repo that you reach for twice in two
different places, write it up here. The format:

1. **When** — the trigger / problem.
2. **The pattern** — the shape of the solution.
3. **Where** — file paths in this repo.
4. Optional: gotchas, why-not-X.

Keep each skill under ~30 lines. If it grows past that, it should probably
be a dedicated doc with this file linking to it.
