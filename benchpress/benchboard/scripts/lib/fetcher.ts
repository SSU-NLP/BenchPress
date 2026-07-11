import type { ScoreRecord, ScoreVariant } from "../../src/lib/types.ts";
import { readJson, writeJson } from "./io.ts";
import { scoresPath } from "./paths.ts";
import { mergeScores } from "./merge.ts";
import { flushUnresolved } from "./normalize.ts";

const normCfg = (c: string) => (c || "default").trim().toLowerCase().replace(/\s+/g, " ");
const decimals = (n: number) => {
  const i = String(n).indexOf(".");
  return i < 0 ? 0 : String(n).length - i - 1;
};
const ROUND_EPS = 0.5;

/**
 * A score file is a single source. If that one source still reports >1 distinct
 * score for the same (model, benchmark, config) it's not a real variant:
 *   - a small spread is a rounding/precision dup → keep the most precise value;
 *   - a wide spread means several different things (model sizes, mixed metrics,
 *     undated snapshots) were collapsed under one label and can't be told apart
 *     → drop the group. (Cross-source disagreements live in separate files and
 *     are merged later, so they're never touched here.)
 */
function dedupeWithinSource(records: ScoreRecord[]): ScoreRecord[] {
  const out: ScoreRecord[] = [];
  for (const rec of records) {
    const entries: Array<ScoreVariant & { extra?: ScoreRecord["extra"] }> = [
      { score: rec.score, config: rec.config, source: rec.source, extra: rec.extra },
      ...(rec.variants ?? []),
    ];
    const groups = new Map<string, typeof entries>();
    for (const e of entries) {
      const k = normCfg(e.config);
      (groups.get(k) ?? groups.set(k, []).get(k)!).push(e);
    }
    for (const [, grp] of groups) {
      const scores = [...new Set(grp.map((e) => e.score))];
      if (scores.length === 1) {
        out.push({ ...rec, score: grp[0].score, config: grp[0].config, source: grp[0].source, variants: undefined });
        continue;
      }
      const spread = Math.max(...scores) - Math.min(...scores);
      if (spread <= ROUND_EPS) {
        const best = grp.slice().sort((a, b) => decimals(b.score) - decimals(a.score))[0];
        out.push({ ...rec, score: best.score, config: best.config, source: best.source, variants: undefined });
      }
      // else: wide-spread collapse → drop the whole group
    }
  }
  // re-fold so per-config survivors of one (model,benchmark) become one record
  return mergeScores([], out);
}

/**
 * Common wrapper: write the current fetch as the source's full state.
 *
 * Each fetch is a complete snapshot of the live source, so we REPLACE the file
 * (dedupe within the incoming batch only) rather than merging into the previous
 * run. Merging across runs let stale rows — e.g. a model_id that was later
 * split or renamed — accumulate forever as phantom variants.
 */
export async function persist(
  sourceId: string,
  incoming: ScoreRecord[],
): Promise<void> {
  const file = scoresPath(sourceId);
  const merged = dedupeWithinSource(mergeScores([], incoming));
  await writeJson(file, merged);
  flushUnresolved(sourceId);
  console.log(
    `[${sourceId}] wrote ${merged.length} records (${incoming.length} incoming).`,
  );
}

export function nowIso(): string {
  return new Date().toISOString();
}
