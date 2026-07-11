/**
 * Chatbot Arena (LMArena) — pulls the daily-refreshed unofficial mirror
 * for the "text" leaderboard. Emits one ScoreRecord per ranked model with
 * benchmark_id "chatbot-arena", extra = {ci, votes, rank}.
 */
import type { ScoreRecord } from "../src/lib/types.ts";
import { persist, nowIso } from "./lib/fetcher.ts";
import { resolveModelId, recordUnresolved } from "./lib/normalize.ts";

const ENDPOINT =
  "https://api.wulong.dev/arena-ai-leaderboards/v1/leaderboard?name=text";
const BENCHMARK_ID = "chatbot-arena";

interface ArenaRow {
  rank: number;
  model: string;
  vendor?: string;
  score: number;
  ci?: number;
  votes?: number;
}

async function main() {
  const res = await fetch(ENDPOINT, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) throw new Error(`Arena fetch failed: ${res.status}`);
  const json = (await res.json()) as { models?: ArenaRow[] };
  const rows = json.models ?? [];

  const fetched_at = nowIso();
  const records: ScoreRecord[] = [];
  for (const row of rows) {
    const id = await resolveModelId(row.model);
    if (!id) {
      recordUnresolved(row.model);
      continue;
    }
    records.push({
      model_id: id,
      benchmark_id: BENCHMARK_ID,
      score: row.score,
      config: "default",
      source: { kind: "arena_api", url: ENDPOINT, fetched_at },
      extra: { rank: row.rank, ci: row.ci, votes: row.votes },
    });
  }
  await persist("arena", records);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
