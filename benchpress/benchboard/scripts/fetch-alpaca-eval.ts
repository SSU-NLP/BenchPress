/**
 * AlpacaEval 2.0 — tatsu-lab maintains a CSV leaderboard in their repo.
 * Length-controlled win rate against GPT-4-Preview.
 */
import type { ScoreRecord } from "../src/lib/types.ts";
import { persist, nowIso } from "./lib/fetcher.ts";
import { resolveModelId, recordUnresolved } from "./lib/normalize.ts";
import { parseCsv } from "./lib/csv.ts";

const ENDPOINT =
  "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/src/alpaca_eval/leaderboards/data_AlpacaEval_2/weighted_alpaca_eval_gpt4_turbo_leaderboard.csv";
const BENCHMARK_ID = "alpacaeval-2";

async function main() {
  const res = await fetch(ENDPOINT);
  if (!res.ok) throw new Error(`AlpacaEval fetch failed: ${res.status}`);
  const csv = await res.text();
  const rows = parseCsv(csv);

  const fetched_at = nowIso();
  const records: ScoreRecord[] = [];
  for (const row of rows) {
    const name = row["model"] ?? row[""] ?? "";
    const lcRaw = row["length_controlled_winrate"] ?? row["win_rate"];
    if (!name || !lcRaw) continue;
    const score = Number(lcRaw);
    if (!Number.isFinite(score)) continue;
    const id = await resolveModelId(name);
    if (!id) {
      recordUnresolved(name);
      continue;
    }
    records.push({
      model_id: id,
      benchmark_id: BENCHMARK_ID,
      score,
      config: row["length_controlled_winrate"]
        ? "length-controlled"
        : "raw win rate",
      source: { kind: "alpaca_csv", url: ENDPOINT, fetched_at },
      extra: {
        avg_length: row["avg_length"] ? Number(row["avg_length"]) : undefined,
      },
    });
  }
  await persist("alpaca-eval", records);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
