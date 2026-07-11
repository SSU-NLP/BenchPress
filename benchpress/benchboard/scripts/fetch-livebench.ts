/**
 * LiveBench — no canonical static leaderboard endpoint exists. Their official
 * flow has you run `python -m livebench.gen_ground_truth_judgment` locally
 * which produces `all_groups.csv`. Drop that file at
 * `data/scores/_livebench-snapshot.csv` and this fetcher will pick it up.
 *
 * Until that file exists this is a graceful no-op so `fetch:all` doesn't fail.
 */
import path from "node:path";
import fs from "node:fs/promises";
import type { ScoreRecord } from "../src/lib/types.ts";
import { persist, nowIso } from "./lib/fetcher.ts";
import { resolveModelId, recordUnresolved } from "./lib/normalize.ts";
import { parseCsv } from "./lib/csv.ts";
import { SCORES_DIR } from "./lib/paths.ts";

const SNAPSHOT = path.join(SCORES_DIR, "_livebench-snapshot.csv");
const BENCHMARK_ID = "livebench";

async function main() {
  let csv: string;
  try {
    csv = await fs.readFile(SNAPSHOT, "utf8");
  } catch {
    console.warn(
      `[livebench] no snapshot at ${SNAPSHOT}; drop the all_groups.csv produced by livebench.gen_ground_truth_judgment to enable.`,
    );
    return;
  }
  const rows = parseCsv(csv);
  const fetched_at = nowIso();
  const records: ScoreRecord[] = [];
  for (const row of rows) {
    const name =
      row["model"] ?? row["Model"] ?? row["model_name"] ?? row[""] ?? "";
    if (!name) continue;
    const id = await resolveModelId(name);
    if (!id) {
      recordUnresolved(name);
      continue;
    }
    const avg = Number(
      row["global_average"] ?? row["average"] ?? row["Average"] ?? "",
    );
    if (!Number.isFinite(avg)) continue;
    records.push({
      model_id: id,
      benchmark_id: BENCHMARK_ID,
      score: avg,
      config: "global average",
      source: {
        kind: "manual",
        url: "local snapshot",
        ref: path.basename(SNAPSHOT),
        fetched_at,
      },
    });
  }
  await persist("livebench", records);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
