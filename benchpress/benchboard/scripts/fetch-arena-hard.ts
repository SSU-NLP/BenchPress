/**
 * Arena-Hard — lmarena org publishes dated CSV snapshots in /leaderboard.
 * Pick the most recent one by name.
 */
import type { ScoreRecord } from "../src/lib/types.ts";
import { persist, nowIso } from "./lib/fetcher.ts";
import { resolveModelId, recordUnresolved } from "./lib/normalize.ts";
import { parseCsv } from "./lib/csv.ts";

const REPO = "lmarena/arena-hard-auto";
const DIR = "leaderboard";
const BENCHMARK_ID = "arena-hard";

interface GhEntry {
  name: string;
  download_url: string | null;
  type: string;
}

async function pickLatest(): Promise<{ name: string; url: string } | null> {
  const res = await fetch(
    `https://api.github.com/repos/${REPO}/contents/${DIR}`,
    { headers: { Accept: "application/vnd.github+json" } },
  );
  if (!res.ok) return null;
  const entries = (await res.json()) as GhEntry[];
  const csvs = entries.filter(
    (e) => e.type === "file" && e.name.endsWith(".csv") && e.download_url,
  );
  if (csvs.length === 0) return null;
  csvs.sort((a, b) => b.name.localeCompare(a.name));
  return { name: csvs[0].name, url: csvs[0].download_url! };
}

async function main() {
  const pick = await pickLatest();
  if (!pick) {
    console.warn("[arena-hard] no CSV snapshot found; skipping.");
    return;
  }
  const res = await fetch(pick.url);
  if (!res.ok) throw new Error(`Arena-Hard fetch ${res.status}`);
  const rows = parseCsv(await res.text());
  const fetched_at = nowIso();
  const records: ScoreRecord[] = [];
  for (const row of rows) {
    const name = row["model"];
    const score = Number(row["score"]);
    if (!name || !Number.isFinite(score)) continue;
    const id = await resolveModelId(name);
    if (!id) {
      recordUnresolved(name);
      continue;
    }
    records.push({
      model_id: id,
      benchmark_id: BENCHMARK_ID,
      score,
      config: "default",
      source: {
        kind: "github_repo",
        url: pick.url,
        ref: pick.name,
        fetched_at,
      },
      extra: {
        ci_lo: row["rating_q025"] ? Number(row["rating_q025"]) : undefined,
        ci_hi: row["rating_q975"] ? Number(row["rating_q975"]) : undefined,
        avg_tokens: row["avg_tokens"] ? Number(row["avg_tokens"]) : undefined,
        snapshot_date: row["date"],
      },
    });
  }
  await persist("arena-hard", records);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
