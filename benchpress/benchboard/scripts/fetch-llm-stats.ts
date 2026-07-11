/**
 * llm-stats.com scraper — fetches individual benchmark leaderboard pages.
 * The site is SSR (Next.js), so raw HTML contains all score data.
 *
 * Benchmarks tracked:
 *   gpqa, hle, mmlu-pro, aime-2025, livecodebench, swe-bench-verified,
 *   ifeval, math, mmmu, arc-agi
 *
 * Run:  npx tsx scripts/fetch-llm-stats.ts
 */
import type { ScoreRecord } from "../src/lib/types.ts";
import { persist, nowIso } from "./lib/fetcher.ts";
import { resolveModelId, recordUnresolved } from "./lib/normalize.ts";

const BASE = "https://llm-stats.com/benchmarks";
const UA =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

// benchmark slug → our canonical benchmark_id
const BENCHMARKS: Record<string, string> = {
  "gpqa":                "gpqa",
  "hle":                 "hle",
  "mmlu-pro":            "mmlu-pro",
  "aime-2025":           "aime-2025",
  "livecodebench":       "livecodebench",
  "swe-bench-verified":  "swe-bench",
  "ifeval":              "ifeval",
  "math":                "math-500",
  "mmmu":                "mmmu",
  "arc-agi":             "arc-agi",
};

/** llm-stats leaderboard pages render a freshness line (e.g. "Last updated:
 *  2026-05-30" or "Updated May 30, 2026"). Capture it once per page so every
 *  row carries the *source's* date in `published`, distinct from our
 *  `fetched_at`. Best-effort: returns undefined when the page omits it.
 *  Page-level is the right grain — a scraped leaderboard has one update stamp,
 *  not a per-row date. */
function parsePublished(html: string): string | undefined {
  const m = html.match(
    /(?:last\s+)?updated\b[^0-9A-Za-z]*((?:[0-9]{4}-[0-9]{2}-[0-9]{2})|(?:[A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4}))/i,
  );
  return m ? m[1].trim() : undefined;
}

function parseLeaderboard(html: string): Array<{ name: string; score: number }> {
  const results: Array<{ name: string; score: number }> = [];
  // Each row: <tr ...><td>rank</td><td>...<a href="/models/...">NAME</a>...org...</td><td>..SCORE..</td>
  // Extract tbody rows
  const tbodyMatch = html.match(/<tbody>([\s\S]*?)<\/tbody>/);
  if (!tbodyMatch) return results;
  const tbody = tbodyMatch[1];

  // Split into rows
  const rows = tbody.split(/<tr[\s>]/);
  for (const row of rows) {
    // Model name: inside <a href="/models/...">NAME</a>
    const nameMatch = row.match(/href="\/models\/[^"]+">([^<]+)<\/a>/);
    // Score: first font-mono cell after the model cell — matches decimal like 0.946 or 94.6
    const scoreMatch = row.match(/font-mono[^>]*>([0-9]+\.?[0-9]*)</);
    if (!nameMatch || !scoreMatch) continue;
    const name = nameMatch[1].trim();
    let score = parseFloat(scoreMatch[1]);
    if (!Number.isFinite(score)) continue;
    // Normalize: if score > 1 it's already a percentage (e.g. 94.6), keep as-is
    // If score <= 1, multiply by 100 for consistency
    if (score <= 1) score = Math.round(score * 1000) / 10; // 0.946 → 94.6
    results.push({ name, score });
  }
  return results;
}

async function fetchBenchmark(slug: string, benchmarkId: string, fetched_at: string): Promise<ScoreRecord[]> {
  const url = `${BASE}/${slug}`;
  const res = await fetch(url, { headers: { "User-Agent": UA } });
  if (!res.ok) {
    console.warn(`[llm-stats] ${slug}: HTTP ${res.status}`);
    return [];
  }
  const html = await res.text();
  const rows = parseLeaderboard(html);
  const published = parsePublished(html);

  const records: ScoreRecord[] = [];
  for (const row of rows) {
    const id = await resolveModelId(row.name);
    if (!id) {
      recordUnresolved(row.name);
      continue;
    }
    records.push({
      model_id: id,
      benchmark_id: benchmarkId,
      score: row.score,
      config: "default",
      source: {
        kind: "github_repo",
        url,
        fetched_at,
        ...(published ? { published } : {}),
      },
    });
  }
  return records;
}

async function main() {
  const fetched_at = nowIso();
  const allRecords: ScoreRecord[] = [];

  for (const [slug, benchmarkId] of Object.entries(BENCHMARKS)) {
    console.log(`Fetching ${slug}...`);
    const records = await fetchBenchmark(slug, benchmarkId, fetched_at);
    console.log(`  → ${records.length} resolved`);
    allRecords.push(...records);
    // Be polite
    await new Promise((r) => setTimeout(r, 500));
  }

  await persist("llm-stats", allRecords);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
