/**
 * τ²-bench (tau2-bench) — Sierra's agentic tool-use benchmark.
 * Source: the public leaderboard submissions committed to the repo
 *   sierra-research/tau2-bench under web/leaderboard/public/submissions/.
 * A manifest.json lists each submission directory; each holds a
 * submission.json with a `results` block keyed by domain
 * (airline / retail / telecom / banking_knowledge). The headline metric
 * Sierra reports is pass^1 (`pass_1`).
 *
 * We fill the three core domains (tau2-airline / tau2-retail / tau2-telecom)
 * with the real pass_1 numbers. We deliberately do NOT synthesize the
 * micro-average — that needs per-domain task counts we don't have, and
 * fabricating it would violate the no-imputed-scores rule.
 */
import type { ScoreRecord } from "../src/lib/types.ts";
import { persist, nowIso } from "./lib/fetcher.ts";
import { resolveModelId, recordUnresolved } from "./lib/normalize.ts";

const RAW =
  "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/web/leaderboard/public/submissions";

// tau2 domain → our canonical benchmark_id. banking_knowledge has no column.
const DOMAIN_MAP: Record<string, string> = {
  airline: "tau2-airline",
  retail: "tau2-retail",
  telecom: "tau2-telecom",
};

interface DomainResult {
  pass_1?: number | null;
}
interface Submission {
  model_name?: string;
  results?: Record<string, DomainResult | null>;
}
interface Manifest {
  submissions?: string[];
  legacy_submissions?: string[];
  voice_submissions?: string[];
}

async function getJson<T>(url: string): Promise<T | null> {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) return null;
  return (await res.json()) as T;
}

async function main() {
  const manifest = await getJson<Manifest>(`${RAW}/manifest.json`);
  if (!manifest) throw new Error("tau2-bench: manifest.json fetch failed");

  // The text-track standard + legacy submissions report the airline/retail/
  // telecom domains. Voice submissions are a separate live-audio track.
  const dirs = [
    ...(manifest.submissions ?? []),
    ...(manifest.legacy_submissions ?? []),
  ];

  const fetched_at = nowIso();
  const records: ScoreRecord[] = [];

  for (const dir of dirs) {
    const url = `${RAW}/${dir}/submission.json`;
    const sub = await getJson<Submission>(url);
    if (!sub?.model_name || !sub.results) continue;

    const id = await resolveModelId(sub.model_name);
    if (!id) {
      recordUnresolved(sub.model_name);
      continue;
    }

    for (const [domain, benchmark_id] of Object.entries(DOMAIN_MAP)) {
      const dr = sub.results[domain];
      const score = dr?.pass_1;
      if (typeof score !== "number" || !Number.isFinite(score)) continue;
      records.push({
        model_id: id,
        benchmark_id,
        score: Math.round(score * 10) / 10,
        config: "pass^1",
        source: { kind: "github_repo", url, ref: dir, fetched_at },
      });
    }
  }

  await persist("tau2-bench", records);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
