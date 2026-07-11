/**
 * Artificial Analysis fetcher — the densest standardized source. AA runs the
 * same eval harness across every model, so this fills the core columns for
 * models that never published a tech-report number.
 *
 * Requires a free API key. Get one at https://artificialanalysis.ai/ (account
 * → API), then expose it to the script:
 *
 *   AA_API_KEY=aa-... npx tsx scripts/fetch-artificial-analysis.ts
 *
 * Without the key the script no-ops (exit 0) so `fetch:all` doesn't hard-fail.
 *
 * API: GET /api/v2/data/llms/models, header `x-api-key`. Each model carries an
 * `evaluations` object keyed by eval slug.
 */
import type { ScoreRecord } from "../src/lib/types.ts";
import { persist, nowIso } from "./lib/fetcher.ts";
import { resolveModelId, recordUnresolved, splitModelName } from "./lib/normalize.ts";

const ENDPOINT = "https://artificialanalysis.ai/api/v2/data/llms/models";

// AA evaluations field → our canonical benchmark_id (+ optional config tag when
// AA's eval is a specific subset of our benchmark).
const EVAL_MAP: Record<string, { id: string; config?: string }> = {
  mmlu_pro: { id: "mmlu-pro" },
  gpqa: { id: "gpqa", config: "Diamond" }, // AA's GPQA is GPQA Diamond
  hle: { id: "hle" },
  livecodebench: { id: "livecodebench" },
  scicode: { id: "scicode" },
  math_500: { id: "math-500" },
  aime: { id: "aime-2025" }, // AA tracks the current-year AIME (2025)
};

interface AAModel {
  name?: string;
  slug?: string;
  model_creator?: { name?: string };
  evaluations?: Record<string, unknown>;
}

/** AA scores arrive either as 0–1 fractions or already as an index (>1).
 *  Normalize to a percentage / index with one decimal, mirroring the other
 *  fetchers. The intelligence index (e.g. 61) is already on a 0–100 scale. */
function normScore(raw: unknown): number | null {
  // Null / undefined → skip. `Number(null) === 0` would otherwise record a fake
  // 0 for models AA exposes the eval slot for but hasn't run yet.
  if (raw == null) return null;
  let inner: unknown = raw;
  if (typeof raw === "object" && "score" in raw) {
    inner = (raw as { score: unknown }).score;
    if (inner == null) return null;
  }
  const n = Number(inner);
  if (!Number.isFinite(n)) return null;
  return Math.round((n > 0 && n <= 1 ? n * 100 : n) * 10) / 10;
}

/** When AA returns an eval as an object it can carry a freshness timestamp
 *  alongside `score`. Pull it into `published` so the record keeps the
 *  *source's* date, distinct from our `fetched_at`. Best-effort: returns
 *  undefined for bare-number evals or when no known date key is present. */
function extractPublished(raw: unknown): string | undefined {
  if (raw == null || typeof raw !== "object") return undefined;
  const o = raw as Record<string, unknown>;
  for (const k of ["last_updated", "updated_at", "evaluated_at", "measured_at", "date"]) {
    const v = o[k];
    if (typeof v === "string" && v) return v;
  }
  return undefined;
}

async function main() {
  const key =
    process.env.AA_API_KEY ??
    process.env.Artificial_Analysis_API_KEY ??
    process.env.ARTIFICIAL_ANALYSIS_API_KEY;
  if (!key) {
    console.warn(
      "[artificial-analysis] AA_API_KEY / Artificial_Analysis_API_KEY not set — skipping.",
    );
    return;
  }

  const fetched_at = nowIso();
  const res = await fetch(ENDPOINT, { headers: { "x-api-key": key } });
  if (!res.ok) {
    console.error(`[artificial-analysis] HTTP ${res.status} ${res.statusText}`);
    process.exit(1);
  }
  const payload = await res.json();
  // The API wraps the list — accept either a bare array or { data: [...] }.
  const models: AAModel[] = Array.isArray(payload)
    ? payload
    : (payload.data ?? payload.models ?? []);

  const records: ScoreRecord[] = [];
  for (const m of models) {
    const rawName = m.name ?? m.slug ?? "";
    // AA lists model variants separately (e.g. "MiniMax M1 40K" / "80K" thinking
    // budgets). Split the budget/effort suffix off so it becomes a config tag
    // instead of collapsing several distinct numbers under one "default".
    const { base, tag } = splitModelName(rawName);
    const id = await resolveModelId(base);
    if (!id) {
      recordUnresolved(rawName);
      continue;
    }
    const evals = m.evaluations ?? {};
    for (const [field, target] of Object.entries(EVAL_MAP)) {
      if (!(field in evals)) continue;
      const score = normScore(evals[field]);
      if (score === null) continue;
      const config = [target.config, tag].filter(Boolean).join(", ") || "default";
      const published = extractPublished(evals[field]);
      records.push({
        model_id: id,
        benchmark_id: target.id,
        score,
        config,
        source: {
          kind: "aggregator_api",
          url: ENDPOINT,
          fetched_at,
          ...(published ? { published } : {}),
        },
      });
    }
  }

  await persist("artificial-analysis", records);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
