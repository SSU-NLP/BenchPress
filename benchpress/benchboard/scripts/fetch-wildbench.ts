/**
 * WildBench — AllenAI's HF Space publishes an aggregated stats JSON keyed by
 * model name. We pull WB_score as the primary; also extract Arena Elo (hard),
 * Arena-Hard v0.1, AE2.0 LC that are cross-tabulated in the same file, and
 * emit those as bonus records for the matching benchmarks.
 */
import type { ScoreRecord } from "../src/lib/types.ts";
import { persist, nowIso } from "./lib/fetcher.ts";
import { resolveModelId, recordUnresolved } from "./lib/normalize.ts";

const ENDPOINT =
  "https://huggingface.co/spaces/allenai/WildBench/raw/main/data_dir/all_stat_wildbench.json";
const BENCHMARK_ID = "wildbench";

type Row = Record<string, number | string>;
type Snapshot = Record<string, Row>;

function num(v: unknown): number | undefined {
  if (typeof v === "number") return v;
  if (typeof v === "string" && v.trim() && v !== "-") {
    const n = Number(v);
    return Number.isFinite(n) ? n : undefined;
  }
  return undefined;
}

async function main() {
  const res = await fetch(ENDPOINT);
  if (!res.ok) {
    console.warn(`[wildbench] ${res.status} — snapshot not found; skipping.`);
    return;
  }
  const snap = (await res.json()) as Snapshot;
  const fetched_at = nowIso();
  const records: ScoreRecord[] = [];

  for (const [name, row] of Object.entries(snap)) {
    const id = await resolveModelId(name);
    if (!id) {
      recordUnresolved(name);
      continue;
    }
    const wb = num(row.WB_score);
    if (wb !== undefined) {
      records.push({
        model_id: id,
        benchmark_id: BENCHMARK_ID,
        score: wb,
        config: "WB-Score",
        source: { kind: "hf_dataset", url: ENDPOINT, fetched_at },
      });
    }
    // Cross-tabulated bonus records — same source kind, marked reported_by wildbench
    const cross: Array<[string, unknown]> = [
      ["arena-hard", row["Arena-Hard v0.1"]],
      ["alpacaeval-2", row["AE2.0 LC"]],
      ["chatbot-arena", row["Arena Elo (hard) - 2024-05-20"]],
    ];
    for (const [bench, raw] of cross) {
      const v = num(raw);
      if (v === undefined) continue;
      records.push({
        model_id: id,
        benchmark_id: bench,
        score: v,
        config: bench === "alpacaeval-2" ? "length-controlled" : "default",
        source: {
          kind: "hf_dataset",
          url: ENDPOINT,
          reported_by: "wildbench",
          fetched_at,
        },
      });
    }
  }
  await persist("wildbench", records);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
