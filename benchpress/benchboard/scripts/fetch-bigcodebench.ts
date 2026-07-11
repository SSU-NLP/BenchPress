/**
 * BigCodeBench — code generation with complex function calls.
 * Fetches from HuggingFace Datasets Server API (no auth needed).
 * Emits benchmark_id "bigcodebench-complete" and "bigcodebench-instruct".
 */
import type { ScoreRecord } from "../src/lib/types.ts";
import { persist, nowIso } from "./lib/fetcher.ts";
import { resolveModelId, recordUnresolved } from "./lib/normalize.ts";

const HF_API =
  "https://datasets-server.huggingface.co/rows?dataset=bigcode%2Fbigcodebench-results&config=default&split=train&offset=0&limit=500";

interface BCBRow {
  model: string;
  link?: string;
  complete?: number | null;
  instruct?: number | null;
}

async function main() {
  const res = await fetch(HF_API, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`BigCodeBench fetch failed: ${res.status}`);
  const json = (await res.json()) as { rows: Array<{ row: BCBRow }> };
  const rows = json.rows.map((r) => r.row);

  const fetched_at = nowIso();
  const complete: ScoreRecord[] = [];
  const instruct: ScoreRecord[] = [];

  for (const row of rows) {
    const id = await resolveModelId(row.model);
    if (!id) {
      recordUnresolved(row.model);
      continue;
    }
    if (row.complete != null) {
      complete.push({
        model_id: id,
        benchmark_id: "bigcodebench-complete",
        score: row.complete,
        config: "default",
        source: { kind: "hf_dataset", url: HF_API, fetched_at },
      });
    }
    if (row.instruct != null) {
      instruct.push({
        model_id: id,
        benchmark_id: "bigcodebench-instruct",
        score: row.instruct,
        config: "default",
        source: { kind: "hf_dataset", url: HF_API, fetched_at },
      });
    }
  }

  await persist("bigcodebench-complete", complete);
  await persist("bigcodebench-instruct", instruct);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
