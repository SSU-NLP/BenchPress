/**
 * EvalPlus — HumanEval+ and MBPP+ pass@1 from evalplus.github.io/results.json.
 * Emits two ScoreRecords per model under the canonical benchmark ids
 * "humaneval+" and "mbpp+" (same columns the tech-report ingests fold into).
 */
import type { ScoreRecord } from "../src/lib/types.ts";
import { persist, nowIso } from "./lib/fetcher.ts";
import { resolveModelId, recordUnresolved } from "./lib/normalize.ts";

const ENDPOINT = "https://evalplus.github.io/results.json";

interface EvalPlusRow {
  link?: string;
  "open-data"?: string;
  "pass@1": {
    humaneval?: number;
    "humaneval+"?: number;
    mbpp?: number;
    "mbpp+"?: number;
  };
  prompted?: boolean;
  size?: number;
}

async function main() {
  const res = await fetch(ENDPOINT);
  if (!res.ok) throw new Error(`EvalPlus fetch failed: ${res.status}`);
  const json = (await res.json()) as Record<string, EvalPlusRow>;

  const fetched_at = nowIso();
  const humaneval: ScoreRecord[] = [];
  const mbpp: ScoreRecord[] = [];

  for (const [name, row] of Object.entries(json)) {
    const id = await resolveModelId(name);
    if (!id) {
      recordUnresolved(name);
      continue;
    }
    const p = row["pass@1"] ?? {};

    const heScore = p["humaneval+"] ?? p["humaneval"];
    if (heScore != null) {
      humaneval.push({
        model_id: id,
        benchmark_id: "humaneval+",
        score: heScore,
        config: p["humaneval+"] != null ? "HumanEval+" : "HumanEval",
        source: { kind: "github_repo", url: ENDPOINT, fetched_at },
      });
    }

    const mbppScore = p["mbpp+"] ?? p["mbpp"];
    if (mbppScore != null) {
      mbpp.push({
        model_id: id,
        benchmark_id: "mbpp+",
        score: mbppScore,
        config: p["mbpp+"] != null ? "MBPP+" : "MBPP",
        source: { kind: "github_repo", url: ENDPOINT, fetched_at },
      });
    }
  }

  await persist("evalplus-humaneval", humaneval);
  await persist("evalplus-mbpp", mbpp);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
