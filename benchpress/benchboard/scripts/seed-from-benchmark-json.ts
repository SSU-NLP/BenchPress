/**
 * One-shot seed: read the existing Desktop/benchmark.json and produce
 * data/benchmarks.json + data/models.json. Idempotent — re-running merges
 * with whatever is already there.
 */
import path from "node:path";
import { readJson, writeJson } from "./lib/io.ts";
import { BENCHMARKS_PATH, MODELS_PATH, ROOT } from "./lib/paths.ts";
import type {
  Benchmark,
  BenchmarkCategory,
  BenchmarkType,
  Model,
} from "../src/lib/types.ts";

const SEED_PATH = path.resolve(ROOT, "..", "benchmark.json");

interface SeedShape {
  llm_benchmark_data: Array<{
    model: string;
    benchmarks: Record<string, string[]>;
  }>;
}

const CATEGORY_MAP: Record<string, BenchmarkCategory> = {
  General: "general",
  "Instruction Following": "instruction",
  "Math/Science": "math",
  Coding: "coding",
  Agent: "agent",
  agent: "agent",
  Multimodal: "multimodal",
  Multilinguality: "multilinguality",
  Long: "long",
  Factuality: "factuality",
  Safety: "safety",
  safety: "safety",
  Health: "health",
  Preference: "preference",
  Finance: "other",
  Law: "other",
  Medical: "health",
  "Practical Reasoning": "other",
  "Biosecurity / safety": "safety",
};

const KOREAN_PREFIXES = ["K", "Ko-", "KBank", "Kor", "Ko "];
const CHINESE_NAMES = new Set([
  "CMMLU",
  "CMRC",
  "C3",
  "CLUEWSC",
  "C-Eval",
  "CMATH",
  "CSimpleQA",
]);

function slug(s: string): string {
  return s
    .toLowerCase()
    .replace(/[()[\]/]/g, " ")
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9.+-]/g, "")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}

const NON_DETERMINISTIC = new Set([
  "arena-hard",
  "arena-hard-v2",
  "ko-arena-hard-v2",
  "writingbench",
  "chatbot-arena",
  "wildbench",
  "alpacaeval",
  "alpacaeval-2",
  "mt-bench",
  "livebench",
]);

function classify(name: string, group: string): {
  category: BenchmarkCategory;
  type: BenchmarkType;
  language?: string;
} {
  const id = slug(name);
  const cat = CATEGORY_MAP[group] ?? "other";
  let language: string | undefined;
  const isChinese =
    CHINESE_NAMES.has(name) ||
    /^(C|Z)h/.test(name) === false && /[一-鿿]/.test(name);
  const isKorean = KOREAN_PREFIXES.some((p) => name.startsWith(p));
  if (isKorean) language = "ko";
  else if (CHINESE_NAMES.has(name) || isChinese) language = "zh";

  let category: BenchmarkCategory = cat;
  if (language === "ko") category = "korean";
  else if (language === "zh") category = "chinese";

  const type: BenchmarkType = NON_DETERMINISTIC.has(id)
    ? "non_deterministic"
    : "deterministic";

  return { category, type, language };
}

// Model names from benchmark.json that are not actual LLMs (toolkits, placeholders, etc.)
const MODEL_BLOCKLIST = new Set([
  "vaetk",
  "gpt-oss-120b-20b",
  "2025_human_panel",
  "human-panel",
  "stem_grad",
  "stem-grad",
  "architects",
  "icecuber",
  "nvarc_2025",
  "nvarc-2025",
]);

function parseModelName(raw: string): {
  id: string;
  name: string;
  vendor: string;
  aliases: string[];
} {
  const stripped = raw.replace(/\s*\([^)]*\)\s*$/, "");
  const id = slug(stripped);
  const vendor = inferVendor(stripped);
  const aliases = Array.from(new Set([raw, stripped, id]));
  return { id, name: stripped, vendor, aliases };
}

function inferVendor(name: string): string {
  const lc = name.toLowerCase();
  if (lc.includes("claude")) return "Anthropic";
  if (lc.startsWith("gpt") || lc.includes("openai")) return "OpenAI";
  if (lc.includes("gemini")) return "Google";
  if (lc.includes("llama")) return "Meta";
  if (lc.includes("qwen")) return "Alibaba";
  if (lc.includes("deepseek")) return "DeepSeek";
  if (lc.includes("kimi")) return "Moonshot";
  if (lc.includes("grok")) return "xAI";
  if (lc.includes("solar")) return "Upstage";
  if (lc.includes("olmo")) return "Allen Institute for AI";
  if (lc.includes("yi")) return "01.AI";
  if (lc.includes("exaone")) return "LG AI Research";
  if (lc.includes("vaetk")) return "VAETKI";
  return "Unknown";
}

async function main() {
  const seed = await readJson<SeedShape>(SEED_PATH, { llm_benchmark_data: [] });
  if (seed.llm_benchmark_data.length === 0) {
    console.error(`No data found at ${SEED_PATH}`);
    process.exit(1);
  }

  const existingBench = await readJson<Benchmark[]>(BENCHMARKS_PATH, []);
  const existingModels = await readJson<Model[]>(MODELS_PATH, []);

  const benchMap = new Map(existingBench.map((b) => [b.id, b]));
  const modelMap = new Map(
    existingModels.filter((m) => !MODEL_BLOCKLIST.has(m.id)).map((m) => [m.id, m]),
  );

  for (const row of seed.llm_benchmark_data) {
    const m = parseModelName(row.model);
    if (MODEL_BLOCKLIST.has(m.id)) continue;
    const prev = modelMap.get(m.id);
    modelMap.set(m.id, {
      id: m.id,
      name: prev?.name ?? m.name,
      vendor: prev?.vendor ?? m.vendor,
      release_date: prev?.release_date,
      report_url: prev?.report_url,
      aliases: Array.from(new Set([...(prev?.aliases ?? []), ...m.aliases])),
    });

    for (const [group, items] of Object.entries(row.benchmarks)) {
      for (const benchName of items) {
        const id = slug(benchName);
        if (!id) continue;
        const meta = classify(benchName, group);
        const prevB = benchMap.get(id);
        benchMap.set(id, {
          id,
          name: prevB?.name ?? benchName,
          category: prevB?.category ?? meta.category,
          type: prevB?.type ?? meta.type,
          language: prevB?.language ?? meta.language,
          source_url: prevB?.source_url,
          description: prevB?.description,
          note: prevB?.note,
        });
      }
    }
  }

  const benchmarks = [...benchMap.values()].sort((a, b) =>
    a.id.localeCompare(b.id),
  );
  const models = [...modelMap.values()].sort((a, b) => a.id.localeCompare(b.id));

  await writeJson(BENCHMARKS_PATH, benchmarks);
  await writeJson(MODELS_PATH, models);

  console.log(
    `Seeded ${benchmarks.length} benchmarks and ${models.length} models.`,
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
