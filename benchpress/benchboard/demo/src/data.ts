import benchmarks from "../../data/benchmarks.json";
import models from "../../data/models.json";
import wellKnownBenchmarks from "../../data/_well-known-benchmarks.json";
import { mergeScores } from "../../scripts/lib/merge";
import { sizeClass, sizeLabel } from "../../src/lib/modelSize";
import type { Benchmark, BenchmarkCategory, Model, ScoreRecord } from "../../src/lib/types";
import type { ExplorerData } from "../../src/lib/loadData";

export type CategorySlug = BenchmarkCategory | "deterministic" | "non_deterministic" | "nd_preference" | "nd_agent" | "nd_safety" | "nd_multilinguality" | "nd_korean" | "all";

export interface CategoryView {
  category: CategorySlug;
  benchmarks: Benchmark[];
  rows: Array<{ model: Model; scores: Record<string, ScoreRecord> }>;
  hiddenBenchmarks: number;
}

export interface TrendPoint {
  model: string;
  vendor: string;
  date: string;
  score: number;
}

export interface TrendData {
  names: Record<string, string>;
  data: Record<string, TrendPoint[]>;
}

export interface ModelScoreCategory {
  category: string;
  scores: Array<{ benchmark: Benchmark; record: ScoreRecord }>;
}

export interface ModelScoreData {
  model: Model | null;
  categories: ModelScoreCategory[];
}

export interface ExpectedScoreRow {
  id: string;
  name: string;
  vendor: string;
  score: number;
}

const scoreModules = import.meta.glob("../../data/scores/*.json", {
  import: "default",
}) as Record<string, () => Promise<ScoreRecord[]>>;

const techReportScoreModules = import.meta.glob("../../data/scores/tech-reports/*.json", {
  import: "default",
}) as Record<string, () => Promise<ScoreRecord[]>>;

const benchmarkIdAliases: Record<string, string> = {
  "arc-agi": "arc-agi-1",
  "arena-hard-v2.0": "arena-hard-v2",
  "crux-i": "cruxeval-i",
  "crux-o": "cruxeval-o",
  "gpqa-diamond": "gpqa",
  "instruction-hierarchy-evaluations": "instruction-hierarchy-evaluation",
  "model-context-mcp-evaluation": "model-context-protocol-mcp-evaluation",
  "multi-if": "multiif",
};
const preferredBenchmarkNames: Record<string, string> = {
  "arc-agi-1": "ARC-AGI-1",
  "arena-hard-v2": "Arena-Hard v2",
  "cruxeval-i": "CRUXEval-I",
  "cruxeval-o": "CRUXEval-O",
  "instruction-hierarchy-evaluation": "Instruction Hierarchy Evaluation",
  "model-context-protocol-mcp-evaluation": "Model Context Protocol (MCP) Evaluation",
  "multiif": "MultiIF",
};

export function canonicalBenchmarkId(id: string): string {
  return benchmarkIdAliases[id] ?? id;
}

function canonicalScore(score: ScoreRecord): ScoreRecord {
  const benchmark_id = canonicalBenchmarkId(score.benchmark_id);
  return benchmark_id === score.benchmark_id ? score : { ...score, benchmark_id };
}

function mergeBenchmarkMeta(existing: Benchmark, incoming: Benchmark): Benchmark {
  return {
    ...existing,
    ...incoming,
    id: existing.id,
    name: preferredBenchmarkNames[existing.id] ?? existing.name,
    source_url: existing.source_url ?? incoming.source_url,
    description: existing.description ?? incoming.description,
    note: existing.note ?? incoming.note,
  };
}

function canonicalBenchmarks(list: Benchmark[]): Benchmark[] {
  const byId = new Map<string, Benchmark>();
  for (const benchmark of list) {
    const id = canonicalBenchmarkId(benchmark.id);
    const next = { ...benchmark, id, name: preferredBenchmarkNames[id] ?? benchmark.name };
    const existing = byId.get(id);
    byId.set(id, existing ? mergeBenchmarkMeta(existing, next) : next);
  }
  return [...byId.values()];
}

const allBenchmarks = canonicalBenchmarks(benchmarks as Benchmark[]);
const allModels = models as Model[];
const modelById = new Map(allModels.map((model) => [model.id, model]));
const benchmarkById = new Map(allBenchmarks.map((benchmark) => [benchmark.id, benchmark]));
const wellKnownIds = new Set((wellKnownBenchmarks as Array<{ id: string }>).map((benchmark) => canonicalBenchmarkId(benchmark.id)));

let allScoresCache: ScoreRecord[] | null = null;

async function scoreFiles(): Promise<ScoreRecord[][]> {
  const topLevel = await Promise.all(Object.values(scoreModules).map((load) => load()));
  const techReports = await Promise.all(
    Object.entries(techReportScoreModules)
      .filter(([path]) => !path.endsWith(".draft.json"))
      .map(([, load]) => load()),
  );
  return [...topLevel, ...techReports];
}

export async function loadAllScoresBrowser(): Promise<ScoreRecord[]> {
  if (!allScoresCache) allScoresCache = mergeScores([], (await scoreFiles()).flat().map(canonicalScore));
  return allScoresCache;
}

export async function buildModelScoreDataBrowser(modelId: string): Promise<ModelScoreData> {
  const model = modelById.get(modelId) ?? null;
  const scores = (await loadAllScoresBrowser()).filter((score) => score.model_id === modelId);
  const grouped = new Map<string, Array<{ benchmark: Benchmark; record: ScoreRecord }>>();

  for (const record of scores) {
    const benchmark = benchmarkById.get(record.benchmark_id) ?? {
      id: record.benchmark_id,
      name: record.benchmark_id,
      category: "other",
      type: "deterministic",
    };
    const category = benchmark.category ?? "other";
    const bucket = grouped.get(category) ?? [];
    bucket.push({ benchmark, record });
    grouped.set(category, bucket);
  }

  const categories = [...grouped.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([category, items]) => ({
      category,
      scores: items.sort((a, b) => a.benchmark.name.localeCompare(b.benchmark.name)),
    }));

  return { model, categories };
}

export async function buildExpectedScoresBrowser(benchIds: string[]): Promise<ExpectedScoreRow[]> {
  const scoreBenchIds = [...new Set(benchIds.map(canonicalBenchmarkId))];
  if (scoreBenchIds.length === 0) return [];

  const scores = await loadAllScoresBrowser();
  const wanted = new Set(scoreBenchIds);
  const byModel = new Map<string, Record<string, number>>();

  for (const score of scores) {
    if (!wanted.has(score.benchmark_id)) continue;
    let bucket = byModel.get(score.model_id);
    if (!bucket) {
      bucket = {};
      byModel.set(score.model_id, bucket);
    }
    bucket[score.benchmark_id] = score.score;
  }

  return [...byModel.entries()]
    .flatMap(([modelId, byBench]) => {
      const model = modelById.get(modelId);
      if (!model) return [];
      const values = scoreBenchIds.map((benchId) => byBench[benchId]);
      if (values.some((value) => value == null)) return [];
      const score = values.reduce((sum, value) => sum + value, 0) / values.length;
      return [{ id: model.id, name: model.name, vendor: model.vendor, score }];
    })
    .sort((a, b) => b.score - a.score);
}

export async function buildDeterministicScoredBenchmarkIdsBrowser(): Promise<Set<string>> {
  const scores = await loadAllScoresBrowser();
  const scoredIds = new Set(scores.map((score) => canonicalBenchmarkId(score.benchmark_id)));
  return new Set(
    allBenchmarks
      .filter((benchmark) => benchmark.type === "deterministic")
      .map((benchmark) => canonicalBenchmarkId(benchmark.id))
      .filter((benchmarkId) => scoredIds.has(benchmarkId)),
  );
}

function inCategory(category: CategorySlug, benchmark: Benchmark): boolean {
  if (category === "all") return true;
  if (category === "deterministic") return benchmark.type === "deterministic";
  if (category === "non_deterministic") return benchmark.type === "non_deterministic";
  if (category === "multimodal") {
    return benchmark.type === "deterministic" && ["multimodal", "vision", "video"].includes(benchmark.category);
  }
  return benchmark.type === "deterministic" && benchmark.category === category;
}

const CORE_MIN_MODELS = 6;
const OVERVIEW_CATEGORIES = new Set<CategorySlug>(["all", "deterministic", "non_deterministic"]);

const generatedScoreViewLoaders = {
  non_deterministic: () => import("./generated/scoreViews/non_deterministic.json"),
  nd_preference: () => import("./generated/scoreViews/nd_preference.json"),
  nd_agent: () => import("./generated/scoreViews/nd_agent.json"),
  nd_safety: () => import("./generated/scoreViews/nd_safety.json"),
  nd_multilinguality: () => import("./generated/scoreViews/nd_multilinguality.json"),
  nd_korean: () => import("./generated/scoreViews/nd_korean.json"),
  deterministic: () => import("./generated/scoreViews/deterministic.json"),
  general: () => import("./generated/scoreViews/general.json"),
  math: () => import("./generated/scoreViews/math.json"),
  coding: () => import("./generated/scoreViews/coding.json"),
  agent: () => import("./generated/scoreViews/agent.json"),
  multimodal: () => import("./generated/scoreViews/multimodal.json"),
  vision: () => import("./generated/scoreViews/vision.json"),
  video: () => import("./generated/scoreViews/video.json"),
  multilinguality: () => import("./generated/scoreViews/multilinguality.json"),
  korean: () => import("./generated/scoreViews/korean.json"),
  all: () => import("./generated/scoreViews/all.json"),
} as const;

const generatedScoreViewCache = new Map<string, CategoryView>();
function canonicalCategoryView(view: CategoryView): CategoryView {
  const benchmarks = new Map<string, Benchmark>();
  for (const benchmark of view.benchmarks) {
    const id = canonicalBenchmarkId(benchmark.id);
    const next = { ...benchmark, id, name: preferredBenchmarkNames[id] ?? benchmark.name };
    const existing = benchmarks.get(id);
    benchmarks.set(id, existing ? mergeBenchmarkMeta(existing, next) : next);
  }

  const rows = view.rows.map((row) => {
    const grouped: Record<string, ScoreRecord[]> = {};
    for (const score of Object.values(row.scores)) {
      const next = canonicalScore(score);
      (grouped[next.benchmark_id] ??= []).push(next);
    }
    const scores = Object.fromEntries(
      Object.entries(grouped).map(([benchmarkId, records]) => [benchmarkId, mergeScores([], records)[0]]),
    );
    return { ...row, scores };
  });

  return {
    ...view,
    benchmarks: [...benchmarks.values()].sort((a, b) => a.name.localeCompare(b.name)),
    rows,
  };
}

export async function buildViewBrowser(category: CategorySlug): Promise<CategoryView> {
  const cached = generatedScoreViewCache.get(category);
  if (cached) return cached;
  const load = generatedScoreViewLoaders[category as keyof typeof generatedScoreViewLoaders];
  if (!load) throw new Error(`Missing generated score view loader: ${category}`);
  const mod = await load();
  const view = canonicalCategoryView(mod.default as CategoryView);
  generatedScoreViewCache.set(category, view);
  return view;
}


let explorerDataCache: ExplorerData | null = null;

export async function buildExplorerDataBrowser(): Promise<ExplorerData> {
  if (explorerDataCache) return explorerDataCache;
  const mod = await import("./generated/explorerData.json");
  explorerDataCache = mod.default as ExplorerData;
  return explorerDataCache;
}

export async function buildTrendDataBrowser(minPoints = 5): Promise<TrendData> {
  const scores = await loadAllScoresBrowser();
  const byBenchmark = new Map<string, TrendPoint[]>();

  for (const score of scores) {
    if (!benchmarkById.has(score.benchmark_id)) continue;
    const model = modelById.get(score.model_id);
    if (!model?.release_date) continue;
    const date = model.release_date.length === 7 ? `${model.release_date}-15` : model.release_date;
    let points = byBenchmark.get(score.benchmark_id);
    if (!points) {
      points = [];
      byBenchmark.set(score.benchmark_id, points);
    }
    points.push({ model: model.name, vendor: model.vendor, date, score: score.score });
  }

  const names: Record<string, string> = {};
  const data: Record<string, TrendPoint[]> = {};
  for (const [benchmarkId, points] of byBenchmark) {
    if (points.length < minPoints) continue;
    points.sort((a, b) => a.date.localeCompare(b.date));
    names[benchmarkId] = benchmarkById.get(benchmarkId)?.name ?? benchmarkId;
    data[benchmarkId] = points;
  }
  return { names, data };
}

const COVERAGE_BENCH: Array<[string, string]> = [
  ["aime-2024", "AIME 2024"],
  ["aime-2025", "AIME 2025"],
  ["arc-challenge", "ARC Challenge"],
  ["bbh", "BBH"],
  ["drop", "Drop"],
  ["gpqa", "GPQA"],
  ["gsm8k", "GSM8K"],
  ["hle", "HLE"],
  ["hmmt-feb-2025", "HMMT Feb 2025"],
  ["humaneval", "HumanEval"],
  ["math", "MATH"],
  ["math-500", "MATH-500"],
  ["mbpp", "MBPP"],
  ["mmlu", "MMLU"],
  ["mmlu-pro", "MMLU-Pro"],
  ["mmlu-redux", "MMLU-Redux"],
  ["simpleqa", "SimpleQA"],
  ["supergpqa", "SuperGPQA"],
  ["scicode", "SciCode"],
  ["livecodebench", "LiveCodeBench"],
];

// BenchPress main set: 13 models × 20 benchmarks (matches
// Benchpress/data/leaderboard_scores.json). COVERAGE_BENCH above holds the 20.
const TARGET_MODELS = [
  "deepseek-v3", "gpt-4o", "qwen-2.5-72b", "qwen3-235b", "claude-3.5-sonnet",
  "gpt-oss-120b", "gpt-oss-20b", "llama-3.1-8b", "phi-4-mini", "claude-sonnet-4",
  "gemma-3-4b", "kimi-k2", "glm-4.7",
];

// Counts shown on the Builder tab so its summary matches the Coverage tab.
export const COVERAGE_MODEL_COUNT = TARGET_MODELS.length;
export const COVERAGE_BENCH_COUNT = COVERAGE_BENCH.length;
// Benchmark ids in the same order as CoverageData row scores, so callers can
// look up per-benchmark scores by id (used by the Builder "Compose dataset" step).
export const COVERAGE_BENCH_IDS = COVERAGE_BENCH.map(([id]) => id);

export interface CoverageRow {
  id: string;
  name: string;
  vendor: string;
  size: string;
  sizeLabel: string;
  n: number;
  scores: Array<number | null>;
}

export interface CoverageData {
  benches: string[];
  rows: CoverageRow[];
  colMin: number[];
  colMax: number[];
}

export async function buildCoverageDataBrowser(): Promise<CoverageData> {
  const scores = await loadAllScoresBrowser();
  const ids = COVERAGE_BENCH.map(([id]) => id);
  const targetIds = new Set(TARGET_MODELS);
  const byModel = new Map<string, Record<string, number>>();

  for (const score of scores) {
    if (!ids.includes(score.benchmark_id)) continue;
    if (!targetIds.has(score.model_id)) continue;
    let bucket = byModel.get(score.model_id);
    if (!bucket) {
      bucket = {};
      byModel.set(score.model_id, bucket);
    }
    bucket[score.benchmark_id] = score.score;
  }

  const rows: CoverageRow[] = [];
  for (const modelId of TARGET_MODELS) {
    const model = modelById.get(modelId);
    if (!model) continue;
    const raw = byModel.get(modelId) ?? {};
    const rowScores = ids.map((id) => (id in raw ? Math.round(raw[id] * 10) / 10 : null));
    rows.push({
      id: modelId,
      name: model.name,
      vendor: model.vendor,
      size: sizeClass(model).key,
      sizeLabel: sizeLabel(model) ?? "",
      n: rowScores.filter((score) => score != null).length,
      scores: rowScores,
    });
  }

  const colMin = ids.map(() => Infinity);
  const colMax = ids.map(() => -Infinity);
  for (const row of rows) {
    row.scores.forEach((score, index) => {
      if (score == null) return;
      colMin[index] = Math.min(colMin[index], score);
      colMax[index] = Math.max(colMax[index], score);
    });
  }

  return { benches: COVERAGE_BENCH.map(([, label]) => label), rows, colMin, colMax };
}
