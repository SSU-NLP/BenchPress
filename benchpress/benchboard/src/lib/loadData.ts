import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type {
  Benchmark,
  BenchmarkCategory,
  Model,
  ScoreRecord,
} from "./types.ts";
import { mergeScores } from "../../scripts/lib/merge.ts";

const here = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(here, "..", "..");
const DATA = path.join(ROOT, "data");
const SCORES = path.join(DATA, "scores");

async function readJson<T>(file: string, fallback: T): Promise<T> {
  try {
    const raw = await fs.readFile(file, "utf8");
    return JSON.parse(raw) as T;
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return fallback;
    throw err;
  }
}

const BENCHMARK_ID_ALIASES: Record<string, string> = {
  "arc-agi": "arc-agi-1",
  "arena-hard-v2.0": "arena-hard-v2",
  "crux-i": "cruxeval-i",
  "crux-o": "cruxeval-o",
  "instruction-hierarchy-evaluations": "instruction-hierarchy-evaluation",
  "model-context-mcp-evaluation": "model-context-protocol-mcp-evaluation",
  "multi-if": "multiif",
};

const PREFERRED_BENCHMARK_NAMES: Record<string, string> = {
  "arc-agi-1": "ARC-AGI-1",
  "arena-hard-v2": "Arena-Hard v2",
  "cruxeval-i": "CRUXEval-I",
  "cruxeval-o": "CRUXEval-O",
  "instruction-hierarchy-evaluation": "Instruction Hierarchy Evaluation",
  "model-context-protocol-mcp-evaluation": "Model Context Protocol (MCP) Evaluation",
  "multiif": "MultiIF",
};

function canonicalBenchmarkId(id: string): string {
  return BENCHMARK_ID_ALIASES[id] ?? id;
}

function canonicalizeBenchmark(benchmark: Benchmark): Benchmark {
  const id = canonicalBenchmarkId(benchmark.id);
  return {
    ...benchmark,
    id,
    name: PREFERRED_BENCHMARK_NAMES[id] ?? benchmark.name,
  };
}

function mergeBenchmark(existing: Benchmark, incoming: Benchmark): Benchmark {
  return {
    ...existing,
    ...incoming,
    id: existing.id,
    name: PREFERRED_BENCHMARK_NAMES[existing.id] ?? existing.name,
    source_url: existing.source_url ?? incoming.source_url,
    description: existing.description ?? incoming.description,
    note: existing.note ?? incoming.note,
  };
}

function canonicalizeScore(score: ScoreRecord): ScoreRecord {
  const benchmark_id = canonicalBenchmarkId(score.benchmark_id);
  return benchmark_id === score.benchmark_id ? score : { ...score, benchmark_id };
}

export async function loadBenchmarks(): Promise<Benchmark[]> {
  const benchmarks = await readJson<Benchmark[]>(path.join(DATA, "benchmarks.json"), []);
  const byId = new Map<string, Benchmark>();
  for (const benchmark of benchmarks.map(canonicalizeBenchmark)) {
    const existing = byId.get(benchmark.id);
    byId.set(benchmark.id, existing ? mergeBenchmark(existing, benchmark) : benchmark);
  }
  return [...byId.values()];
}

/**
 * Hand-curated "important" benchmark ids. These always appear in their
 * category tab even if only a few models have been scored yet.
 */
async function loadWellKnownBenchmarkIds(): Promise<Set<string>> {
  const list = await readJson<Array<{ id: string }>>(
    path.join(DATA, "_well-known-benchmarks.json"),
    [],
  );
  return new Set(list.map((b) => canonicalBenchmarkId(b.id)));
}

/** A benchmark column is shown in a category tab if it has at least this many
 *  scored models — below it the column is mostly empty and only clutters the
 *  view, so it's relegated to the All tab. */
const CORE_MIN_MODELS = 6;

/** Category slugs that show the full long tail (no core filtering). */
const OVERVIEW_CATEGORIES = new Set<CategorySlug>([
  "all",
  "deterministic",
  "non_deterministic",
]);

const CONFIG_WORDS = new Set(["high", "medium", "low", "max", "none"]);

function isConfigToken(value: string): boolean {
  const token = value.trim().toLowerCase();
  return (
    CONFIG_WORDS.has(token) ||
    /^\d+\s*k$/.test(token) ||
    token.includes("thinking") ||
    token.includes("reasoning")
  );
}

function isModelConfigSuffix(value: string): boolean {
  const parts = value.split(",").map((part) => part.trim()).filter(Boolean);
  return parts.length > 0 && parts.every(isConfigToken);
}

function splitModelConfig(name: string): { baseName: string; config?: string } {
  const match = name.match(/^(.*?)\s*\(([^()]*)\)\s*$/);
  if (!match) return { baseName: name };
  const [, baseName, suffix] = match;
  if (!isModelConfigSuffix(suffix)) return { baseName: name };
  return { baseName: baseName.trim(), config: suffix.trim() };
}

function modelGroupKey(vendor: string, name: string): string {
  return `${vendor.trim().toLowerCase()}::${name.trim().toLowerCase().replace(/\s+/g, " ")}`;
}

function mergeConfig(modelConfig: string | undefined, scoreConfig: string): string {
  const cleanScore = scoreConfig.trim();
  if (!modelConfig) return cleanScore;
  if (!cleanScore) return modelConfig;
  if (cleanScore.toLowerCase().includes(modelConfig.toLowerCase())) return cleanScore;
  return `${modelConfig}; ${cleanScore}`;
}

interface ModelGroup {
  id: string;
  model: Model;
  configByModelId: Map<string, string | undefined>;
}

function buildModelGroups(models: Model[]): Map<string, ModelGroup> {
  const modelById = new Map(models.map((model) => [model.id, model]));
  const variantKeys = new Set<string>();
  const splitByModelId = new Map<string, { baseName: string; config?: string }>();

  for (const model of models) {
    const split = splitModelConfig(model.name);
    splitByModelId.set(model.id, split);
    if (split.config) variantKeys.add(modelGroupKey(model.vendor, split.baseName));
  }

  const groups = new Map<string, ModelGroup>();

  function representativeFor(model: Model, baseName: string, key: string): Model {
    const existingBase = models.find(
      (candidate) =>
        candidate.vendor === model.vendor &&
        candidate.name.trim().toLowerCase() === baseName.trim().toLowerCase(),
    );
    const source = existingBase ?? model;
    return {
      ...source,
      id: source.id,
      name: baseName,
      aliases: Array.from(new Set([...(source.aliases ?? []), model.name, model.id])),
    };
  }

  for (const model of models) {
    const split = splitByModelId.get(model.id) ?? { baseName: model.name };
    const strippedKey = modelGroupKey(model.vendor, split.baseName);
    const shouldGroup = Boolean(split.config) || variantKeys.has(strippedKey);
    const key = shouldGroup ? strippedKey : model.id;
    let group = groups.get(key);
    if (!group) {
      const displayModel = shouldGroup
        ? representativeFor(model, split.baseName, strippedKey)
        : modelById.get(model.id) ?? model;
      group = { id: displayModel.id, model: displayModel, configByModelId: new Map() };
      groups.set(key, group);
    }
    group.configByModelId.set(model.id, split.config);
  }

  return groups;
}

function modelGroupForId(
  modelId: string,
  modelById: Map<string, Model>,
  groups: Map<string, ModelGroup>,
): ModelGroup {
  const model = modelById.get(modelId);
  if (!model) {
    return {
      id: modelId,
      model: { id: modelId, name: modelId, vendor: "Unknown", aliases: [] },
      configByModelId: new Map([[modelId, undefined]]),
    };
  }
  const split = splitModelConfig(model.name);
  const grouped = groups.get(modelGroupKey(model.vendor, split.baseName));
  return grouped ?? { id: model.id, model, configByModelId: new Map([[model.id, undefined]]) };
}

function scoreForModelGroup(score: ScoreRecord, group: ModelGroup): ScoreRecord {
  const modelConfig = group.configByModelId.get(score.model_id);
  return {
    ...score,
    model_id: group.id,
    config: mergeConfig(modelConfig, score.config),
    variants: score.variants?.map((variant) => ({
      ...variant,
      config: mergeConfig(modelConfig, variant.config),
    })),
  };
}

export async function loadModels(): Promise<Model[]> {
  return readJson<Model[]>(path.join(DATA, "models.json"), []);
}

export async function loadAllScores(): Promise<ScoreRecord[]> {
  let files: string[];
  try {
    files = await fs.readdir(SCORES);
  } catch {
    return [];
  }
  const out: ScoreRecord[] = [];
  for (const f of files) {
    if (!f.endsWith(".json")) continue;
    if (f.startsWith("_")) continue;
    const recs = await readJson<ScoreRecord[]>(path.join(SCORES, f), []);
    out.push(...recs.map(canonicalizeScore));
  }
  // tech-reports/ — flatten one level deeper
  try {
    const trDir = path.join(SCORES, "tech-reports");
    const trs = await fs.readdir(trDir);
    for (const f of trs) {
      if (!f.endsWith(".json") || f.endsWith(".draft.json")) continue;
      const recs = await readJson<ScoreRecord[]>(path.join(trDir, f), []);
      out.push(...recs.map(canonicalizeScore));
    }
  } catch {
    /* no tech-reports yet */
  }
  // Cross-file merge: variants from different sources fold together so the
  // UI sees one canonical record per (model, benchmark) with the alternates
  // in variants[].
  return mergeScores([], out);
}

export type CategorySlug =
  | BenchmarkCategory
  | "non_deterministic"
  | "nd_preference"
  | "nd_agent"
  | "nd_safety"
  | "nd_korean"
  | "nd_multilinguality"
  | "deterministic"
  | "all";

export interface CategoryView {
  category: CategorySlug;
  benchmarks: Benchmark[];
  models: Model[];
  /** Rows: one per model that has at least one score in this category. */
  rows: Array<{
    model: Model;
    scores: Record<string, ScoreRecord>;
  }>;
  /** How many in-category benchmarks were hidden by core filtering (long
   *  tail with too few scored models). 0 for overview tabs. */
  hiddenBenchmarks: number;
}

export async function buildView(category: CategorySlug): Promise<CategoryView> {
  const [benchmarks, models, allScores, wellKnownIds] = await Promise.all([
    loadBenchmarks(),
    loadModels(),
    loadAllScores(),
    loadWellKnownBenchmarkIds(),
  ]);

  const inCategory = (b: Benchmark) => {
    if (category === "all") return true;
    if (category === "non_deterministic") return b.type === "non_deterministic";
    if (category === "nd_preference") return b.type === "non_deterministic" && b.category === "preference";
    if (category === "nd_agent") return b.type === "non_deterministic" && b.category === "agent";
    if (category === "nd_safety") return b.type === "non_deterministic" && b.category === "safety";
    if (category === "nd_korean") return b.type === "non_deterministic" && b.category === "korean";
    if (category === "nd_multilinguality") return b.type === "non_deterministic" && b.category === "multilinguality";
    if (category === "deterministic") return b.type === "deterministic";
    // "multimodal" is a parent tab over its Vision + Video sub-categories.
    if (category === "multimodal")
      return b.type === "deterministic" && (b.category === "multimodal" || b.category === "vision" || b.category === "video");
    return b.category === category && b.type === "deterministic";
  };

  const allInCat = benchmarks.filter(inCategory);

  // Core filtering: leaf category tabs only show benchmarks that are either
  // hand-curated (well-known) or have enough scored models to be worth a
  // column. Overview tabs (all/deterministic/non_deterministic) show the
  // full long tail.
  const modelCount = new Map<string, Set<string>>();
  for (const s of allScores) {
    let set = modelCount.get(s.benchmark_id);
    if (!set) {
      set = new Set();
      modelCount.set(s.benchmark_id, set);
    }
    set.add(s.model_id);
  }
  const isCore = (b: Benchmark) =>
    wellKnownIds.has(b.id) ||
    (modelCount.get(b.id)?.size ?? 0) >= CORE_MIN_MODELS;

  const curated = !OVERVIEW_CATEGORIES.has(category);
  const benchInCat = curated ? allInCat.filter(isCore) : allInCat;
  const hiddenBenchmarks = curated ? allInCat.length - benchInCat.length : 0;
  const benchIds = new Set(benchInCat.map((b) => b.id));

  const modelById = new Map(models.map((m) => [m.id, m]));
  const modelGroups = buildModelGroups(models);
  const byModel = new Map<string, Record<string, ScoreRecord[]>>();
  const groupedModels = new Map<string, Model>();

  for (const s of allScores) {
    if (!benchIds.has(s.benchmark_id)) continue;
    const group = modelGroupForId(s.model_id, modelById, modelGroups);
    const groupedScore = scoreForModelGroup(s, group);
    let bucket = byModel.get(group.id);
    if (!bucket) {
      bucket = {};
      byModel.set(group.id, bucket);
      groupedModels.set(group.id, group.model);
    }
    (bucket[s.benchmark_id] ??= []).push(groupedScore);
  }

  const rows = [...byModel.entries()]
    .map(([id, scoreLists]) => {
      const scores: Record<string, ScoreRecord> = {};
      for (const [benchmarkId, records] of Object.entries(scoreLists)) {
        scores[benchmarkId] = mergeScores([], records)[0];
      }
      const m = groupedModels.get(id) ?? {
        id,
        name: id,
        vendor: "Unknown",
        aliases: [],
      };
      return { model: m, scores };
    })
    .sort((a, b) => a.model.name.localeCompare(b.model.name));

  const benchSorted = [...benchInCat].sort((a, b) => a.name.localeCompare(b.name));

  return {
    category,
    benchmarks: benchSorted,
    models,
    rows,
    hiddenBenchmarks,
  };
}

// ── Tag explorer (landing page) ──────────────────────────────────────────────

export type SchemeId = "ability" | "survey";

/** benchmark-tags.json shape: { ability: {id:[tags]}, survey: {id:[tags]} } */
export async function loadBenchmarkTags(): Promise<
  Record<SchemeId, Record<string, string[]>>
> {
  return readJson<Record<SchemeId, Record<string, string[]>>>(
    path.join(DATA, "benchmark-tags.json"),
    { ability: {}, survey: {} },
  );
}

export interface ExplorerBenchmark {
  id: string;
  name: string;
  category: BenchmarkCategory;
  type: BenchmarkType;
  modelCount: number;
}

export interface ExplorerScore {
  model: string;
  vendor: string;
  score: number;
  config: string;
}

export interface SchemeData {
  /** tag id → number of benchmarks carrying it */
  tagCounts: Record<string, number>;
  /** benchmark id → tag ids under this scheme */
  benchTags: Record<string, string[]>;
}

export interface ExplorerData {
  schemes: Record<SchemeId, SchemeData>;
  benchmarks: ExplorerBenchmark[];
  /** benchmark id → model scores, sorted desc */
  scores: Record<string, ExplorerScore[]>;
}

/** Everything the landing-page tag explorer needs, precomputed at build time. */
export async function loadExplorerData(): Promise<ExplorerData> {
  const [benchmarks, models, allScores, tagMaps] = await Promise.all([
    loadBenchmarks(),
    loadModels(),
    loadAllScores(),
    loadBenchmarkTags(),
  ]);
  const modelById = new Map(models.map((m) => [m.id, m]));

  // Group scores by benchmark, tracking distinct models for the count.
  const byBench = new Map<string, ExplorerScore[]>();
  const modelsByBench = new Map<string, Set<string>>();
  for (const s of allScores) {
    const m = modelById.get(s.model_id);
    let arr = byBench.get(s.benchmark_id);
    if (!arr) {
      arr = [];
      byBench.set(s.benchmark_id, arr);
    }
    arr.push({
      model: m?.name ?? s.model_id,
      vendor: m?.vendor ?? "Unknown",
      score: s.score,
      config: s.config,
    });
    let set = modelsByBench.get(s.benchmark_id);
    if (!set) {
      set = new Set();
      modelsByBench.set(s.benchmark_id, set);
    }
    set.add(s.model_id);
  }

  const scores: Record<string, ExplorerScore[]> = {};
  for (const [bid, arr] of byBench) {
    scores[bid] = arr.sort((a, b) => b.score - a.score);
  }

  const benchmarksOut: ExplorerBenchmark[] = benchmarks.map((b) => ({
    id: b.id,
    name: b.name,
    category: b.category,
    type: b.type,
    modelCount: modelsByBench.get(b.id)?.size ?? 0,
  }));

  const schemes = {} as Record<SchemeId, SchemeData>;
  for (const scheme of ["ability", "survey"] as const) {
    const map = tagMaps[scheme] ?? {};
    const tagCounts: Record<string, number> = {};
    for (const b of benchmarks) {
      for (const t of map[b.id] ?? []) tagCounts[t] = (tagCounts[t] ?? 0) + 1;
    }
    schemes[scheme] = { tagCounts, benchTags: map };
  }

  return { schemes, benchmarks: benchmarksOut, scores };
}

// ── Trends over time (score vs model release date) ────────────────────────────

export interface TrendPoint {
  model: string;
  vendor: string;
  date: string; // ISO YYYY-MM-DD (release month mid-pointed when only YYYY-MM)
  score: number;
}

export interface TrendData {
  /** benchmark id → display name (only benchmarks with enough dated points) */
  names: Record<string, string>;
  /** benchmark id → dated score points, sorted ascending by date */
  data: Record<string, TrendPoint[]>;
}

/**
 * Score-vs-release-date series per benchmark, for the /trends page. Only models
 * that carry a `release_date` can be placed on the time axis; only benchmarks
 * with at least `minPoints` dated models are included (so the chart grows
 * automatically as more release dates are filled in).
 */
export async function loadTrendData(minPoints = 5): Promise<TrendData> {
  const [benchmarks, models, all] = await Promise.all([
    loadBenchmarks(),
    loadModels(),
    loadAllScores(),
  ]);
  const bname = Object.fromEntries(benchmarks.map((b) => [b.id, b.name]));
  const inCatalog = new Set(benchmarks.map((b) => b.id));
  const byId = new Map(models.map((m) => [m.id, m]));

  const byBench = new Map<string, TrendPoint[]>();
  for (const s of all) {
    if (!inCatalog.has(s.benchmark_id)) continue; // skip blocklisted artifacts
    const m = byId.get(s.model_id);
    if (!m?.release_date) continue;
    const date = m.release_date.length === 7 ? `${m.release_date}-15` : m.release_date;
    let arr = byBench.get(s.benchmark_id);
    if (!arr) {
      arr = [];
      byBench.set(s.benchmark_id, arr);
    }
    arr.push({ model: m.name, vendor: m.vendor, date, score: s.score });
  }

  const names: Record<string, string> = {};
  const data: Record<string, TrendPoint[]> = {};
  for (const [bid, pts] of byBench) {
    if (pts.length < minPoints) continue;
    pts.sort((a, b) => a.date.localeCompare(b.date));
    names[bid] = bname[bid] ?? bid;
    data[bid] = pts;
  }
  return { names, data };
}
