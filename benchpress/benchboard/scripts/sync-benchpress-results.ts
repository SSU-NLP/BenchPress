import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { readJson, writeJson } from "./lib/io.ts";

type AnyRecord = Record<string, any>;

interface FoldSummary {
  fold: number;
  L_align?: number;
  delta_tag?: number;
  rho_align_pearson?: number;
  rho_align_spearman?: number;
  source_benchmarks?: string[];
}

interface ScoreMatrix {
  Y_norm?: Record<string, Record<string, number>>;
}

interface RunCandidate {
  dir: string;
  runId: string;
  runMonth: string;
  qualityStatus?: string;
  folds: FoldSummary[];
  meanLAlign: number;
  meanSpearman: number;
  meanDeltaTag: number;
}

const here = path.dirname(fileURLToPath(import.meta.url));
const boardRoot = path.resolve(here, "..");
// benchboard now lives under benchpress/, so the repo root is two levels up.
const benchpressRoot = path.resolve(boardRoot, "..", "..");
const part2Dir = path.join(benchpressRoot, "results", "part2_experiment");
const dataDir = path.join(boardRoot, "data");

function normalizeId(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/\+/g, "plus")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function mean(values: number[]): number {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) return 0;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function round(value: number, digits = 3): number {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

// Blended cost per 1M tokens (3:1 input:output), from OpenRouter pricing on
// 2026-07-05. Keyed by normalizeId(model_name); unlisted models fall back to 0.
// Refresh from https://openrouter.ai/api/v1/models when the demo run changes.
const MODEL_COST_PER_MTOK: Record<string, number> = {
  "gpt-oss-20b": 0.06, // openai/gpt-oss-20b
  "claude-sonnet-4": 6.0, // anthropic/claude-sonnet-4
  "qwen2-5-72b": 0.37, // qwen/qwen-2.5-72b-instruct
  "deepseek-v3": 0.41, // deepseek/deepseek-chat-v3-0324
  "gpt-4o": 4.38, // openai/gpt-4o
  "gemma-3-4b": 0.06, // google/gemma-3-4b-it
};

async function exists(file: string): Promise<boolean> {
  try {
    await fs.access(file);
    return true;
  } catch {
    return false;
  }
}

async function readJsonLenient<T>(file: string, fallback: T): Promise<T> {
  try {
    const raw = await fs.readFile(file, "utf8");
    return JSON.parse(raw.replace(/:\s*(?:NaN|-?Infinity)(?=\s*[,}\]])/g, ": null")) as T;
  } catch (err: unknown) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return fallback;
    throw err;
  }
}

function finiteMean(values: Array<number | undefined>, fallback = Number.POSITIVE_INFINITY): number {
  const finite = values.filter((value): value is number => Number.isFinite(value));
  if (finite.length === 0) return fallback;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function compareRunCandidates(a: RunCandidate, b: RunCandidate): number {
  const qualityRank = (candidate: RunCandidate) => {
    if (candidate.qualityStatus === "pass") return 0;
    if (candidate.qualityStatus === "fail") return 2;
    return 1;
  };

  return (
    qualityRank(a) - qualityRank(b) ||
    a.meanLAlign - b.meanLAlign ||
    b.meanSpearman - a.meanSpearman ||
    b.meanDeltaTag - a.meanDeltaTag ||
    b.folds.length - a.folds.length ||
    b.runId.localeCompare(a.runId)
  );
}

function runMonth(runId: string): string {
  return runId.match(/\d{8}/)?.[0]?.slice(0, 6) ?? "";
}

async function bestRunDir(): Promise<RunCandidate | undefined> {
  if (process.env.BENCHPRESS_RUN_DIR) {
    const dir = path.resolve(process.env.BENCHPRESS_RUN_DIR);
    // results/ is gitignored, so CI / clean clones won't have this run dir.
    // Fall through to "keep committed demo data" instead of crashing.
    if (!(await exists(dir))) {
      console.warn(`[sync-benchpress-results] BENCHPRESS_RUN_DIR not found (${dir}); keeping existing demo data`);
      return undefined;
    }
    const folds = await loadFoldSummaries(dir);
    const qualityGate = await readJsonLenient<AnyRecord>(path.join(dir, "agg", "quality_gate.json"), {});
    const runId = path.basename(dir);
    return {
      dir,
      runId,
      runMonth: runMonth(runId),
      qualityStatus: qualityGate.status,
      folds,
      meanLAlign: finiteMean(folds.map((fold) => fold.L_align)),
      meanSpearman: finiteMean(folds.map((fold) => fold.rho_align_spearman), Number.NEGATIVE_INFINITY),
      meanDeltaTag: finiteMean(folds.map((fold) => fold.delta_tag), Number.NEGATIVE_INFINITY),
    };
  }

  let entries: string[];
  try {
    entries = await fs.readdir(part2Dir);
  } catch {
    return undefined;
  }

  const candidates: RunCandidate[] = [];
  for (const entry of entries
    .filter((entry) => entry.startsWith("run_") || entry.startsWith("run_cv_"))
    .sort()) {
    const runDir = path.join(part2Dir, entry);
    if (!(await exists(path.join(runDir, "config.json")))) continue;

    const folds = await loadFoldSummaries(runDir);
    if (folds.length === 0) continue;

    const meanLAlign = finiteMean(folds.map((fold) => fold.L_align));
    if (!Number.isFinite(meanLAlign)) continue;

    const qualityGate = await readJsonLenient<AnyRecord>(path.join(runDir, "agg", "quality_gate.json"), {});
    candidates.push({
      dir: runDir,
      runId: entry,
      runMonth: runMonth(entry),
      qualityStatus: qualityGate.status,
      folds,
      meanLAlign,
      meanSpearman: finiteMean(folds.map((fold) => fold.rho_align_spearman), Number.NEGATIVE_INFINITY),
      meanDeltaTag: finiteMean(folds.map((fold) => fold.delta_tag), Number.NEGATIVE_INFINITY),
    });
  }

  const targetMonth =
    process.env.BENCHPRESS_RUN_MONTH ??
    candidates
      .map((candidate) => candidate.runMonth)
      .filter(Boolean)
      .sort()
      .at(-1);
  const monthCandidates = targetMonth
    ? candidates.filter((candidate) => candidate.runMonth === targetMonth)
    : candidates;

  return monthCandidates.sort(compareRunCandidates)[0];
}

async function loadBenchmarkNameMap(): Promise<Map<string, string>> {
  const axisWeights = await readJson<AnyRecord>(
    path.join(dataDir, "benchmark_axis_weights.json"),
    { benchmarks: [] },
  );
  const out = new Map<string, string>();
  for (const benchmark of axisWeights.benchmarks ?? []) {
    out.set(normalizeId(benchmark.id), benchmark.id);
    out.set(normalizeId(benchmark.name), benchmark.id);
  }
  return out;
}

function benchmarkId(name: string, known: Map<string, string>): string {
  return known.get(normalizeId(name)) ?? normalizeId(name);
}

function matrixCoverage(matrix: ScoreMatrix) {
  const rows = matrix.Y_norm ?? {};
  const benchmarks = Object.keys(rows);
  const models = new Set<string>();
  let observedCells = 0;

  for (const scores of Object.values(rows)) {
    for (const [model, value] of Object.entries(scores)) {
      if (Number.isFinite(value)) {
        models.add(model);
        observedCells += 1;
      }
    }
  }

  const possible = benchmarks.length * Math.max(1, models.size);
  return {
    models: models.size,
    benchmarks: benchmarks.length,
    observed_cells: observedCells,
    density: round(observedCells / possible, 3),
  };
}

function rankModels(matrix: ScoreMatrix, selectedNames: string[]) {
  const rows = matrix.Y_norm ?? {};
  const modelIds = new Set<string>();
  for (const scores of Object.values(rows)) {
    Object.keys(scores).forEach((model) => modelIds.add(model));
  }

  const selectedSet = new Set(selectedNames);
  const fullScores = new Map<string, number>();
  const subsetScores = new Map<string, number>();

  for (const model of modelIds) {
    const allValues: number[] = [];
    const selectedValues: number[] = [];
    for (const [benchmark, scores] of Object.entries(rows)) {
      const value = scores[model];
      if (!Number.isFinite(value)) continue;
      allValues.push(value);
      if (selectedSet.has(benchmark)) selectedValues.push(value);
    }
    fullScores.set(model, mean(allValues) * 100);
    subsetScores.set(model, mean(selectedValues.length ? selectedValues : allValues) * 100);
  }

  const fullRank = new Map(
    [...fullScores.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([model], index) => [model, index + 1]),
  );

  return [...subsetScores.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([model, subsetScore], index) => {
      const fullScore = fullScores.get(model) ?? subsetScore;
      const rankDelta = Math.max(0, index + 1 - (fullRank.get(model) ?? index + 1));
      const modelId = normalizeId(model);
      return {
        model_id: modelId,
        model_name: model,
        vendor: model.split(/[-\s]/)[0] || "Unknown",
        subset_rank: index + 1,
        subset_score: round(subsetScore, 1),
        full_suite_rank: fullRank.get(model) ?? index + 1,
        full_suite_score: round(fullScore, 1),
        cost_per_mtok: MODEL_COST_PER_MTOK[modelId] ?? 0,
        regret: round(rankDelta / Math.max(1, modelIds.size), 3),
        reliability: round(1 - Math.min(0.5, Math.abs(subsetScore - fullScore) / 100), 3),
        recommendation: index === 0 ? "Best selected run score" : "Selected run candidate",
      };
    });
}

async function loadFoldSummaries(runDir: string): Promise<FoldSummary[]> {
  const aggregate = await readJsonLenient<FoldSummary[]>(
    path.join(runDir, "agg", "fold_summaries.json"),
    [],
  );
  if (aggregate.length > 0) return aggregate;

  const out: FoldSummary[] = [];
  const entries = await fs.readdir(runDir);
  for (const entry of entries.filter((name) => /^fold\d+$/.test(name)).sort()) {
    const selection = await readJson<AnyRecord>(path.join(runDir, entry, "selection.json"), {});
    const metrics = selection.selected_metrics ?? {};
    out.push({
      fold: Number(entry.replace("fold", "")),
      L_align: metrics.L_align,
      delta_tag: metrics.delta_tag,
      rho_align_pearson: metrics.rho_align_pearson,
      rho_align_spearman: metrics.rho_align_spearman,
    });
  }
  return out;
}

async function main(): Promise<void> {
  const runCandidate = await bestRunDir();
  if (!runCandidate) {
    console.warn("[sync-benchpress-results] no Benchpress run found; keeping existing demo data");
    return;
  }

  const runDir = runCandidate.dir;
  const runId = path.basename(runDir);
  const foldSummaries = runCandidate.folds;
  const primaryFold = foldSummaries[0];
  const primaryFoldDir = path.join(runDir, `fold${primaryFold?.fold ?? 0}`);
  const selection = await readJson<AnyRecord>(path.join(primaryFoldDir, "selection.json"), {});
  const matrix = await readJson<ScoreMatrix>(path.join(primaryFoldDir, "score_matrix.json"), {});
  const qualityGate = await readJson<AnyRecord>(path.join(runDir, "agg", "quality_gate.json"), {});
  const knownBenchmarks = await loadBenchmarkNameMap();

  const selectedNames = (primaryFold?.source_benchmarks ?? Object.keys(matrix.Y_norm ?? {})).slice(0, 5);
  const selectedBenchmarks = selectedNames.map((name) => benchmarkId(name, knownBenchmarks));
  const coverage = matrixCoverage(matrix);
  const selectedMetrics = selection.selected_metrics ?? {};
  const selectedSource =
    selection.selected_vocab_source === "executer" || selection.selected_source === "executer"
      ? "dynamic-vloop"
      : "fixed-vloop";

  await writeJson(path.join(dataDir, "runs", "index.json"), [
    {
      run_id: runId,
      date: runId.match(/\d{8}/)?.[0]?.replace(/(\d{4})(\d{2})(\d{2})/, "$1-$2-$3") ?? null,
      condition: `${selectedSource}, k=${selectedBenchmarks.length}`,
      selected_source: selectedSource,
      score_matrix_coverage: coverage,
      status: qualityGate.status ?? "synced",
      notes: `Synced best ${runCandidate.runMonth} Benchpress run from ${path.relative(benchpressRoot, runDir)} using quality=${runCandidate.qualityStatus ?? "none"}, mean L_align=${round(runCandidate.meanLAlign, 4)}.`,
    },
  ]);

  await writeJson(path.join(dataDir, "subset_selection_results.json"), {
    run_id: runId,
    target_capability: "Capability-targeted evaluation",
    default_request: {
      target_axes: [
        { axis_id: "formal_deduction", weight: 0.5 },
        { axis_id: "procedural_planning", weight: 0.5 },
      ],
      benchmark_count: selectedBenchmarks.length,
      cost_penalty: 0.25,
    },
    results: [
      {
        source_id: selectedSource,
        benchmark_count: selectedBenchmarks.length,
        selected_benchmarks: selectedBenchmarks,
        coverage_score: round(selectedMetrics.delta_tag ?? primaryFold?.delta_tag ?? 0),
        expected_predictive_utility: round(selectedMetrics.rho_align_spearman ?? primaryFold?.rho_align_spearman ?? 0),
        utility_components: {
          relevance: round(selectedMetrics.rho_align_pearson ?? primaryFold?.rho_align_pearson ?? 0),
          redundancy_reduction: round(selectedMetrics.delta_tag ?? primaryFold?.delta_tag ?? 0),
          predictive_gain: round(selectedMetrics.rho_align_spearman ?? primaryFold?.rho_align_spearman ?? 0),
          cost_efficiency: round(coverage.density),
        },
        selection_reasons: selectedNames.map((name) => `${name} appeared in the selected fold source set.`),
        duplicate_notes: [
          `selected_label=${selection.selected_label ?? "unknown"}`,
          `tag_count=${selection.selected_tag_count ?? "unknown"}`,
        ],
        reject_reasons: selection.adoption?.reasons ?? [],
        baseline_subsets: {},
      },
    ],
  });

  await writeJson(path.join(dataDir, "validation_results.json"), {
    run_id: runId,
    target_capability: "Capability-targeted evaluation",
    metrics: foldSummaries.map((fold) => ({
      method_id: `fold-${fold.fold}`,
      label: `Fold ${fold.fold}`,
      spearman: round(fold.rho_align_spearman ?? 0),
      kendall_tau: round(fold.rho_align_pearson ?? 0),
      ndcg_at_5: round(1 - Math.min(1, fold.L_align ?? 1)),
      top_k_overlap: round(fold.delta_tag ?? 0),
      regret: round(fold.L_align ?? 0),
    })),
    held_out_prediction: {
      spearman: round(qualityGate.pooled?.rho_spearman ?? mean(foldSummaries.map((fold) => fold.rho_align_spearman ?? 0))),
      top_k_overlap: round(mean(foldSummaries.map((fold) => fold.delta_tag ?? 0))),
      held_out_benchmarks: selectedBenchmarks,
    },
  });

  const rankings = rankModels(matrix, selectedNames);
  // Best budget = cheapest model by blended cost/Mtok, excluding the top pick,
  // tie-broken by better subset rank.
  const budgetPick = rankings
    .filter((row) => row.model_id !== rankings[0]?.model_id)
    .sort((a, b) => a.cost_per_mtok - b.cost_per_mtok || a.subset_rank - b.subset_rank)[0];
  await writeJson(path.join(dataDir, "model_shortlist.json"), {
    run_id: runId,
    target_capability: "Agentic coding assistant",
    ranking_basis: selectedBenchmarks,
    recommendations: {
      best_overall: rankings[0]?.model_id ?? null,
      best_budget: budgetPick?.model_id ?? rankings.at(-1)?.model_id ?? null,
      most_reliable: rankings.reduce((best, row) => (row.reliability > best.reliability ? row : best), rankings[0])
        ?.model_id ?? null,
    },
    rankings,
  });

  console.log(`[sync-benchpress-results] synced ${runId}`);
}

await main();
