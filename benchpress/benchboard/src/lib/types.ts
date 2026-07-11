export type SourceKind =
  | "tech_report"
  | "arena_api"
  | "alpaca_csv"
  | "github_repo"
  | "hf_dataset"
  | "aggregator_api"
  | "manual";

export interface ScoreSource {
  kind: SourceKind;
  url: string;
  ref?: string;
  reported_by?: string;
  /** When the score itself was produced/published (ISO date) — distinct from
   *  `fetched_at`, which is only when our scraper pulled it. Drives which of
   *  several scores for a (model, benchmark) is the representative. Often
   *  unknown; selection then falls back to source reliability, then
   *  `fetched_at`. */
  published?: string;
  fetched_at: string;
}

export interface ScoreVariant {
  score: number;
  config: string;
  source: ScoreSource;
}

export interface ScoreExtra {
  ci?: number;
  votes?: number;
  rank?: number;
  [k: string]: unknown;
}

export interface ScoreRecord {
  model_id: string;
  benchmark_id: string;
  score: number;
  config: string;
  source: ScoreSource;
  variants?: ScoreVariant[];
  extra?: ScoreExtra;
}

export type BenchmarkType = "deterministic" | "non_deterministic";

export type BenchmarkCategory =
  | "general"
  | "instruction"
  | "math"
  | "coding"
  | "agent"
  | "multimodal"
  | "vision"
  | "video"
  | "multilinguality"
  | "long"
  | "factuality"
  | "safety"
  | "health"
  | "preference"
  | "korean"
  | "chinese"
  | "other";

export interface Benchmark {
  id: string;
  name: string;
  category: BenchmarkCategory;
  type: BenchmarkType;
  language?: string;
  source_url?: string;
  description?: string;
  /** Optional editorial callout shown on the benchmark detail page — e.g. a
   *  deprecation / contamination warning. Distinct from `description`. */
  note?: string;
}

export interface Model {
  id: string;
  name: string;
  vendor: string;
  release_date?: string;
  report_url?: string;
  aliases: string[];
}
