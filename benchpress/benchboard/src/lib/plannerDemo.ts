export const DEFAULT_PLANNER_SOURCE_ID = "dynamic-vloop";

// Live Composer Space for publishing compositions to Hugging Face.
export const COMPOSER_SPACE_URL = "https://huggingface.co/spaces/seonghyeon0408/benchpress-composer"; // update after Space deploy

// Local Composer publish endpoint. Enabled in dev (and via VITE_PUBLISH_API_URL);
// empty in the deployed build so the Builder's publish stays disabled + guidance-only.
export const PUBLISH_API_URL: string =
  (import.meta.env.VITE_PUBLISH_API_URL as string | undefined) ||
  (import.meta.env.DEV ? "http://127.0.0.1:7860/api/publish" : "");
export const DEFAULT_TARGET_AXIS_IDS = ["formal_deduction", "procedural_planning"];
export const DEFAULT_BENCHMARK_COUNT = 5;

export const BUILDER_COPY = {
  eyebrow: "Cognitive-ability-focused evaluation planner",
  title: "Build a compact benchmark suite for your target cognitive abilities.",
  summary:
    "Choose what you want to test, pick how many benchmarks you can run, and BenchPress recommends a smaller suite that still tracks full-suite rankings.",
  selectionPrinciple: "Beyond the most-correlated benchmarks",
};

// Hero call-to-action labels. Primary triggers subset generation in place;
// secondary/tertiary route to the Scores and Axes views.
export const HERO_CTAS = {
  primary: "Build compact suite",
  secondary: "Explore scores",
  tertiary: "View axes",
};

// One-line decision framing surfaced under the hero (key claim #6).
export const KEY_CLAIM =
  "Optimizes for relevance, overlap, ranking signal, and cost instead of only picking the most correlated benchmarks.";

// Top-level ability categories shown in the target ability picker.
// axisIds reference data/benchmark_axis_weights.json; axes not listed here
// render under an "Other" group so new data never disappears from the UI.
export const AXIS_GROUPS: { label: string; axisIds: string[] }[] = [
  {
    label: "Math & Quantitative",
    axisIds: ["numerical_computation", "spatial_geometrical_reasoning", "constraint_satisfaction"],
  },
  {
    label: "Logic & Planning",
    axisIds: ["formal_deduction", "procedural_planning", "pattern_induction", "analogical_reasoning"],
  },
  {
    label: "Knowledge & Comprehension",
    axisIds: ["contextual_retrieval", "external_knowledge_retrieval", "commonsense_causal_reasoning", "temporal_reasoning"],
  },
];

// Plain-language explainers for subset-selection sources, surfaced as a
// hover tooltip on the "Selected source" card for first-time visitors.
export const SOURCE_EXPLAINERS: Record<string, string> = {
  "dynamic-vloop":
    "The Dynamic V-loop is BenchPress's iterative taxonomy-learning procedure: a large language model proposes ability axes from model-by-benchmark score patterns, benchmarks are scored against those axes, and the axis vocabulary is refined until benchmark similarity under the axes best matches held-out model rankings.",
};

// Human labels for model_shortlist.json recommendation keys.
export const RECOMMENDATION_LABELS: Record<string, string> = {
  best_overall: "Best overall",
  best_budget: "Best budget",
  most_reliable: "Most reliable",
};

// Add an ability here only after adding the matching axis id to
// data/benchmark_axis_weights.json. The UI highlights ids listed in
// DEFAULT_TARGET_AXIS_IDS on the first screen.
export const CAPABILITY_OPTIONS = [
  ["formal_deduction", "Formal deduction"],
  ["procedural_planning", "Procedural planning"],
  ["numerical_computation", "Numerical computation"],
  ["contextual_retrieval", "Contextual retrieval"],
  ["external_knowledge_retrieval", "External knowledge retrieval"],
] as const;

// Dynamic V-loop lineage states. Keep this map in sync with
// data/axis_lineage.json when adding new guardrail outcomes.
// Monochrome per the Apple-style palette: status is conveyed by the label
// text; "new" gets the darkest treatment for emphasis.
export const LINEAGE_BADGE_CLASS: Record<string, string> = {
  new: "border-neutral-800 bg-neutral-950 text-white",
  merged: "border-neutral-300 bg-neutral-100 text-neutral-700",
  split: "border-neutral-300 bg-neutral-100 text-neutral-700",
  renamed: "border-neutral-300 bg-neutral-100 text-neutral-700",
  unchanged: "border-neutral-300 bg-neutral-50 text-neutral-600",
};

export const VALIDATION_BAR_SPECS = [
  ["Spearman", "spearman", "bg-emerald-500"],
  ["Kendall tau", "kendall_tau", "bg-cyan-500"],
  ["NDCG@5", "ndcg_at_5", "bg-sky-500"],
] as const;

export function pct(n: number): string {
  return `${Math.round(n * 100)}%`;
}

export function prettyId(id: string): string {
  return id
    .split("-")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function metricLabel(id: string): string {
  return id.replaceAll("_", " ");
}

export function coverageBarWidth(value: number): string {
  return `${Math.max(12, value * 100)}%`;
}

export function metricBarWidth(value: number): string {
  return `${Math.max(0, Math.min(100, value * 100))}%`;
}

export function regretBarWidth(value: number): string {
  return `${Math.max(0, Math.min(100, value * 1000))}%`;
}
