import { Fragment, type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import axisWeights from "../../data/benchmark_axis_weights.json";
import modelShortlist from "../../data/model_shortlist.json";
import modelTagProfilesJson from "../../data/model_tag_profiles.json";
import compositionExamples from "../../data/examples.json";
import LeaderboardTable from "../../src/components/LeaderboardTable";
import ScoreCell from "../../src/components/ScoreCell";
import { vendorSwatch } from "../../src/lib/vendorColors";
import {
  buildModelScoreDataBrowser,
  buildCoverageDataBrowser,
  buildDeterministicScoredBenchmarkIdsBrowser,
  buildExpectedScoresBrowser,
  buildTrendDataBrowser,
  buildViewBrowser,
  canonicalBenchmarkId,
  type CategorySlug,
  type CategoryView,
  type CoverageData,
  type ExpectedScoreRow,
  type ModelScoreData,
  type TrendData,
  type TrendPoint,
} from "./data";
import {
  AXIS_GROUPS,
  BUILDER_COPY,
  PUBLISH_API_URL,
  RECOMMENDATION_LABELS,
  metricBarWidth,
  pct,
} from "../../src/lib/plannerDemo";

// Per-model ability profiles over the 10 learned tags (built offline; may be
// a stub with an empty models list until the export runs).
type ModelTagProfile = { id: string; name: string; vendor: string; profile: Record<string, number> };
const modelTagProfiles = modelTagProfilesJson as unknown as {
  meta: Record<string, unknown>;
  tags: string[];
  models: ModelTagProfile[];
};

const benchById = new Map(axisWeights.benchmarks.map((bench) => [bench.id, bench]));
// Mean tag weight across benchmarks = how prominent an ability is; used to
// order tags (and their categories) by relevance.
const axisMeanWeight = new Map(
  axisWeights.axes.map((axis) => {
    const values = axisWeights.benchmarks.map((bench) => bench.weights[axis.id] ?? 0);
    return [axis.id, values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1)] as const;
  }),
);

// Axes grouped into top-level ability categories; anything the data adds
// beyond AXIS_GROUPS falls into "Other" so it stays selectable. Tags within a
// category and the categories themselves are ordered by relevance (mean weight).
const groupedAxes = (() => {
  const byId = new Map(axisWeights.axes.map((axis) => [axis.id, axis]));
  const byWeight = (a: { id: string }, b: { id: string }) => (axisMeanWeight.get(b.id) ?? 0) - (axisMeanWeight.get(a.id) ?? 0);
  const groups = AXIS_GROUPS.map((group) => ({
    label: group.label,
    axes: group.axisIds.map((id) => byId.get(id)).filter((axis): axis is NonNullable<typeof axis> => axis != null).sort(byWeight),
  }));
  const placed = new Set(AXIS_GROUPS.flatMap((group) => group.axisIds));
  const rest = axisWeights.axes.filter((axis) => !placed.has(axis.id)).sort(byWeight);
  if (rest.length) groups.push({ label: "Other", axes: rest });
  const groupWeight = (group: { axes: { id: string }[] }) => group.axes.reduce((sum, axis) => sum + (axisMeanWeight.get(axis.id) ?? 0), 0) / Math.max(group.axes.length, 1);
  return groups.filter((group) => group.axes.length).sort((a, b) => groupWeight(b) - groupWeight(a));
})();
const shortlistById = new Map(modelShortlist.rankings.map((row) => [row.model_id, row]));
const recommendationCards = Object.entries(modelShortlist.recommendations).map(([key, modelId]) => ({
  key,
  label: RECOMMENDATION_LABELS[key] ?? key,
  row: shortlistById.get(modelId),
}));

// Model shortlist rendered transposed: metrics as row labels, models as columns.
type ShortlistRow = (typeof modelShortlist.rankings)[number];
const SHORTLIST_METRICS: Array<{ label: string; tip?: string; render: (r: ShortlistRow) => ReactNode; badge?: boolean }> = [
  { label: "Full suite", tip: "The model's rank and mean score on the complete benchmark suite, for comparison.", render: (r) => `#${r.full_suite_rank} / ${r.full_suite_score.toFixed(1)}` },
  { label: "Cost", tip: "Approximate price to run the model, in US dollars per one million tokens.", render: (r) => `$${r.cost_per_mtok.toFixed(1)}/M` },
  {
    label: "Ranking mismatch",
    tip: "How far the model's rank on the compact suite is from its rank on the full suite. 0 means identical ordering, so a low value means the small suite reaches the same conclusion as running everything.",
    render: (r) => (
      <span className="inline-flex items-center gap-1.5">
        <span className="tabular-nums">{r.regret.toFixed(3)}</span>
        {r.regret <= 0.03 ? (
          <span className="rounded bg-emerald-50 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-emerald-700">{r.regret === 0 ? "matches full suite" : "≈ full suite"}</span>
        ) : null}
      </span>
    ),
  },
  { label: "Recommendation", render: (r) => r.recommendation, badge: true },
];

type Route = "home" | "builder" | "scores" | "trends" | "model";

function modelIdFromPath(): string | null {
  const path = window.location.pathname.replace(/\/$/, "");
  const match = path.match(/\/models\/([^/]+)$/);
  return match ? decodeURIComponent(match[1]) : null;
}

function routeFromPath(): Route {
  const path = window.location.pathname.replace(/\/$/, "");
  if (path.endsWith("/builder")) return "builder";
  return "home";
}

function pathForRoute(route: Route): string {
  if (route === "home") return import.meta.env.BASE_URL;
  if (route === "model") return `${import.meta.env.BASE_URL}scores`;
  return `${import.meta.env.BASE_URL}${route}`;
}

// ---- Landing page (ported from the BenchPress "Evaluation Profile" design) ----
// Interactive: pick a benchmark chip and the radar, axis bars, model candidates,
// and similar-benchmark list all recompute.
const LANDING_ACCENT = "#00a572";

const LANDING_AXIS_NAMES = axisWeights.axes.map((axis) => axis.name);
const LANDING_AXIS_DESCRIPTIONS = axisWeights.axes.map((axis) => axis.description);

type LandingBench = { id: string; name: string; w: number[] };

const LANDING_BENCHMARKS: LandingBench[] = axisWeights.benchmarks
  .filter((bench) => bench.id !== "math")
  .map((bench) => ({
    id: bench.id,
    name: bench.name,
    w: axisWeights.axes.map((axis) => bench.weights[axis.id] ?? 0),
  }));

type LandingModel = { name: string; vendor: string; aff: number[] };

// Candidate models with a curated per-axis strength profile.
const LANDING_MODELS: LandingModel[] = [
  { name: "GPT-oss-20b", vendor: "GPT", aff: [0.5, 0.8, 0.7, 0.9, 0.6, 0.6, 0.5, 0.4, 0.7, 0.6, 0.9, 0.5] },
  { name: "Claude-Sonnet-4", vendor: "Claude", aff: [0.8, 0.85, 0.8, 0.8, 0.75, 0.85, 0.9, 0.85, 0.7, 0.8, 0.75, 0.85] },
  { name: "Qwen2.5-72B", vendor: "Qwen2.5", aff: [0.7, 0.8, 0.9, 0.7, 0.6, 0.7, 0.6, 0.7, 0.75, 0.85, 0.6, 0.7] },
  { name: "DeepSeek-v3", vendor: "DeepSeek", aff: [0.6, 0.8, 0.85, 0.85, 0.6, 0.65, 0.6, 0.6, 0.8, 0.8, 0.8, 0.6] },
  { name: "GPT-4o", vendor: "GPT", aff: [0.75, 0.7, 0.7, 0.65, 0.7, 0.7, 0.8, 0.8, 0.6, 0.65, 0.65, 0.8] },
  { name: "Gemma-3-4B", vendor: "Gemma", aff: [0.4, 0.45, 0.4, 0.4, 0.4, 0.4, 0.45, 0.4, 0.35, 0.4, 0.4, 0.45] },
];

// Closed Catmull-Rom → cubic Bezier smoothing over a point ring (control
// points p1 ± (p2−p0)/6). Shared by the landing radar blob and the builder's
// interactive radar.
function radarPath(points: Array<[number, number]>): string {
  const n = points.length;
  let d = `M${points[0][0].toFixed(1)},${points[0][1].toFixed(1)}`;
  for (let i = 0; i < n; i++) {
    const p0 = points[(i - 1 + n) % n];
    const p1 = points[i];
    const p2 = points[(i + 1) % n];
    const p3 = points[(i + 2) % n];
    const c1x = p1[0] + (p2[0] - p0[0]) / 6;
    const c1y = p1[1] + (p2[1] - p0[1]) / 6;
    const c2x = p2[0] - (p3[0] - p1[0]) / 6;
    const c2y = p2[1] - (p3[1] - p1[1]) / 6;
    d += ` C${c1x.toFixed(1)},${c1y.toFixed(1)} ${c2x.toFixed(1)},${c2y.toFixed(1)} ${p2[0].toFixed(1)},${p2[1].toFixed(1)}`;
  }
  return `${d} Z`;
}

// Straight-line radar polygon over the axes (0..1 values -> svg path).
function radarBlob(values: number[]): string {
  const cx = 200;
  const cy = 200;
  const radius = 180;
  const pts = values.map((v, i) => {
    const ang = ((-90 + i * (360 / values.length)) * Math.PI) / 180;
    const r = Math.max(0.04, Math.min(1, v)) * radius;
    return [cx + r * Math.cos(ang), cy + r * Math.sin(ang)] as [number, number];
  });
  const d = pts.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  return `${d} Z`;
}

function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

// Small "i" affordance with a hover/focus tooltip. Inline styles keep it usable
// on the inline-styled landing page and the Tailwind builder alike. `align`
// controls which edge the bubble anchors to so it does not clip when the icon
// sits near a container edge (e.g. the sticky left column of a table).
function InfoTip({ text, label, align = "center" }: { text: string; label?: string; align?: "center" | "left" | "right" }) {
  const TOOLTIP_W = 224;
  const triggerRef = useRef<HTMLSpanElement>(null);
  const [coords, setCoords] = useState<{ top: number; left: number } | null>(null);
  // position:fixed (measured from the trigger) so the tooltip escapes any
  // overflow:auto/hidden scroll container — e.g. the model-shortlist table
  // wrapper, whose overflow-x-auto was clipping the upward-popping tooltip.
  function show() {
    const rect = triggerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const rawLeft =
      align === "left" ? rect.left
      : align === "right" ? rect.right - TOOLTIP_W
      : rect.left + rect.width / 2 - TOOLTIP_W / 2;
    const left = Math.max(8, Math.min(rawLeft, window.innerWidth - TOOLTIP_W - 8));
    setCoords({ top: rect.top - 6, left });
  }
  return (
    <span style={{ position: "relative", display: "inline-flex", alignItems: "center", verticalAlign: "middle" }}>
      <span
        ref={triggerRef}
        role="button"
        tabIndex={0}
        aria-label={label ?? "More information"}
        onMouseEnter={show}
        onMouseLeave={() => setCoords(null)}
        onFocus={show}
        onBlur={() => setCoords(null)}
        style={{ cursor: "help", display: "inline-flex", alignItems: "center", justifyContent: "center", width: 14, height: 14, borderRadius: 9999, border: "1px solid currentColor", fontSize: 9, fontWeight: 700, fontStyle: "italic", lineHeight: 1, opacity: 0.55 }}
      >
        i
      </span>
      {coords ? (
        <span role="tooltip" style={{ position: "fixed", top: coords.top, left: coords.left, transform: "translateY(-100%)", zIndex: 60, width: TOOLTIP_W, borderRadius: 8, background: "#191c1d", color: "#ffffff", padding: "8px 10px", fontSize: 11, fontWeight: 400, lineHeight: 1.45, letterSpacing: 0, textTransform: "none", textAlign: "left", whiteSpace: "normal", overflowWrap: "break-word", boxShadow: "0 8px 24px rgba(0,0,0,.2)", pointerEvents: "none" }}>
          {text}
        </span>
      ) : null}
    </span>
  );
}

const APP_NAV: Array<[Route, string]> = [
  ["home", "Home"],
  ["builder", "Builder"],
];

// Shared header for every page, landing included (font-sans/text color set
// explicitly so it doesn't inherit the landing page's Inter styling).
function AppHeader({ route }: { route: Route }) {
  return (
    <header className="sticky top-0 z-50 border-b border-neutral-200 bg-white/90 font-sans text-neutral-900 backdrop-blur">
      <div className="mx-auto grid max-w-[1920px] grid-cols-[1fr_auto_1fr] items-center gap-4 px-4 py-3">
        <a href={pathForRoute("home")} className="justify-self-start font-semibold tracking-tight">BenchPress</a>
        <nav className="flex justify-self-center gap-4 text-sm">
          {APP_NAV.map(([id, label]) => (
            <a
              key={id}
              href={pathForRoute(id)}
              className={`border-b-2 pb-1 ${route === id ? "border-neutral-950 text-neutral-950" : "border-transparent text-neutral-600 hover:text-black"}`}
            >
              {label}
            </a>
          ))}
        </nav>
      </div>
    </header>
  );
}

const RADAR_LABELS: Array<{ x: number; y: number; n: string }> = LANDING_AXIS_NAMES.map((_, i) => {
  const ang = ((-90 + i * (360 / LANDING_AXIS_NAMES.length)) * Math.PI) / 180;
  return {
    x: 200 + 195 * Math.cos(ang),
    y: 200 + 195 * Math.sin(ang),
    n: String(i + 1),
  };
});
const RADAR_SPOKES = RADAR_LABELS.map((label) => {
  const dx = label.x - 200;
  const dy = label.y - 200;
  const scale = 180 / Math.sqrt(dx * dx + dy * dy);
  return {
    x2: 200 + dx * scale,
    y2: 200 + dy * scale,
  };
});

const BUILDER_PRESETS = [
  { label: "General cognitive ability", text: "Balanced across every cognitive ability.", axisIds: axisWeights.axes.map((axis) => axis.id) },
  { label: "Logic-heavy", text: "Formal deduction and procedural planning.", axisIds: ["formal_deduction", "procedural_planning"] },
  { label: "Math-heavy", text: "Numerical, spatial, and constraint reasoning.", axisIds: ["numerical_computation", "spatial_geometrical_reasoning", "constraint_satisfaction"] },
  { label: "Knowledge-heavy", text: "External knowledge and contextual retrieval.", axisIds: ["external_knowledge_retrieval", "contextual_retrieval"] },
  { label: "Reasoning-heavy", text: "Pattern induction, analogy, and causal reasoning.", axisIds: ["pattern_induction", "analogical_reasoning", "commonsense_causal_reasoning"] },
];

// Full item count of each benchmark's evaluation set — caps the per-benchmark
// sampler in Compose dataset. The repo carries no size field, so these are the
// standard public eval-set counts; benchmarks not listed fall back to 5000.
// ponytail: hand-maintained lookup — verify/adjust when a source's size changes.
const BENCHMARK_ITEM_TOTALS: Record<string, number> = {
  "aime-2024": 30,
  "aime-2025": 30,
  "arc-challenge": 1172,
  bbh: 6511,
  drop: 9536,
  gpqa: 198,
  gsm8k: 1319,
  hle: 2500,
  "hmmt-feb-2025": 30,
  humaneval: 164,
  livecodebench: 880,
  "math-500": 500,
  mbpp: 974,
  mmlu: 14042,
  "mmlu-pro": 12032,
  "mmlu-redux": 3000,
  scicode: 338,
  simpleqa: 4326,
  supergpqa: 26529,
};
const benchmarkItemCap = (benchId: string): number => BENCHMARK_ITEM_TOTALS[benchId] ?? 5000;

function LandingPage() {
  const [selectedId, setSelectedId] = useState<string>("mmlu");
  const [hoveredAxisIndex, setHoveredAxisIndex] = useState<number | null>(null);
  const accent = LANDING_ACCENT;
  const cur = LANDING_BENCHMARKS.find((b) => b.id === selectedId) ?? LANDING_BENCHMARKS[0];

  const axisRows = LANDING_AXIS_NAMES.map((name, i) => ({
    num: i + 1,
    name,
    description: LANDING_AXIS_DESCRIPTIONS[i],
    bar: `${Math.round(cur.w[i] * 100)}%`,
  }));
  const hoveredAxis = hoveredAxisIndex == null ? null : axisRows[hoveredAxisIndex];
  // Always show a concrete axis in the explainer box (default to axis 1) so it
  // reads as a worked example rather than an empty prompt on first load.
  const shownAxis = hoveredAxis ?? axisRows[0];

  const models = LANDING_MODELS.map((m) => {
    const denom = cur.w.reduce((s, x) => s + x, 0) || 1;
    const fit = Math.round((cur.w.reduce((s, x, i) => s + x * m.aff[i], 0) / denom) * 100);
    return { name: m.name, vendor: m.vendor, fit };
  })
    .sort((a, b) => b.fit - a.fit)
    .slice(0, 4)
    .map((m, idx) => ({
      name: m.name,
      tag: idx === 0 ? "Best fit" : m.vendor,
      tagColor: idx === 0 ? "#00a572" : "#404943",
      score: String(m.fit),
    }));

  const similar = LANDING_BENCHMARKS.filter((b) => b.id !== cur.id)
    .map((b) => ({ id: b.id, name: b.name, simNum: cosine(cur.w, b.w) }))
    .sort((a, b) => b.simNum - a.simNum)
    .slice(0, 4)
    .map((b) => ({ id: b.id, name: b.name, sim: `${Math.round(b.simNum * 100)}%` }));

  return (
    <div style={{ minHeight: "100vh", background: "#ffffff", fontFamily: "'Geist Mono', ui-monospace, monospace", color: "#191c1d" }}>
      <AppHeader route="home" />

      {/* Hero + radar card */}
      <div style={{ position: "relative", overflow: "hidden", paddingTop: 112, paddingBottom: 72 }}>
        <div style={{ position: "absolute", inset: "-20%", zIndex: 0, background: "radial-gradient(40% 40% at 20% 25%,rgba(110,251,190,.55) 0%,transparent 70%),radial-gradient(45% 45% at 82% 20%,rgba(0,165,114,.30) 0%,transparent 70%),radial-gradient(50% 50% at 60% 90%,rgba(148,244,224,.45) 0%,transparent 70%)", filter: "blur(24px)", animation: "bp-mesh 18s ease-in-out infinite" }} />
        <main style={{ padding: "0 48px", maxWidth: 1440, margin: "0 auto", position: "relative", zIndex: 10 }}>
          <section style={{ textAlign: "center", marginBottom: 48 }}>
            <h1 style={{ fontSize: 48, lineHeight: 1.2, letterSpacing: 0, fontWeight: 700, margin: "0 auto 20px", maxWidth: "56rem", color: "#191c1d" }}>
              Pick a benchmark,<br />
              <span style={{ color: "#006b50" }}>see <span style={{ fontWeight: 900 }}>the cognitive abilities</span> it tests</span>
            </h1>
            <p style={{ fontSize: 18, lineHeight: 1.6, color: "#404943", maxWidth: "46rem", margin: "0 auto" }}>
              BenchPress classifies every benchmark by <strong style={{ color: "#191c1d", fontWeight: 600 }}>cognitive-ability tags</strong> — the reasoning, knowledge, and computation skills it actually measures, learned from model-by-benchmark score patterns. Start with familiar MMLU to explore its cognitive-ability profile, best-fit models, and the benchmarks nearest to it.
            </p>
          </section>

          {/* Primary CTA sits between the hero copy and the radar card. */}
          <div style={{ textAlign: "center", marginBottom: 44 }}>
            <a href={pathForRoute("builder")} style={{ display: "inline-flex", alignItems: "center", gap: 8, borderRadius: 8, background: "#006b50", color: "#ffffff", padding: "12px 22px", fontSize: 15, fontWeight: 600, textDecoration: "none" }}>
              Build your own evaluation suite from these cognitive abilities
              <span aria-hidden="true">→</span>
            </a>
            <p style={{ margin: "10px auto 0", maxWidth: "34rem", fontSize: 13, lineHeight: 1.5, color: "#707972" }}>
              Choose the cognitive abilities you care about and BenchPress recommends a compact benchmark set that still tracks full-suite model rankings.
            </p>
          </div>

          <section style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
            <div style={{ padding: "28px 36px 36px", borderRadius: 16, width: "100%", maxWidth: "64rem", position: "relative", overflow: "hidden", background: "rgba(255,255,255,.62)", backdropFilter: "blur(16px)", WebkitBackdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,.6)", boxShadow: "0 8px 32px rgba(0,107,80,.10)" }}>
              <div style={{ position: "absolute", top: -40, right: -40, width: 200, height: 200, borderRadius: 9999, background: "rgba(0,165,114,.30)", filter: "blur(60px)", animation: "bp-glow 6s ease-in-out infinite" }} />
              <div style={{ position: "relative", zIndex: 10 }}>
                <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 16, marginBottom: 14 }}>
                  <div>
                    <h3 style={{ fontSize: 22, fontWeight: 600, color: "#191c1d", margin: 0 }}>Benchmark Profile</h3>
                    <p style={{ margin: "4px 0 0", fontSize: 13, lineHeight: 1.4, color: "#707972" }}>Each radar axis is a cognitive ability. A longer spoke means this benchmark tests that ability more heavily. Hover a number or row for details.</p>
                  </div>
                  <span style={{ fontFamily: "'Geist Mono',ui-monospace,monospace", fontSize: 12, letterSpacing: ".05em", color: "#006b50" }}>{cur.name}</span>
                </div>

                {/* Benchmark selector chips */}
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6, paddingBottom: 20, marginBottom: 22, borderBottom: "1px solid rgba(0,107,80,.12)" }}>
                  {LANDING_BENCHMARKS.map((b) => {
                    const sel = b.id === cur.id;
                    return (
                      <button
                        key={b.id}
                        type="button"
                        onClick={() => setSelectedId(b.id)}
                        style={{
                          cursor: "pointer",
                          fontFamily: "'Geist Mono',ui-monospace,monospace",
                          fontSize: 11,
                          letterSpacing: ".02em",
                          padding: "5px 10px",
                          borderRadius: 6,
                          transition: "all .15s",
                          border: `1px solid ${sel ? "#006b50" : "rgba(0,107,80,.25)"}`,
                          background: sel ? "#006b50" : "rgba(255,255,255,.8)",
                          color: sel ? "#ffffff" : "#006b50",
                        }}
                      >
                        {b.name}
                      </button>
                    );
                  })}
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "0.95fr 1.05fr", gap: 36, alignItems: "start" }}>
                  {/* Left column: radar + axis legend */}
                  <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <div style={{ position: "relative", width: "100%", aspectRatio: "1 / 1", maxWidth: 340, display: "flex", alignItems: "center", justifyContent: "center" }}>
                        <svg id="bp-radar" viewBox="0 0 400 400" style={{ width: "100%", height: "100%", overflow: "visible", margin: "0 auto" }}>
                          <defs>
                            <filter id="bp-svg-glow">
                              <feGaussianBlur result="cb" stdDeviation="2" />
                              <feMerge>
                                <feMergeNode in="cb" />
                                <feMergeNode in="SourceGraphic" />
                              </feMerge>
                            </filter>
                          </defs>
                          <g fill="none" stroke="#006b50" strokeOpacity="0.1" strokeWidth="1">
                            <circle cx="200" cy="200" r="45" />
                            <circle cx="200" cy="200" r="90" />
                            <circle cx="200" cy="200" r="135" />
                            <circle cx="200" cy="200" r="180" />
                          </g>
                          <g stroke="#006b50" strokeDasharray="4 4" strokeOpacity="0.1" strokeWidth="1">
                            {RADAR_SPOKES.map((spoke, i) => (
                              <line key={i} x1="200" y1="200" x2={spoke.x2} y2={spoke.y2} />
                            ))}
                          </g>
                          <circle cx="200" cy="200" r="180" fill="#006b50" fillOpacity="0.03" stroke="#bfc9c2" strokeWidth="1" />
                          <path d={radarBlob(cur.w)} fill={accent} fillOpacity="0.18" stroke={accent} strokeWidth="2" strokeLinejoin="round" />
                          <g fill="#006b50" fontFamily="Geist Mono" fontSize="13" fontWeight="bold" textAnchor="middle" dominantBaseline="central">
                            {RADAR_LABELS.map((label, index) => (
                              <text key={label.n} x={label.x} y={label.y} tabIndex={0} onMouseEnter={() => setHoveredAxisIndex(index)} onMouseLeave={() => setHoveredAxisIndex(null)} onFocus={() => setHoveredAxisIndex(index)} onBlur={() => setHoveredAxisIndex(null)} fill={hoveredAxisIndex === index ? "#004d3a" : "#006b50"} style={{ cursor: "pointer" }}>{label.n}</text>
                            ))}
                          </g>
                        </svg>
                      </div>
                    </div>
                    <div aria-live="polite" style={{ minHeight: 78, borderRadius: 8, border: "1px solid rgba(0,165,114,.28)", borderLeft: "3px solid #00a572", background: "rgba(0,165,114,.06)", padding: "10px 12px", color: "#404943", fontSize: 13, lineHeight: 1.45 }}>
                      <strong style={{ color: "#006b50" }}>{"Axis " + shownAxis.num + ": " + shownAxis.name}</strong>
                      <div>{shownAxis.description}</div>
                      <div style={{ marginTop: 4, color: "#707972" }}>{"How strongly " + cur.name + " tests it: " + shownAxis.bar + (hoveredAxis ? "" : "  ·  hover any spoke or cognitive-ability row to explore the others")}</div>
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 5 }}>
                      {axisRows.map((a) => (
                        <div key={a.num} onMouseEnter={() => setHoveredAxisIndex(a.num - 1)} onMouseLeave={() => setHoveredAxisIndex(null)} title={"Axis " + a.num + ": " + a.name + ". " + a.description} style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", borderRadius: 6, padding: "2px 3px", background: hoveredAxisIndex === a.num - 1 ? "rgba(0,107,80,.06)" : "transparent" }}>
                          <span style={{ flex: "none", width: 16, height: 16, borderRadius: 9999, background: "#006b50", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, fontFamily: "'Geist Mono',ui-monospace,monospace" }}>{a.num}</span>
                          <span style={{ flex: 1, fontFamily: "'Geist Mono',ui-monospace,monospace", fontSize: 10, textTransform: "uppercase", letterSpacing: ".02em", color: "#404943" }}>{a.name}</span>
                          <div style={{ flex: "none", width: 64, height: 5, background: "rgba(0,107,80,.1)", borderRadius: 3, overflow: "hidden" }}>
                            <div style={{ height: "100%", borderRadius: 3, background: accent, width: a.bar }} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Right column: candidates + similar */}
                  <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
                    <div>
                      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12, justifyContent: "center" }}>
                        <span className="material-symbols-outlined" style={{ color: "#006b50", fontSize: 20 }}>analytics</span>
                        <h4 style={{ fontSize: 20, fontWeight: 600, color: "#191c1d", margin: 0 }}>Model Candidates</h4>
                        <InfoTip text="FIT (0–100) is how well a model's strengths match this benchmark's cognitive-ability profile, weighted by how heavily each cognitive ability is tested. Higher means a better-suited model." label="What FIT means" />
                      </div>
                      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                        {models.map((m) => (
                          <div key={m.name} style={{ width: "71.43%", margin: "0 auto", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, background: "rgba(239,245,242,.7)", padding: "10px 14px", borderRadius: 8, border: "1px solid rgba(191,201,194,.3)" }}>
                            <div style={{ minWidth: 0 }}>
                              <div style={{ fontSize: 14, fontWeight: 600, color: "#191c1d" }}>{m.name}</div>
                              <div style={{ fontFamily: "'Geist Mono',ui-monospace,monospace", fontSize: 9, letterSpacing: ".12em", textTransform: "uppercase", color: m.tagColor, marginTop: 2 }}>{m.tag}</div>
                            </div>
                            <div style={{ textAlign: "right", flex: "none" }}>
                              <span style={{ color: "#006b50", fontFamily: "'Geist Mono',ui-monospace,monospace", fontSize: 20, lineHeight: 1 }}>{m.score}</span>
                              <div style={{ fontFamily: "'Geist Mono',ui-monospace,monospace", fontSize: 8, letterSpacing: ".1em", color: "#707972" }}>FIT</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div>
                      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12, justifyContent: "center" }}>
                        <span className="material-symbols-outlined" style={{ color: "#00a572", fontSize: 20 }}>hub</span>
                        <h4 style={{ fontSize: 20, fontWeight: 600, color: "#191c1d", margin: 0 }}>Similar Benchmarks</h4>
                        <InfoTip text="The percentage is the similarity between two benchmarks' cognitive-ability profiles (cosine similarity). A higher value means they test very similar cognitive abilities. Click one to switch to it." label="What the similarity percentage means" />
                      </div>
                      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                        {similar.map((b) => (
                          <button
                            key={b.id}
                            type="button"
                            onClick={() => setSelectedId(b.id)}
                            className="bp-sim-btn"
                            style={{ width: "71.43%", margin: "0 auto", textAlign: "left", cursor: "pointer", background: "rgba(239,245,242,.7)", padding: "10px 14px", borderRadius: 8, border: "1px solid rgba(191,201,194,.3)", transition: "border-color .15s" }}
                          >
                            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
                              <span style={{ fontSize: 14, fontWeight: 600, color: "#191c1d" }}>{b.name}</span>
                              <span style={{ color: "#00a572", fontFamily: "'Geist Mono',ui-monospace,monospace", fontSize: 14 }}>{b.sim}</span>
                            </div>
                            <div style={{ marginTop: 8, height: 4, background: "rgba(0,107,80,.1)", borderRadius: 3, overflow: "hidden" }}>
                              <div style={{ height: "100%", borderRadius: 3, background: "#00a572", width: b.sim }} />
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </main>
      </div>

      {/* Footer */}
      <footer style={{ borderTop: "1px solid rgba(191,201,194,.2)", width: "100%", padding: "48px 0", position: "relative", zIndex: 10, background: "#fff" }}>
        <div style={{ display: "flex", flexDirection: "row", justifyContent: "space-between", alignItems: "center", width: "100%", padding: "0 48px", maxWidth: 1440, margin: "0 auto", gap: 24, flexWrap: "wrap" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <span style={{ fontSize: 18, fontWeight: 700, color: "#006b50" }}>BenchPress</span>
            <span style={{ fontFamily: "'Geist Mono',ui-monospace,monospace", fontSize: 12, letterSpacing: ".05em", color: "#404943" }}>© 2026 BenchPress Systems. Cognitive-Ability-Focused Model Evaluation &amp; Benchmarking.</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

function PageHeader({ title, desc }: { title: string; desc: string }) {
  return (
    <div className="mb-6 rounded-lg border border-neutral-200 bg-white p-5 shadow-sm">
      <h1 className="text-2xl font-bold tracking-tight text-neutral-950">{title}</h1>
      <p className="mt-1 text-sm leading-6 text-neutral-600">{desc}</p>
    </div>
  );
}

function Surface({ children, className = "", id }: { children: ReactNode; className?: string; id?: string }) {
  return <div id={id} className={`rounded-lg border border-neutral-200 bg-white p-4 shadow-sm ${className}`}>{children}</div>;
}

// Builder wizard stages: select (abilities + benchmarks) → review (scores +
// coverage) → compose (dataset + publish). Each stage renders only its own
// sections; the "On this page" index lists the current stage's sections.
type BuilderStage = "select" | "review" | "compose";
const BUILDER_STAGES: Array<{ id: BuilderStage; label: string; sub: string }> = [
  { id: "select", label: "Build", sub: "abilities & benchmarks" },
  { id: "review", label: "Review", sub: "scores & coverage" },
  { id: "compose", label: "Compose & publish", sub: "items & dataset" },
];
const BUILDER_INDEX: Array<{ id: string; label: string; stage: BuilderStage }> = [
  { id: "abilities", label: "Cognitive Abilities", stage: "select" },
  { id: "benchmarks", label: "Benchmarks", stage: "select" },
  { id: "expected-scores", label: "Expected scores", stage: "review" },
  { id: "coverage", label: "Coverage & provenance", stage: "review" },
  { id: "compose", label: "Compose dataset", stage: "compose" },
];

// Scroll spy: the active section is the last one whose heading has scrolled
// above the activation line (just below the 64px sticky header). Mirrors it
// into the URL hash (replaceState — no history entry, no route change since
// routing keys off pathname not hash).
function useActiveSection(ids: string[]): string {
  const [active, setActive] = useState<string>(ids[0] ?? "");
  const key = ids.join(",");
  useEffect(() => {
    const LINE = 96;
    let frame = 0;
    const compute = () => {
      frame = 0;
      let current = ids[0] ?? "";
      for (const id of ids) {
        const el = document.getElementById(id);
        if (el && el.getBoundingClientRect().top <= LINE) current = id;
      }
      if (!current) return;
      setActive(current);
      if (`#${current}` !== window.location.hash) window.history.replaceState(null, "", `#${current}`);
    };
    const onScroll = () => {
      if (!frame) frame = window.requestAnimationFrame(compute);
    };
    compute();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      if (frame) window.cancelAnimationFrame(frame);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, [key]);
  return active;
}

function BuilderIndex({ entries, activeId }: { entries: Array<{ id: string; label: string }>; activeId: string }) {
  return (
    <nav className="sticky top-16 rounded-lg border border-neutral-200 bg-white p-4 shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">On this page</p>
      <ul className="mt-2 space-y-1 text-sm">
        {entries.map((item) => {
          const active = item.id === activeId;
          return (
            <li key={item.id}>
              <a
                href={`#${item.id}`}
                aria-current={active ? "true" : undefined}
                className={`block rounded px-2 py-1 hover:bg-neutral-100 ${active ? "bg-neutral-100 font-semibold text-neutral-950" : "text-neutral-600 hover:text-neutral-950"}`}
              >
                {item.label}
              </a>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

// Interactive O——O——O stepper in the Builder hero — the wizard's navigation.
// Stage 1 is always clickable; Review and Compose unlock once Build set has
// run with a non-empty selection (canAdvance).
function StageStepper({
  stage,
  canAdvance,
  onSelect,
}: {
  stage: BuilderStage;
  canAdvance: boolean;
  onSelect: (next: BuilderStage) => void;
}) {
  const activeIndex = BUILDER_STAGES.findIndex((item) => item.id === stage);
  return (
    <div className="mt-5 flex items-center gap-3">
      {BUILDER_STAGES.map((item, index) => {
        const active = index === activeIndex;
        const completed = index < activeIndex;
        const reachable = active || index === 0 || canAdvance;
        return (
          <Fragment key={item.id}>
            {index > 0 ? (
              <span aria-hidden="true" className={`h-px flex-1 ${index <= activeIndex ? "bg-neutral-950" : "bg-neutral-200"}`} />
            ) : null}
            <button
              type="button"
              onClick={() => onSelect(item.id)}
              disabled={!reachable}
              aria-current={active ? "step" : undefined}
              title={reachable ? undefined : "Build set first"}
              className={`flex items-center gap-2.5 text-left ${reachable ? "" : "cursor-not-allowed"}`}
            >
              <span
                className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full border text-sm font-bold ${
                  active || completed
                    ? "border-neutral-950 bg-neutral-950 text-white"
                    : reachable
                      ? "border-neutral-950 bg-white text-neutral-950"
                      : "border-neutral-200 bg-white text-neutral-300"
                }`}
              >
                {completed ? "✓" : index + 1}
              </span>
              <span className="min-w-0">
                <span className={`block text-sm leading-5 ${active ? "font-bold text-neutral-950" : reachable ? "font-medium text-neutral-700" : "font-medium text-neutral-300"}`}>{item.label}</span>
                <span className={`block text-xs leading-4 ${reachable ? "text-neutral-500" : "text-neutral-300"}`}>{item.sub}</span>
              </span>
            </button>
          </Fragment>
        );
      })}
    </div>
  );
}

function LoadingPanel({ label }: { label: string }) {
  return <div className="rounded-lg border border-neutral-200 bg-neutral-50 p-4 text-sm text-neutral-500">Loading {label}...</div>;
}

// Minimal centered modal: Esc or a backdrop click closes; the panel scrolls
// internally when the content is taller than 80vh. No portal — z-50 sits above
// the sticky header.
function Modal({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} aria-hidden="true" />
      <div role="dialog" aria-modal="true" aria-label={title} className="relative max-h-[80vh] w-full max-w-2xl overflow-y-auto rounded-lg bg-white p-5 shadow">
        <div className="flex items-start justify-between gap-4">
          <h2 className="text-lg font-semibold text-neutral-950">{title}</h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded-md px-1.5 py-0.5 text-sm text-neutral-500 hover:bg-neutral-100 hover:text-neutral-900"
          >
            ✕
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

// Expected-scores table body, shared by the inline top-10 view and the
// view-all modal.
function ExpectedScoreTable({ rows }: { rows: ExpectedScoreRow[] }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-neutral-200 bg-white">
      <table className="w-full min-w-[420px] border-collapse text-sm">
        <thead>
          <tr className="border-b border-neutral-200 bg-neutral-50 text-left text-xs uppercase tracking-wide text-neutral-500">
            <th className="px-3 py-2 font-semibold">Model</th>
            <th className="px-3 py-2 font-semibold">Vendor</th>
            <th className="px-3 py-2 text-right font-semibold">Expected score</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id} className="border-b border-neutral-100 last:border-b-0">
              <td className="px-3 py-2.5 font-medium text-neutral-900">
                <div className="flex items-center gap-2">
                  <ModelAvatar modelId={row.id} vendor={row.vendor} />
                  <span>{row.name}</span>
                </div>
              </td>
              <td className="px-3 py-2.5 text-neutral-600">{row.vendor}</td>
              <td className="px-3 py-2.5 text-right tabular-nums text-neutral-900">{row.score.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// 20px round model avatar: tries <base>logos/<model_id>.png and falls back to
// the vendor color dot when the logo file is absent.
function ModelAvatar({ modelId, vendor }: { modelId: string; vendor: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) {
    return <span aria-hidden="true" className="inline-block h-5 w-5 shrink-0 rounded-full" style={{ background: vendorSwatch(vendor) }} />;
  }
  return (
    <img
      src={`${import.meta.env.BASE_URL}logos/${modelId}.png`}
      alt=""
      className="h-5 w-5 shrink-0 rounded-full border border-neutral-200 bg-white object-contain"
      onError={() => setFailed(true)}
    />
  );
}

type CompositionExample = (typeof compositionExamples)[number];

function ExampleCard({ example }: { example: CompositionExample }) {
  const [copied, setCopied] = useState(false);
  return (
    <article className="flex flex-col rounded-lg border border-neutral-200 bg-neutral-50 p-4">
      <a
        href={`https://huggingface.co/datasets/${example.repo_id}`}
        target="_blank"
        rel="noreferrer"
        className="font-semibold text-neutral-950 hover:underline"
      >
        {example.title}
      </a>
      <p className="mt-1 text-xs leading-5 text-neutral-500">{example.description}</p>
      <div className="relative mt-3">
        <pre className="overflow-x-auto rounded-md bg-neutral-950 px-3 py-2.5 text-xs leading-5 text-neutral-100">{example.snippet}</pre>
        <button
          type="button"
          onClick={() => {
            navigator.clipboard
              .writeText(example.snippet)
              .then(() => {
                setCopied(true);
                window.setTimeout(() => setCopied(false), 1500);
              })
              .catch(() => {});
          }}
          className="absolute right-2 top-2 rounded border border-neutral-700 bg-neutral-900 px-2 py-0.5 text-xs font-medium text-neutral-200 hover:bg-neutral-800"
        >
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
    </article>
  );
}

// Editable radar over n axes: drag a vertex (or any point along a spoke) to
// set that axis' value in 0.05 steps. Optional overlay renders a second,
// non-interactive profile (a model) for comparison. The legend pairing the
// numbered spokes with axis names is rendered by the parent.
function InteractiveRadar({
  axes,
  values,
  onChange,
  overlay = null,
  overlayLabel = null,
}: {
  axes: Array<{ id: string; name: string }>;
  values: Record<string, number>;
  onChange: (id: string, v: number) => void;
  overlay?: number[] | null;
  overlayLabel?: string | null;
}) {
  const activeAxisRef = useRef<number | null>(null);
  const n = axes.length;
  const angleFor = (i: number) => ((-90 + i * (360 / n)) * Math.PI) / 180;
  const pointFor = (value: number, i: number): [number, number] => {
    const ang = angleFor(i);
    // Keep an empty inner hub: even at value 0 the handle rests on a spread-out
    // ring (not stacked at the center) so each axis stays easy to grab and drag.
    const r = Math.max(0.16, Math.min(1, value)) * 180;
    return [200 + r * Math.cos(ang), 200 + r * Math.sin(ang)];
  };
  const userPoints = axes.map((axis, i) => pointFor(values[axis.id] ?? 0, i));
  const overlayPoints = overlay ? overlay.map((value, i) => pointFor(value, i)) : null;

  function toSvgCoords(event: React.PointerEvent<SVGSVGElement>): [number, number] {
    const rect = event.currentTarget.getBoundingClientRect();
    return [
      ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 400,
      ((event.clientY - rect.top) / Math.max(rect.height, 1)) * 400,
    ];
  }

  // Project the pointer onto the active axis' unit vector → 0..1 in 0.05 steps.
  function applyPointer(event: React.PointerEvent<SVGSVGElement>, axisIndex: number) {
    const [px, py] = toSvgCoords(event);
    const ang = angleFor(axisIndex);
    const raw = ((px - 200) * Math.cos(ang) + (py - 200) * Math.sin(ang)) / 180;
    const value = Math.max(0, Math.min(1, Math.round(raw * 20) / 20));
    onChange(axes[axisIndex].id, value);
  }

  function handlePointerDown(event: React.PointerEvent<SVGSVGElement>) {
    const [px, py] = toSvgCoords(event);
    const pointerAngle = Math.atan2(py - 200, px - 200);
    let nearest = 0;
    let nearestDelta = Infinity;
    for (let i = 0; i < n; i++) {
      const diff = Math.abs(pointerAngle - angleFor(i)) % (2 * Math.PI);
      const delta = Math.min(diff, 2 * Math.PI - diff);
      if (delta < nearestDelta) {
        nearestDelta = delta;
        nearest = i;
      }
    }
    activeAxisRef.current = nearest;
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handlePointerMove(event: React.PointerEvent<SVGSVGElement>) {
    const axisIndex = activeAxisRef.current;
    if (axisIndex == null) return;
    applyPointer(event, axisIndex);
  }

  function handlePointerEnd(event: React.PointerEvent<SVGSVGElement>) {
    activeAxisRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  return (
    <svg
      viewBox="0 0 400 400"
      className="h-auto w-full max-w-[380px] select-none"
      style={{ touchAction: "none" }}
      role="img"
      aria-label={overlayLabel ? `Target cognitive-ability profile compared with ${overlayLabel}` : "Target cognitive-ability profile"}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerEnd}
      onPointerCancel={handlePointerEnd}
    >
      <g fill="none" stroke="#e5e5e5" strokeWidth="1">
        <circle cx="200" cy="200" r="45" />
        <circle cx="200" cy="200" r="90" />
        <circle cx="200" cy="200" r="135" />
        <circle cx="200" cy="200" r="180" />
        {axes.map((axis, i) => {
          const [tx, ty] = [200 + 180 * Math.cos(angleFor(i)), 200 + 180 * Math.sin(angleFor(i))];
          return <line key={axis.id} x1="200" y1="200" x2={tx.toFixed(1)} y2={ty.toFixed(1)} />;
        })}
      </g>
      <g fill="#a3a3a3" fontSize="11" textAnchor="middle" dominantBaseline="central">
        {axes.map((axis, i) => {
          const [lx, ly] = [200 + 194 * Math.cos(angleFor(i)), 200 + 194 * Math.sin(angleFor(i))];
          return (
            <text key={axis.id} x={lx.toFixed(1)} y={ly.toFixed(1)}>
              {i + 1}
            </text>
          );
        })}
      </g>
      {overlayPoints ? (
        <path
          d={radarPath(overlayPoints)}
          fill="rgba(0,165,114,0.12)"
          stroke="#00a572"
          strokeWidth="2"
          strokeLinejoin="round"
          pointerEvents="none"
        />
      ) : null}
      <path d={radarPath(userPoints)} fill="#0a0a0a" fillOpacity="0.08" stroke="#0a0a0a" strokeWidth="2" strokeLinejoin="round" />
      {userPoints.map(([x, y], i) => (
        <circle key={axes[i].id} cx={x.toFixed(1)} cy={y.toFixed(1)} r="6" fill="#ffffff" stroke="#0a0a0a" strokeWidth="2" className="cursor-grab" />
      ))}
    </svg>
  );
}

function BuilderPage() {
  // Ability picker: toggling a domain category selects all of its learned tags.
  // Nothing starts selected — the user opts into the abilities they care about.
  const [selectedAxes, setSelectedAxes] = useState<string[]>([]);
  // Ability input mode: pick discrete tags, or shape a target profile on the radar.
  const [mode, setMode] = useState<"tags" | "profile">("tags");
  const [profileValues, setProfileValues] = useState<Record<string, number>>(() =>
    Object.fromEntries(axisWeights.axes.map((axis) => [axis.id, 0])),
  );
  // Model whose tag profile is overlaid on the radar for comparison (profile mode).
  const [overlayModelId, setOverlayModelId] = useState<string | null>(null);
  // Benchmarks the user has picked for their set (drives models, coverage, compose).
  const [pickedIds, setPickedIds] = useState<string[]>([]);
  // Review/compose stages stay unreachable until the first Build set click;
  // afterwards they live-update as the selection changes.
  const [generated, setGenerated] = useState(false);
  // Wizard stage: select (abilities + benchmarks) → review → compose.
  // Every Build set click restarts at review.
  const [stage, setStage] = useState<BuilderStage>("select");
  // Compose-dataset section: item count sampled from each selected benchmark,
  // set independently per benchmark (defaults to 100 when not explicitly set).
  const [itemsByBenchmark, setItemsByBenchmark] = useState<Record<string, number>>({});
  const itemCount = (benchId: string) => itemsByBenchmark[benchId] ?? 100;
  // Compose-dataset publish: name for the published composition, in-flight flag,
  // and the last publish response (success carries repo_id/url/references).
  const [composeName, setComposeName] = useState("");
  const [publishing, setPublishing] = useState(false);
  const [publishResult, setPublishResult] = useState<{ ok: boolean; url?: string; repo_id?: string; references?: Record<string, number>; error?: string } | null>(null);
  // Expected-scores table: top 10 inline; the full list opens in a modal.
  const [expectedModalOpen, setExpectedModalOpen] = useState(false);
  // Quick-start presets are folded away by default (accordion) to keep the top
  // of the Builder compact; the user expands it when they want a preset.
  const [quickStartOpen, setQuickStartOpen] = useState(false);

  // Full coverage grid powers the Coverage & provenance result section.
  const [coverageData, setCoverageData] = useState<CoverageData | null>(null);
  // Builder only recommends deterministic benchmarks that have at least one
  // score row, so users do not select unscored candidates by default.
  const [selectableBenchmarkIds, setSelectableBenchmarkIds] = useState<Set<string> | null>(null);
  const [expectedScores, setExpectedScores] = useState<ExpectedScoreRow[] | null>([]);
  useEffect(() => {
    let alive = true;
    buildCoverageDataBrowser().then((data) => {
      if (alive) setCoverageData(data);
    });
    return () => { alive = false; };
  }, []);
  useEffect(() => {
    let alive = true;
    buildDeterministicScoredBenchmarkIdsBrowser().then((ids) => {
      if (alive) setSelectableBenchmarkIds(ids);
    });
    return () => { alive = false; };
  }, []);

  const activeAxes = axisWeights.axes.filter((axis) => selectedAxes.includes(axis.id));

  // Target profile as a vector in axis order (profile mode's similarity space).
  const profileVector = useMemo(
    () => axisWeights.axes.map((axis) => profileValues[axis.id] ?? 0),
    [profileValues],
  );
  const profileIsEmpty = profileVector.every((value) => value === 0);

  const rankedBenchmarks = useMemo(() => {
    const keepScoredDeterministic = (
      items: Array<{ bench: (typeof axisWeights.benchmarks)[number]; relevance: number; coverageBreadth: number; utility: number }>,
    ) => selectableBenchmarkIds ? items.filter((item) => selectableBenchmarkIds.has(canonicalBenchmarkId(item.bench.id))) : items;

    if (mode === "profile") {
      if (profileIsEmpty) return [];
      return keepScoredDeterministic(axisWeights.benchmarks
        .map((bench) => {
          const benchVector = axisWeights.axes.map((axis) => (bench.weights as Record<string, number>)[axis.id] ?? 0);
          const relevance = cosine(benchVector, profileVector);
          return { bench, relevance, coverageBreadth: 0, utility: relevance };
        })
        .sort((a, b) => {
          if (b.relevance !== a.relevance) return b.relevance - a.relevance;
          return a.bench.name.localeCompare(b.bench.name);
        }));
    }
    const axisSet = new Set(selectedAxes);
    return keepScoredDeterministic(axisWeights.benchmarks
      .map((bench) => {
        const axisValues = selectedAxes.map((axisId) => bench.weights[axisId] ?? 0);
        const relevance = axisValues.reduce((sum, value) => sum + value, 0) / Math.max(axisValues.length, 1);
        const coverageBreadth = Object.entries(bench.weights).filter(([axisId, value]) => axisSet.has(axisId) && value >= 0.5).length / Math.max(selectedAxes.length, 1);
        const utility = relevance * 0.72 + coverageBreadth * 0.2;
        return { bench, relevance, coverageBreadth, utility };
      })
      .filter((item) => item.relevance > 0.05)
      .sort((a, b) => {
        if (b.utility !== a.utility) return b.utility - a.utility;
        if (b.relevance !== a.relevance) return b.relevance - a.relevance;
        return a.bench.cost - b.bench.cost;
      }));
  }, [selectedAxes, mode, profileVector, profileIsEmpty, selectableBenchmarkIds]);

  // The user's set = ranked benchmarks they picked (kept in rank order). Models,
  // coverage and the composition all derive from this.
  const pickedSet = new Set(pickedIds);
  const selectedBenchmarkItems = rankedBenchmarks.filter((item) => pickedSet.has(item.bench.id));
  const selectedBenchmarks = selectedBenchmarkItems.map((item) => item.bench.id);
  const selectedBenchmarksKey = selectedBenchmarks.join("|");
  useEffect(() => {
    let alive = true;
    if (selectedBenchmarks.length === 0) {
      setExpectedScores([]);
      return () => { alive = false; };
    }
    setExpectedScores(null);
    buildExpectedScoresBrowser(selectedBenchmarks).then((rows) => {
      if (alive) setExpectedScores(rows);
    });
    return () => { alive = false; };
  }, [selectedBenchmarksKey]);

  // Profile mode: every exported model ranked by similarity to the target profile.
  const rankedProfileModels = useMemo(
    () =>
      modelTagProfiles.models
        .map((model) => {
          const vector = axisWeights.axes.map((axis) => model.profile[axis.id] ?? 0);
          return { model, similarity: cosine(vector, profileVector) };
        })
        .sort((a, b) => {
          if (b.similarity !== a.similarity) return b.similarity - a.similarity;
          return a.model.name.localeCompare(b.model.name);
        }),
    [profileVector],
  );
  const overlayModel = overlayModelId ? rankedProfileModels.find(({ model }) => model.id === overlayModelId)?.model ?? null : null;
  const overlayVector = overlayModel ? axisWeights.axes.map((axis) => overlayModel.profile[axis.id] ?? 0) : null;

  // Scroll spy over the current stage's sections (the only ones in the DOM).
  const activeSectionIds = useMemo(() => BUILDER_INDEX.filter((item) => item.stage === stage).map((item) => item.id), [stage]);
  const activeSection = useActiveSection(activeSectionIds);
  // The index sidebar lists the current stage's sections; single-section
  // stages (compose) hide it and let the main column run full-width.
  const indexEntries = BUILDER_INDEX.filter((item) => item.stage === stage);
  const showIndex = indexEntries.length >= 2;

  // Validation reacts to the selected axes (via coverage) and the budget k.
  const weightedCoverage = useMemo(() => {
    return activeAxes.map((axis) => {
      const values = selectedBenchmarks.map((benchId) => benchById.get(benchId)?.weights[axis.id] ?? 0);
      const value = values.reduce((sum, item) => sum + item, 0) / Math.max(values.length, 1);
      return { axis, value };
    });
  }, [activeAxes, selectedBenchmarks]);

  const expectedScoreRows = expectedScores ?? [];
  const resultBenchmarkCount = selectedBenchmarkItems.length;

  // "What this suite measures": target abilities ranked by how strongly the
  // selected benchmarks exercise them, so the results open with a plain-language
  // read of the suite's strengths and blind spots rather than a wall of numbers.
  const rankedAbilities = useMemo(
    () => [...weightedCoverage].sort((a, b) => b.value - a.value),
    [weightedCoverage],
  );
  const bestModelRow = recommendationCards.find((card) => card.key === "best_overall")?.row ?? null;
  const strongAbilities = rankedAbilities.filter((item) => item.value >= 0.4).slice(0, 3);
  const strongShown = strongAbilities.length ? strongAbilities : rankedAbilities.slice(0, 2);
  const weakAbilities = rankedAbilities.filter((item) => item.value < 0.2 && !strongShown.includes(item)).slice(-2);

  function toggleAxis(axisId: string) {
    setSelectedAxes((current) => {
      if (current.includes(axisId)) return current.filter((id) => id !== axisId);
      return [...current, axisId];
    });
  }

  // Toggling a domain category selects/deselects every ability tag under it:
  // if all of its tags are already selected they are removed, otherwise the
  // missing ones are added.
  function toggleGroup(label: string) {
    const group = groupedAxes.find((item) => item.label === label);
    const groupAxisIds = group ? group.axes.map((axis) => axis.id) : [];
    setSelectedAxes((current) => {
      const allSelected = groupAxisIds.length > 0 && groupAxisIds.every((id) => current.includes(id));
      if (allSelected) return current.filter((id) => !groupAxisIds.includes(id));
      return [...current, ...groupAxisIds.filter((id) => !current.includes(id))];
    });
  }

  function applyBuilderPreset(axisIds: readonly string[]) {
    const availableAxisIds = new Set(axisWeights.axes.map((axis) => axis.id));
    const nextAxes = axisIds.filter((axisId) => availableAxisIds.has(axisId));
    setMode("tags");
    setSelectedAxes(nextAxes.length ? [...nextAxes] : axisWeights.axes.map((axis) => axis.id));
    setGenerated(false);
  }

  // Stage navigation guard: review/compose are reachable only after Build set
  // with a non-empty selection. Every stage switch scrolls back to the top so
  // the stepper and the new stage's content are in view.
  const canAdvance = generated && selectedBenchmarks.length > 0;
  function selectStage(next: BuilderStage) {
    if (next === stage) return;
    if (next !== "select" && !canAdvance) return;
    setStage(next);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function togglePick(benchId: string) {
    setPickedIds((current) => (current.includes(benchId) ? current.filter((id) => id !== benchId) : [...current, benchId]));
  }

  const allPicked = rankedBenchmarks.length > 0 && rankedBenchmarks.every((item) => pickedSet.has(item.bench.id));
  function toggleAllPicks() {
    setPickedIds(allPicked ? [] : rankedBenchmarks.map((item) => item.bench.id));
  }

  // Publish the composition to the local Composer backend. Benchmarks keyed by
  // display name, item counts clamped to each benchmark's cap.
  async function handlePublish() {
    const selections = Object.fromEntries(
      selectedBenchmarkItems.map(({ bench }) => [bench.name, Math.min(itemCount(bench.id), benchmarkItemCap(bench.id))]),
    );
    setPublishing(true);
    setPublishResult(null);
    try {
      const res = await fetch(PUBLISH_API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: composeName.trim(), selections, abilities: selectedAxes }),
      });
      const data = await res.json();
      setPublishResult(data);
    } catch {
      setPublishResult({ ok: false, error: "게시 서버에 연결하지 못했습니다. 로컬 Composer(localhost:7860)가 실행 중인지 확인하세요." });
    } finally {
      setPublishing(false);
    }
  }

  return (
    <section className="space-y-4">
      {/* Header: compact copy + the wizard stepper (the only stage navigation). */}
      <div className="rounded-lg border border-neutral-200 bg-white p-5 shadow-sm">
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">{BUILDER_COPY.eyebrow}</p>
          <h1 className="mt-1 text-2xl font-bold tracking-tight text-neutral-950">{BUILDER_COPY.title}</h1>
          <StageStepper stage={stage} canAdvance={canAdvance} onSelect={selectStage} />
        </div>
      </div>

      {stage === "select" ? (
      <Surface className="p-5">
        <button
          type="button"
          onClick={() => setQuickStartOpen((open) => !open)}
          aria-expanded={quickStartOpen}
          className="flex w-full flex-wrap items-center justify-between gap-3 text-left"
        >
          <span className="flex items-center gap-2">
            <span aria-hidden="true" className={`text-2xl leading-none text-neutral-500 transition-transform ${quickStartOpen ? "rotate-180" : ""}`}>▾</span>
            <span className="text-[16px] font-semibold text-neutral-950">Quick start</span>
          </span>
          <span className="rounded-full bg-neutral-100 px-3 py-1 text-xs font-medium text-neutral-600">{selectedAxes.length} cognitive abilities selected</span>
        </button>
        {quickStartOpen ? (
        <div className="mt-4">
          <p className="text-sm font-semibold text-neutral-900">What are you trying to evaluate?</p>
          <p className="mt-1 max-w-3xl text-xs leading-5 text-neutral-500">Pick a preset to auto-select cognitive abilities, then fine-tune them below.</p>
          <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-5">
          {BUILDER_PRESETS.map((preset) => (
            <button
              key={preset.label}
              type="button"
              onClick={() => applyBuilderPreset(preset.axisIds)}
              className="rounded-lg border border-neutral-200 bg-white p-3 text-left transition hover:border-neutral-500 hover:bg-neutral-50"
            >
              <span className="block text-sm font-semibold text-neutral-950">{preset.label}</span>
              <span className="mt-1 block text-xs leading-5 text-neutral-500">{preset.text}</span>
            </button>
          ))}
          </div>
        </div>
        ) : null}
      </Surface>
      ) : null}

      {/* Body: index sidebar (section titles) + main content. Grid items stretch
          (no items-start) so the aside spans the full body height and its sticky
          nav can travel the whole scroll. Single-section stages (compose) skip
          the sidebar and let the main column run full-width. */}
      <div className={showIndex ? "lg:grid lg:grid-cols-[11rem_minmax(0,1fr)] lg:gap-4" : undefined}>
      {showIndex ? (
      <aside className="mb-4 hidden lg:mb-0 lg:block">
        <BuilderIndex entries={indexEntries} activeId={activeSection} />
      </aside>
      ) : null}
      <div className="min-w-0 space-y-4">
      {/* Select stage: abilities + controls | benchmark picker (Build set lives
          in its footer). Review and compose render instead of, not below, it. */}
      {stage === "select" ? (
      <div className="grid gap-4 lg:grid-cols-2">
        <Surface id="abilities" className="scroll-mt-16">
          <h2 className="text-sm font-semibold text-neutral-950"><span className="mr-1.5 inline-flex h-5 w-5 items-center justify-center rounded-full bg-neutral-950 text-[11px] font-bold text-white">1</span>Cognitive Abilities</h2>
          <div className="mt-2 flex gap-1.5">
            {([["tags", "Pick cognitive abilities"], ["profile", "Shape profile"]] as const).map(([id, label]) => (
              <button
                key={id}
                type="button"
                onClick={() => setMode(id)}
                className={`rounded-md border px-2.5 py-1.5 text-xs font-medium leading-4 transition ${
                  mode === id ? "border-neutral-950 bg-neutral-950 text-white" : "border-neutral-300 bg-white text-neutral-600 hover:border-neutral-400 hover:text-neutral-900"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          {mode === "profile" ? (
            <>
              <p className="mt-2 text-xs leading-5 text-neutral-500">Drag the vertices to shape the cognitive-ability profile you need.</p>
              <div className="mt-3 flex flex-col items-center">
                <InteractiveRadar
                  axes={axisWeights.axes.map((axis) => ({ id: axis.id, name: axis.name }))}
                  values={profileValues}
                  onChange={(id, value) => setProfileValues((current) => ({ ...current, [id]: value }))}
                  overlay={overlayVector}
                  overlayLabel={overlayModel?.name ?? null}
                />
                {overlayModel ? (
                  <p className="mt-1 text-xs text-neutral-500">
                    Comparing: <span className="font-medium" style={{ color: "#00a572" }}>{overlayModel.name}</span> (click the model again to clear)
                  </p>
                ) : null}
                <ol className="mt-3 w-full space-y-1 text-xs text-neutral-600">
                  {axisWeights.axes.map((axis, index) => (
                    <li key={axis.id}>
                      <span className="font-semibold text-neutral-500">{index + 1}.</span> {axis.name} —{" "}
                      <span className="tabular-nums">{(profileValues[axis.id] ?? 0).toFixed(2)}</span>
                    </li>
                  ))}
                </ol>
                <div className="mt-4 w-full border-t border-neutral-200 pt-3">
                  <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">Closest models</p>
                  <p className="mt-1 text-xs leading-5 text-neutral-500">Models whose cognitive-ability profile is closest to your target. Click one to overlay it on the radar.</p>
                  {rankedProfileModels.length ? (
                    <ul className="mt-2 max-h-[16rem] space-y-1.5 overflow-y-auto pr-1">
                      {rankedProfileModels.map(({ model, similarity }) => {
                        const active = overlayModelId === model.id;
                        return (
                          <li key={model.id}>
                            <button
                              type="button"
                              onClick={() => setOverlayModelId((current) => (current === model.id ? null : model.id))}
                              className={`flex w-full items-center justify-between gap-2 rounded-md border px-2.5 py-1.5 text-left transition ${
                                active ? "border-neutral-950 bg-neutral-50" : "border-neutral-200 hover:border-neutral-400"
                              }`}
                            >
                              <span className="flex min-w-0 items-center gap-2">
                                <ModelAvatar modelId={model.id} vendor={model.vendor} />
                                <span className="min-w-0">
                                  <span className="block truncate text-sm font-medium text-neutral-900">{model.name}</span>
                                  <span className="block text-xs text-neutral-500">{model.vendor}</span>
                                </span>
                              </span>
                              <span className="text-sm font-semibold tabular-nums text-neutral-900">{similarity.toFixed(2)}</span>
                            </button>
                          </li>
                        );
                      })}
                    </ul>
                  ) : (
                    <p className="mt-2 text-sm text-neutral-400">Model profiles not exported yet.</p>
                  )}
                </div>
              </div>
            </>
          ) : (
          <>
          <p className="mt-2 text-xs leading-5 text-neutral-500">Pick the categories you want to evaluate — each one targets a set of learned cognitive-ability tags. Benchmarks update as you select.</p>
          <div className="mt-3 flex flex-wrap gap-1.5">
            {groupedAxes.map((group) => {
              const allSelected = group.axes.every((axis) => selectedAxes.includes(axis.id));
              return (
                <button
                  key={group.label}
                  type="button"
                  onClick={() => toggleGroup(group.label)}
                  aria-pressed={allSelected}
                  className={`rounded-md border px-2.5 py-1.5 text-left text-xs font-medium leading-4 transition ${
                    allSelected ? "border-neutral-950 bg-neutral-950 text-white" : "border-neutral-300 bg-white text-neutral-600 hover:border-neutral-400 hover:text-neutral-900"
                  }`}
                >
                  {group.label}
                </button>
              );
            })}
          </div>
          <div className="mt-4 border-t border-neutral-200 pt-3">
            <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">Targeted abilities</p>
            <p className="mt-1 text-xs leading-5 text-neutral-500">Learned cognitive-ability tags derived from model×benchmark score patterns — your categories imply these. Fine-tune them after building.</p>
            {activeAxes.length ? (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {activeAxes.map((axis) => (
                  <span key={axis.id} className="inline-flex items-center gap-1 rounded-md border border-neutral-200 bg-neutral-50 px-2 py-1 text-xs font-medium leading-4 text-neutral-600">
                    {axis.name}
                    <InfoTip text={axis.description} label={`About ${axis.name}`} />
                  </span>
                ))}
              </div>
            ) : (
              <p className="mt-2 text-xs leading-4 text-neutral-400">Pick a category above to target its abilities.</p>
            )}
          </div>
          </>
          )}
        </Surface>

        <Surface id="benchmarks" className="scroll-mt-16">
          <div className="flex items-baseline justify-between gap-2">
            <h2 className="text-sm font-semibold text-neutral-950"><span className="mr-1.5 inline-flex h-5 w-5 items-center justify-center rounded-full bg-neutral-950 text-[11px] font-bold text-white">2</span>Benchmarks</h2>
            {rankedBenchmarks.length ? (
              <button type="button" onClick={toggleAllPicks} className="text-xs font-medium text-neutral-500 hover:text-neutral-900">
                {allPicked ? "Clear" : "Select all"}
              </button>
            ) : null}
          </div>
          <p className="mt-1 text-xs leading-5 text-neutral-500">Select the benchmarks to include in your set. The green bar shows each benchmark's relevance to your chosen cognitive abilities — how strongly it exercises them, measured by cosine similarity. A fuller bar means a closer match, and benchmarks are listed most-relevant first.</p>
          {rankedBenchmarks.length ? (
            <ul className="mt-3 max-h-[34rem] space-y-1.5 overflow-y-auto pr-1">
              {rankedBenchmarks.map(({ bench, relevance }) => {
                const picked = pickedSet.has(bench.id);
                return (
                  <li key={bench.id}>
                    <button
                      type="button"
                      onClick={() => togglePick(bench.id)}
                      className={`flex w-full items-center gap-2.5 rounded-md border px-2.5 py-2 text-left transition ${
                        picked ? "border-neutral-950 bg-neutral-50" : "border-neutral-200 hover:border-neutral-400"
                      }`}
                    >
                      <span className={`flex h-4 w-4 shrink-0 items-center justify-center rounded border text-[10px] font-bold ${
                        picked ? "border-neutral-950 bg-neutral-950 text-white" : "border-neutral-300 text-transparent"
                      }`}>✓</span>
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-sm font-medium text-neutral-900">{bench.name}</span>
                        <span className="mt-1 block h-1.5 rounded-full bg-neutral-100">
                          <span className="block h-1.5 rounded-full bg-emerald-500" style={{ width: metricBarWidth(relevance) }} />
                        </span>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          ) : (
            <p className="mt-3 text-sm text-neutral-400">
              {mode === "profile" ? "Adjust the profile to see matches." : "Select at least one target category to see matching benchmarks."}
            </p>
          )}
          <div className="mt-4 flex flex-col items-end gap-1.5 border-t border-neutral-200 pt-3">
            <button
              type="button"
              onClick={() => { setGenerated(true); setStage("review"); window.scrollTo({ top: 0, behavior: "smooth" }); }}
              disabled={selectedBenchmarks.length === 0}
              title={selectedBenchmarks.length === 0 ? "Select at least one benchmark first" : undefined}
              className="rounded-md bg-neutral-950 px-5 py-2.5 text-sm font-semibold text-white hover:bg-neutral-800 disabled:cursor-not-allowed disabled:bg-neutral-300"
            >
              Build set
            </button>
            {selectedBenchmarks.length === 0 ? (
              <span className="rounded-full bg-neutral-100 px-3 py-1 text-xs font-medium text-neutral-600">Pick a benchmark to enable</span>
            ) : (
              <span className="rounded-full bg-neutral-100 px-3 py-1 text-xs font-medium text-neutral-600">{selectedBenchmarks.length} benchmark{selectedBenchmarks.length === 1 ? "" : "s"} selected</span>
            )}
          </div>
        </Surface>
      </div>
      ) : selectedBenchmarks.length === 0 ? (
        // Review/compose with an emptied set (toggling every ability off in
        // coverage can drain it): point the user back to the Build stage.
        <Surface className="p-5">
          <p className="text-sm text-neutral-500">Your set is empty — return to the Build stage and select benchmarks.</p>
        </Surface>
      ) : stage === "review" ? (
          <>
            <Surface id="expected-scores" className="scroll-mt-16 p-5">
              <h2 className="text-lg font-semibold text-neutral-950">Expected scores</h2>
              <p className="mt-1 text-sm leading-6 text-neutral-600">Reference models with a published score on every selected benchmark. The expected score is the mean of the raw scores; each benchmark contributes equally, regardless of its item count.</p>
              {expectedScoreRows.length ? (
                <>
                  <div className="mt-3">
                    <ExpectedScoreTable rows={expectedScoreRows.slice(0, 10)} />
                  </div>
                  {expectedScoreRows.length > 10 ? (
                    <button
                      type="button"
                      onClick={() => setExpectedModalOpen(true)}
                      className="mt-3 text-sm font-medium text-neutral-600 hover:text-neutral-900"
                    >
                      View all {expectedScoreRows.length} models
                    </button>
                  ) : null}
                  {expectedModalOpen ? (
                    <Modal title={`All ${expectedScoreRows.length} models`} onClose={() => setExpectedModalOpen(false)}>
                      <div className="mt-4">
                        <ExpectedScoreTable rows={expectedScoreRows} />
                      </div>
                    </Modal>
                  ) : null}
                </>
              ) : (
                <p className="mt-3 rounded-md border border-neutral-200 bg-neutral-50 px-3 py-2 text-sm text-neutral-600">{expectedScores === null ? "Loading reference scores..." : "Reference scores unavailable for this subset."}</p>
              )}
            </Surface>

            <Surface id="coverage" className="scroll-mt-16 p-5">
              <h2 className="text-lg font-semibold text-neutral-950">Coverage &amp; provenance</h2>
              <p className="mt-1 text-sm leading-6 text-neutral-600">See how strongly each selected benchmark exercises your target cognitive abilities.</p>
              <div className="mt-4">
                <h3 className="text-sm font-semibold text-neutral-950">Target abilities</h3>
                <p className="mt-1 text-xs leading-5 text-neutral-500">Cognitive-ability tags learned from model×benchmark score patterns — toggle a tag to refine what the ranking targets. Benchmarks and table columns update live.</p>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {axisWeights.axes.map((axis) => {
                    const active = selectedAxes.includes(axis.id);
                    return (
                      <button
                        key={axis.id}
                        type="button"
                        onClick={() => toggleAxis(axis.id)}
                        aria-pressed={active}
                        className={`inline-flex items-center gap-1 rounded-md border px-2 py-1 text-xs font-medium leading-4 transition ${
                          active ? "border-neutral-950 bg-neutral-950 text-white" : "border-neutral-300 bg-white text-neutral-600 hover:border-neutral-400 hover:text-neutral-900"
                        }`}
                      >
                        {axis.name}
                        <span onClick={(event) => event.stopPropagation()}>
                          <InfoTip text={axis.description} label={`About ${axis.name}`} />
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
              <div className="mt-4 rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-sm leading-6 text-emerald-950">
                <div className="font-semibold">What this suite measures</div>
                <p className="mt-1">
                  Across {resultBenchmarkCount} selected benchmark{resultBenchmarkCount === 1 ? "" : "s"}, coverage is strongest on{" "}
                  {strongShown.map((item, index) => (
                    <span key={item.axis.id}>
                      {index > 0 ? (index === strongShown.length - 1 ? " and " : ", ") : ""}
                      <span className="font-semibold">{item.axis.name}</span> ({pct(item.value)})
                    </span>
                  ))}
                  {weakAbilities.length ? (
                    <>
                      , and weakest on{" "}
                      {weakAbilities.map((item, index) => (
                        <span key={item.axis.id}>
                          {index > 0 ? " and " : ""}
                          <span className="font-semibold">{item.axis.name}</span> ({pct(item.value)})
                        </span>
                      ))}
                    </>
                  ) : null}
                  .
                </p>
                <p className="mt-2">
                  Use this suite to compare models on {strongShown.map((item) => item.axis.name).join(" and ").toLowerCase()}
                  {weakAbilities.length ? <>; it barely exercises {weakAbilities.map((item) => item.axis.name).join(" and ").toLowerCase()}, so do not draw conclusions there.</> : "."}
                  {bestModelRow ? <> Best-fitting model on this suite: <span className="font-semibold">{bestModelRow.model_name}</span>.</> : null}
                </p>
              </div>
              <div className="mt-5 space-y-6">
                <div>
                  <h3 className="text-sm font-semibold text-neutral-950">Why these benchmarks were selected</h3>
                  <p className="mt-1 text-xs leading-5 text-neutral-500">Each bar shows how strongly a benchmark exercises a target axis. <span className="font-semibold text-neutral-700">Utility</span> is the overall score used to rank and select benchmarks — it blends how strongly a benchmark tests your chosen cognitive abilities (relevance) with how many of them it covers well (breadth), so a higher utility means an all-round better fit for your suite. Cost is the relative run cost (1.0x = average).</p>
                  <div className="mt-3 overflow-x-auto rounded-lg border border-neutral-200 bg-white">
                    <table className="w-full min-w-[560px] border-collapse text-sm">
                      <thead>
                        <tr className="border-b border-neutral-200 bg-neutral-50 text-left text-xs uppercase tracking-wide text-neutral-500">
                          <th className="py-2 pl-3 pr-3 font-semibold">Benchmark</th>
                          {activeAxes.map((axis) => <th key={axis.id} className="px-2 py-2 font-semibold">{axis.name}</th>)}
                          <th className="px-2 py-2 font-semibold">Utility</th>
                          <th className="px-2 py-2 font-semibold">Cost</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selectedBenchmarkItems.map(({ bench, utility }, index) => (
                          <tr key={bench.id} className="border-b border-neutral-100 last:border-b-0">
                            <td className="py-3 pl-3 pr-3 font-medium text-neutral-900">
                              <div className="flex items-center gap-2">
                                <span className="rounded bg-neutral-100 px-1.5 py-0.5 text-xs font-semibold text-neutral-500">#{index + 1}</span>
                                <span>{bench.name}</span>
                              </div>
                            </td>
                            {activeAxes.map((axis) => {
                              const value = bench.weights[axis.id] ?? 0;
                              // Fill length is proportional to the weight (0 -> empty); a small
                              // floor keeps a nonzero value visible. The % sits outside the bar
                              // so the fill length stays honest instead of clamping to fit text.
                              return (
                                <td key={axis.id} className="px-2 py-3">
                                  <div className="flex items-center gap-2">
                                    <div className="h-2 min-w-[40px] flex-1 overflow-hidden rounded-full bg-neutral-100">
                                      <div className="h-2 rounded-full bg-emerald-500" style={{ width: `${value > 0 ? Math.max(4, Math.round(value * 100)) : 0}%` }} />
                                    </div>
                                    <span className="w-7 shrink-0 text-right text-xs font-medium tabular-nums text-neutral-600">{pct(value)}</span>
                                  </div>
                                </td>
                              );
                            })}
                            <td className="px-2 py-3 text-neutral-600 tabular-nums">{utility.toFixed(2)}</td>
                            <td className="px-2 py-3 text-neutral-600">{bench.cost.toFixed(1)}x</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </Surface>

            {/* Review stage exit: hands the reviewed set to the compose stage. */}
            <div className="flex w-full justify-end">
              <button
                type="button"
                onClick={() => selectStage("compose")}
                className="rounded-md bg-neutral-950 px-5 py-2.5 text-sm font-semibold text-white hover:bg-neutral-800"
              >
                Compose this set →
              </button>
            </div>
          </>
        ) : (
          <Surface id="compose" className="scroll-mt-16 p-5">
            <h2 className="text-lg font-semibold text-neutral-950">Compose dataset</h2>
            <p className="mt-1 text-sm leading-6 text-neutral-600">Turn the subset into a fixed, loadable evaluation set — set the item count for each benchmark. Publish guidance is right below.</p>
            <div className="mt-5 space-y-6">
              <div>
                <h3 className="text-sm font-semibold text-neutral-950">Your composition</h3>
                <p className="mt-1 text-xs leading-5 text-neutral-500">Set how many items each selected benchmark contributes to one loadable evaluation set.</p>
                <div className="mt-3 max-w-xs">
                  <label htmlFor="compose-name" className="block text-xs font-medium text-neutral-700">Composition name</label>
                  <input
                    id="compose-name"
                    type="text"
                    value={composeName}
                    onChange={(event) => setComposeName(event.target.value)}
                    placeholder="my-eval-mix"
                    className="mt-1 w-full rounded border border-neutral-300 px-2 py-1 text-sm"
                  />
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  {selectedBenchmarkItems.map(({ bench }) => {
                    const cap = benchmarkItemCap(bench.id);
                    return (
                    <div key={bench.id} className="inline-flex items-center gap-2 rounded-md border border-neutral-200 bg-white px-3 py-1.5">
                      <span className="text-sm font-medium text-neutral-700">{bench.name}</span>
                      <input
                        className="w-20 rounded border border-neutral-300 px-2 py-1 text-sm tabular-nums"
                        type="number"
                        value={Math.min(itemCount(bench.id), cap)}
                        min={1}
                        max={cap}
                        aria-label={`Items for ${bench.name} (max ${cap})`}
                        onChange={(event) => {
                          const next = Math.max(1, Math.min(cap, Math.round(Number(event.target.value)) || 1));
                          setItemsByBenchmark((current) => ({ ...current, [bench.id]: next }));
                        }}
                      />
                      <span className="text-xs text-neutral-400">/ {cap.toLocaleString("en-US")}</span>
                    </div>
                    );
                  })}
                </div>
                <div className="mt-4 flex items-center justify-between border-t border-neutral-200 pt-3">
                  <span className="text-xs font-semibold uppercase tracking-wide text-neutral-500">Total items</span>
                  <span className="text-lg font-semibold tabular-nums text-neutral-950">{selectedBenchmarkItems.reduce((sum, { bench }) => sum + Math.min(itemCount(bench.id), benchmarkItemCap(bench.id)), 0).toLocaleString("en-US")}</span>
                </div>
              </div>

              <div className="border-t border-neutral-200 pt-5">
                <h3 className="text-sm font-semibold text-neutral-950">Pre-published examples</h3>
                <p className="mt-1 text-xs leading-5 text-neutral-500">Compositions already published on Hugging Face — load one to see what a finished dataset looks like.</p>
                <div className="mt-3 grid gap-3 lg:grid-cols-3">
                  {compositionExamples.map((example) => <ExampleCard key={example.repo_id} example={example} />)}
                </div>
              </div>

              {PUBLISH_API_URL ? (
              <div className="border-t border-neutral-200 pt-5">
                <div className="flex flex-col items-end gap-2">
                  <button
                    type="button"
                    onClick={handlePublish}
                    disabled={publishing || composeName.trim() === ""}
                    className="rounded-md bg-neutral-950 px-5 py-2.5 text-sm font-semibold text-white hover:bg-neutral-800 disabled:cursor-not-allowed disabled:bg-neutral-300"
                  >
                    {publishing ? "Publishing…" : "Publish to my Hugging Face"}
                  </button>
                  <p className="text-xs leading-5 text-neutral-500">Publishes to your own Hugging Face namespace using your local token.</p>
                  <p className="text-xs leading-5 text-neutral-500">For self-hosted or reproducible use, the complete source is openly available in the project&rsquo;s <a href="https://github.com/SSU-NLP/Benchpress" target="_blank" rel="noreferrer" className="font-medium text-neutral-700 underline hover:text-neutral-900">GitHub repository</a>.</p>
                </div>
                {publishResult?.ok ? (
                  <div className="mt-4 rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-sm leading-6 text-emerald-950">
                    <div className="font-semibold">✅ Published</div>
                    <p className="mt-1">
                      <a href={publishResult.url} target="_blank" rel="noreferrer" className="font-medium underline hover:no-underline">{publishResult.repo_id}</a>
                    </p>
                    <pre className="mt-3 overflow-x-auto rounded-md bg-neutral-950 px-3 py-2.5 text-xs leading-5 text-neutral-100">{`from benchpress_hub import load_composition\nds = load_composition("${publishResult.repo_id}")`}</pre>
                    {publishResult.references && Object.keys(publishResult.references).length ? (
                      <div className="mt-3">
                        <p className="text-xs font-semibold uppercase tracking-wide text-emerald-800">Expected scores</p>
                        <ul className="mt-1 space-y-0.5">
                          {Object.entries(publishResult.references).slice(0, 5).map(([model, score]) => (
                            <li key={model} className="flex justify-between gap-4 text-xs">
                              <span>{model}</span>
                              <span className="tabular-nums">{score.toFixed(2)}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                  </div>
                ) : publishResult && !publishResult.ok ? (
                  <div className="mt-4 rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm leading-6 text-rose-900">
                    {publishResult.error}
                  </div>
                ) : null}
              </div>
              ) : (
              <div className="flex justify-end border-t border-neutral-200 pt-5">
                <div className="text-right">
                  <button
                    type="button"
                    disabled
                    aria-disabled="true"
                    title="Disabled — run the Composer locally instead (see below)"
                    className="inline-block cursor-not-allowed rounded-md bg-neutral-200 px-4 py-2 text-sm font-semibold text-neutral-400"
                  >
                    Publish via the live Composer
                  </button>
                  <p className="mt-3 text-xs font-semibold uppercase tracking-wide text-neutral-500">How to use this in your local</p>
                  <p className="mt-1 text-xs leading-5 text-neutral-500">Publishing to the live Composer is disabled: just in case, to protect your privacy, composing is available on your own machine only.</p>
                  <p className="mt-1 text-xs leading-5 text-neutral-500">For self-hosted or reproducible use, the complete source is openly available in the project&rsquo;s <a href="https://github.com/SSU-NLP/Benchpress" target="_blank" rel="noreferrer" className="font-medium text-neutral-700 underline hover:text-neutral-900">GitHub repository</a>:</p>
                  <p className="mt-0.5 pl-4 text-xs leading-5 text-neutral-500">clone the repository and run the Composer in your own environment.</p>
                </div>
              </div>
              )}
            </div>
          </Surface>
        )}
      </div>
      </div>
    </section>
  );
}

interface ScoreTab {
  id: CategorySlug;
  slug: string;
  label: string;
  note: string;
  group: "non-deterministic" | "deterministic" | "all";
  indent?: number;
  separator?: boolean;
}

const SCORE_CATEGORIES: ScoreTab[] = [
  { id: "non_deterministic", slug: "non-deterministic", label: "Judge-based", group: "non-deterministic", note: "Scores that can shift from run to run — human preference votes, LLM-as-a-judge, and simulation-based evaluations." },
  { id: "nd_preference", slug: "nd-preference", label: "↳ Preference", group: "non-deterministic", indent: 1, note: "Head-to-head preference / ELO leaderboards such as Chatbot Arena, AlpacaEval, Arena-Hard, and WildBench." },
  { id: "nd_agent", slug: "nd-agent", label: "↳ Agent", group: "non-deterministic", indent: 1, note: "Multi-turn LLM-agent tasks scored in a simulated environment." },
  { id: "nd_safety", slug: "nd-safety", label: "↳ Safety", group: "non-deterministic", indent: 1, note: "Safety and red-team evaluations whose scores depend on a judge." },
  { id: "nd_multilinguality", slug: "nd-multilinguality", label: "↳ Multilingual", group: "non-deterministic", indent: 1, note: "Preference / judge-based evaluations run in multiple languages." },
  { id: "nd_korean", slug: "nd-korean", label: "↳ Korean", group: "non-deterministic", indent: 1, note: "Korean-language judge-based evaluations." },
  { id: "deterministic", slug: "deterministic", label: "Fixed-answer", group: "deterministic", separator: true, note: "Reproducible scores that give the same number every run — from tech reports, system cards, and standard benchmarks with a fixed answer key." },
  { id: "general", slug: "general", label: "↳ General", group: "deterministic", indent: 1, note: "General knowledge and reasoning benchmarks (e.g. MMLU-style exams)." },
  { id: "math", slug: "math", label: "↳ Math / Science", group: "deterministic", indent: 1, note: "Math, science, and quantitative-reasoning benchmarks." },
  { id: "coding", slug: "coding", label: "↳ Coding", group: "deterministic", indent: 1, note: "Code-generation and programming benchmarks." },
  { id: "agent", slug: "agent", label: "↳ Agent", group: "deterministic", indent: 1, note: "Tool-use, software-engineering, and agentic-task benchmarks with fixed graders." },
  { id: "multimodal", slug: "multimodal", label: "↳ Multimodal", group: "deterministic", indent: 1, note: "Multimodal benchmarks covering both images and video." },
  { id: "vision", slug: "vision", label: "↳↳ Vision", group: "deterministic", indent: 2, note: "Image-understanding benchmarks." },
  { id: "video", slug: "video", label: "↳↳ Video", group: "deterministic", indent: 2, note: "Video-understanding benchmarks." },
  { id: "multilinguality", slug: "multilinguality", label: "↳ Multilingual", group: "deterministic", indent: 1, note: "Fixed-answer benchmarks run across multiple languages." },
  { id: "korean", slug: "korean", label: "↳ Korean", group: "deterministic", indent: 1, note: "Korean-language fixed-answer benchmarks." },
  { id: "all", slug: "all", label: "All", group: "all", separator: true, note: "The full benchmark catalog in one table — the largest and slowest view to load." },
];

function ScoresPage() {
  const [category, setCategory] = useState<CategorySlug>("coding");
  const [view, setView] = useState<CategoryView | null>(null);
  const [loadedViews, setLoadedViews] = useState<Record<string, CategoryView>>({});
  const active = SCORE_CATEGORIES.find((item) => item.id === category) ?? SCORE_CATEGORIES[0];

  useEffect(() => {
    let alive = true;
    setView(null);
    buildViewBrowser(category).then((next) => {
      if (!alive) return;
      setView(next);
      setLoadedViews((current) => ({ ...current, [category]: next }));
    });
    return () => { alive = false; };
  }, [category]);

  function countFor(id: CategorySlug) {
    const cached = loadedViews[id];
    if (!cached) return "";
    return cached.benchmarks.length > 0 ? `${cached.rows.length}m·${cached.benchmarks.length}b` : "—";
  }

  return (
    <section>
      <PageHeader title="Scores" desc="Benchmark scores reported for each model, grouped by category. Judge-based scores may vary between runs; fixed-answer scores are reproducible from a fixed answer key." />
      <Surface>
      <nav className="mb-5 border-b border-neutral-200">
        <ul className="flex gap-1 overflow-x-auto">
          {SCORE_CATEGORIES.map((item) => {
            const on = category === item.id;
            return (
              <li key={item.slug} className={item.separator ? "ml-3" : ""}>
                <button
                  type="button"
                  onClick={() => setCategory(item.id)}
                  className={`inline-flex items-baseline gap-1.5 whitespace-nowrap border-b-2 px-3 py-2 text-sm ${
                    on ? "border-black font-semibold text-black" : "border-transparent text-neutral-600 hover:border-neutral-300 hover:text-black"
                  }`}
                >
                  <span className={item.indent === 2 ? "pl-3" : item.indent ? "pl-1" : ""}>{item.label}</span>
                  <span className={`text-xs tabular-nums ${on ? "text-neutral-500" : "text-neutral-400"}`}>{countFor(item.id)}</span>
                </button>
              </li>
            );
          })}
        </ul>
      </nav>
      <p className="mb-4 text-sm text-neutral-500">{active.note}</p>
      {view ? <LeaderboardTable benchmarks={view.benchmarks} rows={view.rows} /> : <LoadingPanel label="scores" />}
      </Surface>
    </section>
  );
}

const CATEGORY_LABELS: Record<string, string> = {
  agent: "Agent",
  coding: "Coding",
  factuality: "Factuality",
  general: "General",
  health: "Health",
  instruction: "Instruction",
  korean: "Korean",
  long: "Long Context",
  math: "Math / Science",
  multimodal: "Multimodal",
  multilinguality: "Multilinguality",
  preference: "Preference",
  safety: "Safety",
  video: "Video",
  vision: "Vision",
  other: "Other",
};

function ModelScoresPage({ modelId }: { modelId: string | null }) {
  const [data, setData] = useState<ModelScoreData | null>(null);

  useEffect(() => {
    let alive = true;
    setData(null);
    if (!modelId) return () => { alive = false; };
    buildModelScoreDataBrowser(modelId).then((next) => {
      if (alive) setData(next);
    });
    return () => { alive = false; };
  }, [modelId]);

  if (!modelId) {
    return (
      <section>
        <a href={pathForRoute("scores")} className="text-xs text-neutral-500 hover:underline">&larr; back to Scores</a>
        <Surface className="mt-4 p-5 text-sm text-neutral-500">No model selected.</Surface>
      </section>
    );
  }

  if (!data) return <LoadingPanel label="model scores" />;

  if (!data.model) {
    return (
      <section>
        <a href={pathForRoute("scores")} className="text-xs text-neutral-500 hover:underline">&larr; back to Scores</a>
        <Surface className="mt-4 p-5 text-sm text-neutral-500">Unknown model: {modelId}</Surface>
      </section>
    );
  }

  const totalScores = data.categories.reduce((sum, category) => sum + category.scores.length, 0);

  return (
    <section>
      <a href={pathForRoute("scores")} className="text-xs text-neutral-500 hover:underline">&larr; back to Scores</a>
      <div className="mt-2 mb-6">
        <h1 className="text-2xl font-bold tracking-tight">{data.model.name}</h1>
        <div className="mt-1 flex flex-wrap items-center gap-2 text-sm text-neutral-500">
          <span className="inline-flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: vendorSwatch(data.model.vendor) }} />
            {data.model.vendor}
          </span>
          {data.model.release_date ? <span>released {data.model.release_date}</span> : null}
          <span>{totalScores} score{totalScores === 1 ? "" : "s"}</span>
          {data.model.report_url ? <a href={data.model.report_url} className="hover:underline" target="_blank" rel="noreferrer">report</a> : null}
        </div>
      </div>

      {data.categories.length === 0 ? (
        <Surface className="p-5 text-sm text-neutral-500">No scores yet.</Surface>
      ) : (
        data.categories.map((category) => (
          <Surface key={category.category} className="mb-5 overflow-hidden p-0">
            <div className="border-b border-neutral-100 px-4 py-3 text-sm font-semibold uppercase tracking-wide text-neutral-500">
              {CATEGORY_LABELS[category.category] ?? category.category}
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <tbody>
                  {category.scores.map(({ benchmark, record }) => (
                    <tr key={record.benchmark_id} className="border-b border-neutral-100 last:border-0">
                      <td className="px-4 py-2.5">
                        <div className="font-medium text-neutral-900">{benchmark.name}</div>
                        <div className="text-xs text-neutral-400">{benchmark.id}</div>
                      </td>
                      <td className="px-4 py-2.5 text-right">
                        <ScoreCell record={record} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Surface>
        ))
      )}

      {data.model.aliases.length > 0 ? (
        <details className="mt-6 text-xs text-neutral-500">
          <summary className="cursor-pointer">Aliases ({data.model.aliases.length})</summary>
          <ul className="mt-2 ml-4 list-disc">
            {data.model.aliases.map((alias) => <li key={alias}>{alias}</li>)}
          </ul>
        </details>
      ) : null}
    </section>
  );
}

function dateToTime(date: string) { return new Date(date).getTime(); }
function fmtMonth(time: number) { const date = new Date(time); return `${date.toLocaleString("en", { month: "short" })} '${String(date.getFullYear()).slice(2)}`; }
function fmtYearMonth(time: number) { const date = new Date(time); return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}`; }
function trendPointTooltipLines(point: TrendPoint & { x: number; y: number }) { return [fmtYearMonth(point.x), point.model, `${point.vendor} · Score ${point.score}`]; }

function trendline(points: Array<{ x: number; y: number }>) {
  const n = points.length;
  if (n < 2) return null;
  let sx = 0, sy = 0, sxy = 0, sxx = 0;
  for (const point of points) { sx += point.x; sy += point.y; sxy += point.x * point.y; sxx += point.x * point.x; }
  const denom = n * sxx - sx * sx;
  if (denom === 0) return null;
  const slope = (n * sxy - sx * sy) / denom;
  const intercept = (sy - slope * sx) / n;
  const xs = points.map((point) => point.x);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  return [{ x: x0, y: slope * x0 + intercept }, { x: x1, y: slope * x1 + intercept }];
}

function TrendsPage() {
  const [trend, setTrend] = useState<TrendData | null>(null);
  const [benchmarkId, setBenchmarkId] = useState("");
  const [vendorLines, setVendorLines] = useState(false);
  const [hiddenVendors, setHiddenVendors] = useState<Set<string>>(new Set());
  const [hoveredPoint, setHoveredPoint] = useState<((TrendPoint & { x: number; y: number }) & { left: number; top: number }) | null>(null);

  useEffect(() => {
    let alive = true;
    buildTrendDataBrowser(5).then((next) => {
      if (!alive) return;
      setTrend(next);
      const ids = Object.keys(next.names);
      setBenchmarkId(ids.includes("gpqa") ? "gpqa" : ids[0] ?? "");
    });
    return () => { alive = false; };
  }, []);

  const rows = benchmarkId && trend ? trend.data[benchmarkId] ?? [] : [];
  const vendors = [...new Set(rows.map((row) => row.vendor))].sort();
  const visiblePoints = rows.filter((row) => !hiddenVendors.has(row.vendor)).map((row) => ({ ...row, x: dateToTime(row.date), y: row.score }));
  const xMin = Math.min(dateToTime("2024-03-01"), ...visiblePoints.map((point) => point.x));
  const xMax = Math.max(Date.now(), ...visiblePoints.map((point) => point.x));
  const yMinRaw = Math.min(...visiblePoints.map((point) => point.y));
  const yMaxRaw = Math.max(...visiblePoints.map((point) => point.y));
  const yPad = Math.max(2, (yMaxRaw - yMinRaw) * 0.12 || 5);
  const yMin = Math.max(0, Math.floor((yMinRaw - yPad) / 10) * 10);
  const yMax = Math.min(100, Math.ceil((yMaxRaw + yPad) / 10) * 10 || 100);
  const line = trendline(visiblePoints);
  const plot = { left: 58, top: 20, width: 780, height: 320 };
  const sx = (x: number) => plot.left + ((x - xMin) / Math.max(1, xMax - xMin)) * plot.width;
  const sy = (y: number) => plot.top + plot.height - ((y - yMin) / Math.max(1, yMax - yMin)) * plot.height;

  return (
    <section>
      <PageHeader title="Benchmark trends" desc="Model scores on the selected benchmark over time. Each point denotes one model, positioned by release date; enable vendor lines to compare providers." />
      {!trend ? <LoadingPanel label="trends" /> : (
        <Surface>
          <div className="mb-3 flex flex-wrap items-center gap-3">
            <label className="text-sm text-neutral-600">Benchmark</label>
            <select className="rounded border border-neutral-300 bg-white px-2 py-1 text-sm" value={benchmarkId} onChange={(event) => { setBenchmarkId(event.target.value); setHiddenVendors(new Set()); setHoveredPoint(null); }}>
              {Object.keys(trend.names).map((id) => <option key={id} value={id}>{trend.names[id]}</option>)}
            </select>
            <label className="ml-2 flex items-center gap-1.5 text-sm text-neutral-600"><input type="checkbox" checked={vendorLines} onChange={(event) => setVendorLines(event.target.checked)} /> Vendor lines</label>
            {line ? <span className="ml-auto text-xs text-neutral-500">Trend: {line[0].y.toFixed(1)} to {line[1].y.toFixed(1)} ({fmtYearMonth(line[0].x)} to {fmtYearMonth(line[1].x)})</span> : null}
          </div>
          <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-neutral-500">
            {vendors.map((vendor) => <button key={vendor} type="button" onClick={() => { setHoveredPoint(null); setHiddenVendors((current) => { const next = new Set(current); next.has(vendor) ? next.delete(vendor) : next.add(vendor); return next; }); }} className="flex items-center gap-1.5" style={{ opacity: hiddenVendors.has(vendor) ? 0.35 : 1 }}><span className="h-2.5 w-2.5 rounded-full" style={{ background: vendorSwatch(vendor) }} />{vendor}</button>)}
          </div>
          <div className="relative rounded-md border border-neutral-200 bg-neutral-50/60 p-3">
            <svg viewBox="0 0 860 380" className="h-[430px] w-full" role="img" aria-label="Benchmark trend scatterplot">
              {[0, 1, 2, 3, 4].map((tick) => { const y = yMin + ((yMax - yMin) * tick) / 4; return <g key={tick}><line x1={plot.left} x2={plot.left + plot.width} y1={sy(y)} y2={sy(y)} stroke="rgba(136,135,128,0.16)" /><text x={plot.left - 10} y={sy(y) + 4} textAnchor="end" className="fill-neutral-500 text-xs">{y.toFixed(0)}</text></g>; })}
              {[0, 1, 2, 3, 4, 5].map((tick) => { const x = xMin + ((xMax - xMin) * tick) / 5; return <g key={tick}><line x1={sx(x)} x2={sx(x)} y1={plot.top} y2={plot.top + plot.height} stroke="rgba(136,135,128,0.10)" /><text x={sx(x)} y={plot.top + plot.height + 24} textAnchor="middle" className="fill-neutral-500 text-xs">{fmtMonth(x)}</text></g>; })}
              {line ? <line x1={sx(line[0].x)} y1={sy(line[0].y)} x2={sx(line[1].x)} y2={sy(line[1].y)} stroke="#888780" strokeWidth="2" strokeDasharray="6 5" /> : null}
              {vendorLines ? vendors.map((vendor) => { const pts = visiblePoints.filter((point) => point.vendor === vendor).sort((a, b) => a.x - b.x); if (pts.length < 2) return null; return <polyline key={vendor} points={pts.map((point) => `${sx(point.x)},${sy(point.y)}`).join(" ")} fill="none" stroke={vendorSwatch(vendor)} strokeWidth="1.5" opacity="0.75" />; }) : null}
              {visiblePoints.map((point) => { const cx = sx(point.x), cy = sy(point.y); return <circle key={`${point.model}-${point.date}-${point.score}`} cx={cx} cy={cy} r="6" fill={vendorSwatch(point.vendor)} stroke="white" strokeWidth="1" tabIndex={0} onMouseEnter={() => setHoveredPoint({ ...point, left: cx, top: cy })} onMouseLeave={() => setHoveredPoint(null)} onFocus={() => setHoveredPoint({ ...point, left: cx, top: cy })} onBlur={() => setHoveredPoint(null)}><title>{trendPointTooltipLines(point).join("\n")}</title></circle>; })}
            </svg>
            {hoveredPoint ? (
              <div className="pointer-events-none absolute z-10 min-w-[12rem] rounded-md border border-neutral-800 bg-neutral-950 px-3 py-2 text-xs leading-5 text-white shadow-lg" style={{ left: `${(hoveredPoint.left / 860) * 100}%`, top: `${(hoveredPoint.top / 380) * 100}%`, transform: "translate(12px, -100%)" }}>
                {trendPointTooltipLines(hoveredPoint).map((line, index) => <div key={index} className={index === 0 ? "font-medium" : ""}>{line}</div>)}
              </div>
            ) : null}
          </div>
        </Surface>
      )}
    </section>
  );
}

export default function App() {
  const [route, setRoute] = useState<Route>(routeFromPath);

  useEffect(() => {
    const syncRoute = () => setRoute(routeFromPath());
    window.addEventListener("popstate", syncRoute);
    window.addEventListener("pageshow", syncRoute);
    return () => {
      window.removeEventListener("popstate", syncRoute);
      window.removeEventListener("pageshow", syncRoute);
    };
  }, []);

  // Home renders the standalone landing design (its own nav/hero/footer), so it
  // bypasses the shared app shell below.
  if (route === "home") return <LandingPage />;

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900 antialiased">
      <AppHeader route={route} />
      <main className="mx-auto max-w-[1920px] px-4 py-6">
        {route === "builder" ? <BuilderPage /> : null}
      </main>
    </div>
  );
}
