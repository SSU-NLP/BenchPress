/**
 * Collapse benchmark variants into canonical entries + config metadata.
 *
 *   npm run canonicalize
 *
 * Catches three sources of noise from PDF extraction:
 *  1. Config in the name: "GPQA Diamond (no tools)", "MMLU (5-shot, CoT)",
 *     "AIME 2024 (Pass@1)". Suffix in parens → moved into score.config.
 *  2. Alias spellings: "AIME'24" ↔ "AIME 2024", "MATH500" ↔ "MATH-500",
 *     "SimpleQA evaluations" ↔ "SimpleQA".
 *  3. Empty catalog entries left over after the merge.
 *
 * Idempotent. Writes back to data/benchmarks.json and data/scores/{**}.json.
 * Score records collide → mergeScores handles variants automatically.
 */
import fs from "node:fs/promises";
import path from "node:path";
import type { Benchmark, ScoreRecord } from "../src/lib/types.ts";
import {
  BENCHMARKS_PATH,
  SCORES_DIR,
  TECH_REPORTS_DIR,
} from "./lib/paths.ts";
import { readJson, writeJson } from "./lib/io.ts";
import { mergeScores } from "./lib/merge.ts";

// Explicit alias map (canonical_id is the value).
//
// Two forms:
//   "foo": "bar"                           — id `foo` becomes canonical `bar`
//   "foo": {to: "bar", config: "tag"}      — id `foo` becomes canonical `bar`
//                                            AND score.config gets `tag` merged
//                                            in (use for sub-tasks like
//                                            "CharXiv Reasoning" of "CharXiv").
//
// Add new entries when the catalog shows the same benchmark under different
// spellings (see CLAUDE.md "Canonicalization").
type AliasTarget = string | { to: string; config: string };
const ALIAS_MAP: Record<string, AliasTarget> = {
  // AIME spellings
  "aime24": "aime-2024",
  "aime'24": "aime-2024",
  "aime25": "aime-2025",
  "aime'25": "aime-2025",
  "mt-aime24": "mt-aime2024",
  // MATH spellings
  "math500": "math-500",
  // MBPP+ is the EvalPlus-augmented MBPP (35× more tests). "MBPP EvalPlus",
  // "MBPP+" and "MBPP-Plus" are the same benchmark; fold the verbose spellings
  // into `mbpp+`. base/plus is a test-split config, not a separate benchmark.
  // Plain `mbpp` (original 3-shot) stays its own column.
  "mbpp-evalplus": "mbpp+",
  "mbpp-plus": "mbpp+",
  // HumanEval+ likewise — the EvalPlus fetcher's older slug folds into `humaneval+`.
  "humaneval-plus": "humaneval+",
  // SWE-Bench is NOT collapsed: "SWE-bench" (full set) and "SWE-bench Verified"
  // (the 500-problem human-validated subset) are distinct benchmarks, so no
  // alias here. Vendor reports almost always mean Verified (slug
  // `swe-bench-verified`); a bare "SWE-bench" stays its own column. See the
  // `swe-bench-verified` note in _well-known-benchmarks.json for why frontier
  // labs are deprecating it (OpenAI, 2026-02).
  // GPQA Diamond is the 198-question hard subset everyone actually reports as
  // "GPQA". Fold into one column; keep the subset distinction in score.config.
  "gpqa-diamond": { to: "gpqa", config: "Diamond" },
  // SimpleQA spellings
  "simpleqa-evaluations": "simpleqa",
  // ARC-Challenge: extractor emits the "ARC-C" acronym as a separate column.
  "arc-c": "arc-challenge",
  // MMMLU
  "mmmlu-average": "mmmlu",
  // BBQ (Bias Benchmark for QA) — Anthropic system cards use the full name,
  // OpenAI/Qwen use the acronym.
  "bbq-evaluation": "bbq",
  "bias-benchmark-for-question-answering-bbq": "bbq",
  "bias-benchmark-for-qa": "bbq",
  "bias-benchmark-for-qa-bbq": "bbq",
  // BrowseComp — extractors swap the modifier order
  "multi-agent-browsecomp": "browsecomp-multi-agent",
  // CharXiv — Reasoning + Descriptive are sub-tasks of the same paper
  "charxiv-reasoning": { to: "charxiv", config: "Reasoning" },
  "charxiv-descriptive": { to: "charxiv", config: "Descriptive" },
  // HLE spellings — straight (') and curly (’) apostrophes slug differently;
  // both are Humanity's Last Exam, fold into the canonical `hle` column.
  // "No tools" / "With tools" are the standard config split — keep one column,
  // tool-use moves into score.config.
  "humanity's-last-exam": "hle",
  "humanitys-last-exam": "hle",
  "hle-no-tools": { to: "hle", config: "No tools" },
  "hle-with-tools": { to: "hle", config: "With tools" },
  "hle-text-only": { to: "hle", config: "text-only" },
  "humanitys-last-exam-text-only": { to: "hle", config: "text-only" },
  // MT-Bench / LiveBench — extractors emit the name without the hyphen, or
  // tack a snapshot date onto LiveBench.
  "mtbench": "mt-bench",
  "livebench-0831": "livebench",
  "livebench-2024-11-25": "livebench",
  // misc spelling collapses
  "metr-time-horizon-score": "metr-time-horizon",
  "financeagent": "finance-agent",
  // BBQ sub-metric spellings
  "bias-benchmark-for-question-answering": "bbq",
  "bias-benchmark-for-question-answering-accuracy": "bbq",
  "bbq-evaluation-accuracy-on-ambiguous-questions": { to: "bbq", config: "Ambiguous" },
  "bbq-evaluation-accuracy-on-unambiguous-questions": { to: "bbq", config: "Unambiguous" },
  "bbq-evaluation-p": "bbq",
  // IFEval strict-prompt split
  "ifeval-strict-prompt": { to: "ifeval", config: "strict prompt" },
  // WebArena pass@k splits
  "webarena-pass1": { to: "webarena", config: "Pass@1" },
  "webarena-pass2": { to: "webarena", config: "Pass@2" },
  "webarena-pass3": { to: "webarena", config: "Pass@3" },
  "webarena-pass4": { to: "webarena", config: "Pass@4" },
  "webarena-pass5": { to: "webarena", config: "Pass@5" },
  "webarena-passk": { to: "webarena", config: "Pass@k" },
};

// Entries that are NOT standalone published benchmarks — they leak in from
// system-card / tech-report extraction. Three kinds:
//  1. generic metric labels (Average, Overall, Coding, Reasoning, …)
//  2. system-card safety / refusal / bias / CBRN capability line-items
//     (Standard refusal evaluation - hate, Ebola FASTA file, Capture the Flag …)
//  3. broken-extraction fragments / duplicate spellings (τ²-bench → "bench")
// They are dropped from the catalog so only real benchmarks form columns. Their
// raw score records stay in the score files (harmless — the UI only renders
// catalog benchmarks), so this is reversible: remove an id here to bring it back.
const BENCHMARK_BLOCKLIST = new Set<string>([
  // generic metric / label fragments
  "average", "overall", "coding", "comprehension", "reasoning", "summarization",
  "tone", "if", "few-shot-learning", "human-feedback", "instruction-following",
  "training-distribution", "llm-training", "production-benchmarks",
  "visual-quality", "single-doc-qa", "time-series-forecasting", "novel-compiler",
  "impossible-tasks", "intelligence-index-v4.0", "internal-suite-2",
  "internal-agentic-coding-evaluation", "agentic-coding", "agentic-tasks",
  "machine-learning-engineering-tasks", "metr-machine-learning-engineering-tasks",
  "openai-interview", "openai-interview-multiple-choice-questions", "openai-prs",
  "openai-proof-qa", "claude-code-evaluation", "computer-use-evaluation",
  "tool-use-evaluation", "image-input-evaluations", "malicious-use-of-claude-code",
  "quadruped-rl", "persuasion", "persuasion-parallel-generation", "fragment-design",
  // system-card safety / refusal / bias / fairness line-items
  "benign-request-evaluation", "benign-request-evaluations",
  "benign-request-evaluations-default", "benign-request-evaluations-extended-thinking",
  "benign-request-evaluations-overall",
  "bias-gender-sexual-orientation", "bias-job", "bias-miscellaneous",
  "bias-political-affiliation", "bias-race-ethnicity-nationality", "bias-region",
  "hate-gender-sexual-orientation", "hate-job", "hate-political-affiliation",
  "hate-race-ethnicity-nationality", "hate-region",
  "challenging-red-teaming-evaluation-1", "challenging-red-teaming-evaluation-2",
  "challenging-refusal-evaluation", "challenging-refusal-evaluation-aggregate",
  "challenging-refusal-evaluation-harassment-threatening",
  "challenging-refusal-evaluation-hate-threatening",
  "challenging-refusal-evaluation-illicit-non-violent",
  "challenging-refusal-evaluation-illicit-violent",
  "challenging-refusal-evaluation-self-harm-instructions",
  "challenging-refusal-evaluation-sexual-exploitative",
  "challenging-refusal-evaluation-sexual-minors",
  "classifier-hack-rate", "classifier-hack-rate-environ-1",
  "classifier-hack-rate-with-anti-hack-prompt", "classifier-hack-rate-with-no-prompt",
  "deception-evaluations", "deception-monitor",
  "first-person-fairness-evaluation", "first-person-fairness-evaluation-netbias",
  "hallucination-evaluations",
  "higher-difficulty-benign-request-evaluations-default",
  "higher-difficulty-benign-request-evaluations-extended-thinking",
  "higher-difficulty-benign-request-evaluations-overall",
  "higher-difficulty-violative-request-evaluations-default",
  "higher-difficulty-violative-request-evaluations-extended-thinking",
  "higher-difficulty-violative-request-evaluations-overall",
  "illegal-illegal",
  "image-generation-refusals", "image-generation-refusals-notoverrefuse",
  "image-generation-refusals-notunsafe",
  "image-editing-character", "image-editing-creative", "image-editing-infographics",
  "image-editing-object-environment", "image-editing-product-recontextualization",
  "image-editing-stylization",
  "misalignment", "misalignment-situational-awareness", "misalignment-stealth",
  "multimodal-refusal-evaluation", "multimodal-refusal-evaluations",
  "multimodal-refusal-evaluations-vision-self-harm-refusal-evaluation",
  "multimodal-refusal-evaluations-vision-sexual-refusal-evaluation",
  "person-identification-and-ungrounded-inference-evaluations",
  "person-identification-and-ungrounded-inference-evaluations-person-identification",
  "person-identification-and-ungrounded-inference-evaluations-ungrounded-inference",
  "ungrounded-inference-and-sensitive-trait-attribution-safe-behavior-accuracy",
  "speaker-identification-safe-behavior-accuracy",
  "voice-output-classifier-performance",
  "political-bias-evaluation", "prompt-injection-evaluation",
  "prompt-injection-evaluations", "refusal-dataset",
  "reward-hack-prone-coding-tasks-v2",
  "sensitiveness-contentious", "sensitiveness-ethical", "sensitiveness-predictive",
  "single-turn-benign-request-evaluation", "single-turn-benign-request-evaluations",
  "single-turn-violative-request-evaluation", "single-turn-violative-request-evaluations",
  "soft-bias", "standard-disallowed-content-evaluation",
  "standard-refusal-evaluation", "standard-refusal-evaluation-aggregate",
  "standard-refusal-evaluation-extremist-propaganda",
  "standard-refusal-evaluation-harassment-threatening",
  "standard-refusal-evaluation-hate", "standard-refusal-evaluation-hate-threatening",
  "standard-refusal-evaluation-illicit-non-violent",
  "standard-refusal-evaluation-illicit-violent",
  "standard-refusal-evaluation-personal-data-extremely-sensitive",
  "standard-refusal-evaluation-personal-data-highly-sensitive",
  "standard-refusal-evaluation-regulated-advice",
  "standard-refusal-evaluation-self-harm-instructions",
  "standard-refusal-evaluation-self-harm-intent",
  "standard-refusal-evaluation-sexual-exploitative",
  "standard-refusal-evaluation-sexual-minors",
  "sycophancy", "sycophancy-evaluation", "unjustified-refusals",
  "violative-request-evaluation", "violative-request-evaluations",
  "violative-request-evaluations-default", "violative-request-evaluations-extended-thinking",
  "violative-request-evaluations-overall",
  "wildchat-toxic", "wildchat-non-toxic", "automated-behavioral-audit",
  "text-to-image-alignment",
  // CBRN / bio / cyber system-card capability evals (not public benchmarks)
  "alphafold", "apollo-sabotage", "attack-planning-red-teaming",
  "attack-planning-red-teaming-win-rate-results",
  "autonomous-replication-and-adaptation", "ara-tasks",
  "capture-the-flag-collegiate", "capture-the-flag-high-school",
  "capture-the-flag-professional", "dna-synthesis-screening-evasion",
  "ebola-fasta-file", "evasion-challenges", "human-pathogen-capabilities-test",
  "lab-automation-for-gibson-assembly", "long-form-virology-task-1",
  "long-form-virology-task-2", "multimodal-troubleshooting-virology",
  "multimodal-virology", "network-attack-simulation-challenges",
  "organic-chemistry", "phylogenetics",
  "pelagic-long-form-virology-plasmid-design",
  "radiological-and-nuclear-expert-knowledge", "screening-evasion",
  "synthesis-screening-evasion", "tacit-knowledge-and-troubleshooting",
  "twist-dna-order", "vulnerability-discovery-and-exploitation-challenges",
  // broken-extraction fragments / duplicate spellings of τ²-bench & others
  "2-bench-airline", "2-bench-retail", "2-bench-section", "2-bench-telecom",
  "bench", "bench-airline", "bench-retail", "bench-telecom", "vmqa-vct",
  "longrag-hotpot-qa-total", "longrag-nq-total",
  // system-card cyber / CTF capability suites (vendor-specific, not public benchmarks)
  "capture-the-flag-ctf-challenges", "collegiate-ctfs", "ctf", "ctf-challenges",
  "cyber-range", "cybersecurity", "cybersecurity-v1", "cybersecurity-v2",
  "high-school-ctfs", "professional-ctfs", "pattern-labs-cybersecurity",
  "pattern-labs-cybersecurity-evasion",
  "pattern-labs-cybersecurity-network-attack-simulation",
  "pattern-labs-cybersecurity-vulnerability-discovery-and-exploitation",
  // system-card safety / jailbreak / instruction-hierarchy line-items + generic metrics
  "image-to-text-safety", "text-to-text-safety", "jailbreak-evaluations",
  "jailbreak-evaluations-human-sourced-jailbreaks",
  "jailbreak-evaluations-strongreject", "multilingual-safety",
  "pairwise-safety-comparison", "safety-evaluations",
  "safety-evaluations-text-vs-audio-not-over-refuse",
  "safety-evaluations-text-vs-audio-not-unsafe", "safety-metric", "safety-metrics",
  "image-editing-overall-preference", "overall-preference",
  "instruction-hierarchy-evaluation-conflicts-between-message-types",
  "instruction-hierarchy-evaluation-phrase-and-password-protection",
  "instruction-hierarchy-evaluation-system-user-message-conflict",
]);

// Real benchmarks that PDF extraction dumped into the catch-all `other`
// category. Move them to their proper category so they surface on the right
// tab instead of only the overview. Durable across re-ingest.
const CATEGORY_OVERRIDE: Record<string, string> = {
  agentdojo: "agent", agentharm: "agent", cybench: "agent",
  "finance-agent": "agent", "finance-agent-v1.1": "agent",
  "metr-time-horizon": "agent", openrca: "agent", paperbench: "agent",
  spreadsheetbench: "agent", "tau-bench": "agent",
  "tau-bench-airline": "agent", "tau-bench-retail": "agent",
  ai2d: "multimodal", docvqa: "multimodal",
  "alignbench-v1.1": "chinese", ccpm: "chinese",
  autologi: "general", "big-bench-hard": "general", "creative-writing-v3": "general",
  wildchat: "general", winogrande: "general", "writing-bench": "general",
  zebralogic: "general",
  "cnmo-2024": "math", "lab-bench": "math", "lab-bench-figqa": "math",
  "lab-bench-subset": "math",
  blend: "multilinguality", "milu-average": "multilinguality",
  "milu-english": "multilinguality", mlogiqa: "multilinguality",
  "uhura-eval": "multilinguality",
  logickor: "korean",
  longrag: "long", "lv-eval": "long", ruler: "long",
  makemepay: "safety", makemesay: "safety", mask: "safety",
  "virology-capabilities-test": "safety", "wmdp-chem": "safety", xstest: "safety",
  personqa: "factuality", truthfulqa: "factuality",
  "medcalc-bench-verified": "health", "medmcqa-dev": "health",
  "medqa-mainland-china": "health", "medqa-taiwan": "health",
  "medqa-usmle-4-options": "health", "medqa-usmle-5-options": "health",
};

function resolveAlias(id: string): { canonical_id: string; configAddon?: string } {
  const t = ALIAS_MAP[id];
  if (!t) return { canonical_id: id };
  if (typeof t === "string") return { canonical_id: t };
  return { canonical_id: t.to, configAddon: t.config };
}

function slug(s: string): string {
  return s
    .toLowerCase()
    .replace(/[()[\]/]/g, " ")
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9.+'-]/g, "")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}

/**
 * Split a benchmark name into the base benchmark and any config suffix.
 * Examples:
 *   "GPQA Diamond (no tools)"           → ("GPQA Diamond", "no tools")
 *   "AIME 2024 (Competition Math) (With Tools)" → ("AIME 2024", "Competition Math, With Tools")
 *   "MMLU (5-shot, CoT)"                → ("MMLU", "5-shot, CoT")
 *   "MMLU-Pro (5-shot, CoT)"            → ("MMLU-Pro", "5-shot, CoT")
 *   "SuperGPQA"                         → ("SuperGPQA", undefined)
 */
function splitName(name: string): { base: string; config?: string } {
  const m = name.match(/^([^(]+?)\s*\((.+)\)\s*$/);
  if (!m) return { base: name.trim() };
  const base = m[1].trim();
  // Collapse multiple parenthesised groups into one comma list.
  const config = m[2].replace(/\)\s*\(/g, ", ").trim();
  return { base, config };
}

function canonicalIdFor(name: string): { canonical_id: string; config?: string } {
  const { base, config } = splitName(name);
  const baseId = slug(base);
  const { canonical_id, configAddon } = resolveAlias(baseId);
  const merged = mergeConfig(config ?? "default", configAddon);
  return {
    canonical_id,
    config: merged === "default" ? undefined : merged,
  };
}

// Canonical spelling for config tokens that arrive under many spellings.
// Keyed by the lowercased token. Only collapses tokens that are unambiguously
// the same thing (agent harness names, tool/effort/thinking wording). Anything
// not listed keeps its original spelling.
const CONFIG_TOKEN_CANON: Record<string, string> = {
  // agent harnesses (Terminal-Bench etc.)
  "codex": "Codex CLI",
  "codex cli": "Codex CLI",
  "forgecode": "Forge Code",
  "forge code": "Forge Code",
  "grok-cli": "Grok CLI",
  "grok cli": "Grok CLI",
  // tool use
  "no tool": "no tools",
  "no tools": "no tools",
  "with tool": "with tools",
  "with tools": "with tools",
  "without tool": "without tools",
  "without tools": "without tools",
  // reasoning effort
  "high": "high",
  "low": "low",
  "medium": "medium",
  "max": "max",
  "minimal": "minimal",
  "xhigh": "xhigh",
  "base": "base",
  "high effort": "high effort",
  "low effort": "low effort",
  "medium effort": "medium effort",
  "max effort": "max effort",
  "default effort": "default effort",
  // thinking / reasoning mode
  "thinking": "thinking",
  "non-thinking": "non-thinking",
  "non thinking": "non-thinking",
  "adaptive thinking": "adaptive thinking",
  "reasoning": "reasoning",
  "non-reasoning": "non-reasoning",
  // subset
  "diamond": "Diamond",
};

/** Normalize a single config token to its canonical spelling. */
function normToken(tok: string): string {
  const t = tok.trim();
  const key = t.toLowerCase();
  if (CONFIG_TOKEN_CANON[key]) return CONFIG_TOKEN_CANON[key];
  if (/^pass@\d+$/i.test(t)) return t.toLowerCase(); // Pass@1 → pass@1
  return t;
}

/** Token-wise normalize + dedupe (case-insensitive). Each token is mapped to a
 *  canonical spelling, then "no tools, No tools" collapses to one. */
function dedupeConfig(s: string): string {
  if (!s || s === "default") return s || "default";
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of s.split(/\s*,\s*/)) {
    const tok = normToken(raw);
    const key = tok.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(tok);
  }
  return out.join(", ");
}

function mergeConfig(prev: string, extracted?: string): string {
  if (!extracted) return dedupeConfig(prev || "default");
  if (!prev || prev === "default") return extracted;
  return dedupeConfig(`${prev}, ${extracted}`);
}

async function listScoreFiles(): Promise<string[]> {
  const out: string[] = [];
  for (const f of await fs.readdir(SCORES_DIR)) {
    if (f.endsWith(".json") && !f.startsWith("_")) out.push(path.join(SCORES_DIR, f));
  }
  try {
    for (const f of await fs.readdir(TECH_REPORTS_DIR)) {
      if (f.endsWith(".json") && !f.endsWith(".draft.json")) {
        out.push(path.join(TECH_REPORTS_DIR, f));
      }
    }
  } catch {
    /* no tech-reports */
  }
  return out;
}

async function main() {
  const benchmarks = await readJson<Benchmark[]>(BENCHMARKS_PATH, []);

  // 1. Build id → canonical mapping based on current catalog entries.
  const idToCanon: Record<string, { canonical_id: string; configFromName?: string }> = {};
  const canonByCanonId = new Map<string, Benchmark>();
  for (const b of benchmarks) {
    const { canonical_id, config } = canonicalIdFor(b.name);
    idToCanon[b.id] = { canonical_id, configFromName: config };
    if (canonical_id === b.id && !config) {
      // Pure canonical entry — keep as-is.
      canonByCanonId.set(canonical_id, b);
    }
  }
  // Second pass: for canonical IDs we haven't found a pure entry yet, promote
  // the first matching variant (strip its config from the name).
  for (const b of benchmarks) {
    const map = idToCanon[b.id];
    if (canonByCanonId.has(map.canonical_id)) continue;
    const { base } = splitName(b.name);
    canonByCanonId.set(map.canonical_id, {
      id: map.canonical_id,
      name: base,
      category: b.category,
      type: b.type,
      language: b.language,
      source_url: b.source_url,
      description: b.description,
      note: b.note,
    });
  }
  // Replace catalog with deduped entries, minus blocklisted non-benchmarks, sorted.
  const deduped = [...canonByCanonId.values()];
  const newCatalog = deduped
    .filter((b) => !BENCHMARK_BLOCKLIST.has(b.id))
    .map((b) => {
      const cat = CATEGORY_OVERRIDE[b.id];
      return cat ? { ...b, category: cat as Benchmark["category"] } : b;
    })
    .sort((a, b) => a.id.localeCompare(b.id));
  const blocked = deduped.length - newCatalog.length;
  await writeJson(BENCHMARKS_PATH, newCatalog);
  console.log(
    `Catalog: ${benchmarks.length} → ${newCatalog.length} entries ` +
      `(${benchmarks.length - deduped.length} collapsed, ${blocked} non-benchmark dropped).`,
  );

  // 2. Rewrite every score file.
  const files = await listScoreFiles();
  let totalIn = 0;
  let totalOut = 0;
  for (const file of files) {
    const recs = await readJson<ScoreRecord[]>(file, []);
    totalIn += recs.length;
    const rewritten: ScoreRecord[] = recs.map((r) => {
      // Primary: catalog-driven remap. Fallback: a score may carry a raw
      // benchmark_id that never had its own catalog entry (e.g. a sub-task id
      // like "charxiv-reasoning"); run it through the alias map directly so it
      // still folds into its canonical column.
      let map = idToCanon[r.benchmark_id];
      if (!map) {
        const { canonical_id, configAddon } = resolveAlias(r.benchmark_id);
        if (canonical_id !== r.benchmark_id) {
          map = { canonical_id, configFromName: configAddon };
        }
      }
      // Always dedupe configs even when the record needs no remap — fixes
      // historical duplicates like "no tools, No tools" left by prior runs.
      const dedupVariants = r.variants?.map((v) => ({
        ...v,
        config: mergeConfig(v.config, map?.configFromName),
      }));
      if (!map) {
        return { ...r, config: mergeConfig(r.config), variants: dedupVariants };
      }
      const next: ScoreRecord = {
        ...r,
        benchmark_id: map.canonical_id,
        config: mergeConfig(r.config, map.configFromName),
        variants: dedupVariants,
      };
      return next;
    });
    const merged = mergeScores([], rewritten);
    totalOut += merged.length;
    await writeJson(file, merged);
  }
  console.log(
    `Scores: ${totalIn} → ${totalOut} across ${files.length} file(s) (variants auto-merged).`,
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
