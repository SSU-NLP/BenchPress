// Bucket every model into a size class — Small / Medium / Large — like the
// Artificial Analysis index, so the leaderboard can be filtered by size.
//
// Total parameter count (in billions) comes from three places, in priority:
//   1. DISCLOSED_B  — officially published totals for open-weight models whose
//                     name doesn't already encode the size (e.g. "DeepSeek-V3"
//                     is 671B). Shown as the size chip; treated as ground truth.
//   2. the model name itself — "Qwen3.6-35B-A3B" -> 35, "gpt-oss-120b" -> 120.
//   3. ESTIMATED_B  — closed models (OpenAI / Anthropic / Google / xAI / Amazon
//                     / Baidu …) publish NO parameter count. These are bucketed
//                     by the best public estimate / product-tier lineage so they
//                     still land in a size class, but are NOT shown as a precise
//                     chip — we don't present a guess as a disclosed number.
import type { Model } from "./types";

type IdName = Pick<Model, "id" | "name">;

/** Officially disclosed total params (B) for open-weight models whose catalog
 *  name omits the figure. Used for bucketing AND shown as the size chip. */
const DISCLOSED_B: Record<string, number> = {
  // DeepSeek — 671B MoE (37B active) across V3/R1/V3.x
  "deepseek-v2.5": 236,
  "deepseek-v3": 671,
  "deepseek-v3-base": 671,
  "deepseek-v3.1": 671,
  "deepseek-v3.2": 671,
  "deepseek-r1": 671,
  "deepseek-r1-zero": 671,
  // Moonshot Kimi — 1T MoE (32B active)
  "kimi-k2": 1000,
  "kimi-k2-base": 1000,
  "kimi-k2-thinking": 1000,
  "kimi-k2.5": 1000,
  "kimi-k2.6": 1000,
  // Meta Llama 4 — name carries no "B"
  "llama-4-maverick": 400,
  "llama-4-maverick-base": 400,
  "llama-4-scout": 109,
  // Z.ai GLM — 355B MoE (32B active); Air is 106B
  "glm-4.5": 355,
  "glm-4.6": 355,
  "glm-4.5-air": 106,
  // Alibaba Qwen — Max is >1T MoE; Plus is 397B (A17B)
  "qwen3-max": 1000,
  "qwen3.5-plus": 397,
  // Mistral (open weights)
  "mistral-large": 123,
  "pixtral-large": 124,
  "mistral-small-3": 24,
  "mistral-small-3.1": 24,
  "mistral-small-3.2": 24,
  "magistral-small": 24,
  // Microsoft Phi
  "phi-4": 14,
  "phi-4-reasoning-plus": 14,
  "phi-4-mini": 3.8,
  "phi-3.5-mini": 3.8,
  "phi-3.5-moe": 42, // 16x3.8B MoE, ~41.9B total
  // MiniMax (open weights)
  "mixtral-8x22b": 141, // 8x22B MoE, 141B total
  "minimax-m1": 456,
  "minimax-m2": 230,
  "minimax-m2.1": 230,
  // Korea
  "phi-2": 2.7, // name omits the "B"
  "phi-3-mini": 3.8,
  "phi-3-small": 7,
  "phi-3-medium": 14,
  "exaone-4": 32,
  "ax-4": 72, // continual-pretrained on Qwen2.5-72B
  "ax-3.1-light": 7,
  "solar-pro-2": 31,
  "solar-open": 11, // Solar 10.7B
};

/** Closed / undisclosed models — no official parameter count exists. Bucketed by
 *  the best public estimate or product-tier lineage so they still get a size
 *  class, but never surfaced as a precise figure. Values are deliberately coarse
 *  — only which bucket they fall in matters. */
const ESTIMATED_B: Record<string, number> = {
  // OpenAI — only GPT-4's ~1.8T MoE has ever leaked; the rest are tier estimates
  "gpt-4": 1800,
  "gpt-4-turbo": 1800,
  "gpt-4.5": 1000,
  "gpt-4o": 200,
  "gpt-4.1": 300,
  "gpt-4o-mini": 8,
  "gpt-4.1-mini": 8,
  "gpt-5": 1000,
  "gpt-5.1": 1000,
  "gpt-5.2": 1000,
  "gpt-5.2-thinking": 1000,
  "gpt-5.4": 1000,
  "gpt-5.5": 1000,
  "gpt-5.4-mini-nano": 8,
  o1: 300,
  o3: 300,
  "o3-mini": 8,
  "o4-mini": 8,
  // Anthropic — undisclosed; bucketed by Haiku / Sonnet / Opus tier
  "claude-3-haiku": 8,
  "claude-3.5-haiku": 8,
  "claude-haiku-4.5": 8,
  "claude-3-sonnet": 70,
  "claude-3.5-sonnet": 70,
  "claude-3.7-sonnet": 70,
  "claude-sonnet-4": 70,
  "claude-sonnet-4.5": 70,
  "claude-sonnet-4.6": 70,
  "claude-3-opus": 300,
  "claude-opus-4": 300,
  "claude-opus-4.1": 300,
  "claude-opus-4.5": 300,
  "claude-opus-4.6": 300,
  "claude-opus-4.7": 300,
  "claude-opus-4.8": 300,
  "claude-fable-5": 300,
  "claude-mythos-5": 300,
  "claude-mythos-preview": 300,
  // Google Gemini — undisclosed; Flash = mid, Flash-Lite = small, Pro = large
  "gemini-1.5-flash": 40,
  "gemini-1.5-pro": 300,
  "gemini-2.0-flash": 40,
  "gemini-2.0-flash-lite": 10,
  "gemini-2.5-flash": 40,
  "gemini-2.5-flash-lite": 10,
  "gemini-2.5-pro": 300,
  "gemini-3-flash": 40,
  "gemini-3-pro": 300,
  "gemini-3.1-pro": 300,
  "gemini-3.5-flash": 40,
  // xAI Grok — undisclosed; Grok-1 was 314B MoE
  "grok-2": 300,
  "grok-3": 300,
  "grok-4": 300,
  "grok-4-heavy": 300,
  "grok-4.1": 300,
  "grok-4.2": 300,
  "grok-4.3": 300,
  "grok-4-fast": 40,
  // Amazon Nova — Micro = small, Lite = mid, Pro/Omni = large
  "nova-micro": 8,
  "nova-lite": 40,
  "nova-pro": 200,
  "nova-2-lite": 40,
  "nova-2-pro": 200,
  "nova-2-omni": 200,
  // Baidu ERNIE — undisclosed frontier MoE
  "ernie-5": 300,
  "ernie-5.1": 300,
  // Moonshot Kimi K1.5 — closed multimodal
  "kimi-k1.5": 200,
  // Mistral (closed tiers)
  "mistral-large-3": 200,
  "mistral-medium-3.5": 100,
  "mistral-small-4": 24,
  "magistral-medium": 100,
  // Z.ai GLM (newer / fictional — follow the 355B family)
  "glm-4.7": 355,
  "glm-4.7-flash": 30,
  "glm-5": 355,
  "glm-5.1": 355,
  // DeepSeek V4 (lineage)
  "deepseek-v4-pro": 671,
  "deepseek-v4-flash": 30,
  // MiniMax (lineage)
  "minimax-m2.5": 230,
  "minimax-m3": 230,
  // Alibaba Qwen (closed Max/Plus lineage)
  "qwen3.6-max": 1000,
  "qwen3.7-max": 1000,
  "qwen3.6-plus": 397,
  // Misc / generic
  "llama-3": 70, // generic entry; flagship of the family
  exaone: 8, // generic EXAONE (3.x was 7.8B)
  "solar-pro-3": 31, // lineage of Solar Pro 2
};

/** Total parameters in billions for the size chip + bucketing: disclosed map,
 *  then the name (e.g. "Qwen3.6-35B-A3B" -> 35). null if not publicly stated.
 *  The FIRST <num>B/T token in the name is the total size; a later "A<num>B" is
 *  active params. */
export function paramsB(model: IdName): number | null {
  if (model.id in DISCLOSED_B) return DISCLOSED_B[model.id];
  const m = model.name.match(/(\d+(?:\.\d+)?)\s*([bt])\b/i);
  if (!m) return null;
  const n = parseFloat(m[1]);
  return m[2].toLowerCase() === "t" ? n * 1000 : n;
}

/** Params used only to choose a size bucket — disclosed/name first, then the
 *  coarse estimate for closed models. null only when even an estimate is
 *  unavailable. */
function bucketParamsB(model: IdName): number | null {
  return paramsB(model) ?? (model.id in ESTIMATED_B ? ESTIMATED_B[model.id] : null);
}

export interface SizeClass {
  key: string;
  label: string;
  /** sort order, smaller = smaller models; Unknown last */
  order: number;
}

// Thresholds (total params, B). Edit here to retune the buckets.
const SMALL_MAX = 15;
const MEDIUM_MAX = 72;

const SMALL: SizeClass = { key: "small", label: "Small (≤15B)", order: 1 };
const MEDIUM: SizeClass = { key: "medium", label: "Medium (15–72B)", order: 2 };
const LARGE: SizeClass = { key: "large", label: "Large (>72B)", order: 3 };
const UNKNOWN: SizeClass = { key: "unknown", label: "Unknown", order: 4 };

export function sizeClass(model: IdName): SizeClass {
  const b = bucketParamsB(model);
  if (b === null) return UNKNOWN;
  if (b <= SMALL_MAX) return SMALL;
  if (b <= MEDIUM_MAX) return MEDIUM;
  return LARGE;
}

/** Short human label for the parsed size, e.g. "35B" or "1T" — disclosed/name
 *  figures only. null for closed models (whose size is merely estimated) so we
 *  never render a guessed parameter count as fact. */
export function sizeLabel(model: IdName): string | null {
  const b = paramsB(model);
  if (b === null) return null;
  return b >= 1000 ? `${b / 1000}T` : `${b}B`;
}

export const SIZE_ORDER = [SMALL, MEDIUM, LARGE, UNKNOWN];
