import type { Model } from "../../src/lib/types.ts";
import { MODELS_PATH } from "./paths.ts";
import { readJson } from "./io.ts";

let modelsCache: Model[] | null = null;

export async function loadModels(): Promise<Model[]> {
  if (!modelsCache) {
    modelsCache = await readJson<Model[]>(MODELS_PATH, []);
  }
  return modelsCache;
}

export function clearModelsCache() {
  modelsCache = null;
}

function flatten(s: string): string {
  return s
    .toLowerCase()
    .replace(/[\s_./\\-]+/g, "")
    .replace(/[()[\]]/g, "");
}

/**
 * Resolve a raw model name from an external source (Arena, AlpacaEval, etc.)
 * into the canonical model_id in data/models.json. Returns null when no entry
 * matches — caller decides whether to skip or surface for review.
 */
export async function resolveModelId(rawName: string): Promise<string | null> {
  if (!rawName) return null;
  const models = await loadModels();
  const needle = flatten(rawName);
  for (const m of models) {
    if (flatten(m.id) === needle) return m.id;
    if (flatten(m.name) === needle) return m.id;
    for (const alias of m.aliases) {
      if (flatten(alias) === needle) return m.id;
    }
  }
  return null;
}

const EFFORT_WORDS =
  /^(high|low|medium|minimal|xhigh|none|unknown|default|think|thinking|reasoning)$/i;

/**
 * Aggregators label the same model many ways: a provider prefix
 * ("openai/gpt-oss-120b"), a thinking-token budget ("claude-opus-4-7_max",
 * "..._32K", "(16K thinking)"), a dated snapshot ("gpt-5-2025-08-07"), or a
 * "-preview"/"-exp" qualifier. Our catalog has one id per model, so all of
 * these should collapse to the same id. This splits a raw name into the part
 * to resolve and a human-readable config tag for the budget/effort it carried.
 *
 * Deliberately conservative: it never strips bare numeric suffixes (so
 * "grok-4-20" / "grok-4-1" stay distinct models), only well-formed
 * date / budget / effort / qualifier tokens.
 */
export function splitModelName(raw: string): { base: string; tag: string | null } {
  let s = raw.trim();
  const tags: string[] = [];

  // leading provider prefix, e.g. "chutes/", "zai-org/", "openai/"
  s = s.replace(/^[A-Za-z0-9._-]+\//, "");

  // trailing parenthetical → tag (e.g. "(16K thinking)", "(high)")
  const paren = s.match(/^(.*?)\s*\(([^)]*)\)\s*$/);
  if (paren) {
    s = paren[1].trim();
    tags.push(paren[2].trim());
  }

  // repeatedly peel trailing budget/effort markers separated by "_", " ", or "-".
  // "-" is included because Scale SEAL emits e.g. "claude-opus-4-5-…-thinking";
  // it stays safe because we only peel EFFORT_WORDS / Nk / max — never things
  // like "pro" / "mini" / "flash" that would split a real model name.
  for (;;) {
    const m = s.match(/^(.*)[_ \-]([^_ \-]+)$/);
    if (!m) break;
    const tok = m[2];
    if (/^\d+[kK]$/.test(tok)) tags.push(tok.toUpperCase());
    else if (/^max$/i.test(tok)) tags.push("max");
    else if (EFFORT_WORDS.test(tok)) tags.push(tok.toLowerCase());
    else break;
    s = m[1].trim();
  }

  // trailing full date snapshot (identity, not config) — drop it.
  // Repeat to handle the "-DATE-thinking" pattern where the effort token
  // peeled first, exposing the date.
  for (;;) {
    const next = s
      .replace(/[-_]\d{4}[-_]\d{2}[-_]\d{2}$/, "")
      .replace(/[-_]\d{8}$/, "");
    if (next === s) break;
    s = next;
  }

  // trailing snapshot qualifiers
  for (;;) {
    const next = s.replace(/[-_](preview|exp|experimental|webapp|web-app)$/i, "");
    if (next === s) break;
    s = next;
  }

  return { base: s.trim(), tag: tags.length ? tags.reverse().join(", ") : null };
}

const unresolved = new Set<string>();
export function recordUnresolved(name: string) {
  unresolved.add(name);
}
export function flushUnresolved(label: string) {
  if (unresolved.size === 0) return;
  console.warn(
    `[${label}] ${unresolved.size} unresolved model name(s) — add aliases to data/models.json:`,
  );
  for (const n of unresolved) console.warn(`  - ${n}`);
  unresolved.clear();
}
