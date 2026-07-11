/**
 * Scale SEAL leaderboards — labs.scale.com hosts ~30 per-category SSR pages
 * (Fortress, MASK, PropensityBench, MultiChallenge, MultiNRC, Korean, …) and
 * each one embeds its full top-N table inside Next.js `__next_f.push(...)`
 * RSC payload chunks. We fetch the HTML, splice the chunks back together,
 * and pull out every `{"model":"...","score":N,...}` record.
 *
 * One fetcher covers all categories because the page structure is uniform.
 * Categories with deprecated=true still expose their last-known scores —
 * Korean is the only Korean leaderboard with frontier-model coverage, so we
 * keep ingesting it.
 *
 * No API key needed.
 */
import type { ScoreRecord } from "../src/lib/types.ts";
import { persist, nowIso } from "./lib/fetcher.ts";
import { resolveModelId, recordUnresolved, splitModelName } from "./lib/normalize.ts";

interface CategoryMap {
  /** SEAL category slug (URL path segment) */
  slug: string;
  /** Our canonical benchmark_id */
  id: string;
  /** Human label for logs */
  label: string;
}

// Only the categories where SEAL has coverage of models we actually track.
// Add more here as needed — the parser handles every page uniformly.
const CATEGORIES: CategoryMap[] = [
  { slug: "fortress", id: "fortress", label: "Fortress" },
  { slug: "mask", id: "mask", label: "MASK" },
  { slug: "propensitybench", id: "propensitybench", label: "PropensityBench" },
  { slug: "multichallenge", id: "multichallenge", label: "MultiChallenge" },
  { slug: "multinrc", id: "multinrc", label: "MultiNRC" },
  { slug: "mcp_atlas", id: "mcp-atlas", label: "MCP Atlas" },
  { slug: "korean", id: "seal-korean", label: "Korean" },
];

interface SealScore {
  model: string;
  version?: string;
  rank: number;
  score: number;
  company?: string;
  maxScore?: number;
  deprecated?: boolean;
  confidenceInterval_upper?: number;
}

/** SEAL-specific name quirks that splitModelName can't see through:
 *   - Period collapsed to "p": `glm-4p5` → `glm-4.5`, `deepseek-v3p1` → `deepseek-v3.1`
 *   - Snapshot qualifiers mid-name: `Gemini 2.5 Pro Preview (May 06 2025)` →
 *     `Gemini 2.5 Pro` (the "Preview / Experimental / Exp" word and the
 *     loose-format date inside the parens are dropped)
 *   - Stacked parens: `o3 (high) (April 2025)` → `o3 (high)` so the existing
 *     single-paren extraction in splitModelName still picks up the effort tag
 *   - Trailing model-family qualifiers like "Instruct" that SEAL spells out
 *     but our catalog ids leave implicit. */
function sealPreNormalize(raw: string): string {
  let s = raw;
  // Date-only paren — drop it entirely (model identity, not config)
  s = s.replace(/\s*\((?:January|February|March|April|May|June|July|August|September|October|November|December)[^)]*\)/gi, "");
  // "p" between version digits stands in for "."
  s = s.replace(/(\d)p(\d)/g, "$1.$2");
  // "preview" / "experimental" / "exp" used as a snapshot qualifier — drop;
  // these aren't a distinct model in our catalog.
  s = s.replace(/[\s\-_](preview|experimental|exp)(?=[\s\-_]|$)/gi, "");
  // Trailing "Instruct" is implicit for chat-tuned models in our catalog
  s = s.replace(/[\s\-_]Instruct$/i, "");
  return s.replace(/\s{2,}/g, " ").trim();
}

/** Reassemble the Next.js RSC payload from the SSR'd HTML so the JSON-ish
 *  score objects become readable strings. The chunks are escaped twice. */
function extractRscPayload(html: string): string {
  const re = /self\.__next_f\.push\(\[1,"([\s\S]*?)"\]\)/g;
  let out = "";
  let m: RegExpExecArray | null;
  while ((m = re.exec(html)) !== null) out += m[1];
  // Undo the double escaping the SSR pipeline applied:
  return out
    .replace(/\\n/g, "\n")
    .replace(/\\"/g, '"')
    .replace(/\\u002F/g, "/");
}

/** Pull every `{"model":"…","rank":N,"score":S,…}` object out of the payload.
 *  We don't fully parse — we regex the leading fields, then scan ahead for the
 *  optional ones, because the surrounding RSC framing isn't valid JSON. */
function extractScores(payload: string): SealScore[] {
  const out: SealScore[] = [];
  const re =
    /\{"model":"([^"]+)","version":"([^"]*)","rank":(\d+),"score":([0-9.]+)([\s\S]*?)\}/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(payload)) !== null) {
    const tail = m[5];
    const company = tail.match(/"company":"([^"]+)"/)?.[1];
    const maxScore = Number(tail.match(/"maxScore":([0-9.]+)/)?.[1]);
    const ciHi = Number(tail.match(/"confidenceInterval_upper":([0-9.]+)/)?.[1]);
    const dep = /"deprecated":true/.test(tail);
    out.push({
      model: m[1].replace(/\\n|\\t/g, "").trim(),
      version: m[2],
      rank: Number(m[3]),
      score: Number(m[4]),
      company,
      maxScore: Number.isFinite(maxScore) ? maxScore : undefined,
      confidenceInterval_upper: Number.isFinite(ciHi) ? ciHi : undefined,
      deprecated: dep,
    });
  }
  return out;
}

async function fetchCategory(cat: CategoryMap): Promise<ScoreRecord[]> {
  const url = `https://labs.scale.com/leaderboard/${cat.slug}`;
  const res = await fetch(url, {
    headers: {
      "User-Agent":
        "Mozilla/5.0 (compatible; benchboard/1.0; +https://github.com/) Safari/537.36",
      Accept: "text/html",
    },
  });
  if (!res.ok) {
    console.warn(`[seal] ${cat.label} HTTP ${res.status} — skipping.`);
    return [];
  }
  const html = await res.text();
  const payload = extractRscPayload(html);
  const scores = extractScores(payload);
  if (scores.length === 0) {
    console.warn(`[seal] ${cat.label} produced 0 scores — page format may have changed.`);
    return [];
  }

  // The same model can appear as multiple entries on one page (e.g. thinking /
  // non-thinking, different snapshots). splitModelName carries those qualifiers
  // into the config tag so mergeScores stores them as variants instead of
  // overwriting one another.
  const fetched_at = nowIso();
  const records: ScoreRecord[] = [];
  for (const s of scores) {
    const { base, tag } = splitModelName(sealPreNormalize(s.model));
    const id = await resolveModelId(base);
    if (!id) {
      recordUnresolved(s.model);
      continue;
    }
    records.push({
      model_id: id,
      benchmark_id: cat.id,
      score: Math.round(s.score * 100) / 100,
      config: tag ?? "default",
      source: {
        kind: "aggregator_api",
        url,
        ref: `SEAL ${cat.label}${s.deprecated ? " (deprecated)" : ""}`,
        fetched_at,
      },
      extra: {
        ci_hi: s.confidenceInterval_upper,
        max_score: s.maxScore,
      },
    });
  }
  console.log(
    `[seal] ${cat.label}: ${records.length}/${scores.length} resolved.`,
  );
  return records;
}

async function main() {
  const all: ScoreRecord[] = [];
  for (const cat of CATEGORIES) {
    const recs = await fetchCategory(cat);
    all.push(...recs);
  }
  await persist("seal", all);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
