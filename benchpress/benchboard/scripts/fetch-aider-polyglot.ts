/**
 * Aider Polyglot — code-editing benchmark across 225 exercises in 6 languages.
 * Source: aider's committed leaderboard YAML in Aider-AI/aider at
 *   aider/website/_data/polyglot_leaderboard.yml
 * The headline number on aider's leaderboard is pass_rate_2 (percent of
 * exercises solved within two attempts). edit_format (diff / whole / ...)
 * is the relevant config, so it goes into score.config.
 *
 * The file is a flat list of scalar-only mappings, machine-generated with a
 * stable shape, so a tiny field-extracting parser is enough — no YAML dep.
 */
import type { ScoreRecord } from "../src/lib/types.ts";
import { persist, nowIso } from "./lib/fetcher.ts";
import { resolveModelId, recordUnresolved } from "./lib/normalize.ts";

const RAW =
  "https://raw.githubusercontent.com/Aider-AI/aider/main/aider/website/_data/polyglot_leaderboard.yml";

function field(block: string, key: string): string | null {
  const m = block.match(new RegExp(`^\\s*${key}:\\s*(.+?)\\s*$`, "m"));
  if (!m) return null;
  return m[1].replace(/^["']|["']$/g, "").trim();
}

async function main() {
  const res = await fetch(RAW, { headers: { Accept: "text/plain" } });
  if (!res.ok) throw new Error(`aider-polyglot fetch failed: ${res.status}`);
  const text = await res.text();

  // Split into list items: each top-level entry begins with "- " at col 0.
  const blocks = text
    .split(/\n(?=- )/)
    .map((b) => b.trim())
    .filter((b) => b.startsWith("- "));

  const fetched_at = nowIso();
  const records: ScoreRecord[] = [];

  for (const block of blocks) {
    const model = field(block, "model");
    const rate = field(block, "pass_rate_2");
    if (!model || rate == null) continue;
    const score = Number(rate);
    if (!Number.isFinite(score)) continue;

    // Architect/combo rows ("o3 + gpt-4.1") aren't attributable to one model.
    if (/\s\+\s/.test(model)) continue;

    // aider often appends a reasoning-effort tag in parens, e.g.
    // "gpt-5 (medium)". Strip it for resolution and fold it into config.
    const tagMatch = model.match(/^(.*?)\s*\(([^)]*)\)\s*$/);
    const baseName = tagMatch ? tagMatch[1].trim() : model;
    const effort = tagMatch ? tagMatch[2].trim() : null;

    const id = await resolveModelId(baseName);
    if (!id) {
      recordUnresolved(model);
      continue;
    }
    const edit = field(block, "edit_format");
    const config = [edit ? `${edit} edit` : null, effort]
      .filter(Boolean)
      .join(", ");
    records.push({
      model_id: id,
      benchmark_id: "aider-polyglot",
      score: Math.round(score * 10) / 10,
      config: config || "default",
      source: { kind: "github_repo", url: RAW, fetched_at },
    });
  }

  await persist("aider-polyglot", records);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
