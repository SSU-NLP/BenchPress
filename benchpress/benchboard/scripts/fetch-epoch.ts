/**
 * Epoch AI Benchmarking Hub — the single densest clean source.
 *
 * Epoch runs each benchmark itself against each model with a fixed harness, so
 * (unlike the SWE-bench experiments repo or raw Terminal-Bench submissions)
 * the numbers are clean per-model evals, not third-party agent-scaffold runs.
 * One zip (updated daily) carries a CSV per benchmark.
 *
 *   https://epoch.ai/data/benchmark_data.zip
 *
 * We pull the subset of CSVs whose benchmark maps to a canonical id we already
 * track. Score columns differ per file (mean_score / Accuracy / EM / ...), so
 * each mapping names its own column. Values arrive as 0–1 fractions or 0–100;
 * normScore handles both. No scores are synthesized — only what Epoch measured.
 */
import { inflateRawSync } from "node:zlib";
import { Buffer } from "node:buffer";
import type { ScoreRecord } from "../src/lib/types.ts";
import { persist, nowIso } from "./lib/fetcher.ts";
import { resolveModelId, recordUnresolved, splitModelName } from "./lib/normalize.ts";

const ZIP_URL = "https://epoch.ai/data/benchmark_data.zip";

interface Mapping {
  file: string;
  id: string;
  col: string; // score column header
  config?: string; // fixed config tag for the whole file
  configCol?: string; // per-row column folded into config (e.g. edit format)
}

const FILES: Mapping[] = [
  { file: "gpqa_diamond.csv", id: "gpqa", col: "mean_score", config: "Diamond" },
  { file: "frontiermath.csv", id: "frontiermath", col: "mean_score" },
  { file: "hle_external.csv", id: "hle", col: "Accuracy" },
  { file: "swe_bench_verified.csv", id: "swe-bench-verified", col: "mean_score" },
  { file: "live_bench_external.csv", id: "livebench", col: "Global average" },
  { file: "aider_polyglot_external.csv", id: "aider-polyglot", col: "Percent correct", configCol: "Edit format" },
  { file: "arc_agi_2_external.csv", id: "arc-agi-2", col: "Score" },
  { file: "simpleqa_verified.csv", id: "simpleqa", col: "mean_score" },
  { file: "mmlu_external.csv", id: "mmlu", col: "EM" },
  { file: "gsm8k_external.csv", id: "gsm8k", col: "EM" },
  { file: "bbh_external.csv", id: "bbh", col: "Average" },
  { file: "terminalbench_external.csv", id: "terminal-bench", col: "Accuracy mean", configCol: "Agent" },
];

/** Minimal ZIP reader: walk the central directory, inflate each entry.
 *  Sufficient for Epoch's well-formed store/deflate archive. */
function unzip(buf: Buffer): Map<string, string> {
  const out = new Map<string, string>();
  // End of Central Directory record: signature 0x06054b50, scan from the tail.
  let eocd = -1;
  for (let i = buf.length - 22; i >= 0; i--) {
    if (buf.readUInt32LE(i) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error("epoch: EOCD not found in zip");
  const count = buf.readUInt16LE(eocd + 10);
  let off = buf.readUInt32LE(eocd + 16); // central directory offset

  for (let n = 0; n < count; n++) {
    if (buf.readUInt32LE(off) !== 0x02014b50) break;
    const method = buf.readUInt16LE(off + 10);
    const compSize = buf.readUInt32LE(off + 20);
    const nameLen = buf.readUInt16LE(off + 28);
    const extraLen = buf.readUInt16LE(off + 30);
    const commentLen = buf.readUInt16LE(off + 32);
    const localOff = buf.readUInt32LE(off + 42);
    const name = buf.toString("utf8", off + 46, off + 46 + nameLen);
    off += 46 + nameLen + extraLen + commentLen;

    // Local file header → real data offset (its own name/extra lengths).
    if (buf.readUInt32LE(localOff) !== 0x04034b50) continue;
    const lNameLen = buf.readUInt16LE(localOff + 26);
    const lExtraLen = buf.readUInt16LE(localOff + 28);
    const dataStart = localOff + 30 + lNameLen + lExtraLen;
    const comp = buf.subarray(dataStart, dataStart + compSize);
    const data = method === 0 ? comp : inflateRawSync(comp);
    out.set(name, data.toString("utf8"));
  }
  return out;
}

/** Parse CSV into array of row objects, honoring quoted fields/newlines. */
function parseCsv(text: string): Array<Record<string, string>> {
  const rows: string[][] = [];
  let field = "";
  let row: string[] = [];
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else inQuotes = false;
      } else field += c;
    } else if (c === '"') inQuotes = true;
    else if (c === ",") {
      row.push(field);
      field = "";
    } else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field);
      field = "";
      if (row.length > 1 || row[0] !== "") rows.push(row);
      row = [];
    } else field += c;
  }
  if (field !== "" || row.length) {
    row.push(field);
    if (row.length > 1 || row[0] !== "") rows.push(row);
  }
  if (!rows.length) return [];
  const header = rows[0];
  return rows.slice(1).map((r) => {
    const o: Record<string, string> = {};
    header.forEach((h, idx) => (o[h] = r[idx] ?? ""));
    return o;
  });
}

/** Pull a publication date out of a CSV row if any recognizable date column is
 *  present. Returns an ISO date (YYYY-MM-DD) or undefined. */
function pickDate(row: Record<string, string>): string | undefined {
  const CANDIDATES = [
    "Publication date",
    "Release date",
    "Date",
    "date",
    "Released",
    "Best score date",
    "Model release date",
  ];
  for (const c of CANDIDATES) {
    const v = row[c]?.trim();
    if (v && /\d{4}-\d{2}/.test(v)) return v.slice(0, 10);
  }
  return undefined;
}

function normScore(raw: string): number | null {
  let n = Number(raw);
  if (!Number.isFinite(n)) return null;
  if (n > 0 && n <= 1) n *= 100;
  return Math.round(n * 10) / 10;
}

async function main() {
  const res = await fetch(ZIP_URL, { headers: { Accept: "application/zip" } });
  if (!res.ok) throw new Error(`epoch fetch failed: ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  const files = unzip(buf);

  const fetched_at = nowIso();

  for (const map of FILES) {
    const csv = files.get(map.file);
    if (!csv) {
      console.warn(`[epoch] ${map.file} not in archive — skipping`);
      continue;
    }
    const records: ScoreRecord[] = [];
    for (const r of parseCsv(csv)) {
      const rawModel = (r["Model version"] ?? r["Name"] ?? "").trim();
      if (!rawModel) continue;
      const score = normScore(r[map.col]);
      if (score === null) continue;

      const { base, tag } = splitModelName(rawModel);
      const id = await resolveModelId(base);
      if (!id) {
        recordUnresolved(rawModel);
        continue;
      }

      const parts = [
        map.config,
        tag,
        map.configCol ? r[map.configCol]?.trim() || null : null,
      ].filter(Boolean);
      // Epoch's CSVs carry the date the result was produced; capture it as the
      // real publication date (distinct from when we fetched the zip).
      const published = pickDate(r);
      records.push({
        model_id: id,
        benchmark_id: map.id,
        score,
        config: parts.length ? parts.join(", ") : "default",
        source: {
          kind: "aggregator_api",
          url: ZIP_URL,
          ref: map.file,
          ...(published ? { published } : {}),
          fetched_at,
        },
      });
    }
    await persist(`epoch-${map.id}`, records);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
