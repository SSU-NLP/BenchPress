/**
 * Tech-report ingest CLI (OpenAI-compatible).
 *
 *   npm run ingest -- <model-id> <pdf-url-or-path>
 *
 * - Reads a PDF, extracts text via pdf-parse.
 * - Sends the text + extraction prompt to an OpenAI-compatible chat endpoint
 *   (Qwen via DashScope, local Ollama, OpenRouter, etc — whichever you point
 *   OPENAI_BASE_URL at).
 * - Every table row becomes a ScoreRecord. Target-model rows → primary; rows
 *   for *comparison* models → source.reported_by = <target-model-id> (cascade).
 * - New benchmark names → auto-registered in data/benchmarks.json with a
 *   heuristic category so nothing is dropped.
 *
 * Env (read from process.env — this script never opens .env directly):
 *   OPENAI_API_KEY    required
 *   OPENAI_BASE_URL   required (e.g. https://dashscope.aliyuncs.com/compatible-mode/v1
 *                     or http://localhost:11434/v1)
 *   OPENAI_MODEL      required (e.g. qwen3-8b, qwen2.5-7b-instruct, qwen3:8b)
 *
 * PDF tables: pdf-parse outputs plain text — column alignment is lost. A 9B
 * model often handles this well for clean tables but may miss rows in
 * multi-column layouts. Always review the draft before publishing.
 */
import fs from "node:fs/promises";
import path from "node:path";
import OpenAI from "openai";
import { PDFParse } from "pdf-parse";
import type {
  Benchmark,
  BenchmarkCategory,
  Model,
  ScoreRecord,
} from "../src/lib/types.ts";
import {
  TECH_REPORTS_DIR,
  MODELS_PATH,
  BENCHMARKS_PATH,
} from "./lib/paths.ts";
import { readJson, writeJson } from "./lib/io.ts";
import { nowIso } from "./lib/fetcher.ts";
import { resolveModelId } from "./lib/normalize.ts";

interface RawExtraction {
  model_name: string;
  benchmark_name: string;
  score: number;
  config: string;
  page?: number;
  is_target?: boolean;
}

function usage(): never {
  console.error(
    "Usage: npm run ingest -- <model-id> <pdf-url-or-path>\n" +
      "Example: npm run ingest -- claude-opus-4.7 ./docs/opus-4-7-system-card.pdf",
  );
  process.exit(2);
}

async function loadPdfBuffer(input: string): Promise<Buffer> {
  if (/^https?:\/\//.test(input)) {
    const res = await fetch(input);
    if (!res.ok) throw new Error(`PDF fetch failed: ${res.status}`);
    return Buffer.from(await res.arrayBuffer());
  }
  return fs.readFile(input);
}

function slug(s: string): string {
  return s
    .toLowerCase()
    .replace(/[()[\]/]/g, " ")
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9.+-]/g, "")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}

function guessCategory(name: string): BenchmarkCategory {
  const n = name.toLowerCase();
  if (/(mmlu|hle|arc|bbh|gpqa|gdpval|trivia|hellaswag)/.test(n)) return "general";
  if (/(aime|math|gsm|hmmt|theorem|usamo|frontier|hidden)/.test(n)) return "math";
  if (/(humaneval|mbpp|swe|livecode|aider|terminal|codeforces|cruxeval|scicode|monorepo|crux)/.test(n)) return "coding";
  if (/(tau2|browsecomp|mcp|osworld|toolathlon|mle|acebench|bfcl|nexus|vending)/.test(n)) return "agent";
  if (/(mmmu|charxiv|videommmu|chart|vibe-eval|zerobench|omnidoc)/.test(n)) return "multimodal";
  if (/(mmmlu|mgsm|include|wmt|multi-|eclektic|global-mmlu)/.test(n)) return "multilinguality";
  if (/(longbench|mrcr|loft|nih|infinite|zeroscrolls|aa-lcr|graphwalks|frames)/.test(n)) return "long";
  if (/(simpleqa|faiths|facts|faithjudge|natural|trivia)/.test(n)) return "factuality";
  if (/(jailbreak|safety|bbq|cve|cyber|ctf)/.test(n)) return "safety";
  if (/(health|bio)/.test(n)) return "health";
  if (/(arena|writingbench|alpaca|wildbench|mt-bench|preference)/.test(n)) return "preference";
  if (/(ifeval|ifbench|collie|multichallenge|protocol)/.test(n)) return "instruction";
  if (/^k|^ko-|^kor|kbank|click|kobalt|hrm8k|kgc/i.test(name)) return "korean";
  if (/^c-?eval|cmmlu|cmath|csimpleqa|cluewsc/i.test(name)) return "chinese";
  return "other";
}

function isNonDeterministic(name: string): boolean {
  return /(arena-hard|chatbot.?arena|alpaca.?eval|wildbench|writingbench|mt-bench|livebench)/i.test(
    name,
  );
}

async function registerNewBenchmarks(
  names: Iterable<string>,
): Promise<string[]> {
  const benchmarks = await readJson<Benchmark[]>(BENCHMARKS_PATH, []);
  const byId = new Map(benchmarks.map((b) => [b.id, b]));
  const byName = new Map(benchmarks.map((b) => [b.name.toLowerCase(), b]));
  const added: string[] = [];
  for (const raw of names) {
    if (byName.has(raw.toLowerCase())) continue;
    const id = slug(raw);
    if (!id || byId.has(id)) continue;
    const entry: Benchmark = {
      id,
      name: raw,
      category: guessCategory(raw),
      type: isNonDeterministic(raw) ? "non_deterministic" : "deterministic",
    };
    byId.set(id, entry);
    byName.set(raw.toLowerCase(), entry);
    added.push(`${raw} → ${id} (${entry.category})`);
  }
  if (added.length) {
    const out = [...byId.values()].sort((a, b) => a.id.localeCompare(b.id));
    await writeJson(BENCHMARKS_PATH, out);
  }
  return added;
}

const SYSTEM = `You are a strict JSON-emitting extractor. Your ONLY output is a single JSON object with one key "rows" whose value is an array of objects. No prose. No "Thinking Process". No analysis. No markdown fences. The first character you emit MUST be { and the last character }.`;

const USER_PREFIX = `Extract every numeric benchmark result from the document below. For each result emit one object inside "rows":
{
  "model_name": "<the model the row reports on, verbatim>",
  "benchmark_name": "<benchmark name, verbatim (e.g. 'MMLU-Pro', 'AIME 2025', 'SWE-Bench Verified')>",
  "score": <number — convert percentages to plain numbers, e.g. 87.2 not "87.2%">,
  "config": "<configuration: '0-shot', 'CoT', 'pass@1', 'high reasoning', etc; 'default' if none>",
  "page": <approximate page number if visible, else 0>,
  "is_target": <true if the row is the model the report is *about*, false if comparison/baseline>
}

Rules:
- Include comparison rows for OTHER models in the same table.
- If the same (model, benchmark) appears with multiple configs, emit separate rows.
- Skip aggregate/composite rows that don't correspond to a published benchmark.

Output schema (NOTHING ELSE): {"rows": [ ... ]}

The target model that this report is about: `;

function recoverTruncatedRows(text: string): RawExtraction[] | null {
  // Find the "rows": [ opener, then collect complete top-level objects until
  // a partial/truncated one. Close cleanly.
  const start = text.search(/"rows"\s*:\s*\[/);
  if (start < 0) return null;
  const arrStart = text.indexOf("[", start);
  if (arrStart < 0) return null;
  const rows: RawExtraction[] = [];
  let i = arrStart + 1;
  while (i < text.length) {
    while (i < text.length && /\s|,/.test(text[i])) i++;
    if (i >= text.length || text[i] === "]") break;
    if (text[i] !== "{") return rows.length ? rows : null;
    // Find matching closing brace, respecting strings.
    let depth = 0;
    let inStr = false;
    let escape = false;
    let j = i;
    for (; j < text.length; j++) {
      const c = text[j];
      if (escape) { escape = false; continue; }
      if (c === "\\") { escape = true; continue; }
      if (c === '"') { inStr = !inStr; continue; }
      if (inStr) continue;
      if (c === "{") depth++;
      else if (c === "}") {
        depth--;
        if (depth === 0) { j++; break; }
      }
    }
    if (depth !== 0) break; // partial last object — discard
    const slice = text.slice(i, j);
    try {
      rows.push(JSON.parse(slice) as RawExtraction);
    } catch {
      break;
    }
    i = j;
  }
  return rows.length ? rows : null;
}

function extractJsonArray(text: string): RawExtraction[] {
  // Try parsing the whole thing as {"rows": [...]} first.
  try {
    const obj = JSON.parse(text);
    if (Array.isArray(obj?.rows)) return obj.rows as RawExtraction[];
    if (Array.isArray(obj)) return obj as RawExtraction[];
  } catch {
    /* fall through */
  }
  // Recover from truncation (common at max_tokens limit).
  const recovered = recoverTruncatedRows(text);
  if (recovered) {
    console.warn(
      `[recover] model output was truncated; salvaged ${recovered.length} complete row(s).`,
    );
    return recovered;
  }
  // Fenced object: ```json {"rows": [...]} ```
  const fencedObj = text.match(/```(?:json)?\s*(\{[\s\S]*?\})\s*```/);
  if (fencedObj) {
    try {
      const obj = JSON.parse(fencedObj[1]);
      if (Array.isArray(obj?.rows)) return obj.rows as RawExtraction[];
    } catch {
      /* fall through */
    }
  }
  // Fenced array.
  const fencedArr = text.match(/```(?:json)?\s*(\[[\s\S]*?\])\s*```/);
  if (fencedArr) return JSON.parse(fencedArr[1]) as RawExtraction[];
  // Bare object containing rows.
  const objMatch = text.match(/\{[\s\S]*"rows"\s*:\s*\[[\s\S]*?\][\s\S]*?\}/);
  if (objMatch) {
    try {
      const obj = JSON.parse(objMatch[0]);
      if (Array.isArray(obj?.rows)) return obj.rows as RawExtraction[];
    } catch {
      /* fall through */
    }
  }
  // Bare array.
  const arrMatch = text.match(/\[[\s\S]*\]/);
  if (arrMatch) return JSON.parse(arrMatch[0]) as RawExtraction[];
  throw new Error("No parseable JSON in model output");
}

async function main() {
  const [modelArg, srcArg] = process.argv.slice(2);
  if (!modelArg || !srcArg) usage();

  const apiKey = process.env.OPENAI_API_KEY;
  const baseURL = process.env.OPENAI_BASE_URL;
  const model = process.env.OPENAI_MODEL;
  if (!apiKey || !baseURL || !model) {
    console.error(
      "OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL must all be set in your shell.",
    );
    console.error(
      "This script does not read .env directly; load it yourself (dotenv-cli, direnv, $env in PowerShell, etc.).",
    );
    process.exit(2);
  }

  const [models] = await Promise.all([readJson<Model[]>(MODELS_PATH, [])]);
  const targetModel = models.find((m) => m.id === modelArg);
  if (!targetModel) {
    console.error(
      `Unknown model id "${modelArg}". Add it to data/models.json first.`,
    );
    process.exit(2);
  }

  console.log(`Loading PDF from ${srcArg}…`);
  const pdfBuf = await loadPdfBuffer(srcArg);
  const parser = new PDFParse({ data: new Uint8Array(pdfBuf) });
  const pdfData = await parser.getText();
  let text = pdfData.text || "";
  const pages = pdfData.pages?.length ?? 0;
  // Crude size cap. Default 80k chars (~20k tokens) is safe for 32k-context
  // small models and avoids gateway 504s on cheap providers. Override with
  // env INGEST_MAX_CHARS for larger-context models (e.g. 200000 for 128k).
  const MAX_CHARS = Number(process.env.INGEST_MAX_CHARS) || 80_000;
  if (text.length > MAX_CHARS) {
    console.warn(
      `PDF text ${text.length} chars; truncating to ${MAX_CHARS}. If important tables are later, split the PDF and ingest each part.`,
    );
    text = text.slice(0, MAX_CHARS);
  }
  console.log(`Extracted ${text.length} chars from ${pages} pages.`);

  console.log(`Calling ${model} via ${baseURL} (streaming)…`);
  const openai = new OpenAI({ apiKey, baseURL, timeout: 600_000 });
  const stream = await openai.chat.completions.create({
    model,
    temperature: 0,
    max_tokens: Number(process.env.INGEST_MAX_TOKENS) || 32_000,
    response_format: { type: "json_object" },
    stream: true,
    messages: [
      { role: "system", content: SYSTEM },
      {
        role: "user",
        content: `${USER_PREFIX}${targetModel.id} (${targetModel.name})\n\n---DOCUMENT TEXT---\n${text}`,
      },
    ],
  } as Parameters<typeof openai.chat.completions.create>[0]);
  let out = "";
  let lastReport = Date.now();
  for await (const chunk of stream as AsyncIterable<{
    choices?: Array<{ delta?: { content?: string | null } }>;
  }>) {
    const piece = chunk.choices?.[0]?.delta?.content ?? "";
    if (piece) {
      out += piece;
      if (Date.now() - lastReport > 3000) {
        process.stdout.write(`\r  …${out.length} chars received`);
        lastReport = Date.now();
      }
    }
  }
  process.stdout.write(`\r  ✓ ${out.length} chars received       \n`);
  out = out.trim();
  if (!out) {
    console.error("Model returned no content.");
    process.exit(1);
  }
  let raw: RawExtraction[];
  try {
    raw = extractJsonArray(out);
  } catch (err) {
    console.error("Could not parse a JSON array from the model output:");
    console.error(out.slice(0, 2000));
    throw err;
  }
  console.log(`Extracted ${raw.length} score rows.`);

  const newBenchNames = new Set<string>();
  for (const row of raw) newBenchNames.add(row.benchmark_name);
  const added = await registerNewBenchmarks(newBenchNames);
  if (added.length) {
    console.log(
      `\nAuto-registered ${added.length} new benchmark(s) in data/benchmarks.json:`,
    );
    for (const a of added) console.log(`  + ${a}`);
    console.log("Review categories and adjust if needed.\n");
  }

  const benchmarks = await readJson<Benchmark[]>(BENCHMARKS_PATH, []);
  const benchByName = new Map(benchmarks.map((b) => [b.name.toLowerCase(), b]));
  const fetched_at = nowIso();
  const records: ScoreRecord[] = [];
  const unknownModels = new Set<string>();

  for (const row of raw) {
    const rowModelId = row.is_target
      ? targetModel.id
      : await resolveModelId(row.model_name);
    if (!rowModelId) {
      unknownModels.add(row.model_name);
      continue;
    }
    const benchmark_id =
      benchByName.get(row.benchmark_name.toLowerCase())?.id ??
      slug(row.benchmark_name);
    records.push({
      model_id: rowModelId,
      benchmark_id,
      score: row.score,
      config: row.config || "default",
      source: {
        // Store only the document name for local PDFs — never the absolute
        // local path, which would leak into the published static site.
        kind: "tech_report",
        url: /^https?:\/\//i.test(srcArg) ? srcArg : path.basename(srcArg),
        ref: row.page ? `p.${row.page}` : undefined,
        reported_by: row.is_target ? undefined : targetModel.id,
        fetched_at,
      },
    });
  }

  await fs.mkdir(TECH_REPORTS_DIR, { recursive: true });
  const draft = path.join(TECH_REPORTS_DIR, `${targetModel.id}.draft.json`);
  await writeJson(draft, records);

  console.log(`\nWrote ${records.length} records → ${draft}`);
  if (unknownModels.size) {
    console.log(
      `\n${unknownModels.size} comparison model(s) had no catalog entry — add aliases to data/models.json or they'll be skipped on merge:`,
    );
    for (const n of unknownModels) console.log(`  - ${n}`);
  }
  console.log(
    `\nReview, then rename: ${path.basename(draft)} → ${targetModel.id}.json`,
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
