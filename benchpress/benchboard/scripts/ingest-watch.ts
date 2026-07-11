/**
 * docs/ → tech-report ingest batch.
 *
 *   npm run ingest:watch
 *
 * Scans `docs/` for *.pdf files and runs the tech-report ingest for each one
 * that hasn't been processed yet. The mapping `pdf → model_id` is recorded
 * in `docs/_index.json`, which is the only thing this script writes outside
 * of the normal ingest outputs.
 *
 * Filename convention (deterministic, override via _index.json):
 *   <model-id>__anything.pdf  →  model_id is the part before "__"
 *   <model-id>.pdf            →  model_id is the basename
 *
 * Examples:
 *   docs/claude-opus-4.7.pdf
 *   docs/claude-opus-4.7__system-card-v2.pdf
 *   docs/gpt-5.4__technical-report.pdf
 *
 * Anything that doesn't resolve to a model_id present in data/models.json is
 * reported and skipped — add the model entry and re-run.
 */
import fs from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { readJson, writeJson } from "./lib/io.ts";
import { MODELS_PATH, ROOT } from "./lib/paths.ts";
import type { Model } from "../src/lib/types.ts";

const here = path.dirname(fileURLToPath(import.meta.url));
const DOCS = path.join(ROOT, "docs");
const INDEX = path.join(DOCS, "_index.json");
const INGEST_SCRIPT = path.join(here, "ingest-tech-report.ts");

interface IndexEntry {
  pdf: string;             // relative to docs/
  model_id: string;
  processed_at?: string;   // ISO when last ingested
  sha?: string;            // optional content hash for change detection
  notes?: string;
}

async function loadIndex(): Promise<IndexEntry[]> {
  return readJson<IndexEntry[]>(INDEX, []);
}

async function saveIndex(entries: IndexEntry[]) {
  await writeJson(INDEX, entries);
}

function deriveModelId(filename: string): string {
  const base = filename.replace(/\.pdf$/i, "");
  const sep = base.indexOf("__");
  return sep > 0 ? base.slice(0, sep) : base;
}

async function sha256(file: string): Promise<string> {
  const { createHash } = await import("node:crypto");
  const buf = await fs.readFile(file);
  return createHash("sha256").update(buf).digest("hex").slice(0, 16);
}

function runIngest(modelId: string, pdfPath: string): Promise<number> {
  return new Promise((resolve) => {
    const child = spawn("npx", ["tsx", INGEST_SCRIPT, modelId, pdfPath], {
      stdio: "inherit",
      shell: process.platform === "win32",
    });
    child.on("exit", (code) => resolve(code ?? 1));
  });
}

async function main() {
  try {
    await fs.mkdir(DOCS, { recursive: true });
  } catch {
    /* noop */
  }

  let files: string[];
  try {
    files = (await fs.readdir(DOCS)).filter((f) => /\.pdf$/i.test(f));
  } catch {
    console.error(`Could not read ${DOCS}`);
    process.exit(1);
  }

  if (files.length === 0) {
    console.log(`docs/ has no PDFs to process. Drop tech reports there and re-run.`);
    return;
  }

  const models = await readJson<Model[]>(MODELS_PATH, []);
  const modelIds = new Set(models.map((m) => m.id));
  const index = await loadIndex();
  const byPdf = new Map(index.map((e) => [e.pdf, e]));

  for (const f of files) {
    let entry = byPdf.get(f);
    const modelId = entry?.model_id ?? deriveModelId(f);
    const pdfPath = path.join(DOCS, f);
    const sha = await sha256(pdfPath);
    if (entry?.processed_at && entry.sha === sha) {
      console.log(`= ${f} — already processed (sha ${sha}). Skipping.`);
      continue;
    }
    if (!modelIds.has(modelId)) {
      console.warn(
        `! ${f} — derived model_id "${modelId}" is not in data/models.json. Add the model entry and re-run, or set the mapping explicitly in docs/_index.json.`,
      );
      continue;
    }
    console.log(`> ${f} → ${modelId}`);
    const code = await runIngest(modelId, pdfPath);
    if (code !== 0) {
      console.warn(`  ingest exited ${code}; not marking as processed.`);
      continue;
    }
    entry = entry ?? { pdf: f, model_id: modelId };
    entry.processed_at = new Date().toISOString();
    entry.sha = sha;
    byPdf.set(f, entry);
  }

  await saveIndex([...byPdf.values()].sort((a, b) => a.pdf.localeCompare(b.pdf)));
  console.log(`\nIndex written: ${INDEX}`);
  console.log(`Review drafts at data/scores/tech-reports/*.draft.json and rename to *.json to publish.`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
