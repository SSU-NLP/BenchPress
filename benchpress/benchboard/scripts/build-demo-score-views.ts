import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildView, loadExplorerData, type CategorySlug } from "../src/lib/loadData.ts";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const outDir = path.join(root, "demo", "src", "generated", "scoreViews");
const oldBundleFile = path.join(root, "demo", "src", "generated", "scoreViews.json");
const indexFile = path.join(outDir, "index.json");
const explorerDataFile = path.join(root, "demo", "src", "generated", "explorerData.json");

const CATEGORIES: CategorySlug[] = [
  "non_deterministic",
  "nd_preference",
  "nd_agent",
  "nd_safety",
  "nd_multilinguality",
  "nd_korean",
  "deterministic",
  "general",
  "math",
  "coding",
  "agent",
  "multimodal",
  "vision",
  "video",
  "multilinguality",
  "korean",
  "all",
];

await fs.rm(outDir, { recursive: true, force: true });
await fs.rm(oldBundleFile, { force: true });
await fs.rm(explorerDataFile, { force: true });
await fs.mkdir(outDir, { recursive: true });

const generatedAt = new Date().toISOString();
for (const category of CATEGORIES) {
  const view = await buildView(category);
  const file = path.join(outDir, `${category}.json`);
  await fs.writeFile(file, JSON.stringify(view) + "\n", "utf8");
  console.log(`wrote ${path.relative(root, file)}`);
}

await fs.writeFile(
  indexFile,
  JSON.stringify({ generated_at: generatedAt, categories: CATEGORIES }, null, 2) + "\n",
  "utf8",
);
console.log(`wrote ${path.relative(root, indexFile)} (${CATEGORIES.length} views)`);

const explorerData = await loadExplorerData();
await fs.writeFile(explorerDataFile, JSON.stringify(explorerData) + "\n", "utf8");
console.log(`wrote ${path.relative(root, explorerDataFile)}`);
