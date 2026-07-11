/**
 * Promote reviewed drafts: rename data/scores/tech-reports/*.draft.json to
 * *.json so loadAllScores picks them up.
 *
 *   npm run publish:drafts                  # interactive list
 *   npm run publish:drafts -- <model-id>    # promote one
 *   npm run publish:drafts -- --all         # promote everything reviewed
 *
 * "Reviewed" here just means "the user is ready" — the draft becomes the
 * canonical file. Re-running ingest for the same model produces a new draft
 * (the existing canonical file is untouched until you republish).
 */
import fs from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { TECH_REPORTS_DIR } from "./lib/paths.ts";

const here = path.dirname(fileURLToPath(import.meta.url));
const CANONICALIZE = path.join(here, "canonicalize-benchmarks.ts");

function runCanonicalize(): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn("npx", ["tsx", CANONICALIZE], {
      stdio: "inherit",
      shell: process.platform === "win32",
    });
    child.on("exit", (code) =>
      code === 0 ? resolve() : reject(new Error(`canonicalize exited ${code}`)),
    );
  });
}

async function listDrafts(): Promise<string[]> {
  try {
    const all = await fs.readdir(TECH_REPORTS_DIR);
    return all.filter((f) => f.endsWith(".draft.json"));
  } catch {
    return [];
  }
}

async function promote(draft: string): Promise<string> {
  const src = path.join(TECH_REPORTS_DIR, draft);
  const dst = path.join(TECH_REPORTS_DIR, draft.replace(/\.draft\.json$/, ".json"));
  await fs.rename(src, dst);
  return dst;
}

async function main() {
  const drafts = await listDrafts();
  const args = process.argv.slice(2);
  if (drafts.length === 0) {
    console.log("No drafts in data/scores/tech-reports/.");
    return;
  }
  if (args.length === 0) {
    console.log("Available drafts:");
    for (const d of drafts) console.log(`  ${d}`);
    console.log(
      "\nPromote one:  npm run publish:drafts -- <model-id>\nPromote all:  npm run publish:drafts -- --all",
    );
    return;
  }
  if (args[0] === "--all") {
    for (const d of drafts) {
      const out = await promote(d);
      console.log(`+ ${path.basename(out)}`);
    }
    console.log("\nRunning canonicalize…");
    await runCanonicalize();
    return;
  }
  const target = `${args[0]}.draft.json`;
  if (!drafts.includes(target)) {
    console.error(`No draft for "${args[0]}" (looked for ${target}).`);
    process.exit(1);
  }
  const out = await promote(target);
  console.log(`+ ${path.basename(out)}`);
  console.log("\nRunning canonicalize…");
  await runCanonicalize();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
