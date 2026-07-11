import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const SCRIPTS = [
  "fetch-arena.ts",
  "fetch-alpaca-eval.ts",
  "fetch-arena-hard.ts",
  "fetch-wildbench.ts",
  "fetch-livebench.ts",
  "fetch-evalplus.ts",
  "fetch-bigcodebench.ts",
  "fetch-llm-stats.ts",
  "fetch-artificial-analysis.ts",
  "fetch-tau2-bench.ts",
  "fetch-aider-polyglot.ts",
  "fetch-epoch.ts",
  "fetch-seal.ts",
];

function run(script: string): Promise<number> {
  return new Promise((resolve) => {
    const child = spawn("npx", ["tsx", path.join(here, script)], {
      stdio: "inherit",
      shell: process.platform === "win32",
    });
    child.on("exit", (code) => resolve(code ?? 1));
  });
}

async function main() {
  let failures = 0;
  for (const s of SCRIPTS) {
    console.log(`\n=== ${s} ===`);
    const code = await run(s);
    if (code !== 0) {
      failures++;
      console.warn(`[fetch-all] ${s} exited ${code}`);
    }
  }
  process.exit(failures === 0 ? 0 : 1);
}

main();
