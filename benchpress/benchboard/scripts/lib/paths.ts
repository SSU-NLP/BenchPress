import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
export const ROOT = path.resolve(here, "..", "..");
export const DATA_DIR = path.join(ROOT, "data");
export const SCORES_DIR = path.join(DATA_DIR, "scores");
export const TECH_REPORTS_DIR = path.join(SCORES_DIR, "tech-reports");

export const BENCHMARKS_PATH = path.join(DATA_DIR, "benchmarks.json");
export const MODELS_PATH = path.join(DATA_DIR, "models.json");

export function scoresPath(name: string): string {
  return path.join(SCORES_DIR, `${name}.json`);
}
