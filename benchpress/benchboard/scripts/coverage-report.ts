/**
 * Coverage diagnostic — quantifies sparsity in the model×benchmark matrix.
 * Read-only. Run: npx tsx scripts/coverage-report.ts
 */
import { loadBenchmarks, loadModels, loadAllScores, buildView, type CategorySlug } from "../src/lib/loadData.ts";

async function main() {
  const [benchmarks, models, scores] = await Promise.all([
    loadBenchmarks(),
    loadModels(),
    loadAllScores(),
  ]);

  console.log(`models:      ${models.length}`);
  console.log(`benchmarks:  ${benchmarks.length}`);
  console.log(`score recs:  ${scores.length}`);
  const cells = models.length * benchmarks.length;
  console.log(`matrix:      ${models.length} × ${benchmarks.length} = ${cells} cells`);
  console.log(`density:     ${((scores.length / cells) * 100).toFixed(2)}%\n`);

  // Benchmarks that actually have ANY score
  const benchWithScore = new Map<string, number>();
  const modelWithScore = new Map<string, number>();
  for (const s of scores) {
    benchWithScore.set(s.benchmark_id, (benchWithScore.get(s.benchmark_id) ?? 0) + 1);
    modelWithScore.set(s.model_id, (modelWithScore.get(s.model_id) ?? 0) + 1);
  }
  console.log(`benchmarks with >=1 score:  ${benchWithScore.size} / ${benchmarks.length}`);
  console.log(`benchmarks with ZERO score: ${benchmarks.length - benchWithScore.size}`);
  console.log(`models with >=1 score:      ${modelWithScore.size} / ${models.length}\n`);

  // Top benchmarks by coverage
  const ranked = [...benchWithScore.entries()].sort((a, b) => b[1] - a[1]);
  console.log("benchmarks by model-coverage (top 25):");
  for (const [bid, n] of ranked.slice(0, 25)) {
    const b = benchmarks.find((x) => x.id === bid);
    console.log(`  ${String(n).padStart(3)}  ${bid}${b ? "" : "  (NOT IN CATALOG)"}`);
  }

  // How many of the score records point at benchmarks NOT in the catalog?
  const benchIds = new Set(benchmarks.map((b) => b.id));
  const orphanBench = ranked.filter(([bid]) => !benchIds.has(bid));
  const orphanCount = orphanBench.reduce((a, [, n]) => a + n, 0);
  console.log(`\nscores pointing at non-catalog benchmark ids: ${orphanCount} (${orphanBench.length} distinct ids)`);
  for (const [bid, n] of orphanBench) console.log(`  ${String(n).padStart(3)}  ${bid}`);

  console.log("\nper-category view after core curation (cols × rows, hidden):");
  const cats: CategorySlug[] = [
    "general", "math", "coding", "agent", "multimodal", "multilinguality", "korean",
    "nd_preference", "nd_agent", "nd_safety", "deterministic", "all",
  ];
  for (const c of cats) {
    const v = await buildView(c);
    const filled = v.rows.reduce((a, r) => a + Object.keys(r.scores).length, 0);
    const cells = v.benchmarks.length * v.rows.length;
    const dens = cells ? ((filled / cells) * 100).toFixed(0) : "0";
    console.log(
      `  ${c.padEnd(16)} ${String(v.benchmarks.length).padStart(3)} cols × ${String(v.rows.length).padStart(2)} rows  ${dens.padStart(3)}% dense  (+${v.hiddenBenchmarks} hidden)`,
    );
  }
}

main();
