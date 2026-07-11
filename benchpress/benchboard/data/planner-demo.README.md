# Planner Demo Data Contract

These files drive the Builder-first demo on `src/pages/index.astro`. They are mock data today, but the intended workflow is to replace them with experiment output using the same shapes.

## Add a new capability axis

1. Add the axis object to `benchmark_axis_weights.json` under `axes`.
2. Add a weight for that axis to every benchmark in `benchmark_axis_weights.json`.
3. If it should appear as a first-screen capability button, add its id and label to `CAPABILITY_OPTIONS` in `src/lib/plannerDemo.ts`.
4. If Dynamic V-loop changed it, add a row in `axis_lineage.json`.

## Add a benchmark to the compact builder

1. Add the benchmark object to `benchmark_axis_weights.json` under `benchmarks`.
2. Add its id to the relevant `selected_benchmarks` list in `subset_selection_results.json`.
3. Add or update baseline subsets in the same file if the validation panel should compare against it.

## Add a model shortlist row

1. Add a row to `model_shortlist.json` under `rankings`.
2. Keep `subset_rank`, `full_suite_rank`, `regret`, and `recommendation` populated so the table remains useful in the demo video.

## Swap in a new run

Update `runs/index.json` first. The home page currently uses the first run in that file and defaults to `dynamic-vloop`, configured in `src/lib/plannerDemo.ts`.

## Fast Vite demo

Use `npm run dev:demo` for the video/demo workflow. It serves the same Builder data through Vite at `http://127.0.0.1:5173/Benchboard/` and avoids the full Astro static route generation path.
