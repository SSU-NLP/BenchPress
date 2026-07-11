# Benchboard Vite Demo

This is the fast Vite React version of Benchboard's demo-facing screens. It includes Builder, Scores, Trends, and Coverage while skipping Astro static route generation for benchmark detail pages.

## Run locally

```bash
npm run dev:demo
```

Open `http://127.0.0.1:5173/Benchboard/`. The SPA routes are:

- `/Benchboard/`
- `/Benchboard/scores`
- `/Benchboard/trends`
- `/Benchboard/coverage`

## Build only the demo

```bash
npm run build:demo
```

The output goes to `dist-demo/` and usually builds much faster than the full Astro site.

## Where to edit

- Demo defaults and labels: `src/lib/plannerDemo.ts`
- UI layout/routes: `demo/src/App.tsx`
- Browser data loaders for Scores/Trends/Coverage: `demo/src/data.ts`
- Mock planner contracts: `data/*.json` and `data/runs/index.json`
