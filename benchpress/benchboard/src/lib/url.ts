// Base-path-aware link helper. Works in both Astro components (server-rendered)
// and Vite-bundled React components, since both expose `import.meta.env.BASE_URL`.
// When astro.config.mjs has `base: "/Benchboard"`, BASE_URL is `/Benchboard/`.
const BASE = import.meta.env.BASE_URL.replace(/\/$/, "");

export function withBase(path: string): string {
  if (!path) return BASE || "/";
  if (!path.startsWith("/")) return path;
  return BASE + path;
}
