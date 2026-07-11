import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "tailwindcss";
import { copyFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";

export default defineConfig({
  root: "demo",
  base: "/Benchpress/",
  plugins: [
    react(),
    {
      name: "github-pages-spa-fallback",
      closeBundle() {
        const indexPath = resolve("dist", "index.html");
        if (existsSync(indexPath)) {
          // GitHub Pages serves 404.html for direct visits to SPA routes.
          copyFileSync(indexPath, resolve("dist", "404.html"));
        }
      },
    },
  ],
  css: {
    postcss: {
      plugins: [tailwindcss()],
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
  },
  preview: {
    host: "127.0.0.1",
    port: 4173,
  },
  build: {
    outDir: "../dist",
    emptyOutDir: true,
  },
});
