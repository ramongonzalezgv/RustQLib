import { defineConfig } from "vite";

export default defineConfig({
  // Relative asset URLs so the build works under any GitHub Pages sub-path.
  base: "./",
  worker: { format: "es" },
  build: { target: "es2022" },
});
