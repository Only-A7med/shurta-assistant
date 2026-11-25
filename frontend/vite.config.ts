import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The API runs as a separate process, so development proxies /api to it and the
// built bundle is served behind a proxy that does the same.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.tsx"],
  },
});
