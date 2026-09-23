import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API is proxied rather than called cross-origin in development: it keeps
// the browser on one origin, so EventSource needs no CORS negotiation and the
// viewer behaves the same whether it is served by Vite or by a static host.
// Set VITE_API_URL to point a built viewer at a remote control plane instead.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/v1": { target: process.env.API_URL ?? "http://localhost:8000", changeOrigin: true },
      "/health": { target: process.env.API_URL ?? "http://localhost:8000", changeOrigin: true },
    },
  },
});
