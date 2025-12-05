import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

// You can later set base: "/ui/" if you want it served under /ui
export default defineConfig({
  plugins: [react()],
base: "/ui/",
  server: {
    port: 4173,
    host: "0.0.0.0"
  }
});