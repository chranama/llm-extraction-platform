// ui/src/components/demo/utils.ts
import { ApiError } from "../../lib/api";
import { DemoActionsConfig, DemoEvidenceConfig, Track } from "./types";

export function defaultActionsConfigForTrack(track: Track): DemoActionsConfig {
  if (track === "extract_gating") {
    // actions still exist, but SLO writing is less relevant
    return {
      sloWindowSeconds: 300,
      sloRoute: "/v1/generate",
      sloModelId: "",
      sloOutPath: "",
    };
  }
  return {
    sloWindowSeconds: 300,
    sloRoute: "/v1/generate",
    sloModelId: "",
    sloOutPath: "",
  };
}

export function defaultEvidenceConfigForTrack(track: Track): DemoEvidenceConfig {
  return {
    route: track === "extract_gating" ? "/v1/extract" : "/v1/generate",
    modelId: "",
    limit: 50,
  };
}

export function toErrorMessage(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return `${e.name}: ${e.message}`;
  try {
    return String(e);
  } catch {
    return "Unknown error";
  }
}

export function fmtBool(x: any): string {
  if (x === true) return "true";
  if (x === false) return "false";
  return "—";
}

export function fmtNum(x: any, digits: number = 2): string {
  const n = typeof x === "number" ? x : Number(x);
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

export function safeArray<T = any>(x: any): T[] {
  return Array.isArray(x) ? (x as T[]) : [];
}

// ---- quick stats helpers for logs ----

export function percentile(values: number[], p: number): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((p / 100) * (sorted.length - 1))));
  const v = sorted[idx];
  return Number.isFinite(v) ? v : null;
}

export function avg(values: number[]): number | null {
  if (!values.length) return null;
  let s = 0;
  let n = 0;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    s += v;
    n += 1;
  }
  return n ? s / n : null;
}

export function clampInt(raw: any, fallback: number, min: number, max: number): number {
  const n = typeof raw === "number" ? raw : Number(raw);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, Math.trunc(n)));
}