// ui/src/components/demo/EvidencePanel.tsx
import React, { useMemo } from "react";
import { AdminLogsPage } from "../../lib/api";
import { DemoEvidenceConfig, Track } from "./types";
import { avg, clampInt, fmtNum, percentile } from "./utils";

export function EvidencePanel(props: {
  track: Track;
  cfg: DemoEvidenceConfig;
  onChangeCfg: (cfg: DemoEvidenceConfig) => void;

  logs: AdminLogsPage | null;
  loading: boolean;
  onRefresh: () => void;
}): JSX.Element {
  const { track, cfg, onChangeCfg, logs, loading, onRefresh } = props;

  const set = (patch: Partial<DemoEvidenceConfig>) => onChangeCfg({ ...cfg, ...patch });

  const items = logs?.items ?? [];

  const stats = useMemo(() => {
    const lat: number[] = [];
    let nErr = 0;

    for (const it of items) {
      const v = it.latency_ms;
      if (typeof v === "number" && Number.isFinite(v)) lat.push(v);
      // best-effort error heuristic: if output is null-ish, or completion_tokens missing and output missing
      // (real error flag might exist server-side; update if you add it)
      const out = (it as any).output;
      if (out == null) nErr += 1;
    }

    const p95 = percentile(lat, 95);
    const a = avg(lat);
    const errorRate = items.length ? nErr / items.length : null;

    return {
      count: items.length,
      errorRate,
      p95,
      avg: a,
    };
  }, [items]);

  return (
    <div
      style={{
        border: "1px solid #e2e8f0",
        borderRadius: 14,
        background: "#ffffff",
        padding: 14,
        boxShadow: "0 1px 2px rgba(15, 23, 42, 0.06)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <h2 style={{ margin: 0, fontSize: 16, fontWeight: 900, color: "#0f172a" }}>Evidence</h2>
        <div style={{ flex: 1 }} />
        <button
          onClick={onRefresh}
          disabled={loading}
          style={{
            padding: "0.35rem 0.6rem",
            borderRadius: 10,
            border: "1px solid #e2e8f0",
            background: loading ? "#f1f5f9" : "#ffffff",
            fontWeight: 900,
            cursor: loading ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Loading…" : "Refresh logs"}
        </button>
      </div>

      <div style={{ marginTop: 10, display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 10 }}>
        <div style={{ border: "1px solid #e2e8f0", borderRadius: 12, padding: 10, background: "#f8fafc" }}>
          <div style={{ fontSize: 12, fontWeight: 900, color: "#334155" }}>count</div>
          <div style={{ fontSize: 18, fontWeight: 950, color: "#0f172a" }}>{stats.count}</div>
        </div>
        <div style={{ border: "1px solid #e2e8f0", borderRadius: 12, padding: 10, background: "#f8fafc" }}>
          <div style={{ fontSize: 12, fontWeight: 900, color: "#334155" }}>error rate*</div>
          <div style={{ fontSize: 18, fontWeight: 950, color: "#0f172a" }}>
            {stats.errorRate === null ? "—" : `${fmtNum(stats.errorRate * 100, 1)}%`}
          </div>
        </div>
        <div style={{ border: "1px solid #e2e8f0", borderRadius: 12, padding: 10, background: "#f8fafc" }}>
          <div style={{ fontSize: 12, fontWeight: 900, color: "#334155" }}>p95 latency</div>
          <div style={{ fontSize: 18, fontWeight: 950, color: "#0f172a" }}>
            {stats.p95 === null ? "—" : `${fmtNum(stats.p95, 0)}ms`}
          </div>
        </div>
        <div style={{ border: "1px solid #e2e8f0", borderRadius: 12, padding: 10, background: "#f8fafc" }}>
          <div style={{ fontSize: 12, fontWeight: 900, color: "#334155" }}>avg latency</div>
          <div style={{ fontSize: 18, fontWeight: 950, color: "#0f172a" }}>
            {stats.avg === null ? "—" : `${fmtNum(stats.avg, 0)}ms`}
          </div>
        </div>
      </div>

      <div style={{ marginTop: 10, color: "#64748b", fontSize: 12, fontWeight: 650 }}>
        * “error rate” here is a UI heuristic (output missing). If you add an explicit error flag in logs, update this.
      </div>

      <div style={{ marginTop: 12, border: "1px solid #e2e8f0", borderRadius: 12, padding: 12, background: "#f8fafc" }}>
        <div style={{ fontWeight: 950, color: "#0f172a", marginBottom: 8 }}>Log filter</div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 10 }}>
          <label style={{ display: "grid", gap: 6, fontWeight: 800, color: "#0f172a", fontSize: 13 }}>
            Route
            <input
              value={cfg.route}
              onChange={(e) => set({ route: e.target.value })}
              placeholder={track === "extract_gating" ? "/v1/extract" : "/v1/generate"}
              style={{
                padding: "0.45rem 0.6rem",
                borderRadius: 10,
                border: "1px solid #e2e8f0",
                fontWeight: 700,
                fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
              }}
            />
          </label>

          <label style={{ display: "grid", gap: 6, fontWeight: 800, color: "#0f172a", fontSize: 13 }}>
            Model ID (optional)
            <input
              value={cfg.modelId}
              onChange={(e) => set({ modelId: e.target.value })}
              placeholder="(any)"
              style={{
                padding: "0.45rem 0.6rem",
                borderRadius: 10,
                border: "1px solid #e2e8f0",
                fontWeight: 700,
                fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
              }}
            />
          </label>

          <label style={{ display: "grid", gap: 6, fontWeight: 800, color: "#0f172a", fontSize: 13 }}>
            Limit
            <input
              type="number"
              min={10}
              max={500}
              value={cfg.limit}
              onChange={(e) => set({ limit: clampInt(e.target.value, cfg.limit, 10, 500) })}
              style={{
                padding: "0.45rem 0.6rem",
                borderRadius: 10,
                border: "1px solid #e2e8f0",
                fontWeight: 800,
                width: 120,
              }}
            />
          </label>
        </div>
      </div>

      <div style={{ marginTop: 12 }}>
        <div style={{ fontWeight: 950, color: "#0f172a", marginBottom: 8 }}>Recent logs</div>

        <div style={{ border: "1px solid #e2e8f0", borderRadius: 12, overflow: "hidden" }}>
          <div style={{ display: "grid", gridTemplateColumns: "140px 1fr 110px 90px", background: "#f1f5f9" }}>
            <div style={{ padding: 10, fontWeight: 900, fontSize: 12 }}>time</div>
            <div style={{ padding: 10, fontWeight: 900, fontSize: 12 }}>model</div>
            <div style={{ padding: 10, fontWeight: 900, fontSize: 12 }}>latency</div>
            <div style={{ padding: 10, fontWeight: 900, fontSize: 12 }}>tokens</div>
          </div>

          {items.map((it) => (
            <div
              key={it.id}
              style={{
                display: "grid",
                gridTemplateColumns: "140px 1fr 110px 90px",
                borderTop: "1px solid #e2e8f0",
              }}
              title={it.route}
            >
              <div style={{ padding: 10, color: "#334155", fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: 12 }}>
                {String(it.created_at).slice(11, 19)}
              </div>
              <div style={{ padding: 10, color: "#0f172a", fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: 12 }}>
                {it.model_id}
              </div>
              <div style={{ padding: 10, color: "#334155", fontWeight: 800 }}>
                {typeof it.latency_ms === "number" ? `${Math.round(it.latency_ms)}ms` : "—"}
              </div>
              <div style={{ padding: 10, color: "#334155", fontWeight: 800 }}>
                {typeof it.completion_tokens === "number" ? it.completion_tokens : "—"}
              </div>
            </div>
          ))}

          {!items.length && <div style={{ padding: 10, color: "#64748b" }}>No logs.</div>}
        </div>
      </div>
    </div>
  );
}