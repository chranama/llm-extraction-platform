// ui/src/components/demo/TrackSelector.tsx
import React from "react";
import { Track } from "./types";

export function TrackSelector(props: {
  track: Track;
  onChangeTrack: (t: Track) => void;

  apiBase: string;
  hasApiKey: boolean;

  autoRefresh: boolean;
  onToggleAutoRefresh: () => void;
  refreshEverySeconds: number;
  onChangeRefreshEverySeconds: (n: number) => void;

  onRefreshNow: () => void;
  isRefreshing: boolean;
}): JSX.Element {
  const {
    track,
    onChangeTrack,
    apiBase,
    hasApiKey,
    autoRefresh,
    onToggleAutoRefresh,
    refreshEverySeconds,
    onChangeRefreshEverySeconds,
    onRefreshNow,
    isRefreshing,
  } = props;

  const Pill = ({ ok, label }: { ok: boolean; label: string }) => (
    <span
      style={{
        fontSize: 12,
        padding: "4px 10px",
        borderRadius: 999,
        border: "1px solid #e2e8f0",
        background: ok ? "#ecfdf5" : "#fff7ed",
        color: ok ? "#065f46" : "#9a3412",
        fontWeight: 800,
      }}
      title={label}
    >
      {label}
    </span>
  );

  const TabButton = ({
    label,
    active,
    onClick,
  }: {
    label: string;
    active: boolean;
    onClick: () => void;
  }) => (
    <button
      onClick={onClick}
      style={{
        padding: "0.45rem 0.85rem",
        borderRadius: 999,
        border: active ? "1px solid #2563eb" : "1px solid #cbd5f5",
        background: active ? "#2563eb" : "white",
        color: active ? "white" : "#0f172a",
        fontWeight: 800,
        cursor: "pointer",
      }}
    >
      {label}
    </button>
  );

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
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <TabButton
            label="Generate clamp (Demo A)"
            active={track === "generate_clamp"}
            onClick={() => onChangeTrack("generate_clamp")}
          />
          <TabButton
            label="Extract gating (Demo B)"
            active={track === "extract_gating"}
            onClick={() => onChangeTrack("extract_gating")}
          />
        </div>

        <div style={{ flex: 1 }} />

        <Pill ok={hasApiKey} label={hasApiKey ? "API key: configured" : "API key: missing"} />

        <span
          style={{
            fontSize: 12,
            padding: "4px 10px",
            borderRadius: 999,
            border: "1px solid #e2e8f0",
            background: "#f8fafc",
            color: "#334155",
            fontWeight: 800,
          }}
          title="API base used by UI fetches"
        >
          API base: {apiBase}
        </span>

        <button
          onClick={onRefreshNow}
          disabled={isRefreshing}
          style={{
            padding: "0.45rem 0.85rem",
            borderRadius: 10,
            border: "1px solid #cbd5f5",
            background: isRefreshing ? "#f1f5f9" : "#ffffff",
            color: "#0f172a",
            fontWeight: 800,
            cursor: isRefreshing ? "not-allowed" : "pointer",
          }}
          title="Refresh models + policy + logs"
        >
          {isRefreshing ? "Refreshing…" : "Refresh now"}
        </button>
      </div>

      <div style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <label style={{ display: "flex", alignItems: "center", gap: 8, fontWeight: 800, color: "#0f172a" }}>
          <input type="checkbox" checked={autoRefresh} onChange={onToggleAutoRefresh} />
          Auto-refresh
        </label>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ color: "#475569", fontWeight: 700, fontSize: 13 }}>every</span>
          <input
            type="number"
            min={2}
            max={60}
            value={refreshEverySeconds}
            onChange={(e) => onChangeRefreshEverySeconds(Number(e.target.value))}
            style={{
              width: 80,
              padding: "0.35rem 0.5rem",
              borderRadius: 10,
              border: "1px solid #e2e8f0",
              fontWeight: 700,
            }}
          />
          <span style={{ color: "#475569", fontWeight: 700, fontSize: 13 }}>seconds</span>
        </div>

        <div style={{ flex: 1 }} />

        <span style={{ color: "#64748b", fontSize: 13, fontWeight: 650 }}>
          Tip: Demo A → run traffic first, then “Write Generate SLO Snapshot”, then reload policy/runtime.
        </span>
      </div>
    </div>
  );
}