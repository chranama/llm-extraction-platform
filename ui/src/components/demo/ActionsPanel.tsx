// ui/src/components/demo/ActionsPanel.tsx
import React from "react";
import { DemoActionsConfig, Track } from "./types";

export function ActionsPanel(props: {
  track: Track;
  cfg: DemoActionsConfig;
  onChangeCfg: (cfg: DemoActionsConfig) => void;

  onWriteGenerateSlo: () => void;
  onReloadPolicy: () => void;
  onReloadRuntime: () => void;

  disabled: boolean;
}): JSX.Element {
  const { track, cfg, onChangeCfg, onWriteGenerateSlo, onReloadPolicy, onReloadRuntime, disabled } = props;

  const set = (patch: Partial<DemoActionsConfig>) => onChangeCfg({ ...cfg, ...patch });

  const Button = ({
    label,
    onClick,
    title,
    tone,
  }: {
    label: string;
    onClick: () => void;
    title?: string;
    tone?: "primary" | "neutral";
  }) => (
    <button
      onClick={onClick}
      disabled={disabled}
      title={disabled ? "VITE_API_KEY missing (admin endpoints require it)" : title}
      style={{
        width: "100%",
        padding: "0.55rem 0.85rem",
        borderRadius: 12,
        border: tone === "primary" ? "1px solid #2563eb" : "1px solid #e2e8f0",
        background: disabled ? "#f1f5f9" : tone === "primary" ? "#2563eb" : "#ffffff",
        color: disabled ? "#94a3b8" : tone === "primary" ? "white" : "#0f172a",
        fontWeight: 900,
        cursor: disabled ? "not-allowed" : "pointer",
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
      <h2 style={{ margin: 0, fontSize: 16, fontWeight: 900, color: "#0f172a" }}>Actions</h2>

      <div style={{ marginTop: 10, color: "#64748b", fontSize: 13, fontWeight: 650, lineHeight: 1.35 }}>
        {track === "generate_clamp" ? (
          <>
            Demo A flow: generate traffic → write SLO snapshot → run policy → reload runtime.
          </>
        ) : (
          <>
            Demo B flow: produce eval → run policy → reload runtime → extract becomes blocked/unblocked.
          </>
        )}
      </div>

      <div style={{ marginTop: 12, border: "1px solid #e2e8f0", borderRadius: 12, padding: 12, background: "#f8fafc" }}>
        <div style={{ fontWeight: 950, marginBottom: 8, color: "#0f172a" }}>
          Write Generate SLO snapshot
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 10 }}>
          <label style={{ display: "grid", gap: 6, fontWeight: 800, color: "#0f172a", fontSize: 13 }}>
            Window (seconds)
            <select
              value={cfg.sloWindowSeconds}
              onChange={(e) => set({ sloWindowSeconds: Number(e.target.value) })}
              style={{
                padding: "0.45rem 0.6rem",
                borderRadius: 10,
                border: "1px solid #e2e8f0",
                fontWeight: 800,
                background: "white",
              }}
            >
              <option value={60}>60</option>
              <option value={120}>120</option>
              <option value={300}>300</option>
            </select>
          </label>

          <label style={{ display: "grid", gap: 6, fontWeight: 800, color: "#0f172a", fontSize: 13 }}>
            Route (optional)
            <input
              value={cfg.sloRoute}
              onChange={(e) => set({ sloRoute: e.target.value })}
              placeholder="/v1/generate"
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
              value={cfg.sloModelId}
              onChange={(e) => set({ sloModelId: e.target.value })}
              placeholder="demo-model"
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
            Out path (optional)
            <input
              value={cfg.sloOutPath}
              onChange={(e) => set({ sloOutPath: e.target.value })}
              placeholder="slo_out/generate/latest.json"
              style={{
                padding: "0.45rem 0.6rem",
                borderRadius: 10,
                border: "1px solid #e2e8f0",
                fontWeight: 700,
                fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
              }}
            />
          </label>

          <Button
            tone="primary"
            label="Write Generate SLO Snapshot"
            onClick={onWriteGenerateSlo}
            title="POST /v1/admin/slo/generate/write"
          />
        </div>
      </div>

      <div style={{ marginTop: 12, display: "grid", gap: 10 }}>
        <Button label="Reload Policy Snapshot" onClick={onReloadPolicy} title="POST /v1/admin/policy/reload" />
        <Button tone="primary" label="Reload Runtime" onClick={onReloadRuntime} title="POST /v1/admin/reload" />
      </div>

      {disabled && (
        <div
          style={{
            marginTop: 12,
            padding: 10,
            borderRadius: 12,
            border: "1px solid #fed7aa",
            background: "#fff7ed",
            color: "#9a3412",
            fontSize: 12,
            fontWeight: 800,
            lineHeight: 1.35,
          }}
        >
          Admin actions are disabled because <code>VITE_API_KEY</code> is missing.
        </div>
      )}
    </div>
  );
}