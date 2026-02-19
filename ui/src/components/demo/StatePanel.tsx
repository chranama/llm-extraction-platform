// ui/src/components/demo/StatePanel.tsx
import React from "react";
import { ModelsResponseBody } from "../../lib/api";
import { DemoPolicySnapshot } from "./types";
import { fmtBool } from "./utils";
import { PolicyCard } from "./PolicyCard";

export function StatePanel(props: {
  models: ModelsResponseBody | null;
  policy: DemoPolicySnapshot | Record<string, any> | null;
  effectiveExtractEnabled: boolean | null;

  loadingModels: boolean;
  loadingPolicy: boolean;

  onRefreshModels: () => void;
  onRefreshPolicy: () => void;
}): JSX.Element {
  const { models, policy, effectiveExtractEnabled, loadingModels, loadingPolicy, onRefreshModels, onRefreshPolicy } =
    props;

  const dep = models?.deployment_capabilities;
  const defaultModel = models?.default_model;

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
        <h2 style={{ margin: 0, fontSize: 16, fontWeight: 900, color: "#0f172a" }}>State</h2>
        <div style={{ flex: 1 }} />
        <button
          onClick={onRefreshModels}
          disabled={loadingModels}
          style={{
            padding: "0.35rem 0.6rem",
            borderRadius: 10,
            border: "1px solid #e2e8f0",
            background: loadingModels ? "#f1f5f9" : "#ffffff",
            fontWeight: 800,
            cursor: loadingModels ? "not-allowed" : "pointer",
          }}
        >
          {loadingModels ? "Loading…" : "Refresh models"}
        </button>
        <button
          onClick={onRefreshPolicy}
          disabled={loadingPolicy}
          style={{
            padding: "0.35rem 0.6rem",
            borderRadius: 10,
            border: "1px solid #e2e8f0",
            background: loadingPolicy ? "#f1f5f9" : "#ffffff",
            fontWeight: 800,
            cursor: loadingPolicy ? "not-allowed" : "pointer",
          }}
        >
          {loadingPolicy ? "Loading…" : "Refresh policy"}
        </button>
      </div>

      <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <div
          style={{
            border: "1px solid #e2e8f0",
            borderRadius: 12,
            padding: 12,
            background: "#f8fafc",
          }}
        >
          <div style={{ fontWeight: 900, color: "#0f172a", marginBottom: 6 }}>Deployment capabilities</div>
          <div style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: 13 }}>
            generate: {fmtBool(dep?.generate)}{" "}
            <span style={{ color: "#94a3b8" }}>|</span> extract: {fmtBool(dep?.extract)}
          </div>
          <div style={{ marginTop: 6, color: "#64748b", fontSize: 13, fontWeight: 650 }}>
            default_model:{" "}
            <span style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}>
              {defaultModel ?? "—"}
            </span>
          </div>
        </div>

        <div
          style={{
            border: "1px solid #e2e8f0",
            borderRadius: 12,
            padding: 12,
            background: "#f8fafc",
          }}
        >
          <div style={{ fontWeight: 900, color: "#0f172a", marginBottom: 6 }}>Effective extract enabled</div>
          <div style={{ fontSize: 22, fontWeight: 950, color: effectiveExtractEnabled ? "#065f46" : "#9a3412" }}>
            {effectiveExtractEnabled === null ? "—" : effectiveExtractEnabled ? "ENABLED" : "DISABLED"}
          </div>
          <div style={{ marginTop: 6, color: "#64748b", fontSize: 12, fontWeight: 650 }}>
            Based on deployment capabilities and policy snapshot (best-effort).
          </div>
        </div>
      </div>

      <div style={{ marginTop: 12 }}>
        <div style={{ fontWeight: 900, color: "#0f172a", marginBottom: 8 }}>Policy snapshot</div>
        <PolicyCard policy={policy} />
      </div>

      <div style={{ marginTop: 12 }}>
        <div style={{ fontWeight: 900, color: "#0f172a", marginBottom: 8 }}>Models</div>
        <div
          style={{
            border: "1px solid #e2e8f0",
            borderRadius: 12,
            overflow: "hidden",
          }}
        >
          <div style={{ display: "grid", gridTemplateColumns: "1.2fr 0.6fr 1fr", background: "#f1f5f9" }}>
            <div style={{ padding: 10, fontWeight: 900, fontSize: 12, color: "#0f172a" }}>model_id</div>
            <div style={{ padding: 10, fontWeight: 900, fontSize: 12, color: "#0f172a" }}>loaded</div>
            <div style={{ padding: 10, fontWeight: 900, fontSize: 12, color: "#0f172a" }}>capabilities</div>
          </div>

          {(models?.models ?? []).slice(0, 30).map((m) => {
            const caps = m.capabilities ?? {};
            const capStr =
              caps && typeof caps === "object"
                ? Object.entries(caps)
                    .map(([k, v]) => `${k}:${v ? "1" : "0"}`)
                    .join(" ")
                : "—";

            return (
              <div
                key={m.id}
                style={{
                  display: "grid",
                  gridTemplateColumns: "1.2fr 0.6fr 1fr",
                  borderTop: "1px solid #e2e8f0",
                }}
              >
                <div style={{ padding: 10, fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}>
                  {m.default ? "★ " : ""}
                  {m.id}
                </div>
                <div style={{ padding: 10, color: "#334155", fontWeight: 700 }}>{fmtBool(m.loaded)}</div>
                <div style={{ padding: 10, color: "#334155", fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: 12 }}>
                  {capStr || "—"}
                </div>
              </div>
            );
          })}

          {!models?.models?.length && (
            <div style={{ padding: 10, color: "#64748b" }}>No models returned.</div>
          )}
        </div>
      </div>
    </div>
  );
}