// ui/src/components/playground/GeneratePanel.tsx
import React from "react";

export function GeneratePanel(props: {
  loading: boolean;
  prompt: string;
  setPrompt: (v: string) => void;
  genMaxNewTokens: number;
  setGenMaxNewTokens: (v: number) => void;
  genTemperature: number;
  setGenTemperature: (v: number) => void;
  onCopyCurl: () => void;
  onRun: () => void;
  activeError: string | null;
  genOutput: string;
  toNumberOr: (prev: number, raw: string) => number;
}) {
  const {
    loading,
    prompt,
    setPrompt,
    genMaxNewTokens,
    setGenMaxNewTokens,
    genTemperature,
    setGenTemperature,
    onCopyCurl,
    onRun,
    activeError,
    genOutput,
    toNumberOr,
  } = props;

  return (
    <div style={{ maxWidth: 960 }}>
      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 12, flexWrap: "wrap" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <label style={{ fontWeight: 500, color: "#334155" }}>max_new_tokens</label>
          <input
            type="number"
            value={genMaxNewTokens}
            onChange={(e) => setGenMaxNewTokens(toNumberOr(genMaxNewTokens, e.target.value))}
            disabled={loading}
            style={{
              width: 110,
              padding: "8px 10px",
              borderRadius: 10,
              border: "1px solid #cbd5f5",
              fontFamily: "inherit",
            }}
          />
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <label style={{ fontWeight: 500, color: "#334155" }}>temperature</label>
          <input
            type="number"
            step="0.05"
            value={genTemperature}
            onChange={(e) => setGenTemperature(toNumberOr(genTemperature, e.target.value))}
            disabled={loading}
            style={{
              width: 90,
              padding: "8px 10px",
              borderRadius: 10,
              border: "1px solid #cbd5f5",
              fontFamily: "inherit",
            }}
          />
        </div>

        <button
          onClick={onCopyCurl}
          disabled={loading}
          style={{
            padding: "0.35rem 0.7rem",
            borderRadius: 10,
            border: "1px solid #cbd5f5",
            background: "white",
            cursor: loading ? "wait" : "pointer",
            fontWeight: 600,
          }}
          title="Copy curl command for this request"
        >
          Copy curl
        </button>
      </div>

      <label style={{ display: "block", fontWeight: 500, marginBottom: 8 }}>Prompt</label>
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        rows={8}
        style={{
          width: "100%",
          padding: 12,
          borderRadius: 8,
          border: "1px solid #cbd5f5",
          fontFamily: "inherit",
        }}
      />

      <button
        onClick={onRun}
        disabled={loading}
        style={{
          marginTop: 12,
          padding: "0.5rem 1rem",
          borderRadius: 999,
          border: "none",
          background: "#2563eb",
          color: "white",
          fontWeight: 600,
          cursor: loading ? "wait" : "pointer",
        }}
      >
        {loading ? "Running..." : "Generate"}
      </button>

      {activeError && (
        <p style={{ marginTop: 12, color: "#b91c1c", whiteSpace: "pre-wrap" }}>Error: {activeError}</p>
      )}

      <div style={{ marginTop: 16 }}>
        <label style={{ display: "block", fontWeight: 500, marginBottom: 8 }}>Output</label>
        <pre
          style={{
            whiteSpace: "pre-wrap",
            padding: 12,
            borderRadius: 8,
            background: "#0f172a",
            color: "#e5e7eb",
            minHeight: 160,
          }}
        >
          {genOutput || (loading ? "Waiting for response..." : "No output yet.")}
        </pre>
      </div>
    </div>
  );
}