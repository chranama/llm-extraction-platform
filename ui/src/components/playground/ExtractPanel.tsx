// ui/src/components/playground/ExtractPanel.tsx
import React from "react";
import type { SchemaIndexItem, JsonSchema } from "../../lib/api";
import type { DiffRow } from "./types";
import { DiffTable } from "./DiffTable";

export function ExtractPanel(props: {
  loading: boolean;

  // schemas
  schemaId: string;
  setSchemaId: (v: string) => void;
  schemasLoading: boolean;
  schemaOptions: SchemaIndexItem[];
  onReloadSchemas: () => void;

  // extract params
  extractCache: boolean;
  setExtractCache: (v: boolean) => void;
  extractRepair: boolean;
  setExtractRepair: (v: boolean) => void;
  extractMaxNewTokens: number;
  setExtractMaxNewTokens: (v: number) => void;
  extractTemperature: number;
  setExtractTemperature: (v: number) => void;

  // input/output
  extractText: string;
  setExtractText: (v: string) => void;
  extractDisabled: boolean;
  onRunExtract: () => void;
  onCopyCurl: () => void;

  activeError: string | null;
  extractOutput: string;

  // diff
  diffRows: DiffRow[];
  filteredDiffRows: DiffRow[];
  diffCounts: { same: number; changed: number; added: number; removed: number };
  diffShowUnchanged: boolean;
  setDiffShowUnchanged: (v: boolean) => void;
  autoBaseline: boolean;
  setAutoBaseline: (v: boolean) => void;
  canSetBaseline: boolean;
  onSetBaseline: () => void;
  canClearBaseline: boolean;
  onClearBaseline: () => void;

  // schema json (for inspector is elsewhere, but we keep types coherent)
  schemaJson: JsonSchema | null;

  // helpers
  toNumberOr: (prev: number, raw: string) => number;
}) {
  const {
    loading,
    schemaId,
    setSchemaId,
    schemasLoading,
    schemaOptions,
    onReloadSchemas,
    extractCache,
    setExtractCache,
    extractRepair,
    setExtractRepair,
    extractMaxNewTokens,
    setExtractMaxNewTokens,
    extractTemperature,
    setExtractTemperature,
    extractText,
    setExtractText,
    extractDisabled,
    onRunExtract,
    onCopyCurl,
    activeError,
    extractOutput,
    diffRows,
    filteredDiffRows,
    diffCounts,
    diffShowUnchanged,
    setDiffShowUnchanged,
    autoBaseline,
    setAutoBaseline,
    canSetBaseline,
    onSetBaseline,
    canClearBaseline,
    onClearBaseline,
    toNumberOr,
  } = props;

  return (
    <div>
      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 12, flexWrap: "wrap" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <label style={{ fontWeight: 500 }}>Schema</label>
          <select
            value={schemaId}
            onChange={(e) => setSchemaId(e.target.value)}
            disabled={schemasLoading || loading}
            style={{
              padding: "8px 10px",
              borderRadius: 10,
              border: "1px solid #cbd5f5",
              minWidth: 260,
            }}
          >
            {schemaOptions.length === 0 ? (
              <option value="">{schemasLoading ? "Loading schemas..." : "No schemas found"}</option>
            ) : (
              schemaOptions.map((s) => (
                <option key={s.schema_id} value={s.schema_id}>
                  {s.schema_id}
                </option>
              ))
            )}
          </select>

          <button
            onClick={onReloadSchemas}
            disabled={schemasLoading || loading}
            style={{
              padding: "0.35rem 0.7rem",
              borderRadius: 10,
              border: "1px solid #cbd5f5",
              background: "white",
              cursor: schemasLoading || loading ? "wait" : "pointer",
            }}
          >
            {schemasLoading ? "Reloading..." : "Reload"}
          </button>
        </div>

        <label style={{ display: "flex", alignItems: "center", gap: 8, color: "#334155" }}>
          <input type="checkbox" checked={extractCache} onChange={(e) => setExtractCache(e.target.checked)} disabled={loading} />
          Cache
        </label>

        <label style={{ display: "flex", alignItems: "center", gap: 8, color: "#334155" }}>
          <input type="checkbox" checked={extractRepair} onChange={(e) => setExtractRepair(e.target.checked)} disabled={loading} />
          Repair
        </label>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <label style={{ fontWeight: 500, color: "#334155" }}>max_new_tokens</label>
          <input
            type="number"
            value={extractMaxNewTokens}
            onChange={(e) => setExtractMaxNewTokens(toNumberOr(extractMaxNewTokens, e.target.value))}
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
            value={extractTemperature}
            onChange={(e) => setExtractTemperature(toNumberOr(extractTemperature, e.target.value))}
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

      <label style={{ display: "block", fontWeight: 500, marginBottom: 8 }}>Input text</label>
      <textarea
        value={extractText}
        onChange={(e) => setExtractText(e.target.value)}
        rows={10}
        style={{
          width: "100%",
          padding: 12,
          borderRadius: 8,
          border: "1px solid #cbd5f5",
          fontFamily: "inherit",
        }}
      />

      <button
        onClick={onRunExtract}
        disabled={extractDisabled}
        style={{
          marginTop: 12,
          padding: "0.5rem 1rem",
          borderRadius: 999,
          border: "none",
          background: extractDisabled ? "#94a3b8" : "#2563eb",
          color: "white",
          fontWeight: 600,
          cursor: extractDisabled ? "not-allowed" : loading ? "wait" : "pointer",
        }}
      >
        {loading ? "Running..." : "Extract"}
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
          {extractOutput || (loading ? "Waiting for response..." : "No output yet.")}
        </pre>
      </div>

      <DiffTable
        rows={diffRows}
        filteredRows={filteredDiffRows}
        diffShowUnchanged={diffShowUnchanged}
        setDiffShowUnchanged={setDiffShowUnchanged}
        autoBaseline={autoBaseline}
        setAutoBaseline={setAutoBaseline}
        diffCounts={diffCounts}
        canSetBaseline={canSetBaseline}
        onSetBaseline={onSetBaseline}
        canClearBaseline={canClearBaseline}
        onClearBaseline={onClearBaseline}
      />
    </div>
  );
}