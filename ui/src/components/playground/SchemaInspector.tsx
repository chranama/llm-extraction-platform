// ui/src/components/playground/SchemaInspector.tsx
import React from "react";
import type { JsonSchema } from "../../lib/api";
import { prettyJson } from "./utils";

export function SchemaInspector(props: {
  schemaId: string;
  schemaJson: JsonSchema | null;
  schemaJsonLoading: boolean;
  schemaJsonError: string | null;
  schemaSummary: {
    title?: string;
    description?: string;
    requiredCount: number;
    propertyCount: number;
    additionalProperties: string;
  };
  loading: boolean;
  onReload: () => void;
  onCopyJson: () => void;
}) {
  const {
    schemaId,
    schemaJson,
    schemaJsonLoading,
    schemaJsonError,
    schemaSummary,
    loading,
    onReload,
    onCopyJson,
  } = props;

  return (
    <div
      style={{
        border: "1px solid #cbd5f5",
        borderRadius: 12,
        padding: 12,
        background: "white",
        position: "sticky",
        top: 12,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
        <h3 style={{ margin: 0, fontSize: 14, fontWeight: 800, color: "#0f172a" }}>Schema Inspector</h3>

        <div style={{ flex: 1 }} />

        <button
          onClick={onReload}
          disabled={loading || !schemaId}
          style={{
            padding: "0.35rem 0.7rem",
            borderRadius: 10,
            border: "1px solid #cbd5f5",
            background: "white",
            cursor: loading ? "wait" : "pointer",
            fontWeight: 700,
          }}
          title="Reload schema JSON"
        >
          Reload
        </button>

        <button
          onClick={onCopyJson}
          disabled={schemaJsonLoading || !schemaJson}
          style={{
            padding: "0.35rem 0.7rem",
            borderRadius: 10,
            border: "1px solid #cbd5f5",
            background: "white",
            cursor: schemaJsonLoading ? "wait" : "pointer",
            fontWeight: 700,
          }}
          title="Copy full schema JSON"
        >
          Copy JSON
        </button>
      </div>

      <div style={{ marginBottom: 10, fontSize: 12, color: "#334155", lineHeight: 1.35 }}>
        <div>
          <strong>schema_id:</strong> <code>{schemaId || "(none)"}</code>
        </div>
        {schemaSummary.title && (
          <div>
            <strong>title:</strong> {schemaSummary.title}
          </div>
        )}
        {schemaSummary.description && (
          <div style={{ marginTop: 6 }}>
            <strong>description:</strong> {schemaSummary.description}
          </div>
        )}
        <div style={{ display: "flex", gap: 10, marginTop: 8, flexWrap: "wrap" }}>
          <span>
            <strong>required:</strong> {schemaSummary.requiredCount}
          </span>
          <span>
            <strong>properties:</strong> {schemaSummary.propertyCount}
          </span>
          <span>
            <strong>additionalProperties:</strong> {schemaSummary.additionalProperties}
          </span>
        </div>
      </div>

      {schemaJsonError && (
        <div
          style={{
            marginTop: 10,
            padding: 10,
            borderRadius: 10,
            border: "1px solid #fecaca",
            background: "#fef2f2",
            color: "#991b1b",
            whiteSpace: "pre-wrap",
            fontSize: 12,
          }}
        >
          {schemaJsonError}
        </div>
      )}

      <div style={{ marginTop: 10 }}>
        <label style={{ display: "block", fontWeight: 700, marginBottom: 8, fontSize: 12, color: "#0f172a" }}>
          Raw schema JSON
        </label>
        <pre
          style={{
            whiteSpace: "pre-wrap",
            padding: 12,
            borderRadius: 10,
            background: "#0b1220",
            color: "#e5e7eb",
            minHeight: 320,
            maxHeight: 520,
            overflow: "auto",
            fontSize: 12,
          }}
        >
          {schemaJsonLoading ? "Loading schema..." : schemaJson ? prettyJson(schemaJson) : "No schema loaded."}
        </pre>
      </div>
    </div>
  );
}