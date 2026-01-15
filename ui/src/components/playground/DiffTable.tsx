// ui/src/components/playground/DiffTable.tsx
import React from "react";
import type { DiffRow } from "./types";
import { valueCell } from "./utils";

export function DiffTable(props: {
  rows: DiffRow[];
  filteredRows: DiffRow[];
  diffShowUnchanged: boolean;
  setDiffShowUnchanged: (v: boolean) => void;
  autoBaseline: boolean;
  setAutoBaseline: (v: boolean) => void;
  diffCounts: { same: number; changed: number; added: number; removed: number };
  canSetBaseline: boolean;
  onSetBaseline: () => void;
  canClearBaseline: boolean;
  onClearBaseline: () => void;
}) {
  const {
    filteredRows,
    diffShowUnchanged,
    setDiffShowUnchanged,
    autoBaseline,
    setAutoBaseline,
    diffCounts,
    canSetBaseline,
    onSetBaseline,
    canClearBaseline,
    onClearBaseline,
  } = props;

  return (
    <div style={{ marginTop: 16, border: "1px solid #cbd5f5", borderRadius: 12, overflow: "hidden" }}>
      <div
        style={{
          padding: 10,
          background: "#f8fafc",
          borderBottom: "1px solid #cbd5f5",
          display: "flex",
          gap: 10,
          alignItems: "center",
          flexWrap: "wrap",
        }}
      >
        <strong style={{ fontSize: 13 }}>Per-field diff</strong>

        <span style={{ fontSize: 12, color: "#334155" }}>
          changed: {diffCounts.changed}, added: {diffCounts.added}, removed: {diffCounts.removed}
        </span>

        <div style={{ flex: 1 }} />

        <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "#334155" }}>
          <input type="checkbox" checked={autoBaseline} onChange={(e) => setAutoBaseline(e.target.checked)} />
          Auto-baseline
        </label>

        <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "#334155" }}>
          <input type="checkbox" checked={diffShowUnchanged} onChange={(e) => setDiffShowUnchanged(e.target.checked)} />
          Show unchanged
        </label>

        <button
          onClick={onSetBaseline}
          disabled={!canSetBaseline}
          style={{
            padding: "0.35rem 0.7rem",
            borderRadius: 10,
            border: "1px solid #cbd5f5",
            background: "white",
            cursor: !canSetBaseline ? "not-allowed" : "pointer",
            fontWeight: 700,
            fontSize: 12,
          }}
          title="Set baseline to the latest successful extract data"
        >
          Set baseline
        </button>

        <button
          onClick={onClearBaseline}
          disabled={!canClearBaseline}
          style={{
            padding: "0.35rem 0.7rem",
            borderRadius: 10,
            border: "1px solid #cbd5f5",
            background: "white",
            cursor: !canClearBaseline ? "not-allowed" : "pointer",
            fontWeight: 700,
            fontSize: 12,
          }}
          title="Clear baseline"
        >
          Clear
        </button>
      </div>

      <div style={{ padding: 10 }}>
        {filteredRows.length === 0 ? (
          <div style={{ fontSize: 12, color: "#64748b" }}>No differences.</div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ textAlign: "left" }}>
                  <th style={{ padding: "8px 6px", borderBottom: "1px solid #e2e8f0" }}>Field</th>
                  <th style={{ padding: "8px 6px", borderBottom: "1px solid #e2e8f0" }}>Baseline</th>
                  <th style={{ padding: "8px 6px", borderBottom: "1px solid #e2e8f0" }}>Latest</th>
                  <th style={{ padding: "8px 6px", borderBottom: "1px solid #e2e8f0" }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {filteredRows.map((r) => {
                  const statusColor =
                    r.status === "changed"
                      ? "#0f172a"
                      : r.status === "added"
                      ? "#065f46"
                      : r.status === "removed"
                      ? "#9a3412"
                      : "#64748b";

                  const badgeBg =
                    r.status === "changed"
                      ? "#e2e8f0"
                      : r.status === "added"
                      ? "#ecfdf5"
                      : r.status === "removed"
                      ? "#fff7ed"
                      : "#f1f5f9";

                  const borderLeft = r.required ? "4px solid #2563eb" : "4px solid transparent";

                  return (
                    <tr key={r.key} style={{ verticalAlign: "top" }}>
                      <td style={{ padding: "8px 6px", borderBottom: "1px solid #f1f5f9", borderLeft }}>
                        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                          <code>{r.key}</code>
                          {r.required && (
                            <span
                              style={{
                                fontSize: 10,
                                padding: "2px 6px",
                                borderRadius: 999,
                                border: "1px solid #cbd5f5",
                                background: "#eff6ff",
                                color: "#1d4ed8",
                                fontWeight: 800,
                              }}
                              title="Required by schema"
                            >
                              required
                            </span>
                          )}
                          {!r.inSchema && (
                            <span
                              style={{
                                fontSize: 10,
                                padding: "2px 6px",
                                borderRadius: 999,
                                border: "1px solid #cbd5f5",
                                background: "#f8fafc",
                                color: "#334155",
                                fontWeight: 800,
                              }}
                              title="Returned by model but not present in schema properties"
                            >
                              extra
                            </span>
                          )}
                        </div>
                      </td>
                      <td style={{ padding: "8px 6px", borderBottom: "1px solid #f1f5f9", whiteSpace: "pre-wrap" }}>
                        {valueCell(r.baseline)}
                      </td>
                      <td style={{ padding: "8px 6px", borderBottom: "1px solid #f1f5f9", whiteSpace: "pre-wrap" }}>
                        {valueCell(r.latest)}
                      </td>
                      <td style={{ padding: "8px 6px", borderBottom: "1px solid #f1f5f9" }}>
                        <span
                          style={{
                            display: "inline-block",
                            fontSize: 11,
                            padding: "3px 8px",
                            borderRadius: 999,
                            border: "1px solid #cbd5f5",
                            background: badgeBg,
                            color: statusColor,
                            fontWeight: 800,
                          }}
                        >
                          {r.status}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}