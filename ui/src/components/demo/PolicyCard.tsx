// ui/src/components/demo/PolicyCard.tsx
import React from "react";
import { DemoPolicyIssue, DemoPolicySnapshot } from "./types";
import { fmtBool } from "./utils";

function IssueList({ title, items }: { title: string; items: DemoPolicyIssue[] }): JSX.Element {
  if (!items.length) {
    return (
      <div style={{ color: "#64748b", fontSize: 12, fontWeight: 700 }}>
        {title}: none
      </div>
    );
  }

  return (
    <div>
      <div style={{ fontWeight: 950, color: "#0f172a", marginBottom: 6 }}>{title}</div>
      <div style={{ display: "grid", gap: 8 }}>
        {items.slice(0, 6).map((it, idx) => (
          <div
            key={`${title}-${idx}`}
            style={{
              border: "1px solid #e2e8f0",
              borderRadius: 10,
              background: "#ffffff",
              padding: 10,
            }}
          >
            <div style={{ fontWeight: 950, color: "#0f172a", fontSize: 13 }}>
              {it.code ?? "issue"}
            </div>
            <div style={{ color: "#334155", fontWeight: 650, fontSize: 13, marginTop: 4 }}>
              {it.message ?? ""}
            </div>
            {it.context && Object.keys(it.context).length > 0 && (
              <pre
                style={{
                  marginTop: 8,
                  marginBottom: 0,
                  padding: 10,
                  borderRadius: 10,
                  background: "#f8fafc",
                  border: "1px solid #e2e8f0",
                  overflow: "auto",
                  fontSize: 12,
                }}
              >
                {JSON.stringify(it.context, null, 2)}
              </pre>
            )}
          </div>
        ))}
        {items.length > 6 && (
          <div style={{ color: "#64748b", fontSize: 12, fontWeight: 700 }}>
            …and {items.length - 6} more
          </div>
        )}
      </div>
    </div>
  );
}

export function PolicyCard(props: { policy: DemoPolicySnapshot | Record<string, any> | null }): JSX.Element {
  const p: any = props.policy ?? null;

  if (!p) {
    return (
      <div
        style={{
          border: "1px solid #e2e8f0",
          borderRadius: 12,
          padding: 12,
          background: "#f8fafc",
          color: "#64748b",
          fontWeight: 700,
        }}
      >
        No policy snapshot loaded.
      </div>
    );
  }

  const reasons: DemoPolicyIssue[] = Array.isArray(p.reasons) ? p.reasons : [];
  const warnings: DemoPolicyIssue[] = Array.isArray(p.warnings) ? p.warnings : [];

  const cap = p.generate_max_new_tokens_cap;

  return (
    <div style={{ border: "1px solid #e2e8f0", borderRadius: 12, padding: 12, background: "#f8fafc" }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <div>
          <div style={{ fontSize: 12, fontWeight: 900, color: "#334155" }}>policy</div>
          <div style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontWeight: 900 }}>
            {p.policy ?? "—"}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 12, fontWeight: 900, color: "#334155" }}>generated_at</div>
          <div style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontWeight: 800, color: "#0f172a" }}>
            {p.generated_at ?? "—"}
          </div>
        </div>

        <div>
          <div style={{ fontSize: 12, fontWeight: 900, color: "#334155" }}>pipeline</div>
          <div style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontWeight: 900 }}>
            {p.pipeline ?? "—"}
          </div>
        </div>

        <div>
          <div style={{ fontSize: 12, fontWeight: 900, color: "#334155" }}>status / ok</div>
          <div style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontWeight: 900 }}>
            {(p.status ?? "—") + " / " + fmtBool(p.ok)}
          </div>
        </div>

        <div>
          <div style={{ fontSize: 12, fontWeight: 900, color: "#334155" }}>enable_extract</div>
          <div style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontWeight: 900 }}>
            {fmtBool(p.enable_extract)}
          </div>
        </div>

        <div>
          <div style={{ fontSize: 12, fontWeight: 900, color: "#334155" }}>generate cap</div>
          <div style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontWeight: 900 }}>
            {cap === null || cap === undefined ? "—" : String(cap)}
          </div>
        </div>

        <div>
          <div style={{ fontSize: 12, fontWeight: 900, color: "#334155" }}>thresholds_profile</div>
          <div style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontWeight: 800 }}>
            {p.thresholds_profile ?? "—"}
          </div>
        </div>

        <div>
          <div style={{ fontSize: 12, fontWeight: 900, color: "#334155" }}>generate_thresholds_profile</div>
          <div style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontWeight: 800 }}>
            {p.generate_thresholds_profile ?? "—"}
          </div>
        </div>
      </div>

      <div style={{ marginTop: 12, display: "grid", gap: 12 }}>
        <IssueList title="Reasons" items={reasons} />
        <IssueList title="Warnings" items={warnings} />
      </div>

      {p.metrics && typeof p.metrics === "object" && (
        <div style={{ marginTop: 12 }}>
          <div style={{ fontWeight: 950, color: "#0f172a", marginBottom: 6 }}>Metrics</div>
          <pre
            style={{
              margin: 0,
              padding: 10,
              borderRadius: 12,
              background: "#ffffff",
              border: "1px solid #e2e8f0",
              overflow: "auto",
              fontSize: 12,
            }}
          >
            {JSON.stringify(p.metrics, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}