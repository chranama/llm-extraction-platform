import React, { useState } from "react";
import { callGenerate } from "../lib/api";

export function Playground() {
  const [prompt, setPrompt] = useState("Write a haiku about autumn leaves.");
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    setOutput("");

    try {
      const res = await callGenerate({ prompt });
      setOutput(res.output);
    } catch (e: any) {
      setError(e?.message ?? "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 960 }}>
      <label style={{ display: "block", fontWeight: 500, marginBottom: 8 }}>Prompt</label>
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        rows={5}
        style={{
          width: "100%",
          padding: 12,
          borderRadius: 8,
          border: "1px solid #cbd5f5",
          fontFamily: "inherit"
        }}
      />

      <button
        onClick={handleRun}
        disabled={loading}
        style={{
          marginTop: 12,
          padding: "0.5rem 1rem",
          borderRadius: 999,
          border: "none",
          background: "#2563eb",
          color: "white",
          fontWeight: 500,
          cursor: loading ? "wait" : "pointer"
        }}
      >
        {loading ? "Running..." : "Generate"}
      </button>

      {error && (
        <p style={{ marginTop: 12, color: "#b91c1c" }}>
          Error: {error}
        </p>
      )}

      <div style={{ marginTop: 20 }}>
        <label style={{ display: "block", fontWeight: 500, marginBottom: 8 }}>Output</label>
        <pre
          style={{
            whiteSpace: "pre-wrap",
            padding: 12,
            borderRadius: 8,
            background: "#0f172a",
            color: "#e5e7eb",
            minHeight: 80
          }}
        >
          {output || (loading ? "Waiting for response..." : "No output yet.")}
        </pre>
      </div>
    </div>
  );
}