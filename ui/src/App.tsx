import React from "react";
import { Playground } from "./components/Playground";

export default function App() {
  return (
    <div style={{ minHeight: "100vh", padding: "1.5rem", fontFamily: "system-ui, sans-serif" }}>
      <header style={{ marginBottom: "1.5rem" }}>
        <h1 style={{ fontSize: "1.75rem", fontWeight: 600 }}>LLM Server Playground</h1>
        <p style={{ color: "#64748b" }}>
          Talk to your FastAPI LLM backend. This is a thin UI on top of <code>/v1/generate</code> and
          <code> /v1/generate/batch</code>.
        </p>
      </header>

      <Playground />
    </div>
  );
}