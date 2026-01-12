// ui/src/components/Playground.tsx
import React, { useEffect, useMemo, useState } from "react";
import {
  callExtract,
  callGenerate,
  listSchemas,
  getSchema,
  ApiError,
  SchemaIndexItem,
  JsonSchema,
} from "../lib/api";

type Mode = "extract" | "generate";

function prettyJson(x: any): string {
  try {
    return JSON.stringify(x, null, 2);
  } catch {
    return String(x);
  }
}

function toNumberOr(prev: number, raw: string): number {
  const s = raw.trim();
  if (!s) return prev;
  const n = Number(s);
  return Number.isFinite(n) ? n : prev;
}

async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      ta.style.top = "-9999px";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return ok;
    } catch {
      return false;
    }
  }
}

function buildCurlExtract(args: {
  schema_id: string;
  text: string;
  cache: boolean;
  repair: boolean;
  max_new_tokens: number;
  temperature: number;
  model?: string;
}): string {
  const baseUrl = "http://localhost:8080/api";

  const body: any = {
    schema_id: args.schema_id,
    text: args.text,
    cache: args.cache,
    repair: args.repair,
    max_new_tokens: args.max_new_tokens,
    temperature: args.temperature,
  };
  if (args.model && args.model.trim()) body.model = args.model.trim();

  return [
    `curl -s ${baseUrl}/v1/extract \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -H "X-API-Key: YOUR_KEY" \\`,
    `  -d '${JSON.stringify(body).replace(/'/g, "'\\''")}' | jq '.'`,
  ].join("\n");
}

function buildCurlGenerate(args: {
  prompt: string;
  max_new_tokens: number;
  temperature: number;
  model?: string;
}): string {
  const baseUrl = "http://localhost:8080/api";
  const body: any = {
    prompt: args.prompt,
    max_new_tokens: args.max_new_tokens,
    temperature: args.temperature,
  };
  if (args.model && args.model.trim()) body.model = args.model.trim();

  return [
    `curl -s ${baseUrl}/v1/generate \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -H "X-API-Key: YOUR_KEY" \\`,
    `  -d '${JSON.stringify(body).replace(/'/g, "'\\''")}' | jq '.'`,
  ].join("\n");
}

function summarizeSchema(schema: JsonSchema | null): {
  title?: string;
  description?: string;
  requiredCount: number;
  propertyCount: number;
  additionalProperties: string;
} {
  if (!schema || typeof schema !== "object") {
    return { requiredCount: 0, propertyCount: 0, additionalProperties: "unknown" };
  }

  const title = typeof (schema as any).title === "string" ? (schema as any).title : undefined;
  const description =
    typeof (schema as any).description === "string" ? (schema as any).description : undefined;

  const req = Array.isArray((schema as any).required) ? (schema as any).required : [];
  const props = (schema as any).properties && typeof (schema as any).properties === "object" ? (schema as any).properties : {};
  const propertyCount = props ? Object.keys(props).length : 0;

  const ap = (schema as any).additionalProperties;
  let additionalProperties: string = "unspecified";
  if (ap === false) additionalProperties = "false (strict)";
  else if (ap === true) additionalProperties = "true";
  else if (ap && typeof ap === "object") additionalProperties = "schema (object)";
  else if (ap === undefined) additionalProperties = "unspecified";

  return {
    title,
    description,
    requiredCount: req.length,
    propertyCount,
    additionalProperties,
  };
}

function stableStringify(x: any): string {
  // Deterministic stringify for equality checks (handles objects/arrays)
  const seen = new WeakSet();
  const helper = (v: any): any => {
    if (v && typeof v === "object") {
      if (seen.has(v)) return "[Circular]";
      seen.add(v);
      if (Array.isArray(v)) return v.map(helper);
      const out: Record<string, any> = {};
      Object.keys(v)
        .sort()
        .forEach((k) => {
          out[k] = helper(v[k]);
        });
      return out;
    }
    return v;
  };
  try {
    return JSON.stringify(helper(x));
  } catch {
    return String(x);
  }
}

function valueCell(v: any): string {
  if (v === undefined) return "(missing)";
  if (v === null) return "null";
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  return prettyJson(v);
}

type DiffRow = {
  key: string;
  required: boolean;
  baseline: any;
  latest: any;
  status: "same" | "changed" | "added" | "removed";
  inSchema: boolean;
};

function buildPerFieldDiff(args: {
  schemaJson: JsonSchema | null;
  baseline: Record<string, any> | null;
  latest: Record<string, any> | null;
}): DiffRow[] {
  const schemaProps =
    args.schemaJson && typeof (args.schemaJson as any).properties === "object"
      ? ((args.schemaJson as any).properties as Record<string, any>)
      : {};

  const requiredSet = new Set<string>(
    args.schemaJson && Array.isArray((args.schemaJson as any).required) ? ((args.schemaJson as any).required as string[]) : []
  );

  const baselineObj = args.baseline ?? {};
  const latestObj = args.latest ?? {};

  // 1) keys from schema first (stable, predictable order)
  const schemaKeys = Object.keys(schemaProps).sort();

  // 2) include any extra keys returned by the model that aren't in schema
  const extraKeys = Array.from(
    new Set([...Object.keys(baselineObj), ...Object.keys(latestObj)].filter((k) => !schemaProps[k]))
  ).sort();

  const keys = [...schemaKeys, ...extraKeys];

  const rows: DiffRow[] = keys.map((k) => {
    const b = (args.baseline ? baselineObj[k] : undefined);
    const l = (args.latest ? latestObj[k] : undefined);

    const bHas = args.baseline ? Object.prototype.hasOwnProperty.call(baselineObj, k) : false;
    const lHas = args.latest ? Object.prototype.hasOwnProperty.call(latestObj, k) : false;

    let status: DiffRow["status"] = "same";
    if (!args.baseline || !args.latest) {
      // if either side missing, we can still label presence
      if (!args.baseline && lHas) status = "added";
      else if (!args.latest && bHas) status = "removed";
      else status = "same";
    } else {
      if (bHas && !lHas) status = "removed";
      else if (!bHas && lHas) status = "added";
      else {
        const same = stableStringify(b) === stableStringify(l);
        status = same ? "same" : "changed";
      }
    }

    return {
      key: k,
      required: requiredSet.has(k),
      baseline: args.baseline ? b : undefined,
      latest: args.latest ? l : undefined,
      status,
      inSchema: Boolean(schemaProps[k]),
    };
  });

  return rows;
}

export function Playground() {
  // -----------------------------
  // Mode (tabs)
  // -----------------------------
  const [mode, setMode] = useState<Mode>("extract");

  // -----------------------------
  // Shared UI state
  // -----------------------------
  const [loading, setLoading] = useState(false);

  // Split errors by mode (less confusing when switching tabs)
  const [extractError, setExtractError] = useState<string | null>(null);
  const [generateError, setGenerateError] = useState<string | null>(null);

  const activeError = mode === "extract" ? extractError : generateError;
  const setActiveError = (msg: string | null) => {
    if (mode === "extract") setExtractError(msg);
    else setGenerateError(msg);
  };

  // -----------------------------
  // Schemas (/v1/schemas)
  // -----------------------------
  const [schemas, setSchemas] = useState<SchemaIndexItem[]>([]);
  const [schemasLoading, setSchemasLoading] = useState(false);

  const schemaOptions = useMemo(() => {
    return [...schemas].sort((a, b) => (a.schema_id || "").localeCompare(b.schema_id || ""));
  }, [schemas]);

  // -----------------------------
  // Schema Inspector (/v1/schemas/{schema_id})
  // -----------------------------
  const [schemaJson, setSchemaJson] = useState<JsonSchema | null>(null);
  const [schemaJsonLoading, setSchemaJsonLoading] = useState(false);
  const [schemaJsonError, setSchemaJsonError] = useState<string | null>(null);

  // -----------------------------
  // Extract state
  // -----------------------------
  const [schemaId, setSchemaId] = useState<string>(""); // filled after schemas load
  const [extractText, setExtractText] = useState<string>(
    "Company: ACME Corp\nDate: 2024-01-01\nTotal: $12.34\nAddress: 123 Main St, Springfield"
  );
  const [extractOutput, setExtractOutput] = useState<string>("");

  const [extractCache, setExtractCache] = useState<boolean>(true);
  const [extractRepair, setExtractRepair] = useState<boolean>(true);
  const [extractMaxNewTokens, setExtractMaxNewTokens] = useState<number>(512);
  const [extractTemperature, setExtractTemperature] = useState<number>(0.0);

  // Keep structured results for diffing
  const [extractDataLatest, setExtractDataLatest] = useState<Record<string, any> | null>(null);
  const [extractDataBaseline, setExtractDataBaseline] = useState<Record<string, any> | null>(null);
  const [autoBaseline, setAutoBaseline] = useState<boolean>(true);

  // Diff filter UI
  const [diffShowUnchanged, setDiffShowUnchanged] = useState<boolean>(false);

  // Copy feedback
  const [copyMsg, setCopyMsg] = useState<string | null>(null);
  useEffect(() => {
    if (!copyMsg) return;
    const t = window.setTimeout(() => setCopyMsg(null), 1500);
    return () => window.clearTimeout(t);
  }, [copyMsg]);

  // -----------------------------
  // Generate state
  // -----------------------------
  const [prompt, setPrompt] = useState("Write a haiku about autumn leaves.");
  const [genOutput, setGenOutput] = useState("");

  const [modelOverride, setModelOverride] = useState<string>("");
  const [genMaxNewTokens, setGenMaxNewTokens] = useState<number>(128);
  const [genTemperature, setGenTemperature] = useState<number>(0.7);

  // -----------------------------
  // Load schema index on mount (+ allow reload)
  // -----------------------------
  async function loadSchemas() {
    setSchemasLoading(true);
    try {
      const items = await listSchemas();
      setSchemas(items);

      setSchemaId((prev) => {
        if (prev) return prev;
        const sorted = [...items].sort((a, b) => (a.schema_id || "").localeCompare(b.schema_id || ""));
        return sorted.length > 0 ? sorted[0].schema_id : "";
      });
    } catch (e: any) {
      setExtractError(e?.message ?? "Failed to load schemas");
    } finally {
      setSchemasLoading(false);
    }
  }

  useEffect(() => {
    let canceled = false;

    (async () => {
      if (canceled) return;
      await loadSchemas();
    })();

    return () => {
      canceled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // -----------------------------
  // Load schema JSON whenever schemaId changes
  // -----------------------------
  useEffect(() => {
    let canceled = false;

    async function loadSchemaJson(id: string) {
      if (!id) {
        setSchemaJson(null);
        setSchemaJsonError(null);
        return;
      }

      setSchemaJsonLoading(true);
      setSchemaJsonError(null);

      try {
        const js = await getSchema(id);
        if (canceled) return;
        setSchemaJson(js);
      } catch (e: any) {
        if (canceled) return;
        if (e instanceof ApiError && e.bodyJson) setSchemaJsonError(prettyJson(e.bodyJson));
        else setSchemaJsonError(e?.message ?? "Failed to load schema JSON");
        setSchemaJson(null);
      } finally {
        if (!canceled) setSchemaJsonLoading(false);
      }
    }

    loadSchemaJson(schemaId);

    return () => {
      canceled = true;
    };
  }, [schemaId]);

  // Reset diff baseline when schema changes (prevents confusing cross-schema comparisons)
  useEffect(() => {
    setExtractDataLatest(null);
    setExtractDataBaseline(null);
  }, [schemaId]);

  const handleRunExtract = async () => {
    setLoading(true);
    setExtractError(null);
    setExtractOutput("");
    setExtractDataLatest(null);

    try {
      if (!schemaId) {
        throw new Error("No schema selected. Is /v1/schemas returning anything?");
      }

      const res = await callExtract({
        schema_id: schemaId,
        text: extractText,
        cache: extractCache,
        repair: extractRepair,
        max_new_tokens: extractMaxNewTokens,
        temperature: extractTemperature,
        ...(modelOverride.trim() ? { model: modelOverride.trim() } : {}),
      });

      const view = {
        schema_id: res.schema_id,
        model: res.model,
        cached: res.cached,
        repair_attempted: res.repair_attempted,
        data: res.data,
      };

      setExtractOutput(prettyJson(view));
      setExtractDataLatest(res.data ?? {});

      setExtractDataBaseline((prev) => {
        if (autoBaseline && !prev) return res.data ?? {};
        return prev;
      });
    } catch (e: any) {
      if (e instanceof ApiError && e.bodyJson) {
        setExtractError(prettyJson(e.bodyJson));
      } else {
        setExtractError(e?.message ?? "Request failed");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleRunGenerate = async () => {
    setLoading(true);
    setGenerateError(null);
    setGenOutput("");

    try {
      const res = await callGenerate({
        prompt,
        max_new_tokens: genMaxNewTokens,
        temperature: genTemperature,
        ...(modelOverride.trim() ? { model: modelOverride.trim() } : {}),
      });

      const view = {
        model: res.model,
        cached: res.cached,
        output: res.output,
      };
      setGenOutput(prettyJson(view));
    } catch (e: any) {
      if (e instanceof ApiError && e.bodyJson) {
        setGenerateError(prettyJson(e.bodyJson));
      } else {
        setGenerateError(e?.message ?? "Request failed");
      }
    } finally {
      setLoading(false);
    }
  };

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
      onClick={() => {
        onClick();
        setActiveError(null);
      }}
      disabled={loading}
      style={{
        padding: "0.4rem 0.8rem",
        borderRadius: 999,
        border: active ? "1px solid #2563eb" : "1px solid #cbd5f5",
        background: active ? "#2563eb" : "white",
        color: active ? "white" : "#0f172a",
        fontWeight: 600,
        cursor: loading ? "wait" : "pointer",
      }}
    >
      {label}
    </button>
  );

  const extractDisabled = loading || schemasLoading || schemaOptions.length === 0 || !schemaId;

  const schemaSummary = useMemo(() => summarizeSchema(schemaJson), [schemaJson]);

  const handleCopyExtractCurl = async () => {
    const curl = buildCurlExtract({
      schema_id: schemaId || "SCHEMA_ID",
      text: extractText,
      cache: extractCache,
      repair: extractRepair,
      max_new_tokens: extractMaxNewTokens,
      temperature: extractTemperature,
      model: modelOverride.trim() ? modelOverride.trim() : undefined,
    });
    const ok = await copyToClipboard(curl);
    setCopyMsg(ok ? "Copied extract curl" : "Copy failed");
  };

  const handleCopyGenerateCurl = async () => {
    const curl = buildCurlGenerate({
      prompt,
      max_new_tokens: genMaxNewTokens,
      temperature: genTemperature,
      model: modelOverride.trim() ? modelOverride.trim() : undefined,
    });
    const ok = await copyToClipboard(curl);
    setCopyMsg(ok ? "Copied generate curl" : "Copy failed");
  };

  const handleCopySchemaJson = async () => {
    const txt = schemaJson ? prettyJson(schemaJson) : "";
    const ok = await copyToClipboard(txt);
    setCopyMsg(ok ? "Copied schema JSON" : "Copy failed");
  };

  const diffRows = useMemo(() => {
    return buildPerFieldDiff({
      schemaJson,
      baseline: extractDataBaseline,
      latest: extractDataLatest,
    });
  }, [schemaJson, extractDataBaseline, extractDataLatest]);

  const filteredDiffRows = useMemo(() => {
    if (diffShowUnchanged) return diffRows;
    return diffRows.filter((r) => r.status !== "same");
  }, [diffRows, diffShowUnchanged]);

  const diffCounts = useMemo(() => {
    const c = { same: 0, changed: 0, added: 0, removed: 0 };
    for (const r of diffRows) c[r.status] += 1;
    return c;
  }, [diffRows]);

  const handleSetBaseline = () => {
    if (!extractDataLatest) return;
    setExtractDataBaseline(extractDataLatest);
    setCopyMsg("Baseline set");
  };

  return (
    <div style={{ maxWidth: 1180 }}>
      {/* Tabs + model override */}
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 16 }}>
        <TabButton label="Extract" active={mode === "extract"} onClick={() => setMode("extract")} />
        <TabButton label="Generate" active={mode === "generate"} onClick={() => setMode("generate")} />

        <div style={{ flex: 1 }} />

        {copyMsg && (
          <span
            style={{
              fontSize: 12,
              padding: "4px 10px",
              borderRadius: 999,
              border: "1px solid #cbd5f5",
              background: "#f1f5f9",
              color: "#0f172a",
              fontWeight: 600,
            }}
          >
            {copyMsg}
          </span>
        )}

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <label style={{ fontWeight: 500, color: "#334155" }}>Model</label>
          <input
            value={modelOverride}
            onChange={(e) => setModelOverride(e.target.value)}
            placeholder="(optional override)"
            style={{
              width: 240,
              padding: "8px 10px",
              borderRadius: 10,
              border: "1px solid #cbd5f5",
              fontFamily: "inherit",
            }}
          />
        </div>
      </div>

      {/* EXTRACT: two-column layout */}
      {mode === "extract" && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1.25fr 0.85fr",
            gap: 16,
            alignItems: "start",
          }}
        >
          {/* Left: Extract input + output + diff */}
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
                  onClick={loadSchemas}
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
                <input
                  type="checkbox"
                  checked={extractCache}
                  onChange={(e) => setExtractCache(e.target.checked)}
                  disabled={loading}
                />
                Cache
              </label>

              <label style={{ display: "flex", alignItems: "center", gap: 8, color: "#334155" }}>
                <input
                  type="checkbox"
                  checked={extractRepair}
                  onChange={(e) => setExtractRepair(e.target.checked)}
                  disabled={loading}
                />
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
                onClick={handleCopyExtractCurl}
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
              onClick={handleRunExtract}
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

            {/* Per-field diff */}
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
                  <input
                    type="checkbox"
                    checked={autoBaseline}
                    onChange={(e) => setAutoBaseline(e.target.checked)}
                  />
                  Auto-baseline
                </label>

                <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "#334155" }}>
                  <input
                    type="checkbox"
                    checked={diffShowUnchanged}
                    onChange={(e) => setDiffShowUnchanged(e.target.checked)}
                  />
                  Show unchanged
                </label>

                <button
                  onClick={handleSetBaseline}
                  disabled={!extractDataLatest}
                  style={{
                    padding: "0.35rem 0.7rem",
                    borderRadius: 10,
                    border: "1px solid #cbd5f5",
                    background: "white",
                    cursor: !extractDataLatest ? "not-allowed" : "pointer",
                    fontWeight: 700,
                    fontSize: 12,
                  }}
                  title="Set baseline to the latest successful extract data"
                >
                  Set baseline
                </button>

                <button
                  onClick={() => {
                    setExtractDataBaseline(null);
                    setCopyMsg("Baseline cleared");
                  }}
                  disabled={!extractDataBaseline}
                  style={{
                    padding: "0.35rem 0.7rem",
                    borderRadius: 10,
                    border: "1px solid #cbd5f5",
                    background: "white",
                    cursor: !extractDataBaseline ? "not-allowed" : "pointer",
                    fontWeight: 700,
                    fontSize: 12,
                  }}
                  title="Clear baseline"
                >
                  Clear
                </button>
              </div>

              <div style={{ padding: 10 }}>
                {!extractDataLatest && !extractDataBaseline ? (
                  <div style={{ fontSize: 12, color: "#64748b" }}>
                    Run an extract to populate <strong>latest</strong>. Use <strong>Set baseline</strong> to compare runs.
                  </div>
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
                        {filteredDiffRows.map((r) => {
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

                          const borderLeft =
                            r.required && !r.inSchema
                              ? "4px solid #ef4444"
                              : r.required
                              ? "4px solid #2563eb"
                              : "4px solid transparent";

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
                        {filteredDiffRows.length === 0 && (
                          <tr>
                            <td colSpan={4} style={{ padding: 10, color: "#64748b" }}>
                              No differences.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right: Schema Inspector */}
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
                onClick={async () => {
                  if (!schemaId) return;
                  setSchemaJsonLoading(true);
                  setSchemaJsonError(null);
                  try {
                    const js = await getSchema(schemaId);
                    setSchemaJson(js);
                    setCopyMsg("Schema reloaded");
                  } catch (e: any) {
                    if (e instanceof ApiError && e.bodyJson) setSchemaJsonError(prettyJson(e.bodyJson));
                    else setSchemaJsonError(e?.message ?? "Failed to reload schema JSON");
                    setSchemaJson(null);
                  } finally {
                    setSchemaJsonLoading(false);
                  }
                }}
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
                onClick={handleCopySchemaJson}
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

            {/* Summary */}
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
                {schemaJsonLoading
                  ? "Loading schema..."
                  : schemaJson
                  ? prettyJson(schemaJson)
                  : "No schema loaded."}
              </pre>
            </div>
          </div>
        </div>
      )}

      {/* GENERATE: single column */}
      {mode === "generate" && (
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
              onClick={handleCopyGenerateCurl}
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
            onClick={handleRunGenerate}
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
      )}
    </div>
  );
}