// ui/src/lib/api.ts

// -----------------------------
// Types: /v1/generate
// -----------------------------
export interface GenerateRequestBody {
  prompt: string;
  max_new_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  stop?: string[];
  model?: string;
  cache?: boolean; // backend supports cache on /v1/generate
}

export interface GenerateResponseBody {
  model: string;
  output: string;
  cached: boolean;
}

// -----------------------------
// Types: /v1/extract
// -----------------------------
export interface ExtractRequestBody {
  schema_id: string;
  text: string;

  model?: string;

  max_new_tokens?: number;
  temperature?: number;

  cache?: boolean;
  repair?: boolean;
}

export interface ExtractResponseBody {
  schema_id: string;
  model: string;
  data: Record<string, any>;
  cached: boolean;
  repair_attempted: boolean;
}

// -----------------------------
// Types: /v1/schemas
// -----------------------------
export interface SchemaIndexItem {
  schema_id: string;
  title?: string;
  description?: string;
}

// -----------------------------
// Types: /v1/schemas/{schema_id}
// -----------------------------
export type JsonSchema = Record<string, any>;

// -----------------------------
// Error type
// -----------------------------
export class ApiError extends Error {
  status: number;
  bodyText: string;
  bodyJson: any | null;

  constructor(status: number, bodyText: string, bodyJson: any | null = null) {
    super(ApiError.makeMessage(status, bodyText, bodyJson));
    this.name = "ApiError";
    this.status = status;
    this.bodyText = bodyText;
    this.bodyJson = bodyJson;
  }

  private static makeMessage(status: number, bodyText: string, bodyJson: any | null) {
    // If backend returns a structured error like:
    // { "code": "...", "message": "...", "extra": {...} }
    if (bodyJson && typeof bodyJson === "object") {
      const code = bodyJson.code ? String(bodyJson.code) : null;
      const msg = bodyJson.message ? String(bodyJson.message) : null;
      if (code || msg) {
        return `HTTP ${status}: ${code ?? "error"}${msg ? ` — ${msg}` : ""}`;
      }
    }
    return `HTTP ${status}: ${bodyText}`;
  }
}

const API_BASE = "/api";
const API_KEY = import.meta.env.VITE_API_KEY as string | undefined;

// ---- Dev UX: warn loudly if missing ----
//
// If you'd rather hard-fail instead of warn, replace with:
//   if (!API_KEY) throw new Error("VITE_API_KEY is not set ...");
//
if (!API_KEY) {
  // eslint-disable-next-line no-console
  console.warn("VITE_API_KEY is not set. Requests will fail with 401 missing_api_key.");
}

function authHeaders(): HeadersInit {
  // Backend expects X-API-Key per src/llm_server/api/deps.py
  return API_KEY ? { "X-API-Key": API_KEY } : {};
}

async function requestJson<T>(path: string, init: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);

  // Read body once
  const bodyText = await res.text();

  // Parse only a bounded preview (prevents slow JSON.parse on huge HTML error pages)
  const preview = bodyText.slice(0, 4000);
  let bodyJson: any | null = null;

  if (preview.trim()) {
    try {
      bodyJson = JSON.parse(preview);
    } catch {
      bodyJson = null;
    }
  }

  if (!res.ok) {
    throw new ApiError(res.status, preview, bodyJson);
  }

  // If empty response, return {} as T
  if (!bodyText.trim()) return {} as T;

  // Prefer parsed JSON if available; otherwise try parsing full body once (success path should be JSON)
  if (bodyJson !== null) return bodyJson as T;

  try {
    const full = JSON.parse(bodyText);
    return full as T;
  } catch {
    throw new ApiError(res.status, `Non-JSON response: ${preview}`, null);
  }
}

// -----------------------------
// API functions
// -----------------------------
export async function callGenerate(body: GenerateRequestBody): Promise<GenerateResponseBody> {
  return requestJson<GenerateResponseBody>("/v1/generate", {
    method: "POST",
    headers: {
      ...authHeaders(),
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
}

export async function callExtract(body: ExtractRequestBody): Promise<ExtractResponseBody> {
  return requestJson<ExtractResponseBody>("/v1/extract", {
    method: "POST",
    headers: {
      ...authHeaders(),
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
}

export async function listSchemas(): Promise<SchemaIndexItem[]> {
  return requestJson<SchemaIndexItem[]>("/v1/schemas", {
    method: "GET",
    headers: {
      ...authHeaders(),
    },
  });
}

export async function getSchema(schemaId: string): Promise<JsonSchema> {
  return requestJson<JsonSchema>(`/v1/schemas/${encodeURIComponent(schemaId)}`, {
    method: "GET",
    headers: {
      ...authHeaders(),
    },
  });
}