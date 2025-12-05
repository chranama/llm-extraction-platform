export interface GenerateRequestBody {
  prompt: string;
  max_new_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  stop?: string[];
  model?: string;
}

export interface GenerateResponseBody {
  model: string;
  output: string;
  cached: boolean;
}

const API_BASE = "/api";
const API_KEY = import.meta.env.VITE_API_KEY;

export async function callGenerate(body: GenerateRequestBody): Promise<GenerateResponseBody> {
  const res = await fetch(`${API_BASE}/v1/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY,
    },
    body: JSON.stringify(body)
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }

  return res.json();
}