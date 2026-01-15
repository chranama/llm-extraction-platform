// ui/src/components/playground/curl.ts
import { getApiBaseUrl } from "../../lib/api";

function resolveCurlBaseUrl(): string {
  const apiBase = getApiBaseUrl(); // "/api" OR "http://host:8000/api"
  const trimmed = apiBase.endsWith("/") ? apiBase.slice(0, -1) : apiBase;

  if (/^https?:\/\//i.test(trimmed)) return trimmed;

  const rel = trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
  return `${window.location.origin}${rel}`;
}

export function buildCurlExtract(args: {
  schema_id: string;
  text: string;
  cache: boolean;
  repair: boolean;
  max_new_tokens: number;
  temperature: number;
  model?: string;
}): string {
  const baseUrl = resolveCurlBaseUrl();

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

export function buildCurlGenerate(args: {
  prompt: string;
  max_new_tokens: number;
  temperature: number;
  model?: string;
}): string {
  const baseUrl = resolveCurlBaseUrl();

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