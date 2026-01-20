// ui/src/components/playground/__tests__/curl.test.ts
import { describe, expect, it, vi, beforeEach } from "vitest";

// Mock api base getter (used by curl.ts)
vi.mock("../../../lib/api", () => ({
  getApiBaseUrl: vi.fn(),
}));

import { getApiBaseUrl } from "../../../lib/api";
import { buildCurlExtract, buildCurlGenerate } from "../curl";

describe("playground/curl", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("buildCurlExtract uses window.location.origin when api base is relative", () => {
    (getApiBaseUrl as any).mockReturnValueOnce("/api");

    const out = buildCurlExtract({
      schema_id: "sroie_receipt_v1",
      text: "O'Hara receipt", // exercise single-quote escaping
      cache: true,
      repair: true,
      max_new_tokens: 512,
      temperature: 0,
    });

    expect(out).toContain(`curl -s ${window.location.origin}/api/v1/extract \\`);
    expect(out).toContain(`-H "Content-Type: application/json" \\`);
    expect(out).toContain(`-H "X-API-Key: YOUR_KEY" \\`);

    expect(out).toContain(`"schema_id":"sroie_receipt_v1"`);
    expect(out).toContain(`"cache":true`);
    expect(out).toContain(`"repair":true`);
    expect(out).toContain(`"max_new_tokens":512`);
    expect(out).toContain(`"temperature":0`);

    expect(out).toContain(`O'\\''Hara`);
  });

  it("buildCurlExtract uses absolute api base as-is (no origin prefix)", () => {
    (getApiBaseUrl as any).mockReturnValueOnce("http://localhost:8000/api/");

    const out = buildCurlExtract({
      schema_id: "sroie_receipt_v1",
      text: "hello",
      cache: false,
      repair: false,
      max_new_tokens: 10,
      temperature: 0.1,
      model: "  my-model  ",
    });

    expect(out).toContain(`curl -s http://localhost:8000/api/v1/extract \\`);
    expect(out).toContain(`"model":"my-model"`);
  });

  it("buildCurlGenerate includes model only when provided, trims model, and uses /v1/generate", () => {
    // IMPORTANT: buildCurlGenerate is called twice here; use persistent mock value
    (getApiBaseUrl as any).mockReturnValue("/api");

    const out1 = buildCurlGenerate({
      prompt: "hi",
      max_new_tokens: 128,
      temperature: 0.7,
    });

    expect(out1).toContain(`${window.location.origin}/api/v1/generate`);
    expect(out1).toContain(`"prompt":"hi"`);
    expect(out1).toContain(`"max_new_tokens":128`);
    expect(out1).toContain(`"temperature":0.7`);
    expect(out1).not.toContain(`"model"`);

    const out2 = buildCurlGenerate({
      prompt: "hi",
      max_new_tokens: 128,
      temperature: 0.7,
      model: "  m  ",
    });

    expect(out2).toContain(`"model":"m"`);
  });
});