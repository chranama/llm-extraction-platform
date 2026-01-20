import { describe, expect, test, vi, beforeEach } from "vitest";
import type { GenerateRequestBody } from "../api";

function mockFetchOnce(status: number, bodyText: string) {
  // @ts-expect-error
  globalThis.fetch = vi.fn(async () => {
    return {
      ok: status >= 200 && status < 300,
      status,
      text: async () => bodyText,
    };
  });
}

describe("api.ts requestJson via exported API functions", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  test("callGenerate posts to /v1/generate and returns parsed JSON", async () => {
    const mod = await import("../api");
    mod.setApiBaseUrl("http://example.com"); // becomes http://example.com/api

    mockFetchOnce(
      200,
      JSON.stringify({ model: "m", output: "hi", cached: false })
    );

    const req: GenerateRequestBody = { prompt: "hello", max_new_tokens: 12 };
    const out = await mod.callGenerate(req);

    expect(out).toEqual({ model: "m", output: "hi", cached: false });

    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    const [url, init] = (globalThis.fetch as any).mock.calls[0];

    expect(url).toBe("http://example.com/api/v1/generate");
    expect(init.method).toBe("POST");
    expect(init.headers["Content-Type"]).toBe("application/json");
    expect(JSON.parse(init.body)).toMatchObject(req);
  });

  test("listSchemas calls GET /v1/schemas", async () => {
    const mod = await import("../api");
    mod.setApiBaseUrl("/api");

    mockFetchOnce(
      200,
      JSON.stringify([{ schema_id: "sroie_receipt_v1", title: "Receipt" }])
    );

    const xs = await mod.listSchemas();
    expect(xs[0].schema_id).toBe("sroie_receipt_v1");

    const [url, init] = (globalThis.fetch as any).mock.calls[0];
    expect(url).toBe("/api/v1/schemas");
    expect(init.method).toBe("GET");
  });

  test("non-200 returns ApiError with structured message when backend uses {code,message,extra}", async () => {
    const mod = await import("../api");
    mod.setApiBaseUrl("/api");

    mockFetchOnce(
      401,
      JSON.stringify({ code: "missing_api_key", message: "No key", extra: { hint: "set X-API-Key" } })
    );

    await expect(mod.listSchemas()).rejects.toMatchObject({
      name: "ApiError",
      status: 401,
    });

    try {
      await mod.listSchemas();
    } catch (e: any) {
      expect(String(e.message)).toContain("HTTP 401");
      expect(String(e.message)).toContain("missing_api_key");
      expect(String(e.message)).toContain("No key");
      expect(e.bodyJson?.extra?.hint).toBe("set X-API-Key");
    }
  });

  test("200 with non-JSON response throws ApiError (success path expects JSON)", async () => {
    const mod = await import("../api");
    mod.setApiBaseUrl("/api");

    mockFetchOnce(200, "<html>oops</html>");

    await expect(mod.listSchemas()).rejects.toMatchObject({
      name: "ApiError",
      status: 200,
    });
  });

  test("empty 200 response returns {} (for endpoints that might do that)", async () => {
    const mod = await import("../api");
    mod.setApiBaseUrl("/api");

    mockFetchOnce(200, "");

    // listSchemas expects SchemaIndexItem[], but requestJson returns {} as T.
    // We won't call listSchemas here; instead call a function that returns object-ish.
    // Easiest: callGenerate and accept it returns {} (contract check is higher-level).
    mockFetchOnce(200, "");
    const out = await mod.callGenerate({ prompt: "x" } as any);
    expect(out).toEqual({});
  });
});