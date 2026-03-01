import { beforeEach, describe, expect, it, vi } from "vitest";
import { makeModelsResponse, makeSchemaIndex } from "../../test/factories/api";

async function importApi() {
  vi.resetModules();
  return await import("../api");
}

describe("api.ts endpoint wrappers and error formatting", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("getSchema encodes schema id in URL path", async () => {
    const api = await importApi();
    api.setApiBaseUrl("/api");

    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ title: "x" }), { status: 200 }) as any
      );

    await api.getSchema("ticket v1/with/slash");

    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain(
      "/api/v1/schemas/ticket%20v1%2Fwith%2Fslash"
    );
    expect((init as any).method).toBe("GET");
  });

  it("listModels and getCapabilities use /v1/models", async () => {
    const api = await importApi();
    api.setApiBaseUrl("/api");

    const payload = makeModelsResponse({
      default_model: "m1",
      deployment_capabilities: { generate: true, extract: false },
    });

    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify(payload), { status: 200 }) as any)
      .mockResolvedValueOnce(new Response(JSON.stringify(payload), { status: 200 }) as any);

    const models = await api.listModels();
    const caps = await api.getCapabilities();

    expect(models.default_model).toBe("m1");
    expect(caps).toEqual({ generate: true, extract: false });

    expect(String(fetchMock.mock.calls[0][0])).toContain("/api/v1/models");
    expect(String(fetchMock.mock.calls[1][0])).toContain("/api/v1/models");
  });

  it("adminLoadModel posts JSON body", async () => {
    const api = await importApi();
    api.setApiBaseUrl("/api");

    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            ok: true,
            already_loaded: false,
            default_model: "m1",
            models: ["m1"],
          }),
          { status: 200 }
        ) as any
      );

    await api.adminLoadModel({ model_id: "m1" });

    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain("/api/v1/admin/models/load");
    expect((init as any).method).toBe("POST");
    expect(JSON.parse((init as any).body)).toEqual({ model_id: "m1" });
  });

  it("policy/admin action wrappers hit expected endpoints", async () => {
    const api = await importApi();
    api.setApiBaseUrl("/api");

    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify({ ok: true }), { status: 200 }) as any)
      .mockResolvedValueOnce(new Response(JSON.stringify({ ok: true }), { status: 200 }) as any)
      .mockResolvedValueOnce(new Response(JSON.stringify({ ok: true }), { status: 200 }) as any)
      .mockResolvedValueOnce(new Response(JSON.stringify({ ok: true }), { status: 200 }) as any);

    await api.adminGetPolicy();
    await api.adminReloadPolicy();
    await api.adminReloadRuntime();
    await api.adminWriteGenerateSlo({
      window_seconds: 300,
      route: "/v1/generate",
      model_id: "m1",
      out_path: "slo_out/generate/latest.json",
    });

    expect(String(fetchMock.mock.calls[0][0])).toContain("/api/v1/admin/policy");
    expect(String(fetchMock.mock.calls[1][0])).toContain("/api/v1/admin/policy/reload");
    expect(String(fetchMock.mock.calls[2][0])).toContain("/api/v1/admin/reload");
    expect(String(fetchMock.mock.calls[3][0])).toContain(
      "/api/v1/admin/slo/generate/write?"
    );
    expect(String(fetchMock.mock.calls[3][0])).toContain("window_seconds=300");
    expect(String(fetchMock.mock.calls[3][0])).toContain("route=%2Fv1%2Fgenerate");
    expect(String(fetchMock.mock.calls[3][0])).toContain("model_id=m1");
    expect(String(fetchMock.mock.calls[3][0])).toContain(
      "out_path=slo_out%2Fgenerate%2Flatest.json"
    );
  });

  it("absolute base joins directly with endpoint path", async () => {
    const api = await importApi();
    api.setApiBaseUrl("http://localhost:9000/");
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify(makeSchemaIndex("sroie_receipt_v1")), {
          status: 200,
        }) as any
      );

    await api.listSchemas();
    expect(String(fetchMock.mock.calls[0][0])).toBe(
      "http://localhost:9000/v1/schemas"
    );
  });

  it("ApiError includes nested detail and request_id", async () => {
    const api = await importApi();
    api.setApiBaseUrl("/api");
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          error: { code: "bad_request", detail: { reason: "bad-input" } },
          request_id: "req-123",
        }),
        { status: 400 }
      ) as any
    );

    const err = await api.listSchemas().catch((e: any) => e);
    expect(err).toMatchObject({
      name: "ApiError",
      status: 400,
    });
    expect(String(err.message)).toContain("bad_request");
    expect(String(err.message)).toContain('{"reason":"bad-input"}');
    expect(String(err.message)).toContain("request_id=req-123");
  });

  it("ApiError falls back to body text when no code/message fields exist", async () => {
    const api = await importApi();
    api.setApiBaseUrl("/api");
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ foo: "bar" }), { status: 500 }) as any
    );

    await expect(api.listSchemas()).rejects.toMatchObject({
      name: "ApiError",
      status: 500,
      message: 'HTTP 500: {"foo":"bar"}',
    });
  });
});
