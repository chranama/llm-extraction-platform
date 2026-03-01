import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
});

describe("runtime_config", () => {
  it("returns merged runtime config when /config.json is available", async () => {
    vi.stubEnv("VITE_API_BASE_URL", "http://fallback.local");
    vi.stubEnv("VITE_API_TIMEOUT_MS", "1234");
    vi.resetModules();

    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          api: { base_url: "http://runtime.local", timeout_ms: 9999 },
          defaults: { mode: "extract" },
        }),
        { status: 200 }
      ) as any
    );

    const mod = await import("../runtime_config");
    const cfg = await mod.loadRuntimeConfig();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0][0])).toContain("/config.json?ts=");
    expect(cfg.api?.base_url).toBe("http://runtime.local");
    expect(cfg.api?.timeout_ms).toBe(9999);
    expect(cfg.defaults?.mode).toBe("extract");
  });

  it("falls back to env defaults when config fetch is non-200", async () => {
    const expectedBase =
      (import.meta as any).env?.VITE_API_BASE_URL || "http://127.0.0.1:8000";
    const expectedTimeout = Number(
      (import.meta as any).env?.VITE_API_TIMEOUT_MS || 60000
    );
    vi.resetModules();

    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response("not found", { status: 404 }) as any
    );

    const mod = await import("../runtime_config");
    const cfg = await mod.loadRuntimeConfig();

    expect(cfg.api?.base_url).toBe(expectedBase);
    expect(cfg.api?.timeout_ms).toBe(expectedTimeout);
  });

  it("falls back to env defaults when fetch throws", async () => {
    const expectedBase =
      (import.meta as any).env?.VITE_API_BASE_URL || "http://127.0.0.1:8000";
    const expectedTimeout = Number(
      (import.meta as any).env?.VITE_API_TIMEOUT_MS || 60000
    );
    vi.resetModules();

    vi.spyOn(globalThis, "fetch").mockRejectedValueOnce(new Error("network down"));

    const mod = await import("../runtime_config");
    const cfg = await mod.loadRuntimeConfig();

    expect(cfg.api?.base_url).toBe(expectedBase);
    expect(cfg.api?.timeout_ms).toBe(expectedTimeout);
  });
});
