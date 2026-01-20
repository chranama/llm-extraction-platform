import { describe, expect, test } from "vitest";

describe("api.ts base URL normalization", async () => {
  test("setApiBaseUrl('/api') keeps '/api'", async () => {
    const mod = await import("../api");
    mod.setApiBaseUrl("/api");
    expect(mod.getApiBaseUrl()).toBe("/api");
  });

  test("setApiBaseUrl('http://localhost:8000') appends /api", async () => {
    const mod = await import("../api");
    mod.setApiBaseUrl("http://localhost:8000");
    expect(mod.getApiBaseUrl()).toBe("http://localhost:8000/api");
  });

  test("setApiBaseUrl('http://localhost:8000/') strips trailing slash and appends /api", async () => {
    const mod = await import("../api");
    mod.setApiBaseUrl("http://localhost:8000/");
    expect(mod.getApiBaseUrl()).toBe("http://localhost:8000/api");
  });

  test("setApiBaseUrl('http://localhost:8000/api') keeps /api (no double append)", async () => {
    const mod = await import("../api");
    mod.setApiBaseUrl("http://localhost:8000/api");
    expect(mod.getApiBaseUrl()).toBe("http://localhost:8000/api");
  });

  test("setApiBaseUrl ignores blank/undefined", async () => {
    const mod = await import("../api");
    mod.setApiBaseUrl("http://x:1");
    const before = mod.getApiBaseUrl();

    mod.setApiBaseUrl("");
    expect(mod.getApiBaseUrl()).toBe(before);

    mod.setApiBaseUrl(undefined);
    expect(mod.getApiBaseUrl()).toBe(before);

    mod.setApiBaseUrl(null);
    expect(mod.getApiBaseUrl()).toBe(before);
  });
});