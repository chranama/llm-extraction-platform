import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const renderMock = vi.fn();
const createRootMock = vi.fn(() => ({ render: renderMock }));
const loadRuntimeConfigMock = vi.fn();
const setApiBaseUrlMock = vi.fn();

vi.mock("react-dom/client", () => ({
  default: {
    createRoot: createRootMock,
  },
}));

vi.mock("../lib/runtime_config", () => ({
  loadRuntimeConfig: loadRuntimeConfigMock,
}));

vi.mock("../lib/api", () => ({
  setApiBaseUrl: setApiBaseUrlMock,
}));

vi.mock("../App", () => ({
  default: ({ runtimeConfig }: any) => (
    <div data-testid="app-root">{runtimeConfig?.api?.base_url ?? "none"}</div>
  ),
}));

describe("main bootstrap", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    document.body.innerHTML = '<div id="root"></div>';
  });

  afterEach(() => {
    vi.resetModules();
  });

  it("loads runtime config, sets API base, and renders App", async () => {
    loadRuntimeConfigMock.mockResolvedValueOnce({
      api: { base_url: "http://runtime.local" },
    });

    await import("../main");

    await Promise.resolve();
    await Promise.resolve();

    expect(loadRuntimeConfigMock).toHaveBeenCalledTimes(1);
    expect(setApiBaseUrlMock).toHaveBeenCalledWith("http://runtime.local");
    expect(createRootMock).toHaveBeenCalledTimes(1);
    expect(createRootMock).toHaveBeenCalledWith(document.getElementById("root"));
    expect(renderMock).toHaveBeenCalledTimes(1);
  });
});

