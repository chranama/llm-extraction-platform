import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

vi.mock("../components/playground/Playground", () => ({
  Playground: () => <div data-testid="playground">Playground</div>,
}));
vi.mock("../components/admin/AdminPage", () => ({
  AdminPage: () => <div data-testid="admin">Admin</div>,
}));
vi.mock("../components/demo/DemoPage", () => ({
  DemoPage: () => <div data-testid="demo">Demo</div>,
}));

afterEach(() => {
  vi.unstubAllEnvs();
  vi.resetModules();
});

describe("App", () => {
  it("uses runtime api base and switches top tabs", async () => {
    vi.stubEnv("VITE_API_BASE_URL", "http://env.local");
    vi.stubEnv("VITE_API_KEY", "dev-key");
    vi.resetModules();

    const { default: App } = await import("../App");

    render(<App runtimeConfig={{ api: { base_url: "http://runtime.local/" } }} />);

    expect(screen.getByText(/API base:\s*http:\/\/runtime.local/i)).toBeInTheDocument();
    expect(screen.getByText(/API key: configured/i)).toBeInTheDocument();

    const docs = screen.getByRole("link", { name: /API docs/i });
    expect(docs).toHaveAttribute("href", "http://runtime.local/docs");

    expect(screen.getByTestId("playground")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Admin" }));
    expect(screen.getByTestId("admin")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Demo" }));
    expect(screen.getByTestId("demo")).toBeInTheDocument();
  });

  it("falls back to env API base when runtime base is missing", async () => {
    vi.stubEnv("VITE_API_BASE_URL", "http://env.local/");
    vi.stubEnv("VITE_API_KEY", "");
    vi.resetModules();

    const { default: App } = await import("../App");

    render(<App runtimeConfig={{}} />);

    expect(screen.getByText(/API base:\s*http:\/\/env.local/i)).toBeInTheDocument();
    expect(screen.getByText(/API key: missing/i)).toBeInTheDocument();
    expect(screen.getByText(/Heads up:/i)).toBeInTheDocument();
  });
});

