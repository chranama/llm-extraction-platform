import React from "react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { StatePanel } from "../StatePanel";

describe("StatePanel", () => {
  it("renders deployment state and model table, and refresh callbacks", () => {
    const onRefreshModels = vi.fn();
    const onRefreshPolicy = vi.fn();

    render(
      <StatePanel
        models={{
          default_model: "m1",
          deployment_capabilities: { generate: true, extract: false },
          models: [
            {
              id: "m1",
              default: true,
              loaded: true,
              capabilities: { generate: true, extract: false },
            },
          ],
        }}
        policy={{ status: "deny", ok: false }}
        effectiveExtractEnabled={false}
        loadingModels={false}
        loadingPolicy={false}
        onRefreshModels={onRefreshModels}
        onRefreshPolicy={onRefreshPolicy}
      />
    );

    expect(screen.getAllByText(/Deployment capabilities/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/default_model:/i)).toBeInTheDocument();
    expect(screen.getByText("m1")).toBeInTheDocument();
    expect(screen.getByText("DISABLED")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Refresh models/i }));
    fireEvent.click(screen.getByRole("button", { name: /Refresh policy/i }));
    expect(onRefreshModels).toHaveBeenCalledTimes(1);
    expect(onRefreshPolicy).toHaveBeenCalledTimes(1);
  });

  it("shows empty model list state and loading button states", () => {
    render(
      <StatePanel
        models={{ default_model: "m1", deployment_capabilities: { generate: true, extract: true }, models: [] }}
        policy={null}
        effectiveExtractEnabled={null}
        loadingModels={true}
        loadingPolicy={true}
        onRefreshModels={vi.fn()}
        onRefreshPolicy={vi.fn()}
      />
    );

    expect(screen.getByText(/No models returned/i)).toBeInTheDocument();
    const loadingButtons = screen.getAllByRole("button", { name: /Loading/i });
    expect(loadingButtons).toHaveLength(2);
    loadingButtons.forEach((btn) => expect(btn).toBeDisabled());
  });
});
