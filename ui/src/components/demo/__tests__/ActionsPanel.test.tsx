import React from "react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ActionsPanel } from "../ActionsPanel";

const baseCfg = {
  sloWindowSeconds: 300,
  sloRoute: "/v1/generate",
  sloModelId: "",
  sloOutPath: "",
};

describe("ActionsPanel", () => {
  it("wires config updates and action callbacks", () => {
    const onChangeCfg = vi.fn();
    const onWriteGenerateSlo = vi.fn();
    const onReloadPolicy = vi.fn();
    const onReloadRuntime = vi.fn();

    render(
      <ActionsPanel
        track="generate_clamp"
        cfg={baseCfg}
        onChangeCfg={onChangeCfg}
        onWriteGenerateSlo={onWriteGenerateSlo}
        onReloadPolicy={onReloadPolicy}
        onReloadRuntime={onReloadRuntime}
        disabled={false}
      />
    );

    fireEvent.change(screen.getByDisplayValue("300"), { target: { value: "120" } });
    expect(onChangeCfg).toHaveBeenCalledWith(expect.objectContaining({ sloWindowSeconds: 120 }));

    fireEvent.change(screen.getByPlaceholderText("/v1/generate"), {
      target: { value: "/v1/custom" },
    });
    expect(onChangeCfg).toHaveBeenCalledWith(expect.objectContaining({ sloRoute: "/v1/custom" }));

    fireEvent.click(screen.getByRole("button", { name: /Write Generate SLO Snapshot/i }));
    expect(onWriteGenerateSlo).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: /Reload Policy Snapshot/i }));
    expect(onReloadPolicy).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: /Reload Runtime/i }));
    expect(onReloadRuntime).toHaveBeenCalledTimes(1);
  });

  it("disables actions when API key is unavailable", () => {
    render(
      <ActionsPanel
        track="extract_gating"
        cfg={baseCfg}
        onChangeCfg={vi.fn()}
        onWriteGenerateSlo={vi.fn()}
        onReloadPolicy={vi.fn()}
        onReloadRuntime={vi.fn()}
        disabled={true}
      />
    );

    expect(screen.getByRole("button", { name: /Write Generate SLO Snapshot/i })).toBeDisabled();
    expect(screen.getByText(/Admin actions are disabled/i)).toBeInTheDocument();
    expect(screen.getByText(/Demo B flow/i)).toBeInTheDocument();
  });
});

