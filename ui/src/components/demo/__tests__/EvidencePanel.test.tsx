import React from "react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { EvidencePanel } from "../EvidencePanel";

describe("EvidencePanel", () => {
  it("renders computed stats, list rows and filter updates", () => {
    const onChangeCfg = vi.fn();
    const onRefresh = vi.fn();

    render(
      <EvidencePanel
        track="generate_clamp"
        cfg={{ route: "/v1/generate", modelId: "", limit: 50 }}
        onChangeCfg={onChangeCfg}
        logs={{
          total: 2,
          limit: 50,
          offset: 0,
          items: [
            {
              id: 1,
              created_at: "2026-01-01T10:11:12Z",
              route: "/v1/generate",
              model_id: "m1",
              latency_ms: 120,
              completion_tokens: 10,
              prompt: "p",
              output: "ok",
            },
            {
              id: 2,
              created_at: "2026-01-01T10:11:13Z",
              route: "/v1/generate",
              model_id: "m2",
              latency_ms: 200,
              completion_tokens: 20,
              prompt: "p",
              output: null,
            },
          ],
        }}
        loading={false}
        onRefresh={onRefresh}
      />
    );

    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText(/50.0%/i)).toBeInTheDocument();
    expect(screen.getByText(/200ms/i)).toBeInTheDocument();
    expect(screen.getByText(/160ms/i)).toBeInTheDocument();
    expect(screen.getByText("m1")).toBeInTheDocument();
    expect(screen.getByText("m2")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Refresh logs/i }));
    expect(onRefresh).toHaveBeenCalledTimes(1);

    fireEvent.change(screen.getByPlaceholderText("/v1/generate"), {
      target: { value: "/v1/custom" },
    });
    expect(onChangeCfg).toHaveBeenCalledWith(expect.objectContaining({ route: "/v1/custom" }));

    fireEvent.change(screen.getByPlaceholderText("(any)"), { target: { value: "model-x" } });
    expect(onChangeCfg).toHaveBeenCalledWith(expect.objectContaining({ modelId: "model-x" }));
  });

  it("clamps limit input and shows no logs state", () => {
    const onChangeCfg = vi.fn();
    render(
      <EvidencePanel
        track="extract_gating"
        cfg={{ route: "/v1/extract", modelId: "", limit: 50 }}
        onChangeCfg={onChangeCfg}
        logs={{ total: 0, limit: 50, offset: 0, items: [] }}
        loading={true}
        onRefresh={vi.fn()}
      />
    );

    expect(screen.getByText(/No logs/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Loading/i })).toBeDisabled();

    fireEvent.change(screen.getByRole("spinbutton"), { target: { value: "1000" } });
    expect(onChangeCfg).toHaveBeenCalledWith(expect.objectContaining({ limit: 500 }));
  });
});

