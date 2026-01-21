// ui/src/components/playground/__tests__/Playground.test.tsx
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// ---- mock API layer (the only real side effects in Playground) ----
vi.mock("../../../lib/api", async () => {
  // We need the real ApiError class for instanceof checks to work
  const mod = await vi.importActual<typeof import("../../../lib/api")>(
    "../../../lib/api"
  );

  return {
    ...mod,
    callExtract: vi.fn(),
    callGenerate: vi.fn(),
    listSchemas: vi.fn(),
    getSchema: vi.fn(),
    getCapabilities: vi.fn(),
  };
});

import { Playground } from "../Playground";
import * as api from "../../../lib/api";
import { ApiError, type SchemaIndexItem } from "../../../lib/api";

// ---- stub child components to avoid testing their UI here ----
vi.mock("../ExtractPanel", () => ({
  ExtractPanel: (props: any) => (
    <div data-testid="extract-panel">
      <div>ExtractPanel</div>
      <div>schemaId:{props.schemaId}</div>

      {/* expose baseline-related flags */}
      <div>autoBaseline:{String(props.autoBaseline)}</div>
      <div>canSetBaseline:{String(props.canSetBaseline)}</div>
      <div>canClearBaseline:{String(props.canClearBaseline)}</div>

      <button onClick={props.onRunExtract} disabled={props.extractDisabled}>
        Run Extract
      </button>
    </div>
  ),
}));

vi.mock("../GeneratePanel", () => ({
  GeneratePanel: (props: any) => (
    <div data-testid="generate-panel">
      <div>GeneratePanel</div>
      <button onClick={props.onRun} disabled={props.loading}>
        Run Generate
      </button>
      {props.activeError ? <div>Error: {props.activeError}</div> : null}
      <pre data-testid="gen-output">{props.genOutput}</pre>
    </div>
  ),
}));

vi.mock("../SchemaInspector", () => ({
  SchemaInspector: (props: any) => (
    <div data-testid="schema-inspector">
      <div>SchemaInspector</div>
      <div>schemaId:{props.schemaId}</div>
      <pre data-testid="schema-json">
        {props.schemaJson ? JSON.stringify(props.schemaJson) : ""}
      </pre>
      {props.schemaJsonError ? (
        <div>SchemaError: {props.schemaJsonError}</div>
      ) : null}
    </div>
  ),
}));

beforeEach(() => {
  vi.clearAllMocks();

  // Default: full capabilities so tests that rely on Extract work unless they override.
  (api.getCapabilities as any).mockResolvedValue({
    generate: true,
    extract: true,
    mode: "full",
  });
});

function schemas(...ids: string[]): SchemaIndexItem[] {
  return ids.map((schema_id) => ({ schema_id }));
}

describe("Playground (capability gating)", () => {
  it("generate-only: does NOT call listSchemas/getSchema, Extract tab is disabled, Generate works", async () => {
    const user = userEvent.setup();

    (api.getCapabilities as any).mockResolvedValueOnce({
      generate: true,
      extract: false,
      mode: "generate-only",
    });

    (api.callGenerate as any).mockResolvedValueOnce({
      model: "m",
      output: "Hello!",
      cached: false,
    });

    render(<Playground />);

    await waitFor(() => expect(api.getCapabilities).toHaveBeenCalledTimes(1));

    expect(api.listSchemas).toHaveBeenCalledTimes(0);
    expect(api.getSchema).toHaveBeenCalledTimes(0);

    const extractTab = screen.getByRole("button", { name: /Extract/i });
    expect(extractTab).toBeDisabled();
    expect(extractTab.textContent || "").toMatch(/disabled/i);

    await waitFor(() => {
      expect(screen.getByTestId("generate-panel")).toBeInTheDocument();
    });

    await user.click(
      within(screen.getByTestId("generate-panel")).getByRole("button", {
        name: "Run Generate",
      })
    );

    await waitFor(() => expect(api.callGenerate).toHaveBeenCalledTimes(1));

    const out =
      within(screen.getByTestId("generate-panel")).getByTestId("gen-output")
        .textContent || "";
    expect(out).toContain('"model": "m"');
    expect(out).toContain('"cached": false');
    expect(out).toContain('"output": "Hello!"');
  });

  it("extract-enabled: loads schemas on mount, sorts them, selects first schemaId, loads schema JSON (after switching to Extract)", async () => {
    (api.listSchemas as any).mockResolvedValueOnce(
      schemas("z_schema", "a_schema", "m_schema")
    );
    (api.getSchema as any).mockResolvedValueOnce({ type: "object" });

    render(<Playground />);

    await waitFor(() => expect(api.getCapabilities).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.listSchemas).toHaveBeenCalledTimes(1));

    // Switch to Extract tab so ExtractPanel + SchemaInspector render
    fireEvent.click(screen.getByRole("button", { name: "Extract" }));

    await waitFor(() => {
      expect(screen.getByTestId("extract-panel")).toBeInTheDocument();
      expect(screen.getByTestId("schema-inspector")).toBeInTheDocument();
    });

    await waitFor(() => {
      const hits = screen.getAllByText("schemaId:a_schema");
      expect(hits.length).toBeGreaterThanOrEqual(1);
    });

    expect(api.getSchema).toHaveBeenCalledWith("a_schema");

    expect(
      within(screen.getByTestId("extract-panel")).getByText("schemaId:a_schema")
    ).toBeInTheDocument();
    expect(
      within(screen.getByTestId("schema-inspector")).getByText("schemaId:a_schema")
    ).toBeInTheDocument();
  });

  it("extract-enabled: Extract tab is clickable and shows ExtractPanel", async () => {
    (api.listSchemas as any).mockResolvedValueOnce(schemas("a_schema"));
    (api.getSchema as any).mockResolvedValueOnce({ type: "object" });

    render(<Playground />);

    await waitFor(() => expect(api.getCapabilities).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.listSchemas).toHaveBeenCalledTimes(1));

    const extractTab = screen.getByRole("button", { name: "Extract" });
    expect(extractTab).not.toBeDisabled();

    fireEvent.click(extractTab);

    await waitFor(() => {
      expect(screen.getByTestId("extract-panel")).toBeInTheDocument();
    });
  });

  it("Generate flow: switches tabs, calls callGenerate with payload, renders output", async () => {
    const user = userEvent.setup();

    (api.listSchemas as any).mockResolvedValueOnce(schemas("sroie_receipt_v1"));
    (api.getSchema as any).mockResolvedValueOnce({ type: "object" });

    (api.callGenerate as any).mockResolvedValueOnce({
      model: "m",
      output: "Hello!",
      cached: false,
    });

    render(<Playground />);

    await waitFor(() => expect(api.getCapabilities).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.listSchemas).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.getSchema).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => {
      expect(screen.getByTestId("generate-panel")).toBeInTheDocument();
    });

    await user.click(
      within(screen.getByTestId("generate-panel")).getByRole("button", {
        name: "Run Generate",
      })
    );

    await waitFor(() => expect(api.callGenerate).toHaveBeenCalledTimes(1));

    const out =
      within(screen.getByTestId("generate-panel")).getByTestId("gen-output")
        .textContent || "";
    expect(out).toContain('"model": "m"');
    expect(out).toContain('"cached": false');
    expect(out).toContain('"output": "Hello!"');
  });

  it("Generate error flow: ApiError with bodyJson is shown via prettyJson(bodyJson)", async () => {
    const user = userEvent.setup();

    (api.listSchemas as any).mockResolvedValueOnce(schemas("sroie_receipt_v1"));
    (api.getSchema as any).mockResolvedValueOnce({ type: "object" });

    const errBody = { code: "bad_request", message: "nope", extra: { why: "x" } };
    (api.callGenerate as any).mockRejectedValueOnce(
      new ApiError(400, JSON.stringify(errBody), errBody)
    );

    render(<Playground />);

    await waitFor(() => expect(api.getCapabilities).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.listSchemas).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.getSchema).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => {
      expect(screen.getByTestId("generate-panel")).toBeInTheDocument();
    });

    await user.click(
      within(screen.getByTestId("generate-panel")).getByRole("button", {
        name: "Run Generate",
      })
    );

    await waitFor(() => expect(api.callGenerate).toHaveBeenCalledTimes(1));

    const errText =
      within(screen.getByTestId("generate-panel")).getByText(/^Error:/)
        .textContent || "";
    expect(errText).toContain('"code": "bad_request"');
    expect(errText).toContain('"message": "nope"');
    expect(errText).toContain('"extra"');
  });

  it("schema JSON load error: ApiError with bodyJson is rendered in SchemaInspector props (extract-enabled, after switching to Extract)", async () => {
    (api.listSchemas as any).mockResolvedValueOnce(schemas("a_schema"));
    const errBody = { code: "not_found", message: "missing schema" };
    (api.getSchema as any).mockRejectedValueOnce(
      new ApiError(404, JSON.stringify(errBody), errBody)
    );

    render(<Playground />);

    await waitFor(() => expect(api.getCapabilities).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.listSchemas).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.getSchema).toHaveBeenCalledTimes(1));

    // SchemaInspector is only rendered in extract mode
    fireEvent.click(screen.getByRole("button", { name: "Extract" }));

    await waitFor(() => {
      expect(screen.getByTestId("schema-inspector")).toBeInTheDocument();
    });

    await waitFor(() => {
      const s =
        within(screen.getByTestId("schema-inspector")).getByText(/^SchemaError:/)
          .textContent || "";
      expect(s).toContain('"code": "not_found"');
      expect(s).toContain('"message": "missing schema"');
    });
  });

  it("Extract happy path (extract-enabled): first successful run sets baseline when autoBaseline=true", async () => {
    const user = userEvent.setup();

    (api.listSchemas as any).mockResolvedValueOnce(schemas("sroie_receipt_v1"));
    (api.getSchema as any).mockResolvedValueOnce({ type: "object" });

    (api.callExtract as any).mockResolvedValueOnce({
      schema_id: "sroie_receipt_v1",
      model: "m",
      cached: false,
      repair_attempted: false,
      data: { company: "ACME", total: "12.34" },
    });

    render(<Playground />);

    await waitFor(() => expect(api.getCapabilities).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.listSchemas).toHaveBeenCalledTimes(1));

    // Now actually render ExtractPanel
    fireEvent.click(screen.getByRole("button", { name: "Extract" }));

    await waitFor(() => {
      expect(screen.getByTestId("extract-panel")).toBeInTheDocument();
    });

    expect(
      within(screen.getByTestId("extract-panel")).getByText("autoBaseline:true")
    ).toBeInTheDocument();

    await user.click(
      within(screen.getByTestId("extract-panel")).getByRole("button", {
        name: "Run Extract",
      })
    );

    await waitFor(() => expect(api.callExtract).toHaveBeenCalledTimes(1));

    await waitFor(() => {
      expect(
        within(screen.getByTestId("extract-panel")).getByText("canSetBaseline:true")
      ).toBeInTheDocument();
      expect(
        within(screen.getByTestId("extract-panel")).getByText("canClearBaseline:true")
      ).toBeInTheDocument();
    });
  });
});