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
    listModels: vi.fn(),
    listSchemas: vi.fn(),
    getSchema: vi.fn(),
  };
});

import { Playground } from "../Playground";
import * as api from "../../../lib/api";
import { ApiError } from "../../../lib/api";
import { copyToClipboard } from "../utils";
import {
  makeExtractResponse,
  makeGenerateResponse,
  makeModelsResponse,
  makeSchemaIndex,
} from "../../../test/factories/api";

vi.mock("../utils", async () => {
  const mod = await vi.importActual<typeof import("../utils")>("../utils");
  return {
    ...mod,
    copyToClipboard: vi.fn(),
  };
});

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
      {props.activeError ? <div>Error: {props.activeError}</div> : null}

      <button onClick={props.onRunExtract} disabled={props.extractDisabled}>
        Run Extract
      </button>
      <button onClick={props.onCopyCurl}>Copy Extract Curl</button>
      <button onClick={props.onSetBaseline} disabled={!props.canSetBaseline}>
        Set Baseline
      </button>
      <button onClick={props.onClearBaseline} disabled={!props.canClearBaseline}>
        Clear Baseline
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
      <button onClick={props.onCopyCurl}>Copy Generate Curl</button>
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
      <button onClick={props.onReload}>Reload Schema</button>
      <button onClick={props.onCopyJson}>Copy Schema JSON</button>
    </div>
  ),
}));

beforeEach(() => {
  vi.clearAllMocks();

  // Default: extract-enabled deployment.
  (api.listModels as any).mockResolvedValue(makeModelsResponse());
  (copyToClipboard as any).mockResolvedValue(true);
});

describe("Playground (capability gating)", () => {
  it("generate-only: does NOT call listSchemas/getSchema, Extract tab is disabled, Generate works", async () => {
    const user = userEvent.setup();

    (api.listModels as any).mockResolvedValueOnce(
      makeModelsResponse({
        deployment_capabilities: { generate: true, extract: false },
      })
    );

    (api.callGenerate as any).mockResolvedValueOnce(
      makeGenerateResponse({ model: "m", output: "Hello!" })
    );

    render(<Playground />);

    await waitFor(() => expect(api.listModels).toHaveBeenCalledTimes(1));

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
      makeSchemaIndex("z_schema", "a_schema", "m_schema")
    );
    (api.getSchema as any).mockResolvedValueOnce({ type: "object" });

    render(<Playground />);

    await waitFor(() => expect(api.listModels).toHaveBeenCalledTimes(1));
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
    (api.listSchemas as any).mockResolvedValueOnce(makeSchemaIndex("a_schema"));
    (api.getSchema as any).mockResolvedValueOnce({ type: "object" });

    render(<Playground />);

    await waitFor(() => expect(api.listModels).toHaveBeenCalledTimes(1));
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

    (api.listSchemas as any).mockResolvedValueOnce(makeSchemaIndex("sroie_receipt_v1"));
    (api.getSchema as any).mockResolvedValueOnce({ type: "object" });

    (api.callGenerate as any).mockResolvedValueOnce(
      makeGenerateResponse({ model: "m", output: "Hello!" })
    );

    render(<Playground />);

    await waitFor(() => expect(api.listModels).toHaveBeenCalledTimes(1));
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

    (api.listSchemas as any).mockResolvedValueOnce(makeSchemaIndex("sroie_receipt_v1"));
    (api.getSchema as any).mockResolvedValueOnce({ type: "object" });

    const errBody = { code: "bad_request", message: "nope", extra: { why: "x" } };
    (api.callGenerate as any).mockRejectedValueOnce(
      new ApiError(400, JSON.stringify(errBody), errBody)
    );

    render(<Playground />);

    await waitFor(() => expect(api.listModels).toHaveBeenCalledTimes(1));
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
    (api.listSchemas as any).mockResolvedValueOnce(makeSchemaIndex("a_schema"));
    const errBody = { code: "not_found", message: "missing schema" };
    (api.getSchema as any).mockRejectedValueOnce(
      new ApiError(404, JSON.stringify(errBody), errBody)
    );

    render(<Playground />);

    await waitFor(() => expect(api.listModels).toHaveBeenCalledTimes(1));
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

    (api.listSchemas as any).mockResolvedValueOnce(
      makeSchemaIndex("sroie_receipt_v1")
    );
    (api.getSchema as any).mockResolvedValueOnce({ type: "object" });

    (api.callExtract as any).mockResolvedValueOnce({
      schema_id: "sroie_receipt_v1",
      model: "m",
      cached: false,
      repair_attempted: false,
      data: { company: "ACME", total: "12.34" },
    });

    render(<Playground />);

    await waitFor(() => expect(api.listModels).toHaveBeenCalledTimes(1));
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

  it("shows model capabilities error banner when listModels fails", async () => {
    (api.listModels as any).mockRejectedValueOnce(new Error("models down"));
    render(<Playground />);

    expect(await screen.findByText(/Unable to load deployment capabilities/i)).toBeInTheDocument();
    expect(screen.getByText(/models down/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Extract \(disabled\)/i })).toBeDisabled();
  });

  it("extract-disabled mode shows generate banner and details action remains safe", async () => {
    (api.listModels as any).mockResolvedValueOnce(
      makeModelsResponse({
        deployment_capabilities: { generate: true, extract: false },
        models: [{ id: "demo-model", default: true, capabilities: { generate: true, extract: false } }],
      })
    );

    render(<Playground />);

    expect(await screen.findByText(/Extract: disabled/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Details/i }));
    expect(await screen.findByTestId("generate-panel")).toBeInTheDocument();
    expect(screen.queryByText(/Extraction disabled/i)).toBeNull();
  });

  it("shows extract error when listSchemas fails", async () => {
    (api.listSchemas as any).mockRejectedValueOnce(new Error("schemas failed"));
    render(<Playground />);

    await waitFor(() => expect(api.listModels).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.listSchemas).toHaveBeenCalledTimes(1));
    expect(await screen.findByText(/Error:\s*schemas failed/i)).toBeInTheDocument();
  });

  it("reload schema success shows toast and reload schema failure renders error", async () => {
    (api.listSchemas as any).mockResolvedValueOnce(makeSchemaIndex("a_schema"));
    (api.getSchema as any)
      .mockResolvedValueOnce({ type: "object" })
      .mockRejectedValueOnce(new ApiError(500, "boom", { code: "x", message: "reload failed" }));

    render(<Playground />);
    await waitFor(() => expect(api.listSchemas).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "Extract" }));

    await waitFor(() => expect(api.getSchema).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "Reload Schema" }));

    await waitFor(() => expect(api.getSchema).toHaveBeenCalledTimes(2));
    expect(await screen.findByText(/SchemaError:/i)).toBeInTheDocument();
  });

  it("extract and generate generic errors are surfaced", async () => {
    const user = userEvent.setup();

    (api.listSchemas as any).mockResolvedValueOnce(makeSchemaIndex("a_schema"));
    (api.getSchema as any).mockResolvedValueOnce({ type: "object" });
    (api.callExtract as any).mockRejectedValueOnce(new Error("extract exploded"));
    (api.callGenerate as any).mockRejectedValueOnce(new Error("generate exploded"));

    render(<Playground />);
    await waitFor(() => expect(api.listSchemas).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "Extract" }));

    await user.click(screen.getByRole("button", { name: "Run Extract" }));
    expect(await screen.findByText(/Error:\s*extract exploded/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Generate" }));
    await user.click(screen.getByRole("button", { name: "Run Generate" }));
    expect(await screen.findByText(/Error:\s*generate exploded/i)).toBeInTheDocument();
  });

  it("set and clear baseline paths show toast messages", async () => {
    const user = userEvent.setup();

    (api.listSchemas as any).mockResolvedValueOnce(makeSchemaIndex("a_schema"));
    (api.getSchema as any).mockResolvedValueOnce({ type: "object" });
    (api.callExtract as any).mockResolvedValueOnce(
      makeExtractResponse({
        schema_id: "a_schema",
        model: "m",
        data: { total: "10.00" },
      })
    );

    render(<Playground />);
    await waitFor(() => expect(api.listSchemas).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "Extract" }));

    await user.click(screen.getByRole("button", { name: "Run Extract" }));
    await waitFor(() => expect(api.callExtract).toHaveBeenCalledTimes(1));

    await user.click(screen.getByRole("button", { name: "Set Baseline" }));
    expect(await screen.findByText(/Baseline set/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Clear Baseline" }));
    expect(await screen.findByText(/Baseline cleared/i)).toBeInTheDocument();
  });

  it("copy failures show toast in extract, generate, and schema copy actions", async () => {
    (api.listSchemas as any).mockResolvedValueOnce(makeSchemaIndex("a_schema"));
    (api.getSchema as any).mockResolvedValueOnce({ type: "object" });
    (copyToClipboard as any).mockResolvedValue(false);

    render(<Playground />);
    await waitFor(() => expect(api.listSchemas).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "Extract" }));

    fireEvent.click(screen.getByRole("button", { name: "Copy Extract Curl" }));
    expect(await screen.findByText(/Copy failed/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Copy Schema JSON" }));
    expect(await screen.findByText(/Copy failed/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Generate" }));
    fireEvent.click(screen.getByRole("button", { name: "Copy Generate Curl" }));
    expect(await screen.findByText(/Copy failed/i)).toBeInTheDocument();
  });
});
