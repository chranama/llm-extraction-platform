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
});

function schemas(...ids: string[]): SchemaIndexItem[] {
  return ids.map((schema_id) => ({ schema_id }));
}

describe("Playground", () => {
  it("loads schemas on mount, sorts them, and selects the first schemaId", async () => {
    (api.listSchemas as any).mockResolvedValueOnce(
      schemas("z_schema", "a_schema", "m_schema")
    );
    (api.getSchema as any).mockResolvedValueOnce({ type: "object" });

    render(<Playground />);

    await waitFor(() => expect(api.listSchemas).toHaveBeenCalledTimes(1));

    // schemaId appears in BOTH stubs: ExtractPanel and SchemaInspector
    await waitFor(() => {
      const hits = screen.getAllByText("schemaId:a_schema");
      expect(hits.length).toBeGreaterThanOrEqual(1);
    });

    expect(api.getSchema).toHaveBeenCalledWith("a_schema");

    // Optional: assert each panel got the right schemaId
    expect(
      within(screen.getByTestId("extract-panel")).getByText("schemaId:a_schema")
    ).toBeInTheDocument();
    expect(
      within(screen.getByTestId("schema-inspector")).getByText(
        "schemaId:a_schema"
      )
    ).toBeInTheDocument();
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

    // Let initial mount effects complete
    await waitFor(() => expect(api.listSchemas).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.getSchema).toHaveBeenCalledTimes(1));

    // Switch to Generate tab (use fireEvent for reliability)
    const tab = screen.getByRole("button", { name: "Generate" });
    fireEvent.click(tab);

    // Ensure generate panel is shown
    await waitFor(() => {
      expect(screen.getByTestId("generate-panel")).toBeInTheDocument();
    });

    // Click the action button INSIDE the generate panel
    await user.click(
      within(screen.getByTestId("generate-panel")).getByRole("button", {
        name: "Run Generate",
      })
    );

    await waitFor(() => expect(api.callGenerate).toHaveBeenCalledTimes(1));

    expect(api.callGenerate).toHaveBeenCalledWith(
      expect.objectContaining({
        prompt: "Write a haiku about autumn leaves.",
        max_new_tokens: 128,
        temperature: 0.7,
      })
    );

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

    await waitFor(() => expect(api.listSchemas).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.getSchema).toHaveBeenCalledTimes(1));

    // Switch to Generate tab (use fireEvent for reliability)
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

  it("schema JSON load error: ApiError with bodyJson is rendered in SchemaInspector props", async () => {
    (api.listSchemas as any).mockResolvedValueOnce(schemas("a_schema"));
    const errBody = { code: "not_found", message: "missing schema" };
    (api.getSchema as any).mockRejectedValueOnce(
      new ApiError(404, JSON.stringify(errBody), errBody)
    );

    render(<Playground />);

    await waitFor(() => expect(api.listSchemas).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.getSchema).toHaveBeenCalledTimes(1));

    await waitFor(() => {
      const s =
        within(screen.getByTestId("schema-inspector")).getByText(/^SchemaError:/)
          .textContent || "";
      expect(s).toContain('"code": "not_found"');
      expect(s).toContain('"message": "missing schema"');
    });
  });

  it("Extract happy path: first successful run sets baseline when autoBaseline=true", async () => {
    const user = userEvent.setup();

    (api.listSchemas as any).mockResolvedValueOnce(
      schemas("sroie_receipt_v1")
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

    await waitFor(() => expect(api.listSchemas).toHaveBeenCalledTimes(1));

    // defaults visible via ExtractPanel stub
    expect(
      within(screen.getByTestId("extract-panel")).getByText("autoBaseline:true")
    ).toBeInTheDocument();
    expect(
      within(screen.getByTestId("extract-panel")).getByText(
        "canClearBaseline:false"
      )
    ).toBeInTheDocument();
    expect(
      within(screen.getByTestId("extract-panel")).getByText(
        "canSetBaseline:false"
      )
    ).toBeInTheDocument();

    await user.click(
      within(screen.getByTestId("extract-panel")).getByRole("button", {
        name: "Run Extract",
      })
    );

    await waitFor(() => expect(api.callExtract).toHaveBeenCalledTimes(1));

    expect(api.callExtract).toHaveBeenCalledWith(
      expect.objectContaining({
        schema_id: "sroie_receipt_v1",
        cache: true,
        repair: true,
        max_new_tokens: 512,
        temperature: 0,
      })
    );

    // after successful run: latest exists -> canSetBaseline true; autoBaseline sets baseline -> canClearBaseline true
    await waitFor(() => {
      expect(
        within(screen.getByTestId("extract-panel")).getByText(
          "canSetBaseline:true"
        )
      ).toBeInTheDocument();
      expect(
        within(screen.getByTestId("extract-panel")).getByText(
          "canClearBaseline:true"
        )
      ).toBeInTheDocument();
    });
  });
});