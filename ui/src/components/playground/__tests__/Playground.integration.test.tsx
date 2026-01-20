// ui/src/components/playground/__tests__/Playground.integration.test.tsx
import React from "react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, act, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Playground } from "../Playground";

// ---------------------------
// Mocks
// ---------------------------
vi.mock("../../../lib/api", () => {
  class ApiError extends Error {
    bodyJson?: any;
    constructor(message: string, bodyJson?: any) {
      super(message);
      this.name = "ApiError";
      this.bodyJson = bodyJson;
    }
  }

  return {
    callExtract: vi.fn(),
    callGenerate: vi.fn(),
    listSchemas: vi.fn(),
    getSchema: vi.fn(),
    getApiBaseUrl: vi.fn(),
    ApiError,
  };
});

vi.mock("../utils", async () => {
  const actual = await vi.importActual<typeof import("../utils")>("../utils");
  return {
    ...actual,
    copyToClipboard: vi.fn(),
  };
});

import { callExtract, callGenerate, listSchemas, getSchema, getApiBaseUrl } from "../../../lib/api";
import { copyToClipboard } from "../utils";

function mockSchemaJson() {
  return {
    title: "Receipt",
    description: "A schema",
    type: "object",
    required: ["total", "date"],
    properties: {
      total: { type: "string" },
      date: { type: "string" },
      merchant: { type: "string" },
      address: { type: "string" },
      items: { type: "array" },
    },
    additionalProperties: false,
  };
}

async function bootstrapWithSchemas() {
  (listSchemas as any).mockResolvedValueOnce([{ schema_id: "b" }, { schema_id: "a" }]);
  (getSchema as any).mockResolvedValueOnce(mockSchemaJson());

  render(<Playground />);

  await waitFor(() => expect(listSchemas).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(getSchema).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(getSchema).toHaveBeenCalledWith("a"));
}

function getTabButton(label: "Extract" | "Generate") {
  // Tab buttons are rendered before panel action buttons, so index 0 is the tab.
  return screen.getAllByRole("button", { name: label })[0];
}

function getActionButton(label: "Extract" | "Generate") {
  // Action button is the last match (tab comes first).
  const all = screen.getAllByRole("button", { name: label });
  return all[all.length - 1];
}

describe("Playground integration", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (getApiBaseUrl as any).mockReturnValue("/api");
  });

  afterEach(() => {
    try {
      vi.useRealTimers();
    } catch {
      // ignore
    }
  });

  it("boot sequence: listSchemas -> selects first sorted schema -> loads schema JSON", async () => {
    await bootstrapWithSchemas();

    expect(screen.getByDisplayValue("a")).toBeInTheDocument();
    expect(screen.getByText("Schema Inspector")).toBeInTheDocument();
    expect(screen.getByText("Receipt")).toBeInTheDocument();
  });

  it("extract happy path: calls callExtract with correct payload, renders output, auto-baseline enables Clear", async () => {
    await bootstrapWithSchemas();

    (callExtract as any).mockResolvedValueOnce({
      schema_id: "a",
      model: "mock-model",
      cached: false,
      repair_attempted: false,
      data: { total: "$12.34" },
    });

    const user = userEvent.setup();

    await user.click(getActionButton("Extract"));

    await waitFor(() => expect(callExtract).toHaveBeenCalledTimes(1));
    const args = (callExtract as any).mock.calls[0][0];

    expect(args).toEqual(
      expect.objectContaining({
        schema_id: "a",
        cache: true,
        repair: true,
        max_new_tokens: 512,
        temperature: 0,
      })
    );
    expect(args.model).toBeUndefined();

    expect(await screen.findByText(/"schema_id"\s*:\s*"a"/)).toBeInTheDocument();
    expect(screen.getByText(/"data"\s*:/)).toBeInTheDocument();

    expect(screen.getByRole("button", { name: "Clear" })).not.toBeDisabled();
  });

  it("mode switch clears active error (generate error -> switch to extract clears it)", async () => {
    await bootstrapWithSchemas();

    const user = userEvent.setup();

    // go to Generate tab
    await user.click(getTabButton("Generate"));

    (callGenerate as any).mockRejectedValueOnce(new Error("boom"));

    // run generate (action button)
    await user.click(getActionButton("Generate"));

    expect(await screen.findByText(/Error:\s*boom/i)).toBeInTheDocument();

    // switching tabs clears activeError
    await user.click(getTabButton("Extract"));

    await waitFor(() => {
      expect(screen.queryByText(/Error:\s*boom/i)).toBeNull();
    });
  });

 
  it("copy curl toast appears and auto-clears after 1500ms", async () => {
    // IMPORTANT: fake timers before render
    vi.useFakeTimers();

    (listSchemas as any).mockResolvedValueOnce([{ schema_id: "b" }, { schema_id: "a" }]);
    (getSchema as any).mockResolvedValueOnce(mockSchemaJson());
    (getApiBaseUrl as any).mockReturnValue("/api");
    (copyToClipboard as any).mockResolvedValueOnce(true);

    const flush = async (n = 4) => {
      // Run a few microtask ticks to let async effects settle
      for (let i = 0; i < n; i++) {
        await act(async () => {
          await Promise.resolve();
        });
      }
    };

    render(<Playground />);

    // let: listSchemas resolve -> schemaId set -> getSchema effect fires/resolves
    await flush(6);

    // Sanity: schema select should be populated with "a" and "b"
    // (and default selected should be "a" since sorted)
    expect(screen.getByRole("combobox")).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "a" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "b" })).toBeInTheDocument();
    expect(screen.getByRole("combobox")).toHaveValue("a");

    // Click Copy curl (Extract panel)
    fireEvent.click(screen.getByRole("button", { name: "Copy curl" }));

    // let: copyToClipboard resolve -> setCopyMsg
    await flush(2);

    expect(screen.getByText("Copied extract curl")).toBeInTheDocument();

    // Let the 1500ms auto-clear timer fire
    act(() => {
      vi.advanceTimersByTime(1500);
    });

    // flush state update caused by timer callback
    await flush(2);

    expect(screen.queryByText("Copied extract curl")).toBeNull();

    // Prevent timer mode leaking into other tests
    vi.useRealTimers();
  });
});