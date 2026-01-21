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
    getCapabilities: vi.fn(),
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

import {
  callExtract,
  callGenerate,
  listSchemas,
  getSchema,
  getApiBaseUrl,
  getCapabilities,
} from "../../../lib/api";
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

function getTabButton(label: "Extract" | "Generate") {
  // Tab buttons are rendered before panel action buttons, so index 0 is the tab.
  return screen.getAllByRole("button", { name: label })[0];
}

function getActionButton(label: "Extract" | "Generate") {
  // Action button is the last match (tab comes first).
  const all = screen.getAllByRole("button", { name: label });
  return all[all.length - 1];
}

async function ensureExtractTabVisible(user?: ReturnType<typeof userEvent.setup>) {
  const tab = getTabButton("Extract");
  // In generate-only mode this would be disabled; in these tests we set extract=true.
  expect(tab).not.toBeDisabled();
  if (user) await user.click(tab);
  else fireEvent.click(tab);
  await waitFor(() => {
    // Extract mode renders the schema combobox
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });
}

async function bootstrapWithSchemas() {
  // Full mode by default unless overridden in a test
  (getCapabilities as any).mockResolvedValueOnce({
    generate: true,
    extract: true,
    mode: "full",
  });

  (listSchemas as any).mockResolvedValueOnce([{ schema_id: "b" }, { schema_id: "a" }]);
  (getSchema as any).mockResolvedValueOnce(mockSchemaJson());

  render(<Playground />);

  // Capabilities gate happens before schema boot now
  await waitFor(() => expect(getCapabilities).toHaveBeenCalledTimes(1));

  // Schemas only load when extract is enabled, but Extract UI is only rendered in Extract mode.
  // So: click Extract tab first, then assert schema boot.
  fireEvent.click(getTabButton("Extract"));

  await waitFor(() => expect(listSchemas).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(getSchema).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(getSchema).toHaveBeenCalledWith("a"));
}

describe("Playground integration", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (getApiBaseUrl as any).mockReturnValue("/api");
    // Default for tests that *don’t* use bootstrapWithSchemas()
    (getCapabilities as any).mockResolvedValue({
      generate: true,
      extract: true,
      mode: "full",
    });
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

    // We are already in Extract mode from bootstrapWithSchemas(), so action button is available.
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

    (getCapabilities as any).mockResolvedValueOnce({
      generate: true,
      extract: true,
      mode: "full",
    });

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

    // let capabilities + initial effects settle
    await flush(4);

    // Ensure we are in Extract mode so the combobox + Copy curl exist
    fireEvent.click(getTabButton("Extract"));
    await flush(6);

    // Sanity: schema select should be populated with "a" and "b"
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