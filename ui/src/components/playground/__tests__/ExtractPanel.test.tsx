// ui/src/components/playground/__tests__/ExtractPanel.test.tsx
import React from "react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import type { DiffRow } from "../types";
import type { SchemaIndexItem } from "../../../lib/api";
import { ExtractPanel } from "../ExtractPanel";

// Stub DiffTable so we only verify wiring + presence
vi.mock("../DiffTable", () => ({
  DiffTable: (props: any) => (
    <div data-testid="diff-table">
      <div>DiffTable</div>
      <div>filteredRows:{props.filteredRows?.length ?? 0}</div>
      <div>autoBaseline:{String(props.autoBaseline)}</div>
      <div>diffShowUnchanged:{String(props.diffShowUnchanged)}</div>
      <button onClick={props.onSetBaseline} disabled={!props.canSetBaseline}>
        Set baseline
      </button>
      <button onClick={props.onClearBaseline} disabled={!props.canClearBaseline}>
        Clear
      </button>
    </div>
  ),
}));

function renderWith(
  overrides: Partial<React.ComponentProps<typeof ExtractPanel>> = {}
): React.ComponentProps<typeof ExtractPanel> {
  const props: React.ComponentProps<typeof ExtractPanel> = {
    loading: false,

    schemaId: "sroie_receipt_v1",
    setSchemaId: vi.fn(),
    schemasLoading: false,
    schemaOptions: [{ schema_id: "sroie_receipt_v1" }, { schema_id: "other" }] as SchemaIndexItem[],
    onReloadSchemas: vi.fn(),

    extractCache: true,
    setExtractCache: vi.fn(),
    extractRepair: true,
    setExtractRepair: vi.fn(),
    extractMaxNewTokens: 512,
    setExtractMaxNewTokens: vi.fn(),
    extractTemperature: 0,
    setExtractTemperature: vi.fn(),

    extractText: "hello",
    setExtractText: vi.fn(),
    extractDisabled: false,
    onRunExtract: vi.fn(),
    onCopyCurl: vi.fn(),

    activeError: null,
    extractOutput: "",

    diffRows: [] as DiffRow[],
    filteredDiffRows: [] as DiffRow[],
    diffCounts: { same: 0, changed: 0, added: 0, removed: 0 },
    diffShowUnchanged: false,
    setDiffShowUnchanged: vi.fn(),
    autoBaseline: true,
    setAutoBaseline: vi.fn(),
    canSetBaseline: false,
    onSetBaseline: vi.fn(),
    canClearBaseline: false,
    onClearBaseline: vi.fn(),

    schemaJson: null,

    toNumberOr: (prev, raw) => {
      const s = raw.trim();
      if (!s) return prev;
      const n = Number(s);
      return Number.isFinite(n) ? n : prev;
    },

    ...overrides,
  };

  render(<ExtractPanel {...props} />);
  return props;
}

describe("ExtractPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    cleanup(); // prevent duplicate DOM when multiple renderWith() happen across tests
  });

  it("renders schema select, options, and reload button", () => {
    renderWith();

    expect(screen.getByText("Schema")).toBeInTheDocument();
    expect(screen.getByRole("combobox")).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "sroie_receipt_v1" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reload" })).toBeInTheDocument();
  });

  it("changes schema via select", async () => {
    const user = userEvent.setup();
    const props = renderWith({ schemaId: "sroie_receipt_v1" });

    await user.selectOptions(screen.getByRole("combobox"), "other");
    expect(props.setSchemaId).toHaveBeenCalledTimes(1);
    expect(props.setSchemaId).toHaveBeenCalledWith("other");
  });

  it("reload button calls onReloadSchemas", async () => {
    const user = userEvent.setup();
    const props = renderWith();

    await user.click(screen.getByRole("button", { name: "Reload" }));
    expect(props.onReloadSchemas).toHaveBeenCalledTimes(1);
  });

  it("toggles cache and repair checkboxes", async () => {
    const user = userEvent.setup();
    const props = renderWith({ extractCache: false, extractRepair: false });

    const cache = screen.getByRole("checkbox", { name: "Cache" });
    const repair = screen.getByRole("checkbox", { name: "Repair" });

    await user.click(cache);
    expect(props.setExtractCache).toHaveBeenCalledWith(true);

    await user.click(repair);
    expect(props.setExtractRepair).toHaveBeenCalledWith(true);
  });

  it("uses toNumberOr for numeric inputs", async () => {
    const user = userEvent.setup();

    // deterministic spy: returns parsed number when possible
    const toNumberOr = vi.fn((prev: number, raw: string) => {
      const s = String(raw).trim();
      if (!s) return prev;
      const n = Number(s);
      return Number.isFinite(n) ? n : prev;
    });

    const props = renderWith({
      extractMaxNewTokens: 512,
      extractTemperature: 0,
      toNumberOr,
    });

    const [maxTokens, temp] = screen.getAllByRole("spinbutton");

    // ---- max_new_tokens ----
    await user.clear(maxTokens);
    await user.type(maxTokens, "999");

    // We should call toNumberOr with prev=512 at least once
    const maxParserCalls = (toNumberOr as any).mock.calls.filter((c: any[]) => c[0] === 512);
    expect(maxParserCalls.length).toBeGreaterThan(0);

    // And we should eventually call the setter with whatever toNumberOr returned
    // for one of those prev=512 calls.
    const maxSetterArgs = new Set((props.setExtractMaxNewTokens as any).mock.calls.map((c: any[]) => c[0]));
    const maxExpected = new Set(maxParserCalls.map((c: any[]) => toNumberOr(512, c[1])));
    expect([...maxExpected].some((v) => maxSetterArgs.has(v))).toBe(true);

    // ---- temperature ----
    await user.clear(temp);
    await user.type(temp, "999");

    const tempParserCalls = (toNumberOr as any).mock.calls.filter((c: any[]) => c[0] === 0);
    expect(tempParserCalls.length).toBeGreaterThan(0);

    const tempSetterArgs = new Set((props.setExtractTemperature as any).mock.calls.map((c: any[]) => c[0]));
    const tempExpected = new Set(tempParserCalls.map((c: any[]) => toNumberOr(0, c[1])));
    expect([...tempExpected].some((v) => tempSetterArgs.has(v))).toBe(true);
  });

  it("copy curl calls onCopyCurl", async () => {
    const user = userEvent.setup();
    const props = renderWith();

    await user.click(screen.getByRole("button", { name: "Copy curl" }));
    expect(props.onCopyCurl).toHaveBeenCalledTimes(1);
  });

  it("extract button calls onRunExtract and respects disabled/loading label", async () => {
    const user = userEvent.setup();

    const props = renderWith({ loading: false, extractDisabled: false });
    await user.click(screen.getByRole("button", { name: "Extract" }));
    expect(props.onRunExtract).toHaveBeenCalledTimes(1);

    cleanup();
    renderWith({ loading: true, extractDisabled: true });
    expect(screen.getByRole("button", { name: "Running..." })).toBeDisabled();
  });

  it("renders output placeholder and error", () => {
    renderWith({ loading: false, extractOutput: "" });
    expect(screen.getByText("No output yet.")).toBeInTheDocument();

    cleanup();
    renderWith({ loading: true, extractOutput: "" });
    expect(screen.getByText("Waiting for response...")).toBeInTheDocument();

    cleanup();
    renderWith({ activeError: "boom" });
    expect(screen.getByText(/Error: boom/)).toBeInTheDocument();
  });

  it("wires DiffTable props and baseline buttons", async () => {
    const user = userEvent.setup();
    const props = renderWith({
      filteredDiffRows: [{ key: "a", required: false, baseline: 1, latest: 2, status: "changed", inSchema: true }],
      autoBaseline: true,
      diffShowUnchanged: false,
      canSetBaseline: true,
      canClearBaseline: true,
    });

    expect(screen.getByTestId("diff-table")).toBeInTheDocument();
    expect(screen.getByText("filteredRows:1")).toBeInTheDocument();
    expect(screen.getByText("autoBaseline:true")).toBeInTheDocument();
    expect(screen.getByText("diffShowUnchanged:false")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Set baseline" }));
    expect(props.onSetBaseline).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(props.onClearBaseline).toHaveBeenCalledTimes(1);
  });
});