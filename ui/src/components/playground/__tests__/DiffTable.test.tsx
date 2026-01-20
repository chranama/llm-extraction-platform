// ui/src/components/playground/__tests__/DiffTable.test.tsx
import React from "react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import type { DiffRow } from "../types";
import { DiffTable } from "../DiffTable";

// Make valueCell deterministic / easy to assert calls
vi.mock("../utils", () => ({
  valueCell: vi.fn((v: any) => `CELL(${String(v)})`),
}));

import { valueCell } from "../utils";

function row(overrides: Partial<DiffRow> = {}): DiffRow {
  return {
    key: "field",
    required: false,
    baseline: "b",
    latest: "l",
    status: "same",
    inSchema: true,
    ...overrides,
  };
}

function renderWith(overrides: Partial<React.ComponentProps<typeof DiffTable>> = {}) {
  const props: React.ComponentProps<typeof DiffTable> = {
    rows: [],
    filteredRows: [],
    diffShowUnchanged: false,
    setDiffShowUnchanged: vi.fn(),
    autoBaseline: true,
    setAutoBaseline: vi.fn(),
    diffCounts: { same: 0, changed: 0, added: 0, removed: 0 },
    canSetBaseline: false,
    onSetBaseline: vi.fn(),
    canClearBaseline: false,
    onClearBaseline: vi.fn(),
    ...overrides,
  };

  render(<DiffTable {...props} />);
  return props;
}

describe("DiffTable", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    cleanup(); // <-- important: ensure no duplicate DOM between tests
  });

  it("renders header counts", () => {
    renderWith({
      diffCounts: { same: 10, changed: 1, added: 2, removed: 3 },
    });

    expect(screen.getByText("Per-field diff")).toBeInTheDocument();
    expect(screen.getByText("changed: 1, added: 2, removed: 3")).toBeInTheDocument();
  });

  it("shows 'No differences.' when filteredRows is empty", () => {
    renderWith({ filteredRows: [] });
    expect(screen.getByText("No differences.")).toBeInTheDocument();
  });

  it("toggles Auto-baseline checkbox", async () => {
    const user = userEvent.setup();
    const props = renderWith({ autoBaseline: false });

    const cb = screen.getByRole("checkbox", { name: "Auto-baseline" });
    expect(cb).not.toBeChecked();

    await user.click(cb);
    expect(props.setAutoBaseline).toHaveBeenCalledTimes(1);
    expect(props.setAutoBaseline).toHaveBeenCalledWith(true);
  });

  it("toggles Show unchanged checkbox", async () => {
    const user = userEvent.setup();
    const props = renderWith({ diffShowUnchanged: false });

    const cb = screen.getByRole("checkbox", { name: "Show unchanged" });
    expect(cb).not.toBeChecked();

    await user.click(cb);
    expect(props.setDiffShowUnchanged).toHaveBeenCalledTimes(1);
    expect(props.setDiffShowUnchanged).toHaveBeenCalledWith(true);
  });

  it("Set baseline and Clear buttons call handlers and respect disabled state", async () => {
    const user = userEvent.setup();

    // Disabled path
    renderWith({
      canSetBaseline: false,
      canClearBaseline: false,
    });

    const setBtn = screen.getByRole("button", { name: "Set baseline" });
    const clearBtn = screen.getByRole("button", { name: "Clear" });

    expect(setBtn).toBeDisabled();
    expect(clearBtn).toBeDisabled();

    // IMPORTANT: cleanup before rendering another DiffTable
    cleanup();

    // Enabled path
    const props2 = renderWith({
      canSetBaseline: true,
      canClearBaseline: true,
    });

    const setBtn2 = screen.getByRole("button", { name: "Set baseline" });
    const clearBtn2 = screen.getByRole("button", { name: "Clear" });

    expect(setBtn2).not.toBeDisabled();
    expect(clearBtn2).not.toBeDisabled();

    await user.click(setBtn2);
    expect(props2.onSetBaseline).toHaveBeenCalledTimes(1);

    await user.click(clearBtn2);
    expect(props2.onClearBaseline).toHaveBeenCalledTimes(1);
  });

  it("renders rows, required/extra badges, and status labels", () => {
    const rows: DiffRow[] = [
      row({ key: "a", status: "same", required: true, inSchema: true, baseline: "x", latest: "x" }),
      row({ key: "b", status: "changed", required: false, inSchema: true, baseline: 1, latest: 2 }),
      row({ key: "c", status: "added", required: false, inSchema: false, baseline: undefined, latest: "y" }),
      row({ key: "d", status: "removed", required: false, inSchema: true, baseline: "z", latest: undefined }),
    ];

    renderWith({ filteredRows: rows });

    // Table headers
    expect(screen.getByText("Field")).toBeInTheDocument();
    expect(screen.getByText("Baseline")).toBeInTheDocument();
    expect(screen.getByText("Latest")).toBeInTheDocument();
    expect(screen.getByText("Status")).toBeInTheDocument();

    // Keys render as code (textContent still matches)
    expect(screen.getByText("a")).toBeInTheDocument();
    expect(screen.getByText("b")).toBeInTheDocument();

    // Badges
    expect(screen.getByText("required")).toBeInTheDocument();
    expect(screen.getByText("extra")).toBeInTheDocument();

    // Status pills
    expect(screen.getByText("same")).toBeInTheDocument();
    expect(screen.getByText("changed")).toBeInTheDocument();
    expect(screen.getByText("added")).toBeInTheDocument();
    expect(screen.getByText("removed")).toBeInTheDocument();

    // valueCell called for baseline+latest cells for each row
    // (4 rows * 2 cells = 8 calls)
    expect(valueCell).toHaveBeenCalled();
    expect((valueCell as any).mock.calls.length).toBe(8);

    // Avoid "Found multiple elements" by using getAllByText when duplicates are expected
    expect(screen.getAllByText("CELL(x)").length).toBe(2); // baseline + latest for row a
    expect(screen.getByText("CELL(1)")).toBeInTheDocument();
    expect(screen.getByText("CELL(2)")).toBeInTheDocument();
    expect(screen.getAllByText("CELL(undefined)").length).toBe(2); // baseline of added + latest of removed
  });
});