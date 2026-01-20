// ui/src/components/playground/__tests__/SchemaInspector.test.tsx
import React from "react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// Make prettyJson deterministic and assert it was called
vi.mock("../utils", () => ({
  prettyJson: vi.fn((x: any) => `PJ(${JSON.stringify(x)})`),
}));

import { prettyJson } from "../utils";
import { SchemaInspector } from "../SchemaInspector";

function renderWith(overrides: Partial<React.ComponentProps<typeof SchemaInspector>> = {}) {
  const props: React.ComponentProps<typeof SchemaInspector> = {
    schemaId: "sroie_receipt_v1",
    schemaJson: { type: "object" },
    schemaJsonLoading: false,
    schemaJsonError: null,
    schemaSummary: {
      title: "Receipt",
      description: "A schema",
      requiredCount: 2,
      propertyCount: 5,
      additionalProperties: "false (strict)",
    },
    loading: false,
    onReload: vi.fn(),
    onCopyJson: vi.fn(),
    ...overrides,
  };

  render(<SchemaInspector {...props} />);
  return props;
}

describe("SchemaInspector", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    cleanup();
  });

  it("renders schema metadata and summary fields", () => {
    renderWith();

    expect(screen.getByText("Schema Inspector")).toBeInTheDocument();
    expect(screen.getByText("schema_id:")).toBeInTheDocument();
    expect(screen.getByText("sroie_receipt_v1")).toBeInTheDocument();

    expect(screen.getByText("title:")).toBeInTheDocument();
    expect(screen.getByText("Receipt")).toBeInTheDocument();

    expect(screen.getByText("description:")).toBeInTheDocument();
    expect(screen.getByText("A schema")).toBeInTheDocument();

    expect(screen.getByText(/required:/)).toBeInTheDocument();
    expect(screen.getByText(/properties:/)).toBeInTheDocument();
    expect(screen.getByText(/additionalProperties:/)).toBeInTheDocument();
  });

  it("reload button disabled when loading or schemaId empty, and calls onReload", async () => {
    const user = userEvent.setup();

    const props = renderWith({ loading: false, schemaId: "x" });
    await user.click(screen.getByRole("button", { name: "Reload" }));
    expect(props.onReload).toHaveBeenCalledTimes(1);

    cleanup();
    renderWith({ loading: true });
    expect(screen.getByRole("button", { name: "Reload" })).toBeDisabled();

    cleanup();
    renderWith({ schemaId: "" });
    expect(screen.getByRole("button", { name: "Reload" })).toBeDisabled();
  });

  it("copy JSON button disabled when schemaJsonLoading or no schemaJson, and calls onCopyJson", async () => {
    const user = userEvent.setup();

    const props = renderWith({ schemaJsonLoading: false, schemaJson: { type: "object" } });
    await user.click(screen.getByRole("button", { name: "Copy JSON" }));
    expect(props.onCopyJson).toHaveBeenCalledTimes(1);

    cleanup();
    renderWith({ schemaJsonLoading: true });
    expect(screen.getByRole("button", { name: "Copy JSON" })).toBeDisabled();

    cleanup();
    renderWith({ schemaJson: null });
    expect(screen.getByRole("button", { name: "Copy JSON" })).toBeDisabled();
  });

  it("shows schemaJsonError block when provided", () => {
    renderWith({ schemaJsonError: "boom" });
    expect(screen.getByText("boom")).toBeInTheDocument();
  });

  it("shows loading/empty states and calls prettyJson when schemaJson present", () => {
    renderWith({ schemaJsonLoading: true, schemaJson: null });
    expect(screen.getByText("Loading schema...")).toBeInTheDocument();

    cleanup();
    renderWith({ schemaJsonLoading: false, schemaJson: null });
    expect(screen.getByText("No schema loaded.")).toBeInTheDocument();

    cleanup();
    renderWith({ schemaJsonLoading: false, schemaJson: { type: "object" } });
    expect(prettyJson).toHaveBeenCalledTimes(1);
    expect(screen.getByText('PJ({"type":"object"})')).toBeInTheDocument();
  });

  it("shows (none) when schemaId is empty", () => {
    renderWith({ schemaId: "" });
    expect(screen.getByText("(none)")).toBeInTheDocument();
  });
});