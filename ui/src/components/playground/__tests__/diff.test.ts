// ui/src/components/playground/__tests__/diff.test.ts
import { describe, expect, it } from "vitest";

import { buildPerFieldDiff } from "../diff";

describe("playground/diff", () => {
  it("uses schema properties order (sorted) and marks required fields", () => {
    const schemaJson: any = {
      type: "object",
      required: ["b"],
      properties: {
        b: { type: "string" },
        a: { type: "string" },
      },
    };

    const rows = buildPerFieldDiff({
      schemaJson,
      baseline: { a: "x", b: "y" },
      latest: { a: "x", b: "y" },
    });

    expect(rows.map((r) => r.key)).toEqual(["a", "b"]);
    const bRow = rows.find((r) => r.key === "b")!;
    expect(bRow.required).toBe(true);
    expect(bRow.inSchema).toBe(true);
    expect(bRow.status).toBe("same");
  });

  it("adds extra keys not in schema after schema keys (sorted) and marks inSchema=false", () => {
    const schemaJson: any = {
      type: "object",
      properties: { a: { type: "string" } },
    };

    const rows = buildPerFieldDiff({
      schemaJson,
      baseline: { a: "x", z: 1 },
      latest: { a: "x", m: 2 },
    });

    expect(rows.map((r) => r.key)).toEqual(["a", "m", "z"]);
    expect(rows.find((r) => r.key === "m")!.inSchema).toBe(false);
    expect(rows.find((r) => r.key === "z")!.inSchema).toBe(false);
  });

  it("computes status for added/removed/changed/same when both baseline and latest exist", () => {
    const schemaJson: any = {
      type: "object",
      properties: { a: {}, b: {}, c: {}, d: {} },
    };

    const rows = buildPerFieldDiff({
      schemaJson,
      baseline: { a: 1, b: 2, c: { x: 1 } },
      latest: { a: 1, b: 999, d: true, c: { x: 1 } },
    });

    const byKey = new Map(rows.map((r) => [r.key, r]));

    expect(byKey.get("a")!.status).toBe("same");
    expect(byKey.get("b")!.status).toBe("changed");
    expect(byKey.get("c")!.status).toBe("same"); // stableStringify makes nested equality stable
    expect(byKey.get("d")!.status).toBe("added");
  });

  it("handles missing baseline or latest (one-sided diffs)", () => {
    const schemaJson: any = { type: "object", properties: { a: {}, b: {} } };

    const rowsAdded = buildPerFieldDiff({
      schemaJson,
      baseline: null,
      latest: { a: 1, b: 2 },
    });
    expect(rowsAdded.find((r) => r.key === "a")!.status).toBe("added");
    expect(rowsAdded.find((r) => r.key === "a")!.baseline).toBeUndefined();
    expect(rowsAdded.find((r) => r.key === "a")!.latest).toBe(1);

    const rowsRemoved = buildPerFieldDiff({
      schemaJson,
      baseline: { a: 1, b: 2 },
      latest: null,
    });
    expect(rowsRemoved.find((r) => r.key === "b")!.status).toBe("removed");
    expect(rowsRemoved.find((r) => r.key === "b")!.baseline).toBe(2);
    expect(rowsRemoved.find((r) => r.key === "b")!.latest).toBeUndefined();
  });

  it("treats schemaJson without properties as empty schema (only extra keys)", () => {
    const rows = buildPerFieldDiff({
      schemaJson: { type: "object" } as any,
      baseline: { z: 1 },
      latest: { z: 1, a: 2 },
    });

    expect(rows.map((r) => r.key)).toEqual(["a", "z"]);
    expect(rows.find((r) => r.key === "a")!.inSchema).toBe(false);
    expect(rows.find((r) => r.key === "z")!.inSchema).toBe(false);
  });
});