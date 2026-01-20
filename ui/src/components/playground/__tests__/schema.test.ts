// ui/src/components/playground/__tests__/schema.test.ts
import { describe, expect, it } from "vitest";
import { summarizeSchema } from "../schema";

describe("playground/schema", () => {
  it("returns defaults for null or non-object schema", () => {
    expect(summarizeSchema(null)).toEqual({
      requiredCount: 0,
      propertyCount: 0,
      additionalProperties: "unknown",
    });

    // non-object
    expect(summarizeSchema("x" as any)).toEqual({
      requiredCount: 0,
      propertyCount: 0,
      additionalProperties: "unknown",
    });
  });

  it("extracts title/description, counts required and properties", () => {
    const out = summarizeSchema({
      title: "Receipt",
      description: "Schema for receipts",
      required: ["a", "b"],
      properties: { a: { type: "string" }, b: { type: "number" }, c: {} },
    } as any);

    expect(out.title).toBe("Receipt");
    expect(out.description).toBe("Schema for receipts");
    expect(out.requiredCount).toBe(2);
    expect(out.propertyCount).toBe(3);
    expect(out.additionalProperties).toBe("unspecified");
  });

  it("handles additionalProperties variants", () => {
    expect(
      summarizeSchema({ properties: {}, additionalProperties: false } as any).additionalProperties
    ).toBe("false (strict)");

    expect(
      summarizeSchema({ properties: {}, additionalProperties: true } as any).additionalProperties
    ).toBe("true");

    expect(
      summarizeSchema({ properties: {}, additionalProperties: { type: "string" } } as any)
        .additionalProperties
    ).toBe("schema (object)");

    expect(summarizeSchema({ properties: {} } as any).additionalProperties).toBe("unspecified");
  });

  it("treats missing/invalid required/properties gracefully", () => {
    const out = summarizeSchema({
      required: "not-an-array",
      properties: "not-an-object",
    } as any);

    expect(out.requiredCount).toBe(0);
    expect(out.propertyCount).toBe(0);
  });
});