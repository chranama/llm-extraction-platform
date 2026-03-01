import { describe, expect, it } from "vitest";
import { ApiError } from "../../../lib/api";
import {
  avg,
  clampInt,
  defaultActionsConfigForTrack,
  defaultEvidenceConfigForTrack,
  fmtBool,
  fmtNum,
  percentile,
  safeArray,
  toErrorMessage,
} from "../utils";

describe("demo/utils", () => {
  it("returns default action config for both tracks", () => {
    expect(defaultActionsConfigForTrack("generate_clamp")).toEqual({
      sloWindowSeconds: 300,
      sloRoute: "/v1/generate",
      sloModelId: "",
      sloOutPath: "",
    });
    expect(defaultActionsConfigForTrack("extract_gating")).toEqual({
      sloWindowSeconds: 300,
      sloRoute: "/v1/generate",
      sloModelId: "",
      sloOutPath: "",
    });
  });

  it("returns track-specific evidence defaults", () => {
    expect(defaultEvidenceConfigForTrack("generate_clamp").route).toBe("/v1/generate");
    expect(defaultEvidenceConfigForTrack("extract_gating").route).toBe("/v1/extract");
  });

  it("formats errors from ApiError and generic Error", () => {
    expect(toErrorMessage(new ApiError(400, "bad"))).toContain("HTTP 400");
    expect(toErrorMessage(new Error("boom"))).toBe("Error: boom");
  });

  it("formats bool/number and safeArray helpers", () => {
    expect(fmtBool(true)).toBe("true");
    expect(fmtBool(false)).toBe("false");
    expect(fmtBool("x")).toBe("—");
    expect(fmtNum("12.34", 1)).toBe("12.3");
    expect(fmtNum("x")).toBe("—");
    expect(safeArray([1, 2])).toEqual([1, 2]);
    expect(safeArray("x")).toEqual([]);
  });

  it("computes percentile and avg with edge handling", () => {
    expect(percentile([], 95)).toBeNull();
    expect(percentile([1, 10, 5], 50)).toBe(5);
    expect(avg([])).toBeNull();
    expect(avg([1, 2, 3])).toBe(2);
  });

  it("clamps and truncates ints with fallback", () => {
    expect(clampInt("7.9", 5, 0, 10)).toBe(7);
    expect(clampInt("200", 5, 0, 100)).toBe(100);
    expect(clampInt("x", 5, 0, 100)).toBe(5);
  });
});

