// ui/src/components/playground/__tests__/utils.test.ts
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

import { prettyJson, toNumberOr, copyToClipboard, stableStringify, valueCell, shorten } from "../utils";

describe("playground/utils", () => {
  describe("prettyJson", () => {
    it("stringifies objects with indentation", () => {
      expect(prettyJson({ a: 1 })).toBe('{\n  "a": 1\n}');
    });

    it("falls back to String(x) for circular objects", () => {
      const x: any = {};
      x.self = x;
      expect(prettyJson(x)).toBe("[object Object]");
    });
  });

  describe("toNumberOr", () => {
    it("returns prev for empty/whitespace strings", () => {
      expect(toNumberOr(5, "")).toBe(5);
      expect(toNumberOr(5, "   ")).toBe(5);
    });

    it("returns prev for non-finite numbers", () => {
      expect(toNumberOr(5, "NaN")).toBe(5);
      expect(toNumberOr(5, "Infinity")).toBe(5);
      expect(toNumberOr(5, "-Infinity")).toBe(5);
    });

    it("parses valid numbers", () => {
      expect(toNumberOr(5, "0")).toBe(0);
      expect(toNumberOr(5, "12.5")).toBe(12.5);
      expect(toNumberOr(5, "  7  ")).toBe(7);
    });
  });

  describe("copyToClipboard", () => {
    const origClipboard = (navigator as any).clipboard;
    const origExecCommand = (document as any).execCommand;

    beforeEach(() => {
      vi.restoreAllMocks();
    });

    afterEach(() => {
      // restore globals so tests don't leak state
      Object.defineProperty(navigator, "clipboard", {
        value: origClipboard,
        configurable: true,
        writable: true,
      });
      if (origExecCommand === undefined) {
        // remove if we added it
        try {
          delete (document as any).execCommand;
        } catch {
          // ignore
        }
      } else {
        (document as any).execCommand = origExecCommand;
      }
      vi.restoreAllMocks();
    });

    it("uses navigator.clipboard.writeText when available and succeeds", async () => {
      const writeText = vi.fn().mockResolvedValueOnce(undefined);

      Object.defineProperty(navigator, "clipboard", {
        value: { writeText },
        configurable: true,
      });

      await expect(copyToClipboard("hello")).resolves.toBe(true);
      expect(writeText).toHaveBeenCalledTimes(1);
      expect(writeText).toHaveBeenCalledWith("hello");
    });

    it("falls back to document.execCommand('copy') when clipboard fails", async () => {
      const writeText = vi.fn().mockRejectedValueOnce(new Error("nope"));
      Object.defineProperty(navigator, "clipboard", {
        value: { writeText },
        configurable: true,
      });

      // JSDOM may not define execCommand; define it so we can spy on it.
      Object.defineProperty(document, "execCommand", {
        value: () => true,
        configurable: true,
      });

      const execCommand = vi.spyOn(document, "execCommand").mockReturnValueOnce(true);

      await expect(copyToClipboard("fallback")).resolves.toBe(true);

      expect(writeText).toHaveBeenCalledTimes(1);
      expect(execCommand).toHaveBeenCalledTimes(1);
      expect(execCommand).toHaveBeenCalledWith("copy");
    });

    it("returns false if both clipboard and fallback fail", async () => {
      const writeText = vi.fn().mockRejectedValueOnce(new Error("nope"));
      Object.defineProperty(navigator, "clipboard", {
        value: { writeText },
        configurable: true,
      });

      Object.defineProperty(document, "execCommand", {
        value: () => true,
        configurable: true,
      });

      vi.spyOn(document, "execCommand").mockImplementationOnce(() => {
        throw new Error("blocked");
      });

      await expect(copyToClipboard("fail")).resolves.toBe(false);
    });
  });

  describe("stableStringify", () => {
    it("sorts object keys and is stable across key ordering", () => {
      const a = { b: 1, a: 2 };
      const b = { a: 2, b: 1 };
      expect(stableStringify(a)).toBe(stableStringify(b));
      expect(stableStringify(a)).toBe('{"a":2,"b":1}');
    });

    it("replaces circular references with [Circular]", () => {
      const x: any = { a: 1 };
      x.self = x;
      expect(stableStringify(x)).toContain('"self":"[Circular]"');
    });

    it("handles arrays and nested objects", () => {
      expect(stableStringify({ z: [2, { b: 1, a: 2 }] })).toBe('{"z":[2,{"a":2,"b":1}]}');
    });
  });

  describe("valueCell", () => {
    it("formats primitives and missing values", () => {
      expect(valueCell(undefined)).toBe("(missing)");
      expect(valueCell(null)).toBe("null");
      expect(valueCell("x")).toBe("x");
      expect(valueCell(3)).toBe("3");
      expect(valueCell(true)).toBe("true");
    });

    it("pretty prints objects", () => {
      expect(valueCell({ a: 1 })).toBe('{\n  "a": 1\n}');
    });
  });

  describe("shorten", () => {
    it("trims and returns as-is when short enough", () => {
      expect(shorten("  hello  ", 10)).toBe("hello");
    });

    it("shortens and adds ellipsis when needed", () => {
      expect(shorten("abcdefghij", 7)).toBe("abcd...");
      expect(shorten("   abcdefghij   ", 7)).toBe("abcd...");
    });
  });
});