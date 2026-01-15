// ui/src/components/playground/diff.ts
import type { JsonSchema } from "../../lib/api";
import type { DiffRow } from "./types";
import { stableStringify } from "./utils";

export function buildPerFieldDiff(args: {
  schemaJson: JsonSchema | null;
  baseline: Record<string, any> | null;
  latest: Record<string, any> | null;
}): DiffRow[] {
  const schemaProps =
    args.schemaJson && typeof (args.schemaJson as any).properties === "object"
      ? ((args.schemaJson as any).properties as Record<string, any>)
      : {};

  const requiredSet = new Set<string>(
    args.schemaJson && Array.isArray((args.schemaJson as any).required)
      ? ((args.schemaJson as any).required as string[])
      : []
  );

  const baselineObj = args.baseline ?? {};
  const latestObj = args.latest ?? {};

  const schemaKeys = Object.keys(schemaProps).sort();

  const extraKeys = Array.from(
    new Set([...Object.keys(baselineObj), ...Object.keys(latestObj)].filter((k) => !schemaProps[k]))
  ).sort();

  const keys = [...schemaKeys, ...extraKeys];

  return keys.map((k) => {
    const b = args.baseline ? baselineObj[k] : undefined;
    const l = args.latest ? latestObj[k] : undefined;

    const bHas = args.baseline ? Object.prototype.hasOwnProperty.call(baselineObj, k) : false;
    const lHas = args.latest ? Object.prototype.hasOwnProperty.call(latestObj, k) : false;

    let status: DiffRow["status"] = "same";
    if (!args.baseline || !args.latest) {
      if (!args.baseline && lHas) status = "added";
      else if (!args.latest && bHas) status = "removed";
      else status = "same";
    } else {
      if (bHas && !lHas) status = "removed";
      else if (!bHas && lHas) status = "added";
      else status = stableStringify(b) === stableStringify(l) ? "same" : "changed";
    }

    return {
      key: k,
      required: requiredSet.has(k),
      baseline: args.baseline ? b : undefined,
      latest: args.latest ? l : undefined,
      status,
      inSchema: Boolean(schemaProps[k]),
    };
  });
}