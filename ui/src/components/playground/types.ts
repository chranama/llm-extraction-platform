// ui/src/components/playground/types.ts
export type Mode = "extract" | "generate";

export type DiffRow = {
  key: string;
  required: boolean;
  baseline: any;
  latest: any;
  status: "same" | "changed" | "added" | "removed";
  inSchema: boolean;
};