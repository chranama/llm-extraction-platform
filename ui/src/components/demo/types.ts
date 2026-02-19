// ui/src/components/demo/types.ts
export type Track = "generate_clamp" | "extract_gating";

export type DemoPolicyIssue = {
  code?: string;
  message?: string;
  context?: Record<string, any>;
};

export type DemoPolicySnapshot = {
  // common-ish fields (policy_decision_v2-like)
  schema_version?: string;
  generated_at?: string;

  policy?: string;
  pipeline?: string;

  status?: string;
  ok?: boolean;

  enable_extract?: boolean;
  generate_max_new_tokens_cap?: number | null;

  contract_errors?: number;
  contract_warnings?: number;

  thresholds_profile?: string | null;
  thresholds_version?: string | null;

  generate_thresholds_profile?: string | null;

  eval_run_dir?: string | null;
  eval_task?: string | null;
  eval_run_id?: string | null;

  model_id?: string | null;

  reasons?: DemoPolicyIssue[];
  warnings?: DemoPolicyIssue[];

  metrics?: Record<string, any>;
};

export type DemoActionsConfig = {
  // Generate clamp action
  sloWindowSeconds: number;
  sloRoute: string;
  sloModelId: string;
  sloOutPath: string;
};

export type DemoEvidenceConfig = {
  route: string;
  modelId: string;
  limit: number;
};

export type DemoState = {
  // placeholder if you want a single view model later
};