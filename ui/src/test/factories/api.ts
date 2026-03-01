import type {
  AdminLogsPage,
  AdminPolicySnapshot,
  ExtractResponseBody,
  GenerateResponseBody,
  ModelsResponseBody,
  SchemaIndexItem,
} from "../../lib/api";

export function makeModelsResponse(
  overrides: Partial<ModelsResponseBody> = {}
): ModelsResponseBody {
  return {
    default_model: "demo-model",
    models: [],
    deployment_capabilities: { generate: true, extract: true },
    ...overrides,
  };
}

export function makeSchemaIndex(...schemaIds: string[]): SchemaIndexItem[] {
  return schemaIds.map((schema_id) => ({ schema_id }));
}

export function makeGenerateResponse(
  overrides: Partial<GenerateResponseBody> = {}
): GenerateResponseBody {
  return {
    model: "demo-model",
    output: "ok",
    cached: false,
    ...overrides,
  };
}

export function makeExtractResponse(
  overrides: Partial<ExtractResponseBody> = {}
): ExtractResponseBody {
  return {
    schema_id: "sroie_receipt_v1",
    model: "demo-model",
    data: {},
    cached: false,
    repair_attempted: false,
    ...overrides,
  };
}

export function makeAdminPolicySnapshot(
  overrides: Partial<AdminPolicySnapshot> = {}
): AdminPolicySnapshot {
  return {
    status: "allow",
    ok: true,
    enable_extract: true,
    ...overrides,
  };
}

export function makeAdminLogsPage(
  overrides: Partial<AdminLogsPage> = {}
): AdminLogsPage {
  return {
    total: 0,
    limit: 50,
    offset: 0,
    items: [],
    ...overrides,
  };
}

