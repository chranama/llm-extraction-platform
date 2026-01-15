// ui/src/components/playground/schema.ts
import type { JsonSchema } from "../../lib/api";

export function summarizeSchema(schema: JsonSchema | null): {
  title?: string;
  description?: string;
  requiredCount: number;
  propertyCount: number;
  additionalProperties: string;
} {
  if (!schema || typeof schema !== "object") {
    return { requiredCount: 0, propertyCount: 0, additionalProperties: "unknown" };
  }

  const title = typeof (schema as any).title === "string" ? (schema as any).title : undefined;
  const description =
    typeof (schema as any).description === "string" ? (schema as any).description : undefined;

  const req = Array.isArray((schema as any).required) ? (schema as any).required : [];
  const props =
    (schema as any).properties && typeof (schema as any).properties === "object"
      ? (schema as any).properties
      : {};
  const propertyCount = props ? Object.keys(props).length : 0;

  const ap = (schema as any).additionalProperties;
  let additionalProperties: string = "unspecified";
  if (ap === false) additionalProperties = "false (strict)";
  else if (ap === true) additionalProperties = "true";
  else if (ap && typeof ap === "object") additionalProperties = "schema (object)";
  else if (ap === undefined) additionalProperties = "unspecified";

  return {
    title,
    description,
    requiredCount: req.length,
    propertyCount,
    additionalProperties,
  };
}