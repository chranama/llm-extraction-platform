# Extraction Contract

This document defines the runtime contract for `POST /v1/extract`.

It focuses on API behavior and validation guarantees, not model quality claims.

## Purpose

`/v1/extract` converts unstructured text into structured JSON that must satisfy a registered schema.

Contract summary:
- Input: plain text + `schema_id`
- Output: validated JSON object
- Gate: capability and policy checks may deny extract before model inference

## Endpoint

```text
POST /v1/extract
```

## Request shape

```json
{
  "schema_id": "sroie_receipt_v1",
  "text": "Vendor: ACME\nTotal: 10.00",
  "model": "optional-model-id",
  "cache": true,
  "repair": true,
  "max_new_tokens": 512,
  "temperature": 0.0
}
```

Required:
- `schema_id`
- `text`

Common optional fields:
- `model`
- `cache`
- `repair`
- generation controls (`max_new_tokens`, `temperature`, etc.)

## Schema and contracts split

- `schemas/` contains schema specification files.
  - Current model-output schema path: `schemas/model_output/sroie_receipt_v1.json`
- `contracts/` contains Python type/validation enforcement used by services and jobs.

This split is intentional and used across server, eval, and policy workflows.

## Runtime validation order

1. Capability/policy enforcement (may block extract).
2. Strict JSON parsing of model output.
3. Schema validation against the selected schema.
4. Optional repair pass (if enabled) with one bounded retry strategy.

## Success guarantees

On success:
- response body includes validated structured data,
- data conforms to selected schema,
- response includes model/cache metadata fields used by clients and diagnostics.

## Failure semantics

Common categories:
- `400/501` capability disabled or unsupported for the selected model/deployment
- `422` parse or schema validation failures
- `5xx` dependency/runtime failures (for example model runtime unavailable)

Important: model quality problems should surface as extraction failure semantics, not silent success.

## Capability gating

Extract availability is conditional on model capability and configuration.

Primary controls:
- model capabilities loaded from `MODELS_YAML` + `MODELS_PROFILE`
- runtime toggles (for example `ENABLE_EXTRACT`)
- optional policy decision overlays

The extract-gate demo in [Project Demos](02-project-demos.md) shows PASS/FAIL artifacts driving this behavior.

## Stability notes

This contract is stable at the behavioral level for current `v1` extraction APIs. If payload semantics or status-code semantics change, update this document and corresponding integration tests together.

## Related docs

- [Testing and CI](00-testing.md)
- [Project Demos](02-project-demos.md)
- [Deployment Modes](03-deployment-modes.md)
