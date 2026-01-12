# Extraction Contract

**Status:** Phase 1 (v1.x)  
**Applies to:** `/v1/extract`  
**Audience:** Backend engineers, platform engineers, future contributors

---

## 1. Purpose

The extraction service converts **unstructured plaintext input** into **structured JSON output** that conforms to a predefined JSON Schema.

This document defines the **hard contract** of the extraction system:

- what inputs are accepted
- what outputs are guaranteed
- how failures are represented
- what is *explicitly out of scope*

If an interaction satisfies this contract, the system is behaving correctly.

---

## 2. High-Level Contract

**Input**
- Plaintext (`string`)
- Interpreted as natural language or OCR output
- Not required to be valid JSON

**Output**
- A single JSON object
- Must strictly conform to a declared JSON Schema
- No additional keys
- No partial objects
- No commentary or metadata mixed into the JSON payload

Formally:

> **plaintext → JSON(schema)**

---

## 3. Endpoint

```text
POST /v1/extract
```

---

## 4. Request Schema

```json
{
  "schema_id": "string",
  "text": "string",
  "model": "string | null",
  "max_new_tokens": "integer",
  "temperature": "number",
  "cache": "boolean",
  "repair": "boolean"
}
```

### Required Fields

| Field | Description |
|------|-------------|
| `schema_id` | Identifier of a registered JSON Schema |
| `text` | Raw input text to extract structured data from |

### Optional Fields

| Field | Default | Description |
|------|--------|-------------|
| `model` | system default | Override model (if allowed) |
| `max_new_tokens` | `512` | Generation cap |
| `temperature` | `0.0` | Generation randomness |
| `cache` | `true` | Enable cache read/write |
| `repair` | `true` | Allow one repair attempt |

---

## 5. Schema Resolution

- `schema_id` must reference a schema registered in the schema registry
- Each schema is a **JSON Schema Draft 2020-12 document**
- The schema defines:
  - required fields
  - field types
  - constraints
  - allowed values

If the schema cannot be loaded, the request fails with **500**.

---

## 6. Output Guarantees (Success Path)

On success, the service guarantees:

1. The response body contains a JSON object
2. The object:
   - parses under strict JSON rules
   - conforms exactly to the declared schema
3. No additional fields are present
4. No schema violations exist

### Success Response

```json
{
  "schema_id": "ticket_v1",
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "data": { },
  "cached": false,
  "repair_attempted": false
}
```

### Semantics

| Field | Meaning |
|-----|--------|
| `data` | Fully validated extraction result |
| `cached` | Result was returned from cache |
| `repair_attempted` | A repair pass was required |

---

## 7. JSON Strictness

The system enforces **strict JSON parsing**:

- No markdown
- No comments
- No trailing commas
- No code fences
- No partial objects

If the model output cannot be parsed as strict JSON, the attempt is considered a failure.

---

## 8. Validation Rules

Validation occurs in this order:

1. **JSON parsing**
2. **Top-level object check**
3. **JSON Schema validation**

Failure at any stage produces a **422 Unprocessable Entity**.

---

## 9. Repair Semantics

If `repair=true`:

- The system allows **one** repair attempt
- The repair prompt includes:
  - the schema
  - the original input
  - the previous invalid output
  - a structured error hint
- Repair is executed at `temperature=0.0`
- No further retries are allowed

If repair fails, the request fails.

---

## 10. Error Contract

### 422 – Client-Visible Extraction Failure

All extraction failures return a structured 422 error:

```json
{
  "code": "invalid_json | schema_validation_failed",
  "message": "Human-readable explanation",
  "request_id": "optional",
  "errors": [ ],
  "raw_preview": "optional"
}
```

Common error codes:

| Code | Meaning |
|----|--------|
| `invalid_json` | Model output was not valid JSON |
| `schema_validation_failed` | JSON did not conform to schema |

### 500 – Server Error

Returned only for:
- missing dependencies (e.g. jsonschema)
- model registry misconfiguration
- internal infrastructure failures

Model behavior errors **should not** return 500.

---

## 11. Caching Contract

When `cache=true`:

1. Redis cache is checked first
2. Database cache is checked second
3. New valid outputs are written to both

Cache writes are **best-effort**:
- cache failures do not fail the request
- duplicate cache inserts are ignored

When `cache=false`:
- no cache reads
- no cache writes

---

## 12. Observability Guarantees

Each request records:
- latency
- model ID
- schema ID
- cache hit/miss
- repair attempt status
- validated output only

Invalid or partial outputs are never persisted.

---

## 13. Explicit Non-Goals (v1.x)

The extraction contract **does not guarantee**:

- semantic correctness
- factual accuracy
- completeness beyond schema requirements
- confidence calibration
- probabilistic scores (except if schema includes them)

These are **model-level concerns**, not contract concerns.

---

## 14. Contract Stability

This contract is considered **stable for v1.x**.

Any change that:
- alters failure semantics
- relaxes validation rules
- changes cache behavior
- allows partial JSON

**requires a v2.0 contract revision**.

---

## 15. Summary

The extraction service is a **schema-enforced transformation layer** with:

- plaintext input
- validated JSON output
- deterministic failure modes
- explicit repair semantics

If the service returns `200`, the output is safe to consume downstream.

If it returns `422`, the output is unusable by definition.

There are no ambiguous states.