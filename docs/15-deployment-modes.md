# Deployment Modes — Capability-Based LLM Platform

This document defines the supported **deployment modes** of the LLM Extraction Platform and explains how system capabilities vary by environment.

The platform treats **generation** and **extraction** as distinct capabilities.  
Extraction is schema-constrained and reliability-critical, and therefore **not guaranteed** in all deployments.

Instead of failing silently, the platform **detects, reports, and enforces** capability availability.

---

## Capability Model

The platform exposes its active capabilities via:

GET /v1/capabilities

Example response:

{
  "generate": true,
  "extract": false,
  "mode": "generate-only"
}

Capabilities are determined by:

- Configuration (ENABLE_GENERATE, ENABLE_EXTRACT)
- Model backend suitability
- Optional runtime probes (future extension)

---

## Mode 1 — Generate-Only (CPU-Safe, Default)

### Purpose

This mode is designed for:

- Local development
- CPU-only environments
- CI and reproducible testing
- Kubernetes kind demo

Generation is always supported.  
Extraction is explicitly disabled.

This prevents misleading or unreliable extraction behavior on weak models.

---

### Activation

Configuration:

ENABLE_GENERATE=true  
ENABLE_EXTRACT=false  

Or via YAML:

enable_generate: true  
enable_extract: false  

---

### Expected API Behavior

| Endpoint | Behavior |
|-------|---------|
| GET /v1/capabilities | generate=true, extract=false |
| POST /v1/generate | Returns valid response |
| POST /v1/extract | Returns 501 CAPABILITY_DISABLED |

Example extract response:

{
  "error": {
    "code": "CAPABILITY_DISABLED",
    "message": "Extraction is disabled in this deployment mode.",
    "hint": "Enable ENABLE_EXTRACT with a compatible backend."
  }
}

---

### UI Behavior

- Extract panel is hidden or disabled.
- UI displays a message:  
  “Extraction is disabled in this deployment mode.”

Generate remains fully functional.

---

### Eval Behavior

- Extract-required tasks are skipped.
- Skip reason is recorded explicitly.
- At least one generate-based task must be runnable.
- Eval run produces an artifact.

---

### Verification

curl /v1/capabilities  
curl /v1/generate  
curl /v1/extract  

Expected:

- extract=false in capabilities
- extract endpoint returns CAPABILITY_DISABLED

---

## Mode 2 — Extract-Enabled (Capability-Dependent)

### Purpose

This mode enables schema-constrained extraction when a capable backend is available.

Typical requirements:

- GPU-backed local model
- Strong hosted LLM backend
- High schema conformance reliability

---

### Activation

Configuration:

ENABLE_GENERATE=true  
ENABLE_EXTRACT=true  

---

### Expected API Behavior (Successful Backend)

| Endpoint | Behavior |
|-------|---------|
| GET /v1/capabilities | generate=true, extract=true |
| POST /v1/extract | Returns schema-valid output |
| Golden fixtures | Pass contract tests |

---

### Expected API Behavior (Insufficient Backend)

If the backend cannot reliably perform extraction:

| Endpoint | Behavior |
|-------|---------|
| GET /v1/capabilities | extract=false with reason |
| POST /v1/extract | Returns 503 CAPABILITY_UNAVAILABLE |

Example:

{
  "error": {
    "code": "CAPABILITY_UNAVAILABLE",
    "message": "Extraction backend does not meet reliability requirements.",
    "hint": "Use generate-only mode or configure a stronger backend."
  }
}

This allows the system to remain operational while clearly reporting limitations.

---

## Capability Enforcement Philosophy

The platform does **not** assume:

- All models can perform extraction.
- All deployments support all endpoints.

Instead, it guarantees:

- Capabilities are explicit.
- Failures are structured.
- Behavior is testable.
- Limitations are observable.

---

## Why Extraction Is Conditional

Extraction requires:

- Schema conformance
- Field completeness
- Deterministic structure
- Repair loop success

Small CPU-only models frequently fail these constraints, producing:

- Invalid JSON
- Missing required fields
- Hallucinated keys
- Inconsistent structure

For this reason, extraction is treated as a **conditional capability**, not a guaranteed feature.

---

## Verification Matrix

| Mode | /v1/generate | /v1/extract | /v1/capabilities |
|------|-------------|-------------|------------------|
| generate-only | OK | CAPABILITY_DISABLED | extract=false |
| extract-enabled (good backend) | OK | OK | extract=true |
| extract-enabled (bad backend) | OK | CAPABILITY_UNAVAILABLE | extract=false + reason |

---

## Kubernetes Alignment

The Kubernetes demo overlay uses **generate-only mode** by default.

This ensures:

- Deterministic local deployment
- CPU compatibility
- Stable CI behavior
- Honest capability reporting

Extract-enabled overlays may be added when a suitable backend is available.

---

## CI Alignment

CI pipelines run in generate-only mode by default.

This ensures:

- Reproducibility
- Fast execution
- Clear capability behavior
- Stable integration results

Extract-enabled CI runs are optional and gated.

---

## Summary

This platform does not pretend all capabilities work everywhere.

Instead, it:

- Declares capabilities explicitly
- Enforces them deterministically
- Tests both availability and unavailability
- Documents expected degradation behavior

This approach ensures the system remains reliable, honest, and production-credible across environments.

---

End of deployment modes documentation.