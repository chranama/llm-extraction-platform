# Demo Acceptance Checklist — Production LLM Extraction Platform

This document defines the **acceptance criteria** for the current “production demo” phase of the project.  
Every item below must be verifiable by command and observable behavior.

This file is the audit trail for completion.

---

## Phase Scope

This phase proves that the platform:

- Supports deterministic multi-mode deployment.
- Enforces capability-based behavior (generate vs extract).
- Is testable end-to-end.
- Is operationally credible.
- Is deployable in Kubernetes (kind) in generate-only mode.
- Is CI-validated.

---

## 1. Docker Compose Deployment Modes

### 1.1 Local

Checklist:
- [ ] `make up-local` completes without errors
- [ ] API responds to `/healthz`
- [ ] API responds to `/readyz`
- [ ] UI is reachable
- [ ] Grafana is reachable
- [ ] Prometheus is reachable

Verification:
- [ ] `make smoke-local` passes

---

### 1.2 Dev

Checklist:
- [ ] `make up-dev` completes without errors
- [ ] API responds to `/healthz`
- [ ] API responds to `/readyz`

Verification:
- [ ] `make smoke-dev` passes (if defined)

---

### 1.3 Eval

Checklist:
- [ ] `make up-eval` completes without errors
- [ ] Eval container can reach API base URL
- [ ] Model download / cache logic works in eval mode
- [ ] Eval config matches compose environment

Verification:
- [ ] `make smoke-eval` produces at least one eval artifact

---

## 2. Capability-Based Deployment Modes

### 2.1 Generate-Only Mode

Checklist:
- [ ] `/v1/capabilities` returns:
  - generate = true
  - extract = false
- [ ] `/v1/generate` returns a valid response
- [ ] `/v1/extract` returns structured CAPABILITY_DISABLED error
- [ ] UI hides or disables Extract panel
- [ ] Eval skips extract tasks with explicit reason
- [ ] Eval runs at least one generate task

Verification commands:
- [ ] `curl /v1/capabilities`
- [ ] `curl /v1/generate`
- [ ] `curl /v1/extract` (expected disabled response)
- [ ] `llm_eval cli run` produces artifact

---

### 2.2 Extract-Enabled Mode (Optional / Gated)

Checklist:
- [ ] `/v1/capabilities` returns extract=true
- [ ] `/v1/extract` returns schema-valid output on golden fixture
- [ ] Golden fixture tests pass

If backend is insufficient:
- [ ] `/v1/capabilities` reports extract unavailable with reason
- [ ] `/v1/extract` returns CAPABILITY_UNAVAILABLE error

---

## 3. Integrations Test Suite

Checklist:
- [ ] Integration tests run against live stack
- [ ] Capability gating behavior is tested
- [ ] Generate-only suite passes
- [ ] Extract suite is gated or optional
- [ ] Artifacts are written for failed or skipped extract runs

Verification:
- [ ] `make test-integrations`
- [ ] `make test-integrations-compose`

---

## 4. Kubernetes (kind) Demo

### 4.1 Cluster

Checklist:
- [ ] `make kind-up` creates cluster
- [ ] kubeconfig is active
- [ ] nodes are Ready

---

### 4.2 Deployment

Checklist:
- [ ] `make k8s-apply` applies overlay
- [ ] Migrations job completes
- [ ] API pod becomes Ready
- [ ] Postgres pod Ready
- [ ] Redis pod Ready
- [ ] Ingress or port-forward works

---

### 4.3 Generate-Only Acceptance

Checklist:
- [ ] `/v1/capabilities` shows extract=false
- [ ] `/v1/generate` works
- [ ] `/v1/extract` returns CAPABILITY_DISABLED

Verification:
- [ ] Smoke script passes

---

## 5. Continuous Integration

Checklist:
- [ ] Backend unit tests run
- [ ] Eval unit tests run
- [ ] UI unit tests run
- [ ] Integration tests (generate-only) run
- [ ] Artifacts uploaded:
  - junit xml
  - coverage summary
  - eval artifact

Optional:
- [ ] kind smoke job runs in nightly or manual workflow

---

## 6. Operational Hardening

Checklist:
- [ ] Request timeouts enforced
- [ ] Concurrency limits enforced
- [ ] Payload size limits enforced
- [ ] Rate limits / quotas enforced
- [ ] Structured error codes used consistently
- [ ] API keys not logged
- [ ] Containers run as non-root where applicable
- [ ] Base images pinned or stable

---

## 7. Documentation

Checklist:
- [ ] docs/14-demo-acceptance.md complete (this file)
- [ ] docs/15-deployment-modes.md complete
- [ ] Commands in docs are runnable
- [ ] Expected outputs are documented

---

## 8. Final Acceptance Statement

This phase is considered complete when **all checked items above are verified locally** and at least one full CI run reproduces the same results.

---

## Notes

This checklist is intentionally strict.

Anything not checked is not considered part of the demo guarantee.

---

End of acceptance criteria.