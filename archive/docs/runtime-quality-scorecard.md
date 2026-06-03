# Runtime Quality Scorecard

This document defines the Phase 2.2.9 operational quality signals that should influence runtime choice for `llm-extraction-platform`.

Gateway-side decision contract:

- `/Users/chranama/career/inference-serving-gateway/docs/runtime-decision-contract.md`

Eval modernization reference:

- `/Users/chranama/career/job-search/audit/2026-03-27__eval-modernization-plan.md`

## Purpose

The integrated stack already proves service boundaries, trace continuity, and telemetry wiring.

Phase 2.2.9 makes one additional claim explicit:

- the AI side of the runtime should be judged with operational quality signals, not only with systems metrics

For this system, "quality" means:

- contract-valid structured outputs
- stable async completion behavior
- understandable repair behavior
- policy decisions that are visible and explainable

This document does not try to define general LLM benchmark quality.
It defines the runtime quality signals that matter for this bounded extraction service.

## What Is In Scope

In scope now:

- extract contract pass rate
- malformed-output or validation-failure rate
- repair attempt and repair success rates
- policy rejection rate
- async completion-before-timeout rate
- rough request cost when token or provider-cost data is available

Explicitly not in scope yet:

- retrieval quality metrics
- agent trajectory quality
- first-class runtime fallback invocation metrics

Those should only be added when the underlying runtime paths truly exist.

## Quality Signal Scorecard

| Signal | Definition | Current raw sources | Default window | Main decisions |
| --- | --- | --- | --- | --- |
| `extract_contract_pass_rate` | Share of runs that finish with valid structured output and no contract failure | sync response bodies, async job status, trace `extract.completed`, validation-failure counters | `1h` and per-run proof pack | route, provider, prompt, and release judgment |
| `structured_output_invalid_rate` | Share of extract runs that fail parse, schema, or truncation validation | `llm_extraction_validation_failures_total` by `stage`, AppError codes, trace stages | `1h` and replay pack | prompt, provider, and policy tuning |
| `repair_attempt_rate` | Share of extract runs that require repair after the first validation pass | `llm_extraction_repair_total{outcome="attempted"}` divided by extract requests | `1h` and replay pack | latency versus quality tradeoff, repair toggle decisions |
| `repair_success_rate` | Share of repair attempts that end in valid structured output | `llm_extraction_repair_total{outcome="success"}` vs attempted | `1h` and replay pack | whether repair is earning its latency and cost |
| `policy_rejection_rate` | Share of requests intentionally blocked by capability or policy gates rather than by malformed output | request or trace error codes today; first-class metric later if needed | `1h` and post-change review | policy tuning versus runtime-health diagnosis |
| `async_completion_before_timeout_rate` | Share of async jobs that reach success within the bounded timeout budget | async job timestamps, status responses, trace timelines | `1h` and proof pack | async-path readiness and queue or worker health |
| `rough_request_cost_usd` | Estimated request cost from token counts and known provider pricing assumptions where available | inference logs, token counters, provider config, usage endpoints | per-run and `1h` usage summary | cost-aware runtime choice and platform-owner review |
| `route_fallback_invocation_rate` | Reserved signal for future multi-route fallback behavior | not first-class today | not active yet | future route or fallback evaluation only |

## Derivation Rules

### 1. Contract pass means more than HTTP 200

For sync paths, treat a run as a quality pass only when all of the following are true:

- the request succeeds
- the output is schema-valid
- no validation-failure signal was emitted for the run

For async paths, treat a run as a quality pass only when:

- submit succeeds
- the job reaches `succeeded`
- the trace or job record does not show a contract failure

### 2. Malformed-output rate comes from validation semantics

Use validation-stage information as the primary malformed-output signal.

In the current backend that means paying attention to stages such as:

- `validate_output`
- `truncation_check`
- `repair_validate`

This is a better signal than a generic HTTP error rate because it isolates the AI-side failure mode.

### 3. Repair should be justified, not assumed

Repair is useful only if:

- contract pass rate improves materially
- repair success is meaningfully above zero
- latency and cost remain acceptable for the bounded workflow

If repair attempt rate is high but repair success is weak, that is evidence of runtime instability, not resilience.

### 4. Policy rejection should stay distinguishable from model failure

Policy or capability rejections answer a different question than malformed outputs.

Use them to answer:

- "did the runtime intentionally refuse?"

Do not merge them into:

- "the model could not produce valid output"

### 5. Rough cost should be good enough for runtime choice

The first quality-and-cost loop does not require a full billing subsystem.

It requires enough cost visibility to answer:

- did the better quality path cost materially more?
- did a latency improvement come from a lower-cost runtime choice or just a less-safe one?

## Runtime Comparison Rules

When comparing prompt, provider, route, or policy variants:

- use the same replay set or proof inputs
- keep schema constant
- record whether `repair` was enabled
- note whether results were cached
- compare quality before celebrating latency or cost wins

Current practical rule:

- a runtime variant should not be considered better unless it preserves or improves contract pass rate while also improving latency, cost, or policy clarity

## Current Usage And Metering Surfaces

The backend already has bounded usage surfaces that Phase 2.2.9 can treat as real:

- `GET /v1/me/usage`
- `GET /v1/admin/usage`
- inference logs with `api_key`, token counts, and latency
- API-key quota fields

That means the current usage-scope seed can remain:

- API-key-backed report surfaces

Important metric rule:

- do not push raw `api_key` values into default Prometheus labels
- keep per-key usage in DB-backed admin or proof surfaces instead

## Why This Improves The AI Story

Without this scorecard, the system can look like:

- strong observability
- strong service decomposition
- but weakly specified AI runtime judgment

With this scorecard, the system can be described more accurately as:

- an AI runtime that measures contract validity, repair behavior, async completion, policy effects, latency, and cost together

That is the intended outcome of Phase 2.2.9.

## Related Docs

- [06) Eval Methodology](06-eval-methodology.md)
- [Application Runtime Architecture](application-runtime-architecture.md)
- [AWS Deployment Contract](aws-deployment-contract.md)
- [12) Trace Identity Contract](12-trace-identity-contract.md)
