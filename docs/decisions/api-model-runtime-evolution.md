# API And Model Runtime Evolution

This decision note explains how the system moved from a single service that ran
the API and model runtime together toward the current split between API
orchestration and external model-serving backends.

## Starting Point

The earliest runtime shape kept the API and model execution inside the same
service. The backend loaded models directly through the legacy Transformers
runtime and served generation and extraction from one process.

That shape was useful for early development because it reduced deployment
surface area. It also made the runtime boundary hard to inspect: API behavior,
model loading, generation, validation, and operational concerns were all
collapsed into one service.

## Package Separation

The next step was to separate supporting responsibilities:

- `contracts/` and `schemas/` define shared data contracts and output schemas.
- `eval/` owns evaluation jobs and scoring workflows.
- `policy/` owns capability and runtime control decisions.
- `server/` owns API behavior, auth, routing, validation, async jobs, traces,
  health, readiness, and admin surfaces.

This did not fully split model serving, but it made the API service less
monolithic and made eval/policy artifacts reviewable outside the request path.

## Backend Abstraction

The model runtime was then abstracted behind model backend configuration. The
API no longer needs to know whether generation comes from an in-process fake
backend, legacy Transformers, `llama.cpp`, or another OpenAI-compatible runtime.

The current backend types are:

- `fake` for deterministic local checks.
- `transformers` for the legacy in-process runtime.
- `llamacpp` for direct llama.cpp integration where supported.
- `remote` for model servers reached over HTTP.

For vLLM, the system uses:

```yaml
backend: remote
remote:
  provider: vllm
```

This keeps vLLM as a model server rather than treating it as a separate API
backend type.

## Current Boundary

The API service owns:

- authentication and API keys;
- schema loading and output validation;
- capability and policy gates;
- sync extraction;
- async extraction job state;
- trace and log inspection;
- readiness and model status reporting.

The model server owns:

- model loading;
- token generation;
- runtime-specific acceleration or quantization;
- model-server health and model listing.

The gateway owns:

- request admission and limits;
- edge route policy;
- request and trace identity propagation;
- forwarding to LLMEP;
- gateway logs, metrics, and OpenTelemetry propagation.

## Runtime Consequences

The promoted local workflow uses a live CPU-only `llama.cpp` model server inside
kind because it is portable and reviewable on a developer machine.

The promoted AWS live-model workflow uses vLLM on a GPU node group because cloud
GPU infrastructure can support an accelerated OpenAI-compatible model runtime.

The fake backend remains valuable for deterministic checks, failure proofs, and
CI-friendly workflows where model variance would obscure API behavior.

## Decision

Keep the API service separate from model-serving implementation details. Treat
live model runtimes as replaceable backends behind a stable runtime contract.
Use fake backends for deterministic proof of API and gateway behavior, CPU-only
`llama.cpp` for the promoted local kind workflow, and vLLM for the promoted AWS
live-model workflow.

## Tradeoffs

This design has more deployment parts than the original single-process
Transformers runtime. It requires clearer docs, preflight checks, and runbooks.

The benefit is a more production-relevant boundary: API orchestration,
validation, jobs, traces, gateway controls, and model serving can be inspected
independently.

