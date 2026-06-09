# Documentation

This directory contains the current technical documentation for the repository.

Start with the root [`README.md`](../README.md). Use these documents when you
want more detail about system structure, APIs, tests, operations, artifacts, or
scope boundaries.

## Documents

- [Architecture](architecture.md): system shape, component boundaries, and main flows.
- [API](api.md): public runtime endpoints and contract surfaces.
- [Testing](testing.md): test layout, CI lanes, and behavior coverage.
- [Operations](operations.md): local runtime modes, diagnostics, and common failures.
- [Runbook](runbook.md): start, verify, observe, and shut down the local system.
- [Runtime Setup](runtime-setup.md): local hardware, model, and AWS prep requirements.
- [Inference Gateway Integration](inference-gateway-integration.md): promoted joint kind workflow and evidence workflows with the companion gateway.
- [Artifacts](artifacts.md): generated evidence files and validation commands.
- [AWS Deployment Contract](aws-deployment-contract.md): backend participation in the bounded joint AWS deployment.
- [API And Model Runtime Evolution](decisions/api-model-runtime-evolution.md): decision note for the API/model-serving boundary.
- [Scope](scope.md): current claims, non-claims, and known limits.

Archived documentation lives in [`../archive/docs/`](../archive/docs/). Treat it
as historical context, not as current implementation guidance.
