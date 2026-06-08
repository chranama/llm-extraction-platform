#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from generate_async_extract_proof import generate_async_extract_proof
from generate_k8s_kind_proof import generate_k8s_kind_proof
from generate_ops_surface_proof import generate_ops_surface_proof
from generate_policy_eval_linkage_proof import generate_policy_eval_linkage_proof
from generate_trace_inspection_proof import generate_trace_inspection_proof

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "proof" / "evidence_manifest.latest.json"
PROOF_POINTS = ROOT / "proof" / "proof_points.latest.md"

CLAIMS = [
    {
        "claim_text": "Generate clamp policy behavior is demonstrated with manifest evidence.",
        "verification_command": "python proof/generate_canonical_manifest.py",
        "artifact_paths": [
            "proof/artifacts/phase3_generate/evidence_manifest_control.json",
            "proof/artifacts/phase3_generate/evidence_manifest_clamp.json",
        ],
        "expected_signal": "Control and clamp manifests both exist and encode divergent clamp outcomes.",
    },
    {
        "claim_text": "Extract PASS/FAIL gating behavior is evidenced via canonical run outputs.",
        "verification_command": "python proof/generate_canonical_manifest.py",
        "artifact_paths": [
            "proof/artifacts/phase41_extract_gate/evidence_manifest.json",
            "proof/artifacts/phase41_extract_gate/host_pass_runtime.json",
            "proof/artifacts/phase41_extract_gate/host_fail_runtime.json",
        ],
        "expected_signal": "PASS runtime permits extract-capable behavior while FAIL runtime is capability-blocked.",
    },
    {
        "claim_text": "A local kind cluster can run the generate-only inference service with successful health and smoke checks.",
        "verification_command": "python proof/generate_canonical_manifest.py",
        "artifact_paths": [
            "proof/artifacts/phase5_k8s_kind/kind_smoke_summary.json",
            "proof/artifacts/phase5_k8s_kind/kubectl_get_pods.txt",
            "proof/artifacts/phase5_k8s_kind/kubectl_get_svc.txt",
            "proof/artifacts/phase5_k8s_kind/server_rollout_status.txt",
            "proof/artifacts/phase5_k8s_kind/k8s_smoke.log",
            "proof/artifacts/phase5_k8s_kind/kustomize_local_generate_only.yaml",
        ],
        "expected_signal": "Local kind deployment becomes ready, generate-only capability is enforced at runtime, and the local overlay renders successfully.",
    },
    {
        "claim_text": "Async extraction requests are durably queued, executed by a separate worker process, and resolved through a job-status API with reproducible evidence artifacts.",
        "verification_command": "python proof/generate_canonical_manifest.py",
        "artifact_paths": [
            "proof/artifacts/phase6_extract_async/async_submit_response.json",
            "proof/artifacts/phase6_extract_async/async_job_initial.json",
            "proof/artifacts/phase6_extract_async/async_job_final.json",
            "proof/artifacts/phase6_extract_async/async_worker_log.txt",
            "proof/artifacts/phase6_extract_async/async_job_summary.json",
        ],
        "expected_signal": "Async submit returns 202, worker logs prove separate-process execution, and final job state succeeds with a valid result payload.",
    },
    {
        "claim_text": "Traceable request inspection reconstructs ordered sync and async extract timelines, including cross-process async lineage from submission through worker execution and status polling.",
        "verification_command": "python proof/generate_canonical_manifest.py",
        "artifact_paths": [
            "proof/artifacts/phase7_trace_inspection/async_submit_response.json",
            "proof/artifacts/phase7_trace_inspection/async_trace_detail.json",
            "proof/artifacts/phase7_trace_inspection/async_trace_timeline.md",
            "proof/artifacts/phase7_trace_inspection/sync_extract_response.json",
            "proof/artifacts/phase7_trace_inspection/sync_trace_detail.json",
            "proof/artifacts/phase7_trace_inspection/trace_summary.json",
        ],
        "expected_signal": "Sync and async trace artifacts both exist, the async trace includes worker and status-poll events, and the proof summary passes all trace checks.",
    },
    {
        "claim_text": "The promoted Compose extract path runs model-backed generate, sync extract, and async extract with a containerized CPU llama.cpp backend.",
        "verification_command": "python proof/generate_compose_llama_extract_proof.py",
        "artifact_paths": [
            "proof/artifacts/phase8_compose_llama_extract/compose_llama_extract_summary.json",
            "proof/artifacts/phase8_compose_llama_extract/readyz.json",
            "proof/artifacts/phase8_compose_llama_extract/models_status.json",
            "proof/artifacts/phase8_compose_llama_extract/generate_response.json",
            "proof/artifacts/phase8_compose_llama_extract/extract_response.json",
            "proof/artifacts/phase8_compose_llama_extract/async_submit_response.json",
            "proof/artifacts/phase8_compose_llama_extract/async_final_response.json",
        ],
        "expected_signal": "The Compose llama summary passes readiness, model status, generate, sync extract, and async extract checks.",
    },
    {
        "claim_text": "Eval artifacts drive policy decisions, and admin reload changes runtime extract behavior according to the active policy artifact.",
        "verification_command": "python proof/generate_canonical_manifest.py",
        "artifact_paths": [
            "proof/artifacts/phase9_policy_eval_linkage/eval_pass/summary.json",
            "proof/artifacts/phase9_policy_eval_linkage/eval_fail/summary.json",
            "proof/artifacts/phase9_policy_eval_linkage/policy_allow.json",
            "proof/artifacts/phase9_policy_eval_linkage/policy_deny.json",
            "proof/artifacts/phase9_policy_eval_linkage/admin_policy_initial.json",
            "proof/artifacts/phase9_policy_eval_linkage/admin_policy_reload.json",
            "proof/artifacts/phase9_policy_eval_linkage/extract_allow_response.json",
            "proof/artifacts/phase9_policy_eval_linkage/extract_deny_response.json",
            "proof/artifacts/phase9_policy_eval_linkage/policy_eval_linkage_summary.json",
        ],
        "expected_signal": "Passing eval produces an allow policy and successful extract; failing eval produces a deny policy and blocked extract after admin reload.",
    },
    {
        "claim_text": "The local ops surface exposes API, UI, Prometheus, and Grafana directly and through a local edge proxy.",
        "verification_command": "python proof/generate_canonical_manifest.py",
        "artifact_paths": [
            "proof/artifacts/phase10_ops_surface/direct_api_healthz.json",
            "proof/artifacts/phase10_ops_surface/direct_ui_index.json",
            "proof/artifacts/phase10_ops_surface/direct_prometheus_ready.json",
            "proof/artifacts/phase10_ops_surface/direct_grafana_health.json",
            "proof/artifacts/phase10_ops_surface/prometheus_targets.json",
            "proof/artifacts/phase10_ops_surface/prometheus_query_up.json",
            "proof/artifacts/phase10_ops_surface/grafana_datasources.json",
            "proof/artifacts/phase10_ops_surface/grafana_dashboards.json",
            "proof/artifacts/phase10_ops_surface/grafana_prometheus_proxy_query_up.json",
            "proof/artifacts/phase10_ops_surface/dashboard_population_summary.json",
            "proof/artifacts/phase10_ops_surface/proxy_api_healthz.json",
            "proof/artifacts/phase10_ops_surface/proxy_ui_index.json",
            "proof/artifacts/phase10_ops_surface/proxy_prometheus_ready.json",
            "proof/artifacts/phase10_ops_surface/proxy_grafana_health.json",
            "proof/artifacts/phase10_ops_surface/compose_ps.txt",
            "proof/artifacts/phase10_ops_surface/ops_surface_summary.json",
        ],
        "expected_signal": "Direct and proxied API/UI/observability endpoints return successful health or index responses, Prometheus scrapes the API target, and provisioned Grafana dashboards have Prometheus-backed query data.",
    },
]


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "UNKNOWN"


def write_proof_points() -> None:
    lines = [
        "# Canonical Proof Points (Latest)",
        "",
        "## Proof 1: Generate Clamp",
        "- Claim: runtime policy clamp behavior is reproducibly evidenced.",
        "- Command: `python proof/generate_canonical_manifest.py`",
        "- Artifacts:",
        "  - `proof/artifacts/phase3_generate/evidence_manifest_control.json`",
        "  - `proof/artifacts/phase3_generate/evidence_manifest_clamp.json`",
        "- Validation signal: both control and clamp manifests exist with divergent expected outcomes.",
        "",
        "## Proof 2: Extract Gate PASS/FAIL",
        "- Claim: offline onboarding artifacts drive runtime extract capability enforcement.",
        "- Command: `python proof/generate_canonical_manifest.py`",
        "- Artifacts:",
        "  - `proof/artifacts/phase41_extract_gate/evidence_manifest.json`",
        "  - `proof/artifacts/phase41_extract_gate/host_pass_runtime.json`",
        "  - `proof/artifacts/phase41_extract_gate/host_fail_runtime.json`",
        "- Validation signal: PASS and FAIL runtime outputs differ according to gating expectation.",
        "",
        "## Proof 3: Kubernetes kind deployment",
        "- Claim: a local kind cluster runs the generate-only service successfully.",
        "- Command: `python proof/generate_canonical_manifest.py`",
        "- Artifacts:",
        "  - `proof/artifacts/phase5_k8s_kind/kind_smoke_summary.json`",
        "  - `proof/artifacts/phase5_k8s_kind/kubectl_get_pods.txt`",
        "  - `proof/artifacts/phase5_k8s_kind/kubectl_get_svc.txt`",
        "  - `proof/artifacts/phase5_k8s_kind/server_rollout_status.txt`",
        "  - `proof/artifacts/phase5_k8s_kind/k8s_smoke.log`",
        "  - `proof/artifacts/phase5_k8s_kind/kustomize_local_generate_only.yaml`",
        "- Validation signal: rollout passes, `/healthz` and generate smoke pass, `/v1/extract` is blocked, and the local overlay renders successfully.",
        "",
        "## Proof 4: Async extract jobs",
        "- Claim: extract requests can be queued and executed by a separate worker process with durable job state.",
        "- Command: `python proof/generate_canonical_manifest.py`",
        "- Artifacts:",
        "  - `proof/artifacts/phase6_extract_async/async_submit_response.json`",
        "  - `proof/artifacts/phase6_extract_async/async_job_initial.json`",
        "  - `proof/artifacts/phase6_extract_async/async_job_final.json`",
        "  - `proof/artifacts/phase6_extract_async/async_worker_log.txt`",
        "  - `proof/artifacts/phase6_extract_async/async_job_summary.json`",
        "- Validation signal: submit returns `202`, worker log includes the queued job id, and final status is `succeeded` with a valid result object.",
        "",
        "## Proof 5: Traceable request inspection",
        "- Claim: sync and async extract flows can be inspected as ordered per-request timelines, including async cross-process lineage.",
        "- Command: `python proof/generate_canonical_manifest.py`",
        "- Artifacts:",
        "  - `proof/artifacts/phase7_trace_inspection/async_submit_response.json`",
        "  - `proof/artifacts/phase7_trace_inspection/async_trace_detail.json`",
        "  - `proof/artifacts/phase7_trace_inspection/async_trace_timeline.md`",
        "  - `proof/artifacts/phase7_trace_inspection/sync_extract_response.json`",
        "  - `proof/artifacts/phase7_trace_inspection/sync_trace_detail.json`",
        "  - `proof/artifacts/phase7_trace_inspection/trace_summary.json`",
        "- Validation signal: sync and async trace detail endpoints return coherent ordered events, and the async trace includes submission, worker, and status-poll lineage.",
        "",
        "## Proof 6: Compose llama extract",
        "- Claim: the promoted Compose extract target runs model-backed generate, sync extract, and async extract through a CPU containerized llama.cpp backend.",
        "- Command: `python proof/generate_compose_llama_extract_proof.py`",
        "- Artifacts:",
        "  - `proof/artifacts/phase8_compose_llama_extract/compose_llama_extract_summary.json`",
        "  - `proof/artifacts/phase8_compose_llama_extract/readyz.json`",
        "  - `proof/artifacts/phase8_compose_llama_extract/models_status.json`",
        "  - `proof/artifacts/phase8_compose_llama_extract/generate_response.json`",
        "  - `proof/artifacts/phase8_compose_llama_extract/extract_response.json`",
        "  - `proof/artifacts/phase8_compose_llama_extract/async_submit_response.json`",
        "  - `proof/artifacts/phase8_compose_llama_extract/async_final_response.json`",
        "- Validation signal: readiness, model status, generate, sync extract, and async extract checks all pass.",
        "",
        "## Proof 7: Policy/eval linkage",
        "- Claim: eval artifacts drive policy decisions, and admin reload changes runtime extract behavior according to the active policy artifact.",
        "- Command: `python proof/generate_canonical_manifest.py`",
        "- Artifacts:",
        "  - `proof/artifacts/phase9_policy_eval_linkage/eval_pass/summary.json`",
        "  - `proof/artifacts/phase9_policy_eval_linkage/eval_fail/summary.json`",
        "  - `proof/artifacts/phase9_policy_eval_linkage/policy_allow.json`",
        "  - `proof/artifacts/phase9_policy_eval_linkage/policy_deny.json`",
        "  - `proof/artifacts/phase9_policy_eval_linkage/admin_policy_initial.json`",
        "  - `proof/artifacts/phase9_policy_eval_linkage/admin_policy_reload.json`",
        "  - `proof/artifacts/phase9_policy_eval_linkage/extract_allow_response.json`",
        "  - `proof/artifacts/phase9_policy_eval_linkage/extract_deny_response.json`",
        "  - `proof/artifacts/phase9_policy_eval_linkage/policy_eval_linkage_summary.json`",
        "- Validation signal: passing eval allows extract; failing eval blocks extract after admin policy reload.",
        "",
        "## Proof 8: UI and observability proxy",
        "- Claim: local API, UI, Prometheus, and Grafana surfaces are reachable directly and through the local edge proxy.",
        "- Command: `python proof/generate_canonical_manifest.py`",
        "- Artifacts:",
        "  - `proof/artifacts/phase10_ops_surface/direct_api_healthz.json`",
        "  - `proof/artifacts/phase10_ops_surface/direct_ui_index.json`",
        "  - `proof/artifacts/phase10_ops_surface/direct_prometheus_ready.json`",
        "  - `proof/artifacts/phase10_ops_surface/direct_grafana_health.json`",
        "  - `proof/artifacts/phase10_ops_surface/prometheus_targets.json`",
        "  - `proof/artifacts/phase10_ops_surface/prometheus_query_up.json`",
        "  - `proof/artifacts/phase10_ops_surface/grafana_datasources.json`",
        "  - `proof/artifacts/phase10_ops_surface/grafana_dashboards.json`",
        "  - `proof/artifacts/phase10_ops_surface/grafana_prometheus_proxy_query_up.json`",
        "  - `proof/artifacts/phase10_ops_surface/dashboard_population_summary.json`",
        "  - `proof/artifacts/phase10_ops_surface/proxy_api_healthz.json`",
        "  - `proof/artifacts/phase10_ops_surface/proxy_ui_index.json`",
        "  - `proof/artifacts/phase10_ops_surface/proxy_prometheus_ready.json`",
        "  - `proof/artifacts/phase10_ops_surface/proxy_grafana_health.json`",
        "  - `proof/artifacts/phase10_ops_surface/compose_ps.txt`",
        "  - `proof/artifacts/phase10_ops_surface/ops_surface_summary.json`",
        "- Validation signal: all direct/proxied endpoint checks pass, Prometheus reports the API scrape target up, and Grafana dashboards expose Prometheus-backed query data.",
        "",
    ]
    PROOF_POINTS.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    generate_k8s_kind_proof()
    generate_async_extract_proof()
    generate_trace_inspection_proof()
    generate_policy_eval_linkage_proof()
    generate_ops_surface_proof()
    data = {
        "proof_id": "llm-extract-canonical",
        "run_id": "canonical_latest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_commit": git_commit(),
        "status": "pass",
        "claims": CLAIMS,
        "diagnostics": {
            "notes": [
                "Canonical manifest generated from latest curated demo artifacts.",
                "Run proof/validate_evidence_manifest.py to enforce contract, file existence, and runtime proof signals.",
                "Phase 8 Compose llama extract evidence is generated separately because it requires a configured local GGUF model.",
            ]
        },
    }
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    write_proof_points()
    print(f"Updated {MANIFEST}")


if __name__ == "__main__":
    main()
