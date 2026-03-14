#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "proof" / "evidence_manifest.latest.json"
K8S_SUMMARY = ROOT / "proof" / "artifacts" / "phase5_k8s_kind" / "kind_smoke_summary.json"
ASYNC_SUMMARY = ROOT / "proof" / "artifacts" / "phase6_extract_async" / "async_job_summary.json"
TRACE_SUMMARY = ROOT / "proof" / "artifacts" / "phase7_trace_inspection" / "trace_summary.json"

REQUIRED_TOP = [
    "proof_id",
    "run_id",
    "generated_at",
    "repo_commit",
    "status",
    "claims",
    "diagnostics",
]
REQUIRED_CLAIM = ["claim_text", "verification_command", "artifact_paths", "expected_signal"]
REQUIRED_K8S_CHECKS = [
    "rollout_status",
    "healthz",
    "models_capabilities",
    "generate_smoke",
    "extract_disabled",
    "local_overlay_render",
    "prod_overlay_render",
]


def fail(msg: str) -> None:
    print(f"ERROR: {msg}")
    sys.exit(1)


def validate_rendered_manifest(path: Path, *, require_probe: bool) -> None:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        fail(f"rendered manifest is empty: {path.relative_to(ROOT)}")
    if "kind: Deployment" not in text:
        fail(f"rendered manifest missing Deployment: {path.relative_to(ROOT)}")
    if require_probe and "kind: Service" not in text:
        fail(f"local rendered manifest missing Service: {path.relative_to(ROOT)}")
    if require_probe and "readinessProbe:" not in text:
        fail(f"local rendered manifest missing readinessProbe: {path.relative_to(ROOT)}")
    if require_probe and "livenessProbe:" not in text:
        fail(f"local rendered manifest missing livenessProbe: {path.relative_to(ROOT)}")
    if require_probe and "configMap:" not in text and "env:" not in text:
        fail(f"local rendered manifest missing config/env wiring: {path.relative_to(ROOT)}")


def validate_k8s_summary() -> None:
    if not K8S_SUMMARY.exists():
        fail(f"missing Kubernetes summary artifact: {K8S_SUMMARY.relative_to(ROOT)}")

    data = json.loads(K8S_SUMMARY.read_text(encoding="utf-8"))
    required_top = [
        "proof_phase",
        "cluster_name",
        "namespace",
        "overlay",
        "generated_at",
        "status",
        "checks",
    ]
    for key in required_top:
        if key not in data:
            fail(f"k8s summary missing key: {key}")

    if data["proof_phase"] != "phase5_k8s_kind":
        fail("k8s summary proof_phase must be phase5_k8s_kind")
    if data["cluster_name"] != "llm":
        fail("k8s summary cluster_name must be llm")
    if data["namespace"] != "llm":
        fail("k8s summary namespace must be llm")
    if data["overlay"] != "local-generate-only":
        fail("k8s summary overlay must be local-generate-only")
    if data["status"] != "pass":
        fail("k8s summary status must be pass")

    checks = data.get("checks")
    if not isinstance(checks, dict):
        fail("k8s summary checks must be an object")
    for key in REQUIRED_K8S_CHECKS:
        if checks.get(key) != "pass":
            fail(f"k8s summary check {key} must be pass")

    dep_caps = data.get("deployment_capabilities")
    if dep_caps is not None:
        if dep_caps.get("generate") is not True:
            fail("k8s summary deployment_capabilities.generate must be true")
        if dep_caps.get("extract") is not False:
            fail("k8s summary deployment_capabilities.extract must be false")

    validate_rendered_manifest(
        ROOT / "proof" / "artifacts" / "phase5_k8s_kind" / "kustomize_local_generate_only.yaml",
        require_probe=True,
    )
    validate_rendered_manifest(
        ROOT / "proof" / "artifacts" / "phase5_k8s_kind" / "kustomize_prod_gpu_full.yaml",
        require_probe=False,
    )


def validate_async_summary() -> None:
    if not ASYNC_SUMMARY.exists():
        fail(f"missing async summary artifact: {ASYNC_SUMMARY.relative_to(ROOT)}")

    data = json.loads(ASYNC_SUMMARY.read_text(encoding="utf-8"))
    for key in (
        "proof_phase",
        "generated_at",
        "status",
        "job_id",
        "submission_status",
        "final_status",
        "worker_claimed",
        "result_valid",
        "schema_id",
        "resolved_model_id",
    ):
        if key not in data:
            fail(f"async summary missing key: {key}")

    if data["proof_phase"] != "phase6_extract_async":
        fail("async summary proof_phase must be phase6_extract_async")
    if data["status"] != "pass":
        fail("async summary status must be pass")
    if data["submission_status"] != "queued":
        fail("async summary submission_status must be queued")
    if data["final_status"] != "succeeded":
        fail("async summary final_status must be succeeded")
    if data["worker_claimed"] is not True:
        fail("async summary worker_claimed must be true")
    if data["result_valid"] is not True:
        fail("async summary result_valid must be true")

    submit_path = (
        ROOT / "proof" / "artifacts" / "phase6_extract_async" / "async_submit_response.json"
    )
    final_path = ROOT / "proof" / "artifacts" / "phase6_extract_async" / "async_job_final.json"
    worker_log = ROOT / "proof" / "artifacts" / "phase6_extract_async" / "async_worker_log.txt"
    submit = json.loads(submit_path.read_text(encoding="utf-8"))
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if submit.get("status_code") != 202:
        fail("async submit response must record status_code 202")
    final_body = final.get("body") or {}
    if final.get("status_code") != 200:
        fail("async final job response must record status_code 200")
    if final_body.get("status") != "succeeded":
        fail("async final job body must be succeeded")
    result = final_body.get("result")
    if not isinstance(result, dict) or not result:
        fail("async final job body must include non-empty result object")
    log_text = worker_log.read_text(encoding="utf-8")
    if str(data["job_id"]) not in log_text:
        fail("async worker log must include the job id")


def validate_trace_summary() -> None:
    if not TRACE_SUMMARY.exists():
        fail(f"missing trace summary artifact: {TRACE_SUMMARY.relative_to(ROOT)}")

    data = json.loads(TRACE_SUMMARY.read_text(encoding="utf-8"))
    for key in (
        "proof_phase",
        "generated_at",
        "status",
        "sync_trace_complete",
        "async_trace_complete",
        "async_worker_claimed",
        "async_status_polled",
        "sync_contains_generate_or_cache_path",
        "trace_ids_present",
        "sync_trace_id",
        "async_trace_id",
    ):
        if key not in data:
            fail(f"trace summary missing key: {key}")

    if data["proof_phase"] != "phase7_trace_inspection":
        fail("trace summary proof_phase must be phase7_trace_inspection")
    if data["status"] != "pass":
        fail("trace summary status must be pass")

    for key in (
        "sync_trace_complete",
        "async_trace_complete",
        "async_worker_claimed",
        "async_status_polled",
        "sync_contains_generate_or_cache_path",
        "trace_ids_present",
    ):
        if data[key] is not True:
            fail(f"trace summary {key} must be true")

    sync_trace = ROOT / "proof" / "artifacts" / "phase7_trace_inspection" / "sync_trace_detail.json"
    async_trace = (
        ROOT / "proof" / "artifacts" / "phase7_trace_inspection" / "async_trace_detail.json"
    )
    async_timeline = (
        ROOT / "proof" / "artifacts" / "phase7_trace_inspection" / "async_trace_timeline.md"
    )
    sync_resp = (
        ROOT / "proof" / "artifacts" / "phase7_trace_inspection" / "sync_extract_response.json"
    )
    async_submit = (
        ROOT / "proof" / "artifacts" / "phase7_trace_inspection" / "async_submit_response.json"
    )
    for path in (sync_trace, async_trace, async_timeline, sync_resp, async_submit):
        if not path.exists():
            fail(f"missing trace artifact: {path.relative_to(ROOT)}")

    sync_payload = json.loads(sync_trace.read_text(encoding="utf-8"))
    async_payload = json.loads(async_trace.read_text(encoding="utf-8"))
    if sync_payload.get("status_code") != 200:
        fail("sync trace detail must record status_code 200")
    if async_payload.get("status_code") != 200:
        fail("async trace detail must record status_code 200")
    sync_events = [x.get("event_name") for x in (sync_payload.get("body") or {}).get("events", [])]
    async_events = [
        x.get("event_name") for x in (async_payload.get("body") or {}).get("events", [])
    ]
    if "extract.completed" not in sync_events:
        fail("sync trace detail must include extract.completed")
    for name in (
        "extract_job.worker_claimed",
        "extract_job.completed",
        "extract_job.status_polled",
    ):
        if name not in async_events:
            fail(f"async trace detail must include {name}")


def main() -> None:
    if not MANIFEST.exists():
        fail(f"missing manifest: {MANIFEST}")

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for key in REQUIRED_TOP:
        if key not in data:
            fail(f"missing top-level key: {key}")

    if data["status"] not in {"pass", "fail"}:
        fail("status must be pass|fail")

    claims = data.get("claims", [])
    if not isinstance(claims, list) or not claims:
        fail("claims must be non-empty list")

    for idx, claim in enumerate(claims, start=1):
        for key in REQUIRED_CLAIM:
            if key not in claim:
                fail(f"claim[{idx}] missing key: {key}")
        paths = claim["artifact_paths"]
        if not isinstance(paths, list) or not paths:
            fail(f"claim[{idx}] artifact_paths must be non-empty list")
        for raw in paths:
            p = ROOT / raw
            if not p.exists():
                fail(f"claim[{idx}] missing artifact path: {raw}")

    if any(
        "phase5_k8s_kind/kind_smoke_summary.json" in path
        for claim in claims
        for path in claim["artifact_paths"]
    ):
        validate_k8s_summary()
    if any(
        "phase6_extract_async/async_job_summary.json" in path
        for claim in claims
        for path in claim["artifact_paths"]
    ):
        validate_async_summary()
    if any(
        "phase7_trace_inspection/trace_summary.json" in path
        for claim in claims
        for path in claim["artifact_paths"]
    ):
        validate_trace_summary()

    print("OK: evidence manifest contract and artifact paths validated")


if __name__ == "__main__":
    main()
