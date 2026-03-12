#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from generate_k8s_kind_proof import generate_k8s_kind_proof

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
        "claim_text": "A local kind cluster can run the generate-only inference service with successful health and smoke checks, and the production Kubernetes scaffold renders cleanly with deployment primitives expected for a production-oriented ML service.",
        "verification_command": "python proof/generate_canonical_manifest.py",
        "artifact_paths": [
            "proof/artifacts/phase5_k8s_kind/kind_smoke_summary.json",
            "proof/artifacts/phase5_k8s_kind/kubectl_get_pods.txt",
            "proof/artifacts/phase5_k8s_kind/kubectl_get_svc.txt",
            "proof/artifacts/phase5_k8s_kind/server_rollout_status.txt",
            "proof/artifacts/phase5_k8s_kind/k8s_smoke.log",
            "proof/artifacts/phase5_k8s_kind/kustomize_local_generate_only.yaml",
            "proof/artifacts/phase5_k8s_kind/kustomize_prod_gpu_full.yaml",
        ],
        "expected_signal": "Local kind deployment becomes ready, generate-only capability is enforced at runtime, and both local/prod overlays render successfully.",
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
        "- Claim: a local kind cluster runs the generate-only service successfully, while the production overlay renders as a production-oriented scaffold.",
        "- Command: `python proof/generate_canonical_manifest.py`",
        "- Artifacts:",
        "  - `proof/artifacts/phase5_k8s_kind/kind_smoke_summary.json`",
        "  - `proof/artifacts/phase5_k8s_kind/kubectl_get_pods.txt`",
        "  - `proof/artifacts/phase5_k8s_kind/kubectl_get_svc.txt`",
        "  - `proof/artifacts/phase5_k8s_kind/server_rollout_status.txt`",
        "  - `proof/artifacts/phase5_k8s_kind/k8s_smoke.log`",
        "  - `proof/artifacts/phase5_k8s_kind/kustomize_local_generate_only.yaml`",
        "  - `proof/artifacts/phase5_k8s_kind/kustomize_prod_gpu_full.yaml`",
        "- Validation signal: rollout passes, `/healthz` and generate smoke pass, `/v1/extract` is blocked, and both overlays render successfully.",
        "",
    ]
    PROOF_POINTS.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    generate_k8s_kind_proof()
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
                "Run proof/validate_evidence_manifest.py to enforce contract, file existence, and Kubernetes proof signals.",
            ]
        },
    }
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    write_proof_points()
    print(f"Updated {MANIFEST}")


if __name__ == "__main__":
    main()
