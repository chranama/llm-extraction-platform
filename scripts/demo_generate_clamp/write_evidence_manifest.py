#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description="Write generate-clamp demo evidence manifest")
    p.add_argument("--slo", required=True, help="path to runtime_generate_slo_v1 json")
    p.add_argument("--policy", required=True, help="path to policy_decision_v2 json")
    p.add_argument("--out", required=True, help="output manifest path")
    p.add_argument("--expected-clamp", choices=["yes", "no"], default="yes")
    args = p.parse_args()

    slo = _load(Path(args.slo))
    policy = _load(Path(args.policy))

    cap = policy.get("generate_max_new_tokens_cap")
    clamped = cap is not None
    want = args.expected_clamp == "yes"

    payload = {
        "schema_version": "demo_evidence_manifest_v1",
        "demo": "generate_clamp",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": clamped == want,
        "control": {
            "expected_clamp": want,
            "threshold_profile": policy.get("generate_thresholds_profile"),
        },
        "observed": {
            "policy_cap": cap,
            "policy_status": policy.get("status"),
            "latency_p95_ms": policy.get("metrics", {}).get("generate_slo_latency_p95_ms")
            or slo.get("totals", {}).get("latency_ms", {}).get("p95"),
            "error_rate": policy.get("metrics", {}).get("generate_slo_error_rate")
            or slo.get("totals", {}).get("errors", {}).get("rate"),
            "total_requests": policy.get("metrics", {}).get("generate_slo_total_requests")
            or slo.get("totals", {}).get("requests", {}).get("total"),
        },
        "evidence_files": {
            "slo": str(Path(args.slo).resolve()),
            "policy": str(Path(args.policy).resolve()),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
