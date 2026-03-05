#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _status_for_extract(payload: dict[str, Any], expected: str) -> tuple[bool, str]:
    probe = payload.get("extract_probe") or {}
    code = str(probe.get("code") or "").lower()
    status = probe.get("status")
    ok_flag = bool(payload.get("ok"))

    if expected == "allow":
        blocked_codes = {"capability_disabled", "capability_not_supported"}
        ok = ok_flag and code not in blocked_codes and int(status or 0) not in {400, 501}
    else:
        blocked_codes = {"capability_disabled", "capability_not_supported"}
        ok = ok_flag and (code in blocked_codes or int(status or 0) in {400, 501})
    observed = f"status={status},code={code or 'none'},ok={ok_flag}"
    return ok, observed


def main() -> int:
    p = argparse.ArgumentParser(description="Write extract-gate demo evidence manifest")
    p.add_argument("--run-dir", required=True, help="traffic_out/<run> directory")
    p.add_argument("--out", default="", help="output path (defaults to <run-dir>/evidence_manifest.json)")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out = Path(args.out).resolve() if args.out else run_dir / "evidence_manifest.json"

    required = {
        "host_pass_runtime": run_dir / "host_pass_runtime.json",
        "host_fail_runtime": run_dir / "host_fail_runtime.json",
        "host_pass_extract": run_dir / "host_pass_extract.json",
        "host_fail_extract": run_dir / "host_fail_extract.json",
    }

    missing = [k for k, v in required.items() if not v.exists()]
    if missing:
        payload = {
            "schema_version": "demo_evidence_manifest_v1",
            "demo": "extract_gate",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir),
            "ok": False,
            "error": f"missing required evidence files: {', '.join(missing)}",
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return 1

    host_pass_rt = _load_json(required["host_pass_runtime"])
    host_fail_rt = _load_json(required["host_fail_runtime"])
    host_pass_ex = _load_json(required["host_pass_extract"])
    host_fail_ex = _load_json(required["host_fail_extract"])

    pass_ok, pass_observed = _status_for_extract(host_pass_ex, "allow")
    fail_ok, fail_observed = _status_for_extract(host_fail_ex, "block")

    pass_cap = bool((host_pass_rt.get("runtime") or {}).get("model_extract_capability"))
    fail_cap = bool((host_fail_rt.get("runtime") or {}).get("model_extract_capability"))

    verdict_ok = pass_ok and fail_ok and pass_cap and (not fail_cap)

    payload = {
        "schema_version": "demo_evidence_manifest_v1",
        "demo": "extract_gate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "ok": verdict_ok,
        "control": {
            "pass_artifact": "extract capability expected true",
            "fail_artifact": "extract capability expected false",
        },
        "observed": {
            "host_pass": {
                "extract_status": pass_observed,
                "model_extract_capability": pass_cap,
            },
            "host_fail": {
                "extract_status": fail_observed,
                "model_extract_capability": fail_cap,
            },
        },
        "evidence_files": {k: str(v) for k, v in required.items()},
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
