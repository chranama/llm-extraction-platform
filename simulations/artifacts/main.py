# simulations/artifacts/main.py
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import shutil

from simulations.paths import resolve_under_repo

from simulations.artifacts.slo_fixtures import (
    bad_generate_slo_payload,
    good_generate_slo_payload,
    write_generate_slo_latest,
)
from simulations.artifacts.policy_fixtures import (
    allow_no_clamp_policy,
    allow_with_clamp_policy,
    deny_policy,
    unknown_policy,
    write_policy_latest,
)
from simulations.artifacts.contracts import (
    verify_policy_payload,
    verify_slo_payload,
)

# Eval fixtures (formerly Demo B fixtures)
from simulations.artifacts.eval_fixtures import (
    eval_fixture_fail,
    eval_fixture_pass,
    write_eval_pointer_for_run,
)

# Demo models.yaml generator
from simulations.artifacts.models_yaml_demo import (
    DemoModelsYamlError,
    build_demo_models_yaml,
)
from llm_policy.onboarding.demo import apply_model_onboarding

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


class SimError(Exception):
    def __init__(self, message: str, *, code: int = 2):
        super().__init__(message)
        self.code = int(code)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SimError(f"File not found: {path}", code=2)
    except Exception as e:
        raise SimError(f"Failed to read JSON from {path}: {type(e).__name__}: {e}", code=2)


def _maybe_print_write(rt: Any, *, dest: Path, payload: dict[str, Any], label: str) -> None:
    if getattr(rt, "dry_run", False):
        print(f"[dry-run] would write {label} -> {dest}")
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def _build_extract_only_policy_payload(
    *,
    enable_extract: bool,
    thresholds_profile: str,
    eval_run_dir: str,
    model_id: Optional[str] = None,
    policy_name: str = "extract_enablement",
    status: str = "allow",  # allow|deny|unknown
    ok: bool = True,
    contract_errors: int = 0,
    generated_at: Optional[str] = None,
) -> dict[str, Any]:
    """
    Build a policy_decision_v2 payload for extract gating:
      pipeline = extract_only
    """
    if status not in ("allow", "deny", "unknown"):
        raise SimError("status must be one of: allow, deny, unknown", code=2)

    if not generated_at:
        generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    payload: dict[str, Any] = {
        "schema_version": "policy_decision_v2",
        "generated_at": generated_at,
        "policy": policy_name,
        "pipeline": "extract_only",
        "status": status,
        "ok": bool(ok),
        "contract_errors": int(contract_errors),
        "reasons": [],
        "warnings": [],
        "enable_extract": bool(enable_extract),
        "thresholds_profile": str(thresholds_profile),
        "thresholds_version": "fixtures",
        "eval_run_dir": str(eval_run_dir),
        "eval_task": "extract",
        "eval_run_id": None,
        "model_id": model_id,
        "generate_max_new_tokens_cap": None,
        "generate_thresholds_profile": None,
        "metrics": {"extract_gate": {"enable_extract": bool(enable_extract)}},
        "contract_warnings": 0,
    }

    res = verify_policy_payload(payload)
    if not res.ok:
        raise SimError(f"extract_only policy payload failed verification: {res.error}", code=2)
    return payload


# ---------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------


def _artifacts_write_slo(rt: Any, args: argparse.Namespace) -> int:
    out_path = resolve_under_repo(rt.repo_root, getattr(args, "out", None)) or rt.slo_out_path
    fixture = str(getattr(args, "fixture"))
    model_id = str(getattr(args, "model_id") or "").strip() or "demo-model"
    window_seconds = int(getattr(args, "window_seconds") or 300)

    if fixture == "good":
        payload = good_generate_slo_payload(model_id=model_id, window_seconds=window_seconds)
    elif fixture == "bad":
        payload = bad_generate_slo_payload(model_id=model_id, window_seconds=window_seconds)
    else:
        raise SimError(f"Unknown SLO fixture: {fixture}", code=2)

    _maybe_print_write(rt, dest=out_path, payload=payload, label=f"SLO fixture '{fixture}'")
    if rt.dry_run:
        return 0

    write_generate_slo_latest(out_path, payload)
    print(f"✅ wrote SLO fixture '{fixture}' -> {out_path}")
    return 0


def _artifacts_verify_slo(rt: Any, args: argparse.Namespace) -> int:
    path = resolve_under_repo(rt.repo_root, getattr(args, "path", None)) or rt.slo_out_path
    payload = _read_json(path)
    res = verify_slo_payload(payload, source_path=str(path))
    if not res.ok:
        raise SimError(f"SLO verify failed: {res.error}", code=2)

    snap = res.snapshot
    print(
        json.dumps(
            {
                "ok": True,
                "kind": res.kind,
                "source_path": res.source_path,
                "schema_version": getattr(snap, "schema_version", ""),
                "window_seconds": getattr(snap, "window_seconds", None),
                "routes": getattr(snap, "routes", []),
                "total_requests": getattr(snap, "total_requests", None),
                "error_rate": getattr(snap, "error_rate", None),
                "latency_p95_ms": getattr(snap, "latency_p95_ms", None),
                "completion_tokens_p95": getattr(snap, "completion_tokens_p95", None),
            },
            indent=2,
        )
    )
    return 0


def _artifacts_write_policy(rt: Any, args: argparse.Namespace) -> int:
    out_path = resolve_under_repo(rt.repo_root, getattr(args, "out", None)) or rt.policy_out_path
    fixture = str(getattr(args, "fixture"))
    cap = getattr(args, "cap", None)
    gprof = str(getattr(args, "generate_thresholds_profile") or "default").strip() or "default"
    model_id = str(getattr(args, "model_id") or "").strip() or None

    if fixture == "allow_no_clamp":
        payload = allow_no_clamp_policy(generate_thresholds_profile=gprof, model_id=model_id)
    elif fixture == "allow_clamp":
        if cap is None or int(cap) <= 0:
            raise SimError("--cap must be a positive integer for allow_clamp", code=2)
        payload = allow_with_clamp_policy(cap=int(cap), generate_thresholds_profile=gprof, model_id=model_id)
    elif fixture == "deny":
        payload = deny_policy(generate_thresholds_profile=gprof, model_id=model_id)
    elif fixture == "unknown":
        payload = unknown_policy(generate_thresholds_profile=gprof, model_id=model_id)
    else:
        raise SimError(f"Unknown policy fixture: {fixture}", code=2)

    _maybe_print_write(rt, dest=out_path, payload=payload, label=f"policy fixture '{fixture}'")
    if rt.dry_run:
        return 0

    write_policy_latest(out_path, payload)
    print(f"✅ wrote policy fixture '{fixture}' -> {out_path}")
    return 0


def _artifacts_verify_policy(rt: Any, args: argparse.Namespace) -> int:
    path = resolve_under_repo(rt.repo_root, getattr(args, "path", None)) or rt.policy_out_path
    payload = _read_json(path)
    res = verify_policy_payload(payload, source_path=str(path))
    if not res.ok:
        raise SimError(f"Policy verify failed: {res.error}", code=2)

    snap = res.snapshot
    print(
        json.dumps(
            {
                "ok": True,
                "kind": res.kind,
                "source_path": res.source_path,
                "schema_version": getattr(snap, "schema_version", ""),
                "policy_ok": getattr(snap, "ok", None),
                "pipeline": getattr(snap, "pipeline", None),
                "status": getattr(snap, "status", None),
                "generate_max_new_tokens_cap": getattr(snap, "generate_max_new_tokens_cap", None),
                "enable_extract": getattr(snap, "enable_extract", None),
                "contract_errors": getattr(snap, "contract_errors", None),
                "model_id": getattr(snap, "model_id", None),
                "generate_thresholds_profile": getattr(snap, "generate_thresholds_profile", None),
                "thresholds_profile": getattr(snap, "thresholds_profile", None),
                "eval_run_dir": getattr(snap, "eval_run_dir", None),
            },
            indent=2,
        )
    )
    return 0


def _artifacts_demo_eval(rt: Any, args: argparse.Namespace) -> int:
    """
    Writes an eval fixture run + pointer + an extract_only policy artifact.

    This is your "Demo Eval" artifact writer (renamed from demo-b).
    """
    fixture = str(getattr(args, "fixture") or "pass")
    run_id = str(getattr(args, "run_id") or "").strip()
    model_id = str(getattr(args, "model_id") or "").strip() or None
    schema_id = str(getattr(args, "schema_id") or "sroie_receipt_v1").strip() or "sroie_receipt_v1"
    thresholds_profile = str(getattr(args, "thresholds_profile") or "default").strip() or "default"

    if not run_id:
        raise SimError("--run-id is required", code=2)
    if fixture not in ("pass", "fail"):
        raise SimError("fixture must be one of: pass, fail", code=2)

    pointer_out = getattr(rt, "eval_out_path", None)

    if rt.dry_run:
        print("[dry-run] demo-eval will write:")
        print(f"  - results/extract/{run_id}/summary.json (+ results.jsonl)")
        print(f"  - eval_out/extract/latest.json (or override) -> {pointer_out or '(default)'}")
        print(f"  - policy_out/latest.json -> {rt.policy_out_path}")
        print(f"  - enable_extract = {fixture == 'pass'}")
        return 0

    if fixture == "pass":
        paths, pointer = eval_fixture_pass(
            repo_root=rt.repo_root,
            task="extract",
            run_id=run_id,
            model_id=model_id,
            schema_id=schema_id,
            thresholds_profile=thresholds_profile,
        )
    else:
        paths, pointer = eval_fixture_fail(
            repo_root=rt.repo_root,
            task="extract",
            run_id=run_id,
            model_id=model_id,
            schema_id=schema_id,
            thresholds_profile=thresholds_profile,
        )

    if isinstance(pointer_out, Path):
        pointer = write_eval_pointer_for_run(
            task="extract",
            run_id=run_id,
            run_dir=paths.run_dir,
            summary_path=paths.summary_json,
            notes={"fixture": f"eval_{fixture}", "overridden_pointer_path": str(pointer_out)},
            out_path=pointer_out,
        )

    enable_extract = fixture == "pass"
    policy_payload = _build_extract_only_policy_payload(
        enable_extract=enable_extract,
        thresholds_profile=thresholds_profile,
        eval_run_dir=str(paths.run_dir),
        model_id=model_id,
        status="allow",
        ok=True,
        contract_errors=0,
    )
    write_policy_latest(rt.policy_out_path, policy_payload)

    print(
        json.dumps(
            {
                "ok": True,
                "fixture": fixture,
                "enable_extract": enable_extract,
                "run_dir": str(paths.run_dir),
                "summary_json": str(paths.summary_json),
                "results_jsonl": str(paths.results_jsonl),
                "eval_pointer": str(pointer),
                "policy_out": str(rt.policy_out_path),
            },
            indent=2,
        )
    )
    return 0


def _artifacts_build_demo_models_yaml(rt: Any, args: argparse.Namespace) -> int:
    """
    Generate a demo models.yaml containing ONLY one profile + one model entry.

    Writes to repo-level artifact_out/ by default.
    """
    src = resolve_under_repo(rt.repo_root, getattr(args, "src", None)) or resolve_under_repo(rt.repo_root, "config/models.yaml")

    # NEW default: repo-level artifact_out/
    out_path = resolve_under_repo(rt.repo_root, getattr(args, "out", None)) or resolve_under_repo(
        rt.repo_root, "artifact_out/demo_models.yaml"
    )

    profile = str(getattr(args, "profile") or "").strip()
    model_id = str(getattr(args, "model_id") or "").strip()
    extract_enabled = bool(getattr(args, "extract_enabled") or False)

    # NEW: assessed flag
    assessed = bool(getattr(args, "assessed") or False)

    if not profile:
        raise SimError("--profile is required", code=2)
    if not model_id:
        raise SimError("--model-id is required", code=2)

    if rt.dry_run:
        print("[dry-run] would build demo models.yaml")
        print(
            json.dumps(
                {
                    "src": str(src),
                    "out": str(out_path),
                    "profile": profile,
                    "model_id": model_id,
                    "assessed": assessed,
                    "extract_enabled": extract_enabled,
                },
                indent=2,
            )
        )
        return 0

    try:
        res = build_demo_models_yaml(
            src_models_yaml=Path(src),
            out_models_yaml=Path(out_path),
            profile=profile,
            model_id=model_id,
            assessed=assessed,
            extract_enabled=extract_enabled,
        )
    except DemoModelsYamlError as e:
        raise SimError(f"build-demo-models-yaml failed: {e}", code=2)

    print(
        json.dumps(
            {
                "ok": True,
                "src": str(res.source_path),
                "out": str(res.out_path),
                "profile": res.profile,
                "model_id": res.model_id,
                "assessed": assessed,
                "extract_enabled": extract_enabled,
            },
            indent=2,
        )
    )
    return 0


def _read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise SimError("PyYAML is not available; cannot read models YAML.", code=2)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raise SimError(f"File not found: {path}", code=2)
    except Exception as e:
        raise SimError(f"Failed to read YAML from {path}: {type(e).__name__}: {e}", code=2)
    if not isinstance(data, dict):
        raise SimError(f"YAML payload must be a mapping: {path}", code=2)
    return data


def _extract_model_caps_from_models_yaml(path: Path, *, model_id: str, profile: str) -> dict[str, Any]:
    doc = _read_yaml(path)

    if isinstance(doc.get("profiles"), dict):
        prof = doc["profiles"].get(profile)
        if not isinstance(prof, dict):
            raise SimError(f"profile not found in models yaml: {profile}", code=2)
        models_any = prof.get("models")
    else:
        models_any = doc.get("models")

    if not isinstance(models_any, list):
        raise SimError(f"models list missing for profile={profile} in {path}", code=2)

    for entry in models_any:
        if not isinstance(entry, dict):
            continue
        mid = entry.get("id")
        if isinstance(mid, str) and mid.strip() == model_id:
            caps = entry.get("capabilities")
            assessment = entry.get("assessment")
            return {
                "capabilities": caps if isinstance(caps, dict) else {},
                "assessment": assessment if isinstance(assessment, dict) else {},
            }

    raise SimError(f"model_id not found in profile={profile}: {model_id}", code=2)


def _artifacts_verify_models_cap(rt: Any, args: argparse.Namespace) -> int:
    path = resolve_under_repo(rt.repo_root, getattr(args, "path", None)) or resolve_under_repo(rt.repo_root, "config/models.yaml")
    model_id = str(getattr(args, "model_id") or "").strip()
    profile = str(getattr(args, "models_profile") or "").strip()

    if not model_id:
        raise SimError("--model-id is required", code=2)
    if not profile:
        raise SimError("--models-profile is required", code=2)

    payload = _extract_model_caps_from_models_yaml(Path(path), model_id=model_id, profile=profile)
    caps = payload.get("capabilities", {}) or {}
    assessment = payload.get("assessment", {}) or {}

    ext = caps.get("extract")
    if isinstance(ext, dict):
        ext_enabled = ext.get("enabled") if isinstance(ext.get("enabled"), bool) else None
    elif isinstance(ext, bool):
        ext_enabled = ext
    else:
        ext_enabled = None

    assessed = assessment.get("assessed") if isinstance(assessment.get("assessed"), bool) else None

    expected_extract = getattr(args, "expect_extract", None)
    expected_assessed = getattr(args, "expect_assessed", None)

    if expected_extract is not None and ext_enabled is not bool(expected_extract):
        raise SimError(
            f"extract capability mismatch for {model_id} in {profile}: expected={bool(expected_extract)} actual={ext_enabled}",
            code=2,
        )
    if expected_assessed is not None and assessed is not bool(expected_assessed):
        raise SimError(
            f"assessment.assessed mismatch for {model_id} in {profile}: expected={bool(expected_assessed)} actual={assessed}",
            code=2,
        )

    print(
        json.dumps(
            {
                "ok": True,
                "path": str(path),
                "model_id": model_id,
                "models_profile": profile,
                "extract_enabled": ext_enabled,
                "assessed": assessed,
                "assessment": assessment,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _artifacts_onboarding_apply(rt: Any, args: argparse.Namespace) -> int:
    """
    Run policy model-onboarding apply and write deterministic models artifact.
    """
    in_models = resolve_under_repo(rt.repo_root, getattr(args, "models_yaml", None)) or resolve_under_repo(rt.repo_root, "config/models.yaml")
    out_models = resolve_under_repo(rt.repo_root, getattr(args, "out_models_yaml", None))
    model_id = str(getattr(args, "model_id") or "").strip()
    models_profile = str(getattr(args, "models_profile") or "").strip()
    eval_run_dir = str(getattr(args, "eval_run_dir") or "").strip()
    threshold_profile = str(getattr(args, "threshold_profile") or "").strip()
    thresholds_root = str(getattr(args, "thresholds_root") or "").strip()

    if not model_id:
        raise SimError("--model-id is required", code=2)
    if not models_profile:
        raise SimError("--models-profile is required", code=2)
    if not eval_run_dir:
        raise SimError("--eval-run-dir is required", code=2)
    if not threshold_profile:
        raise SimError("--threshold-profile is required", code=2)
    if out_models is None:
        raise SimError("--out-models-yaml is required", code=2)

    if rt.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "models_yaml": str(in_models),
                    "out_models_yaml": str(out_models),
                    "model_id": model_id,
                    "models_profile": models_profile,
                    "eval_run_dir": eval_run_dir,
                    "threshold_profile": threshold_profile,
                    "thresholds_root": thresholds_root or None,
                },
                indent=2,
            )
        )
        return 0

    src = Path(in_models)
    dst = Path(out_models)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)

    apply_res = apply_model_onboarding(
        models_yaml=dst,
        model_id=model_id,
        eval_run_dir=eval_run_dir,
        threshold_profile=threshold_profile,
        thresholds_root=(thresholds_root or None),
        patch_profile=models_profile,
        verbose=not bool(getattr(args, "quiet", False)),
    )

    allow_deny = bool(getattr(args, "allow_deny", False))
    acceptable = bool(apply_res.ok) or (allow_deny and bool(apply_res.patch_ok))
    if not acceptable:
        raise SimError(
            f"onboarding apply failed: decision_ok={apply_res.decision_ok} patch_ok={apply_res.patch_ok} message={apply_res.message}",
            code=2,
        )

    print(
        json.dumps(
            {
                "ok": True,
                "models_yaml": str(src),
                "out_models_yaml": str(dst),
                "model_id": model_id,
                "models_profile": models_profile,
                "eval_run_dir": eval_run_dir,
                "threshold_profile": threshold_profile,
                "enable_extract": apply_res.enable_extract,
                "changed": apply_res.changed,
                "deployment_key": apply_res.deployment_key,
                "policy": apply_res.policy_name,
                "pipeline": apply_res.pipeline,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _artifacts_onboarding_demo(rt: Any, args: argparse.Namespace) -> int:
    """
    Convenience wrapper for deterministic PASS/FAIL onboarding artifacts.
    """
    fixture = str(getattr(args, "fixture") or "").strip().lower()
    if fixture not in {"pass", "fail"}:
        raise SimError("--fixture must be one of: pass, fail", code=2)

    model_id = str(getattr(args, "model_id") or "").strip()
    models_profile = str(getattr(args, "models_profile") or "").strip()
    eval_run_dir = str(getattr(args, "eval_run_dir") or "").strip()
    if not model_id:
        raise SimError("--model-id is required", code=2)
    if not models_profile:
        raise SimError("--models-profile is required", code=2)
    if not eval_run_dir:
        raise SimError("--eval-run-dir is required", code=2)

    in_models = resolve_under_repo(rt.repo_root, getattr(args, "models_yaml", None)) or resolve_under_repo(rt.repo_root, "config/models.yaml")
    out_default = f"config/models.patched.{fixture}.yaml"
    out_models = resolve_under_repo(rt.repo_root, getattr(args, "out_models_yaml", None)) or resolve_under_repo(rt.repo_root, out_default)

    if fixture == "pass":
        threshold_profile = str(getattr(args, "threshold_profile_pass") or "extract/default").strip()
        expect_extract = True
    else:
        threshold_profile = str(getattr(args, "threshold_profile_fail") or "extract/strict").strip()
        expect_extract = False

    fake_args = argparse.Namespace(
        models_yaml=str(in_models),
        out_models_yaml=str(out_models),
        model_id=model_id,
        models_profile=models_profile,
        eval_run_dir=eval_run_dir,
        threshold_profile=threshold_profile,
        thresholds_root=getattr(args, "thresholds_root", None),
        allow_deny=(fixture == "fail"),
        quiet=bool(getattr(args, "quiet", False)),
    )
    rc = _artifacts_onboarding_apply(rt, fake_args)
    if rc != 0:
        return rc

    verify_args = argparse.Namespace(
        path=str(out_models),
        model_id=model_id,
        models_profile=models_profile,
        expect_extract=expect_extract,
        expect_assessed=True,
    )
    return _artifacts_verify_models_cap(rt, verify_args)


# ---------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------


def register_artifacts_subcommands(root_subparsers: argparse._SubParsersAction) -> None:
    """
    Adds:
      sim artifacts write-slo / verify-slo / write-policy / verify-policy / demo-eval / build-demo-models-yaml
      sim artifacts onboarding-apply / onboarding-demo / verify-models-cap
    """
    art = root_subparsers.add_parser("artifacts", help="Write / verify deterministic artifact fixtures.")
    art_sub = art.add_subparsers(dest="art_cmd", required=True)

    # --- SLO
    slo = art_sub.add_parser("write-slo", help="Write slo_out/generate/latest.json fixture (runtime_generate_slo_v1).")
    slo.add_argument("--fixture", default="good", choices=["good", "bad"])
    slo.add_argument("--model-id", default="demo-model")
    slo.add_argument("--window-seconds", type=int, default=300)
    slo.add_argument("--out", default=None, help="Optional explicit output path (relative to repo root ok).")
    slo.set_defaults(_handler=_artifacts_write_slo)

    v_slo = art_sub.add_parser("verify-slo", help="Verify a runtime_generate_slo_v1 artifact (default: latest).")
    v_slo.add_argument("--path", default=None, help="Path to SLO artifact (relative ok).")
    v_slo.set_defaults(_handler=_artifacts_verify_slo)

    # --- Policy (generate_clamp_only)
    pol = art_sub.add_parser("write-policy", help="Write policy_out/latest.json fixture (policy_decision_v2).")
    pol.add_argument(
        "--fixture",
        default="allow_no_clamp",
        choices=["allow_no_clamp", "allow_clamp", "deny", "unknown"],
    )
    pol.add_argument("--cap", type=int, default=None, help="Required for allow_clamp (positive integer).")
    pol.add_argument("--generate-thresholds-profile", default="default")
    pol.add_argument("--model-id", default=None)
    pol.add_argument("--out", default=None, help="Optional explicit output path (relative to repo root ok).")
    pol.set_defaults(_handler=_artifacts_write_policy)

    v_pol = art_sub.add_parser("verify-policy", help="Verify a policy_decision_v2 artifact (default: latest).")
    v_pol.add_argument("--path", default=None, help="Path to policy artifact (relative ok).")
    v_pol.set_defaults(_handler=_artifacts_verify_policy)

    # --- Demo Eval (extract gating fixtures)
    de = art_sub.add_parser(
        "demo-eval",
        help="Write eval artifacts: eval run + eval pointer + extract_only policy.",
    )
    de.add_argument("--fixture", default="pass", choices=["pass", "fail"])
    de.add_argument("--run-id", default=None, required=True, help="Run id under results/extract/<run_id>/")
    de.add_argument("--model-id", default=None)
    de.add_argument("--schema-id", default="sroie_receipt_v1")
    de.add_argument("--thresholds-profile", default="extract/default")
    de.set_defaults(_handler=_artifacts_demo_eval)

    # --- Demo models.yaml generator (single model/profile)
    dm = art_sub.add_parser(
        "build-demo-models-yaml",
        help="Generate a demo models.yaml containing ONLY one profile+model (repo-level artifact_out/ by default).",
    )
    dm.add_argument(
        "--profile",
        required=True,
        help="One of: host-transformers, docker-transformers, host-llama, docker-llama, test",
    )
    dm.add_argument("--model-id", required=True, help="Exact model id present in that profile")
    dm.add_argument(
        "--src",
        default="config/models.yaml",
        help="Source models.yaml (relative ok). Default: config/models.yaml",
    )
    dm.add_argument(
        "--out",
        default="artifact_out/demo_models.yaml",
        help="Output path (relative ok). Default: artifact_out/demo_models.yaml",
    )
    dm.add_argument(
        "--extract-enabled",
        action="store_true",
        help="Set capabilities.extract=true (bool clamp).",
    )
    dm.add_argument(
        "--assessed",
        action="store_true",
        help="Set assessment.assessed=true (default is false).",
    )
    dm.set_defaults(_handler=_artifacts_build_demo_models_yaml)

    # --- Onboarding apply (direct)
    oa = art_sub.add_parser(
        "onboarding-apply",
        help="Run model-onboarding apply and write a patched models artifact.",
    )
    oa.add_argument("--models-yaml", default="config/models.yaml")
    oa.add_argument("--out-models-yaml", required=True)
    oa.add_argument("--model-id", required=True)
    oa.add_argument("--models-profile", required=True)
    oa.add_argument("--eval-run-dir", required=True)
    oa.add_argument("--threshold-profile", required=True, help="e.g. extract/default or extract/strict")
    oa.add_argument("--thresholds-root", default=None)
    oa.add_argument(
        "--allow-deny",
        action="store_true",
        help="Treat deny decisions as acceptable when patching succeeded (useful for fail fixtures).",
    )
    oa.add_argument("--quiet", action="store_true")
    oa.set_defaults(_handler=_artifacts_onboarding_apply)

    # --- Onboarding demo convenience
    od = art_sub.add_parser(
        "onboarding-demo",
        help="Generate deterministic PASS/FAIL models artifacts for extract-gate demos.",
    )
    od.add_argument("--fixture", required=True, choices=["pass", "fail"])
    od.add_argument("--models-yaml", default="config/models.yaml")
    od.add_argument("--out-models-yaml", default=None)
    od.add_argument("--model-id", required=True)
    od.add_argument("--models-profile", required=True)
    od.add_argument("--eval-run-dir", required=True)
    od.add_argument("--threshold-profile-pass", default="extract/default")
    od.add_argument("--threshold-profile-fail", default="extract/strict")
    od.add_argument("--thresholds-root", default=None)
    od.add_argument("--quiet", action="store_true")
    od.set_defaults(_handler=_artifacts_onboarding_demo)

    # --- Verify models capability in artifact
    vm = art_sub.add_parser(
        "verify-models-cap",
        help="Verify extract capability + assessed state for a model in a models YAML artifact.",
    )
    vm.add_argument("--path", default="config/models.yaml")
    vm.add_argument("--model-id", required=True)
    vm.add_argument("--models-profile", required=True)
    vm.add_argument("--expect-extract", dest="expect_extract", action="store_true")
    vm.add_argument("--expect-no-extract", dest="expect_extract", action="store_false")
    vm.set_defaults(expect_extract=None)
    vm.add_argument("--expect-assessed", dest="expect_assessed", action="store_true")
    vm.add_argument("--expect-not-assessed", dest="expect_assessed", action="store_false")
    vm.set_defaults(expect_assessed=None)
    vm.set_defaults(_handler=_artifacts_verify_models_cap)
