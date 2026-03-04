# simulations/traffic/main.py
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from simulations.paths import ensure_parent_dir, find_repo_root
from simulations.traffic.client import ClientConfig, SimClient, SimClientError
from simulations.traffic.models import TrafficConfig
from simulations.traffic.runner import run_scenario
from simulations.traffic.scenarios import demo_baseline, demo_clamp

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


class SimError(Exception):
    def __init__(self, message: str, *, code: int = 2):
        super().__init__(message)
        self.code = int(code)


def _utc_now_slug() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")


def _require_api_key(rt: Any) -> None:
    if not getattr(rt, "api_key", None):
        raise SimError("Missing API key. Set API_KEY env var or pass --api-key.", code=2)


def _repo_root_from_rt(rt: Any) -> Path:
    rr = getattr(rt, "repo_root", None)
    if isinstance(rr, Path):
        return rr
    return find_repo_root(Path.cwd())


def _traffic_root_out_dir(rt: Any) -> Path:
    rr = _repo_root_from_rt(rt)
    root = (rr / "traffic_out").resolve()
    ensure_parent_dir(root / "placeholder.json")
    return root


def _client_from_rt(rt: Any, *, model: Optional[str] = None) -> SimClient:
    _require_api_key(rt)
    base_url = str(getattr(rt, "base_url", "") or "").strip() or "http://127.0.0.1:8000"
    timeout_s = float(getattr(rt, "timeout_s", None) or 20.0)
    cfg = ClientConfig(
        base_url=base_url.rstrip("/"),
        api_key=str(rt.api_key),
        timeout_s=float(timeout_s),
        model=(model or getattr(rt, "model", None)),
    )
    return SimClient(cfg)


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


def _artifact_model_expectations(path: Path, *, model_id: str, profile: str) -> dict[str, Any]:
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

            ext = (caps or {}).get("extract") if isinstance(caps, dict) else None
            if isinstance(ext, dict):
                ext_enabled = ext.get("enabled") if isinstance(ext.get("enabled"), bool) else None
            elif isinstance(ext, bool):
                ext_enabled = ext
            else:
                ext_enabled = None

            deploy_key = None
            if isinstance(assessment, dict):
                dk = assessment.get("deployment_key")
                if isinstance(dk, str) and dk.strip():
                    deploy_key = dk.strip()

            return {
                "extract_enabled": ext_enabled,
                "deployment_key": deploy_key,
            }

    raise SimError(f"model_id not found in profile={profile}: {model_id}", code=2)


# -----------------------------------------------------------------------------
# Admin helpers
# -----------------------------------------------------------------------------

def _admin_policy_reload(rt: Any, args: argparse.Namespace) -> int:
    client = _client_from_rt(rt)
    if getattr(rt, "dry_run", False):
        base_url = client.cfg.base_url.rstrip("/")
        print(
            f'[dry-run] would POST {base_url}/v1/admin/policy/reload '
            f'with header X-API-Key: {client.cfg.api_key[:6]}…'
        )
        return 0

    try:
        resp = client.post_admin_policy_reload()
    except SimClientError as e:
        raise SimError(f"policy reload failed: {e} :: {e.response_text or ''}".strip(), code=2)

    print(
        json.dumps(
            {"ok": 200 <= resp.status < 300, "status": resp.status, "elapsed_ms": resp.elapsed_ms, "json": resp.json},
            indent=2,
        )
    )
    return 0


def _admin_reload(rt: Any, args: argparse.Namespace) -> int:
    client = _client_from_rt(rt)
    if getattr(rt, "dry_run", False):
        base_url = client.cfg.base_url.rstrip("/")
        print(
            f'[dry-run] would POST {base_url}/v1/admin/reload '
            f'with header X-API-Key: {client.cfg.api_key[:6]}…'
        )
        return 0

    try:
        resp = client.post_admin_reload()
    except SimClientError as e:
        raise SimError(f"admin reload failed: {e} :: {e.response_text or ''}".strip(), code=2)

    print(
        json.dumps(
            {"ok": 200 <= resp.status < 300, "status": resp.status, "elapsed_ms": resp.elapsed_ms, "json": resp.json},
            indent=2,
        )
    )
    return 0


def _extract_gate_check(rt: Any, args: argparse.Namespace) -> int:
    _require_api_key(rt)
    client = _client_from_rt(rt, model=(args.model_id or None))

    if getattr(rt, "dry_run", False):
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "base_url": client.cfg.base_url,
                    "model_id": args.model_id,
                    "schema_id": args.schema_id,
                    "expect": args.expect,
                    "expect_code": args.expect_code,
                    "expect_model_extract": args.expect_model_extract,
                },
                indent=2,
            )
        )
        return 0

    # 1) Read /v1/models for deployment + model capability snapshot.
    try:
        models_resp = client.get_models()
    except SimClientError as e:
        raise SimError(f"GET /v1/models failed: {e} :: {e.response_text or ''}".strip(), code=2)

    models_payload = models_resp.json if isinstance(models_resp.json, dict) else {}
    default_model = models_payload.get("default_model")
    model_id = str((args.model_id or default_model or "")).strip()
    if not model_id:
        raise SimError("Could not resolve model id; pass --model-id explicitly.", code=2)

    model_extract_cap = None
    models = models_payload.get("models")
    if isinstance(models, list):
        for row in models:
            if not isinstance(row, dict):
                continue
            if str(row.get("id", "")) != model_id:
                continue
            caps = row.get("capabilities")
            if isinstance(caps, dict) and "extract" in caps:
                model_extract_cap = bool(caps.get("extract"))
            break

    # 2) Probe /v1/extract for behavior.
    extract_status = None
    extract_json = None
    extract_text = None
    try:
        r = client.post_extract(
            schema_id=str(args.schema_id),
            text=str(args.text),
            model=model_id,
            max_new_tokens=int(args.max_new_tokens) if args.max_new_tokens is not None else None,
            temperature=(float(args.temperature) if args.temperature is not None else None),
            cache=bool(args.cache),
            repair=bool(args.repair),
        )
        extract_status = int(r.status)
        extract_json = r.json if isinstance(r.json, dict) else None
        extract_text = r.text
    except SimClientError as e:
        extract_status = int(e.status) if e.status is not None else None
        extract_json = e.payload if isinstance(e.payload, dict) else None
        extract_text = e.response_text

    body_code = extract_json.get("code") if isinstance(extract_json, dict) else None
    passed = True
    failures: list[str] = []

    expected_mode = str(args.expect).strip().lower()
    if expected_mode == "allow":
        if extract_status != 200:
            allow_model_errors = bool(getattr(args, "allow_model_errors", False))
            gated_codes = {"capability_disabled", "extract_disabled", "policy_denied"}
            if allow_model_errors and str(body_code or "") not in gated_codes and int(extract_status or 0) != 501:
                pass
            else:
                passed = False
                failures.append(f"expected allow (HTTP 200), got status={extract_status}")
    elif expected_mode == "block":
        if extract_status == 200:
            passed = False
            failures.append("expected block (non-200), got status=200")

    if args.expect_code:
        if str(body_code) != str(args.expect_code):
            passed = False
            failures.append(f"expected error code={args.expect_code!r}, got {body_code!r}")

    exp_cap = str(args.expect_model_extract).strip().lower()
    if exp_cap in {"true", "false"}:
        want = exp_cap == "true"
        if model_extract_cap is None:
            passed = False
            failures.append(
                f"expected model capabilities.extract={want}, but /v1/models did not provide an extract capability for {model_id!r}"
            )
        elif bool(model_extract_cap) != want:
            passed = False
            failures.append(
                f"expected model capabilities.extract={want}, got {bool(model_extract_cap)} for {model_id!r}"
            )

    out = {
        "ok": passed,
        "base_url": client.cfg.base_url,
        "model_id": model_id,
        "models": {
            "status": int(models_resp.status),
            "default_model": default_model,
            "model_extract_capability": model_extract_cap,
            "deployment_capabilities": models_payload.get("deployment_capabilities"),
        },
        "extract_probe": {
            "status": extract_status,
            "code": body_code,
            "json": extract_json,
            "text": extract_text,
        },
        "expectations": {
            "expect": expected_mode,
            "expect_code": args.expect_code,
            "expect_model_extract": exp_cap,
            "allow_model_errors": bool(getattr(args, "allow_model_errors", False)),
        },
        "failures": failures,
    }
    print(json.dumps(out, indent=2))

    if not passed:
        raise SimError("extract gate check failed; see JSON output for details", code=2)
    return 0


def _runtime_proof(rt: Any, args: argparse.Namespace) -> int:
    _require_api_key(rt)
    client = _client_from_rt(rt, model=(args.model_id or None))

    if getattr(rt, "dry_run", False):
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "base_url": client.cfg.base_url,
                    "model_id": args.model_id,
                    "artifact_models_yaml": args.artifact_models_yaml,
                    "artifact_models_profile": args.artifact_models_profile,
                    "expect_policy_source": args.expect_policy_source,
                    "expect_policy_enable_extract": args.expect_policy_enable_extract,
                },
                indent=2,
            )
        )
        return 0

    modelz_status = None
    modelz = {}
    try:
        modelz_resp = client._request("GET", "/modelz")
        modelz_status = int(modelz_resp.status)
        modelz = modelz_resp.json if isinstance(modelz_resp.json, dict) else {}
    except SimClientError as e:
        # /modelz may return 503 with a useful JSON payload; keep parsing for proof output.
        if e.status is not None and isinstance(e.payload, dict):
            modelz_status = int(e.status)
            modelz = dict(e.payload)
        else:
            raise SimError(f"GET /modelz failed: {e} :: {e.response_text or ''}".strip(), code=2)
    try:
        models_resp = client.get_models()
    except SimClientError as e:
        raise SimError(f"GET /v1/models failed: {e} :: {e.response_text or ''}".strip(), code=2)

    models = models_resp.json if isinstance(models_resp.json, dict) else {}

    default_model = models.get("default_model")
    model_id = str((args.model_id or default_model or "")).strip()
    if not model_id:
        raise SimError("Could not resolve model id; pass --model-id explicitly.", code=2)

    model_extract_cap = None
    rows = models.get("models")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("id", "")) != model_id:
                continue
            caps = row.get("capabilities")
            if isinstance(caps, dict) and "extract" in caps:
                model_extract_cap = bool(caps.get("extract"))
            break

    policy = modelz.get("policy") if isinstance(modelz.get("policy"), dict) else {}
    deployment = modelz.get("deployment") if isinstance(modelz.get("deployment"), dict) else {}
    assessed_gate = modelz.get("assessed_gate") if isinstance(modelz.get("assessed_gate"), dict) else {}
    ag_snap = assessed_gate.get("snapshot") if isinstance(assessed_gate.get("snapshot"), dict) else {}

    policy_source = policy.get("source_path")
    policy_enable_extract = policy.get("enable_extract")
    deployment_key_live = deployment.get("deployment_key")
    assessed_key_live = ag_snap.get("selected_deployment_key")
    models_profile_selected = (
        (deployment.get("profiles") or {}).get("models_profile_selected")
        if isinstance(deployment.get("profiles"), dict)
        else None
    )

    failures: list[str] = []
    passed = True

    exp_policy_source = str(args.expect_policy_source).strip().lower()
    if exp_policy_source == "none":
        if policy_source not in (None, ""):
            passed = False
            failures.append(f"expected no policy source_path, got {policy_source!r}")
    elif exp_policy_source == "set":
        if policy_source in (None, ""):
            passed = False
            failures.append("expected policy source_path to be set, got empty")

    exp_policy_extract = str(args.expect_policy_enable_extract).strip().lower()
    if exp_policy_extract == "none":
        if policy_enable_extract is not None:
            passed = False
            failures.append(f"expected policy enable_extract=None, got {policy_enable_extract!r}")
    elif exp_policy_extract in {"true", "false"}:
        want = exp_policy_extract == "true"
        if bool(policy_enable_extract) != want:
            passed = False
            failures.append(f"expected policy enable_extract={want}, got {policy_enable_extract!r}")

    exp_model_extract = str(args.expect_model_extract).strip().lower()
    if exp_model_extract in {"true", "false"}:
        want = exp_model_extract == "true"
        if model_extract_cap is None:
            passed = False
            failures.append(
                f"expected model capabilities.extract={want}, but /v1/models did not provide extract capability for {model_id!r}"
            )
        elif bool(model_extract_cap) != want:
            passed = False
            failures.append(f"expected model capabilities.extract={want}, got {bool(model_extract_cap)}")

    artifact_expect = None
    artifact_path = str(getattr(args, "artifact_models_yaml", "") or "").strip()
    if artifact_path:
        artifact_profile = str(getattr(args, "artifact_models_profile", "") or "").strip() or str(
            models_profile_selected or ""
        )
        if not artifact_profile:
            raise SimError(
                "artifact profile is required when --artifact-models-yaml is set and runtime did not expose models_profile_selected",
                code=2,
            )
        artifact_expect = _artifact_model_expectations(
            Path(artifact_path), model_id=model_id, profile=artifact_profile
        )

        art_extract = artifact_expect.get("extract_enabled")
        if art_extract is not None and model_extract_cap is not None and bool(model_extract_cap) != bool(art_extract):
            passed = False
            failures.append(
                f"artifact/runtime extract mismatch for {model_id}: artifact={bool(art_extract)} runtime={bool(model_extract_cap)}"
            )

        art_key = artifact_expect.get("deployment_key")
        if isinstance(art_key, str) and art_key.strip():
            if isinstance(assessed_key_live, str) and assessed_key_live.strip() and assessed_key_live.strip() != art_key:
                passed = False
                failures.append(
                    f"artifact/runtime assessed deployment_key mismatch: artifact={art_key!r} runtime={assessed_key_live!r}"
                )
            if isinstance(deployment_key_live, str) and deployment_key_live.strip() and deployment_key_live.strip() != art_key:
                passed = False
                failures.append(
                    f"artifact/runtime deployment metadata key mismatch: artifact={art_key!r} runtime={deployment_key_live!r}"
                )

    out = {
        "ok": passed,
        "base_url": client.cfg.base_url,
        "model_id": model_id,
        "runtime": {
            "modelz_status": modelz_status,
            "models_status": int(models_resp.status),
            "models_profile_selected": models_profile_selected,
            "policy_source_path": policy_source,
            "policy_enable_extract": policy_enable_extract,
            "deployment_key": deployment_key_live,
            "assessed_selected_deployment_key": assessed_key_live,
            "model_extract_capability": model_extract_cap,
        },
        "artifact_expectation": artifact_expect,
        "expectations": {
            "expect_policy_source": exp_policy_source,
            "expect_policy_enable_extract": exp_policy_extract,
            "expect_model_extract": exp_model_extract,
        },
        "failures": failures,
    }
    print(json.dumps(out, indent=2))

    if not passed:
        raise SimError("runtime proof failed; see JSON output for details", code=2)
    return 0


# -----------------------------------------------------------------------------
# Scenario runs
# -----------------------------------------------------------------------------

def _run_demo_baseline(rt: Any, args: argparse.Namespace) -> int:
    _require_api_key(rt)

    run_id = (args.run_id or "").strip() or f"demo_baseline_{_utc_now_slug()}"
    scenario = demo_baseline.build_demo_baseline()

    cfg = TrafficConfig(
        run_id=run_id,
        scenario=scenario.name,
        base_url=str(getattr(rt, "base_url", "") or "http://127.0.0.1:8000"),
        api_key=str(rt.api_key),
        timeout_s=float(getattr(rt, "timeout_s", None) or 20.0),
        duration_s=float(args.seconds),
        rps=float(args.rps),
        max_in_flight=int(args.concurrency),
        seed=int(args.seed),
        model=(args.model_id or None),
        cache=bool(args.cache),
        prompt_size=str(args.prompt_size),
    )

    if getattr(rt, "dry_run", False):
        print(json.dumps({"dry_run": True, "cfg": cfg.model_dump()}, indent=2))
        return 0

    res = run_scenario(
        repo_root_out_dir=str(_traffic_root_out_dir(rt)),
        cfg=cfg,
        scenario=scenario,
        dry_run=False,
    )
    print(json.dumps(res, indent=2))
    return 0


def _run_demo_clamp(rt: Any, args: argparse.Namespace) -> int:
    _require_api_key(rt)

    run_id = (args.run_id or "").strip() or f"demo_clamp_{_utc_now_slug()}"
    scenario = demo_clamp.build_demo_clamp()

    cfg = TrafficConfig(
        run_id=run_id,
        scenario=scenario.name,
        base_url=str(getattr(rt, "base_url", "") or "http://127.0.0.1:8000"),
        api_key=str(rt.api_key),
        timeout_s=float(getattr(rt, "timeout_s", None) or 20.0),
        duration_s=float(args.seconds),
        rps=float(args.rps),
        max_in_flight=int(args.concurrency),
        seed=int(args.seed),
        model=(args.model_id or None),
        cache=bool(args.cache),
        prompt_size=str(args.prompt_size),
    )

    if getattr(rt, "dry_run", False):
        print(json.dumps({"dry_run": True, "cfg": cfg.model_dump()}, indent=2))
        return 0

    res = run_scenario(
        repo_root_out_dir=str(_traffic_root_out_dir(rt)),
        cfg=cfg,
        scenario=scenario,
        dry_run=False,
    )
    print(json.dumps(res, indent=2))
    return 0


def register_traffic_subcommands(root_subparsers: argparse._SubParsersAction) -> None:
    tr = root_subparsers.add_parser("traffic", help="Run traffic scenarios against a running server.")
    tr_sub = tr.add_subparsers(dest="traffic_cmd", required=True)

    adm = tr_sub.add_parser("admin", help="Admin helpers (real HTTP).")
    adm_sub = adm.add_subparsers(dest="admin_cmd", required=True)

    pr = adm_sub.add_parser("policy-reload", help="POST /v1/admin/policy/reload")
    pr.set_defaults(_handler=_admin_policy_reload)

    ar = adm_sub.add_parser("reload", help="POST /v1/admin/reload")
    ar.set_defaults(_handler=_admin_reload)

    db = tr_sub.add_parser("demo-baseline", help="Run baseline generate traffic (regression + stability).")
    db.add_argument("--run-id", default=None)
    db.add_argument("--seconds", type=int, default=15)
    db.add_argument("--rps", type=float, default=1.0)
    db.add_argument("--concurrency", type=int, default=1)
    db.add_argument("--seed", type=int, default=13)
    db.add_argument("--model-id", default=None)
    db.add_argument("--prompt-size", default="short", choices=["short", "medium", "long"])
    db.add_argument("--no-cache", dest="cache", action="store_false")
    db.set_defaults(cache=True)
    db.set_defaults(_handler=_run_demo_baseline)

    dc = tr_sub.add_parser("demo-clamp", help="Run clamp visibility traffic (policy reload + clamp proof).")
    dc.add_argument("--run-id", default=None)
    dc.add_argument("--seconds", type=int, default=15)
    dc.add_argument("--rps", type=float, default=2.0)
    dc.add_argument("--concurrency", type=int, default=4)
    dc.add_argument("--seed", type=int, default=13)
    dc.add_argument("--model-id", default=None)
    dc.add_argument("--prompt-size", default="short", choices=["short", "medium", "long"])
    dc.add_argument("--no-cache", dest="cache", action="store_false")
    dc.set_defaults(cache=True)
    dc.set_defaults(_handler=_run_demo_clamp)

    eg = tr_sub.add_parser(
        "extract-gate-check",
        help="Probe /v1/models + /v1/extract and assert extract allow/block behavior.",
    )
    eg.add_argument("--model-id", default=None, help="Model id under test. Defaults to /v1/models default_model.")
    eg.add_argument("--schema-id", default="sroie_receipt_v1")
    eg.add_argument("--text", default="Vendor: ACME Corp\nTotal: 42.13\nDate: 2026-03-01")
    eg.add_argument("--max-new-tokens", type=int, default=256)
    eg.add_argument("--temperature", type=float, default=0.0)
    eg.add_argument("--no-cache", dest="cache", action="store_false")
    eg.set_defaults(cache=True)
    eg.add_argument("--no-repair", dest="repair", action="store_false")
    eg.set_defaults(repair=True)
    eg.add_argument("--expect", choices=["any", "allow", "block"], default="any")
    eg.add_argument("--expect-code", default=None, help="Optional expected JSON error code for blocked calls.")
    eg.add_argument(
        "--allow-model-errors",
        action="store_true",
        help="For --expect allow, accept non-200 model/parse errors as long as call is not capability-gated.",
    )
    eg.add_argument(
        "--expect-model-extract",
        choices=["any", "true", "false"],
        default="any",
        help="Optional expectation for model capabilities.extract from /v1/models.",
    )
    eg.set_defaults(_handler=_extract_gate_check)

    rp = tr_sub.add_parser(
        "runtime-proof",
        help="Collect /modelz + /v1/models evidence and assert active runtime wiring for extract-gate demos.",
    )
    rp.add_argument("--model-id", default=None, help="Model id under test. Defaults to /v1/models default_model.")
    rp.add_argument(
        "--artifact-models-yaml",
        default=None,
        help="Optional local models artifact path to compare against runtime evidence.",
    )
    rp.add_argument(
        "--artifact-models-profile",
        default=None,
        help="Profile in --artifact-models-yaml. Defaults to runtime models_profile_selected when available.",
    )
    rp.add_argument(
        "--expect-policy-source",
        choices=["any", "none", "set"],
        default="any",
        help="Assert whether /modelz policy.source_path should be empty (none) or populated (set).",
    )
    rp.add_argument(
        "--expect-policy-enable-extract",
        choices=["any", "none", "true", "false"],
        default="any",
        help="Assert /modelz policy.enable_extract value; use 'none' for offline demos.",
    )
    rp.add_argument(
        "--expect-model-extract",
        choices=["any", "true", "false"],
        default="any",
        help="Optional expected model capabilities.extract value from /v1/models.",
    )
    rp.set_defaults(_handler=_runtime_proof)
