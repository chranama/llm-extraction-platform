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
    db.add_argument("--rps", type=float, default=2.0)
    db.add_argument("--concurrency", type=int, default=4)
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