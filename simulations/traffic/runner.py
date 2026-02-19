# simulations/traffic/runner.py
from __future__ import annotations

import concurrent.futures as futures
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from simulations.traffic.client import ClientConfig, SimClient, SimClientError
from simulations.traffic.models import Event, RequestSpec, Scenario, TrafficConfig
from simulations.traffic.scheduler import OpenLoopScheduler
from simulations.traffic.stats import compute_summary
from simulations.traffic.writers import RunPaths, write_events_jsonl, write_summary_json


def _now() -> float:
    return time.time()


def _sleep_until(t_unix: float) -> None:
    while True:
        dt = t_unix - _now()
        if dt <= 0:
            return
        time.sleep(min(0.01, dt))


def _extract_common_fields(resp_json: Any) -> Tuple[Optional[bool], Optional[int], Optional[int], Dict[str, Any]]:
    """
    Returns:
      (cached, prompt_tokens, completion_tokens, extra_tags)

    extra_tags includes clamp evidence when present:
      requested_max_new_tokens, effective_max_new_tokens,
      policy_generate_max_new_tokens_cap, clamped
    """
    cached = None
    pt = None
    ct = None
    extra: Dict[str, Any] = {}

    if isinstance(resp_json, dict):
        if "cached" in resp_json and isinstance(resp_json["cached"], bool):
            cached = resp_json["cached"]

        usage = resp_json.get("usage")
        if isinstance(usage, dict):
            v = usage.get("prompt_tokens")
            if isinstance(v, int):
                pt = v
            v = usage.get("completion_tokens")
            if isinstance(v, int):
                ct = v

        meta = resp_json.get("meta")
        if isinstance(meta, dict):
            if cached is None and isinstance(meta.get("cached"), bool):
                cached = bool(meta.get("cached"))

        # clamp evidence from /v1/generate + /v1/generate/batch
        for k in (
            "requested_max_new_tokens",
            "effective_max_new_tokens",
            "policy_generate_max_new_tokens_cap",
            "clamped",
        ):
            if k in resp_json:
                extra[k] = resp_json.get(k)

    return cached, pt, ct, extra


def _do_one(client: SimClient, cfg: TrafficConfig, req: RequestSpec) -> Event:
    t0 = _now()
    started = t0
    try:
        if req.endpoint == "generate":
            r = client.post_generate(
                prompt=str(req.payload.get("prompt", "")),
                model=req.payload.get("model"),
                cache=bool(req.payload.get("cache", True)),
                max_new_tokens=req.payload.get("max_new_tokens"),
                temperature=req.payload.get("temperature"),
                top_p=req.payload.get("top_p"),
                top_k=req.payload.get("top_k"),
                stop=req.payload.get("stop"),
            )
        else:
            r = client.post_extract(
                schema_id=str(req.payload.get("schema_id", cfg.schema_id)),
                text=str(req.payload.get("text", "")),
                model=req.payload.get("model"),
                max_new_tokens=req.payload.get("max_new_tokens"),
                temperature=req.payload.get("temperature"),
                cache=bool(req.payload.get("cache", True)),
                repair=bool(req.payload.get("repair", True)),
            )

        cached, pt, ct, extra = _extract_common_fields(r.json)
        tags = dict(req.tags or {})
        if extra:
            tags.update(extra)

        return Event(
            run_id=cfg.run_id,
            scenario=cfg.scenario,
            idx=req.idx,
            endpoint=req.endpoint,
            started_at_unix=started,
            elapsed_ms=float(r.elapsed_ms),
            ok=(200 <= r.status < 300),
            status=int(r.status),
            cached=cached,
            prompt_tokens=pt,
            completion_tokens=ct,
            error=None,
            error_payload=None,
            model=cfg.model,
            tags=tags,
        )

    except SimClientError as e:
        elapsed_ms = (_now() - t0) * 1000
        return Event(
            run_id=cfg.run_id,
            scenario=cfg.scenario,
            idx=req.idx,
            endpoint=req.endpoint,
            started_at_unix=started,
            elapsed_ms=float(elapsed_ms),
            ok=False,
            status=e.status,
            cached=None,
            prompt_tokens=None,
            completion_tokens=None,
            error=str(e),
            error_payload=e.payload,
            model=cfg.model,
            tags=dict(req.tags or {}),
        )


def run_scenario(
    *,
    repo_root_out_dir: str,
    cfg: TrafficConfig,
    scenario: Scenario,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Orchestrates:
      setup -> schedule -> (hooks + send) -> collect -> write outputs

    Output layout:
      <repo_root_out_dir>/<run_id>/{events.jsonl, summary.json}

    Hook contract (back-compat):
      - Prefer: hook.run(client, cfg, repo_root=<Path>)
      - Fallback: hook.run(client, cfg)
    """
    client = SimClient(
        ClientConfig(
            base_url=cfg.base_url.rstrip("/"),
            api_key=cfg.api_key,
            timeout_s=float(cfg.timeout_s),
            model=cfg.model,
            extra_headers={"X-Sim-Run": cfg.run_id, "X-Sim-Scenario": scenario.name},
        )
    )

    if scenario.setup is not None and not dry_run:
        scenario.setup(client, cfg)

    reqs = list(scenario.build_requests(cfg))
    if not reqs:
        return {"ok": False, "error": "scenario produced no requests"}

    if dry_run:
        return {"ok": True, "dry_run": True, "run_id": cfg.run_id, "scenario": scenario.name, "n_requests": len(reqs)}

    out_root = Path(repo_root_out_dir).resolve()
    # traffic_out is under repo root => parent is repo root
    repo_root = out_root.parent.resolve()

    rp = RunPaths.under(out_root, run_id=cfg.run_id)

    sched = OpenLoopScheduler(duration_s=float(cfg.duration_s), rps=float(cfg.rps))
    schedule_points = list(sched.iter())

    n = min(len(schedule_points), len(reqs))
    schedule_points = schedule_points[:n]
    reqs = reqs[:n]

    # ---- hook prep ----
    hooks = list(getattr(scenario, "hooks", None) or [])
    hooks_sorted = sorted(hooks, key=lambda h: float(getattr(h, "at_s", 0.0)))
    next_hook_idx = 0
    t_start = _now()

    def _run_hook(h: Any) -> Tuple[bool, Optional[str]]:
        """
        Run hook with back-compat:
          - try calling with repo_root kwarg
          - fall back to old 2-arg signature
        """
        fn = getattr(h, "run", None)
        if not callable(fn):
            return False, "hook.run is not callable"
        try:
            fn(client, cfg, repo_root=repo_root)  # preferred
            return True, None
        except TypeError:
            # older signature
            fn(client, cfg)
            return True, None
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def _maybe_run_hooks(now_unix: float, events: List[Event]) -> None:
        nonlocal next_hook_idx
        if not hooks_sorted:
            return
        elapsed = now_unix - t_start
        while next_hook_idx < len(hooks_sorted):
            h = hooks_sorted[next_hook_idx]
            at_s = float(getattr(h, "at_s", 0.0))
            if elapsed < at_s:
                break

            h_t0 = _now()
            ok, err = _run_hook(h)
            h_elapsed_ms = (_now() - h_t0) * 1000

            # record hook event as a normal Event (negative idx reserved for hooks)
            events.append(
                Event(
                    run_id=cfg.run_id,
                    scenario=cfg.scenario,
                    idx=-(next_hook_idx + 1),
                    endpoint=scenario.endpoint,
                    started_at_unix=h_t0,
                    elapsed_ms=float(h_elapsed_ms),
                    ok=bool(ok),
                    status=200 if ok else 500,
                    cached=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    error=err,
                    error_payload=None,
                    model=cfg.model,
                    tags={
                        "kind": "hook",
                        "hook": getattr(h, "name", "hook"),
                        "at_s": at_s,
                        "repo_root": str(repo_root),
                    },
                )
            )

            next_hook_idx += 1

    events: List[Event] = []
    in_flight: Dict[futures.Future[Event], int] = {}

    with futures.ThreadPoolExecutor(max_workers=max(1, int(cfg.max_in_flight))) as ex:
        for sp, req in zip(schedule_points, reqs):
            _sleep_until(sp.at_unix)

            # run hooks before submitting this request
            _maybe_run_hooks(_now(), events)

            while len(in_flight) >= int(cfg.max_in_flight):
                done, _ = futures.wait(in_flight.keys(), timeout=0.01, return_when=futures.FIRST_COMPLETED)
                for fut in done:
                    events.append(fut.result())
                    in_flight.pop(fut, None)

            fut = ex.submit(_do_one, client, cfg, req)
            in_flight[fut] = req.idx

        for fut in futures.as_completed(list(in_flight.keys())):
            events.append(fut.result())

    # run any remaining hooks after traffic completes
    _maybe_run_hooks(_now(), events)

    # deterministic ordering: hooks first (negative idx), then requests
    events.sort(key=lambda e: e.idx)

    summary = compute_summary(
        run_id=cfg.run_id,
        scenario=scenario.name,
        duration_s=float(cfg.duration_s),
        events=[e for e in events if e.idx >= 0],  # exclude hook events from summary math
    )
    write_events_jsonl(rp.events_jsonl, events)
    write_summary_json(rp.summary_json, summary)

    return {
        "ok": True,
        "run_dir": str(rp.run_dir),
        "events_jsonl": str(rp.events_jsonl),
        "summary_json": str(rp.summary_json),
        "summary": summary.model_dump(),
    }