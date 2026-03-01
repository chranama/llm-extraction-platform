from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import simulations.traffic.runner as runner_mod
from simulations.traffic.models import Event, RequestSpec, Scenario, TrafficConfig


def _cfg(run_id: str = "run_1") -> TrafficConfig:
    return TrafficConfig(
        run_id=run_id,
        scenario="unit_scenario",
        base_url="http://127.0.0.1:8000",
        api_key="k",
        timeout_s=1.0,
        duration_s=1.0,
        rps=2.0,
        max_in_flight=2,
        seed=1,
    )


def test_run_scenario_returns_error_when_no_requests(tmp_path: Path) -> None:
    sc = Scenario(name="empty", endpoint="generate", build_requests=lambda _cfg: [])
    out = runner_mod.run_scenario(repo_root_out_dir=str(tmp_path), cfg=_cfg(), scenario=sc, dry_run=False)
    assert out["ok"] is False
    assert "scenario produced no requests" in out["error"]


def test_run_scenario_dry_run_returns_request_count(tmp_path: Path) -> None:
    reqs = [RequestSpec(idx=0, endpoint="generate", payload={"prompt": "x"})]
    sc = Scenario(name="s1", endpoint="generate", build_requests=lambda _cfg: reqs)

    out = runner_mod.run_scenario(repo_root_out_dir=str(tmp_path), cfg=_cfg(), scenario=sc, dry_run=True)
    assert out["ok"] is True
    assert out["dry_run"] is True
    assert out["n_requests"] == 1


def test_run_scenario_writes_events_summary_and_runs_hook(tmp_path: Path, monkeypatch) -> None:
    # Avoid real sleep/network.
    monkeypatch.setattr(runner_mod, "_sleep_until", lambda _t: None)
    monkeypatch.setattr(runner_mod, "SimClient", lambda _cfg: object())

    def _fake_do_one(_client, cfg, req):
        return Event(
            run_id=cfg.run_id,
            scenario=cfg.scenario,
            idx=req.idx,
            endpoint=req.endpoint,
            started_at_unix=1.0,
            elapsed_ms=10.0 + req.idx,
            ok=True,
            status=200,
            cached=False,
            prompt_tokens=1,
            completion_tokens=2,
            error=None,
            error_payload=None,
            model=cfg.model,
            tags=dict(req.tags or {}),
        )

    monkeypatch.setattr(runner_mod, "_do_one", _fake_do_one)

    hook_called = {"n": 0}

    @dataclass
    class _Hook:
        name: str = "h1"
        at_s: float = 0.0

        def run(self, _client, _cfg, repo_root=None):
            hook_called["n"] += 1
            assert repo_root is not None

    reqs = [
        RequestSpec(idx=0, endpoint="generate", payload={"prompt": "a"}),
        RequestSpec(idx=1, endpoint="generate", payload={"prompt": "b"}),
    ]
    sc = Scenario(name="with_hook", endpoint="generate", build_requests=lambda _cfg: reqs, hooks=[_Hook()])
    cfg = _cfg(run_id="run_hook")
    out = runner_mod.run_scenario(repo_root_out_dir=str(tmp_path), cfg=cfg, scenario=sc, dry_run=False)

    assert out["ok"] is True
    assert hook_called["n"] == 1

    events_path = Path(out["events_jsonl"])
    summary_path = Path(out["summary_json"])
    assert events_path.exists()
    assert summary_path.exists()

    events = [json.loads(x) for x in events_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    # 2 request events + 1 hook event (negative idx).
    assert len(events) == 3
    assert events[0]["idx"] < 0
    assert events[1]["idx"] == 0
    assert events[2]["idx"] == 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    # Hook event is excluded from summary math.
    assert summary["sent"] == 2
