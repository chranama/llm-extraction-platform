from __future__ import annotations

import json
from pathlib import Path

from simulations.traffic.models import Event, Summary
from simulations.traffic.prompts import extract_text, generate_prompt
from simulations.traffic.scheduler import ClosedLoopBackoff, OpenLoopScheduler
from simulations.traffic.stats import compute_summary
from simulations.traffic.writers import RunPaths, write_events_jsonl, write_summary_json


def test_generate_prompt_is_deterministic_for_same_seed() -> None:
    p1 = generate_prompt(3, seed=42)
    p2 = generate_prompt(3, seed=42)
    assert p1 == p2


def test_extract_text_contains_expected_fields() -> None:
    txt = extract_text(1, seed=7)
    assert "TICKET:" in txt
    assert "REQUESTER:" in txt
    assert "PRIORITY:" in txt


def test_open_loop_scheduler_iter_count_and_timestamps() -> None:
    sched = OpenLoopScheduler(duration_s=2.0, rps=2.0)
    pts = list(sched.iter(start_unix=100.0))
    assert len(pts) == 4
    assert pts[0].at_unix == 100.0
    assert pts[1].at_unix == 100.5


def test_closed_loop_backoff_bad_and_good_windows() -> None:
    b = ClosedLoopBackoff(base_rps=10.0)
    assert b.current_rps() == 10.0
    b.observe_window(error_rate=0.10, latency_p95_ms=3000)
    assert b.current_rps() < 10.0
    b.observe_window(error_rate=0.0, latency_p95_ms=100.0)
    assert b.current_rps() <= 10.0


def test_compute_summary_breakdown_and_gate_metrics() -> None:
    events = [
        Event(
            run_id="r1",
            scenario="s1",
            idx=0,
            endpoint="generate",
            started_at_unix=1.0,
            elapsed_ms=100.0,
            ok=True,
            status=200,
            cached=True,
            prompt_tokens=10,
            completion_tokens=20,
            error=None,
            error_payload=None,
            model=None,
            tags={},
        ),
        Event(
            run_id="r1",
            scenario="s1",
            idx=1,
            endpoint="generate",
            started_at_unix=2.0,
            elapsed_ms=200.0,
            ok=False,
            status=429,
            cached=False,
            prompt_tokens=None,
            completion_tokens=None,
            error="quota",
            error_payload=None,
            model=None,
            tags={},
        ),
    ]
    s = compute_summary(run_id="r1", scenario="s1", duration_s=3.0, events=events)
    assert s.sent == 2
    assert s.completed == 2
    assert s.error_rate == 0.5
    assert s.breakdown["gate"]["count_429"] == 1
    assert s.total_prompt_tokens == 10
    assert s.total_completion_tokens == 20


def test_writers_write_expected_files(tmp_path: Path) -> None:
    rp = RunPaths.under(tmp_path, run_id="run_a")
    assert rp.run_dir == tmp_path / "run_a"

    events = [
        Event(
            run_id="r",
            scenario="s",
            idx=0,
            endpoint="generate",
            started_at_unix=1.0,
            elapsed_ms=5.0,
            ok=True,
            status=200,
            cached=None,
            prompt_tokens=None,
            completion_tokens=None,
            error=None,
            error_payload=None,
            model=None,
            tags={},
        )
    ]
    summary = Summary(
        run_id="r",
        scenario="s",
        duration_s=1.0,
        sent=1,
        completed=1,
        ok=True,
        error_rate=0.0,
        breakdown={},
    )

    write_events_jsonl(rp.events_jsonl, events)
    write_summary_json(rp.summary_json, summary)

    lines = rp.events_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["run_id"] == "r"
    assert json.loads(rp.summary_json.read_text(encoding="utf-8"))["sent"] == 1
