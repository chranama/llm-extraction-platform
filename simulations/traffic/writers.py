# simulations/traffic/writers.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from simulations.paths import ensure_parent_dir
from simulations.traffic.models import Event, Summary


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    events_jsonl: Path
    summary_json: Path

    @staticmethod
    def under(out_root: Path, *, run_id: str) -> "RunPaths":
        rd = out_root / run_id
        return RunPaths(
            run_dir=rd,
            events_jsonl=rd / "events.jsonl",
            summary_json=rd / "summary.json",
        )


def write_events_jsonl(path: Path, events: Iterable[Event]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e.model_dump(), ensure_ascii=False) + "\n")


def write_summary_json(path: Path, summary: Summary) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")