# simulations/traffic/scheduler.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class SchedulePoint:
    idx: int
    at_unix: float


class OpenLoopScheduler:
    """
    Emits (idx, target_time) at fixed RPS for duration.
    """
    def __init__(self, *, duration_s: float, rps: float):
        self.duration_s = max(0.0, float(duration_s))
        self.rps = max(0.0, float(rps))

    def iter(self, *, start_unix: Optional[float] = None) -> Iterator[SchedulePoint]:
        t0 = time.time() if start_unix is None else float(start_unix)
        if self.rps <= 0.0 or self.duration_s <= 0.0:
            return iter(())
        dt = 1.0 / self.rps
        n = int(self.duration_s * self.rps)
        for i in range(n):
            yield SchedulePoint(idx=i, at_unix=t0 + i * dt)


class ClosedLoopBackoff:
    """
    Tiny closed-loop add-on: if things look bad, slow down by a multiplier.
    This is intentionally simple (no PID), and remains deterministic.
    """
    def __init__(self, *, base_rps: float, min_rps: float = 0.2, max_rps: float = 50.0):
        self.base_rps = max(0.0, float(base_rps))
        self.min_rps = float(min_rps)
        self.max_rps = float(max_rps)
        self.mult = 1.0

    def current_rps(self) -> float:
        rps = self.base_rps * self.mult
        if rps < self.min_rps:
            return self.min_rps
        if rps > self.max_rps:
            return self.max_rps
        return rps

    def observe_window(self, *, error_rate: float, latency_p95_ms: Optional[float]) -> None:
        """
        Heuristic:
          - if error_rate >= 5% or p95 >= 2000ms -> back off (halve)
          - if clean (error_rate < 1% and p95 < 1200ms) -> recover (x1.2)
        """
        er = float(error_rate)
        p95 = float(latency_p95_ms) if latency_p95_ms is not None else None

        bad = er >= 0.05 or (p95 is not None and p95 >= 2000.0)
        good = er < 0.01 and (p95 is None or p95 < 1200.0)

        if bad:
            self.mult = max(self.mult * 0.5, 0.1)
        elif good:
            self.mult = min(self.mult * 1.2, 1.0)