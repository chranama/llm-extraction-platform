# server/src/llm_server/services/limits/generate_gating.py
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, TypeVar

from llm_server.core.errors import AppError
from llm_server.services.limits.config import GenerateGateConfig, load_generate_gate_config
from llm_server.services.limits.metrics import (
    GENERATE_EXECUTION_SECONDS,
    GENERATE_GATE_ENTERS,
    GENERATE_GATE_REJECTS,
    GENERATE_GATE_STARTS,
    GENERATE_GATE_TIMEOUTS,
    GENERATE_IN_FLIGHT,
    GENERATE_QUEUE_DEPTH,
    GENERATE_QUEUE_WAIT_SECONDS,
)

T = TypeVar("T")


@dataclass
class _GateState:
    sem: asyncio.Semaphore
    queue_slots: asyncio.Semaphore
    cfg: GenerateGateConfig


@dataclass(frozen=True)
class GenerateGateSnapshot:
    """
    Best-effort view of gate pressure for early-reject middleware.
    Values are approximate under contention (good enough for overload rejection).
    """
    enabled: bool
    max_concurrent: int
    max_queue: int
    timeout_seconds: float
    in_flight_estimate: int
    queue_depth_estimate: int


class GenerateGate:
    """
    Async gate to protect the server from:
      - excessive concurrent model work
      - unbounded queue growth
      - long tail latency under load

    timeout_seconds covers BOTH queue wait + execution.

    Important behavior:
      - fail_fast applies to BOTH queue admission and execution-slot acquisition.
      - if max_queue == 0, there is no waiting room: concurrency slot is the admission control.
      - stage labels in timeouts match metrics.py: queue_wait | execution
    """

    def __init__(self, cfg: GenerateGateConfig) -> None:
        cfg = GenerateGateConfig(
            enabled=bool(cfg.enabled),
            max_concurrent=max(1, int(cfg.max_concurrent)),
            max_queue=max(0, int(cfg.max_queue)),
            timeout_seconds=float(cfg.timeout_seconds),
            fail_fast=bool(cfg.fail_fast),
            count_queued_as_in_flight=bool(cfg.count_queued_as_in_flight),
        )

        self._state = _GateState(
            sem=asyncio.Semaphore(cfg.max_concurrent),
            queue_slots=asyncio.Semaphore(cfg.max_queue if cfg.max_queue > 0 else 1),
            cfg=cfg,
        )

        # Initialize gauges (best-effort)
        try:
            GENERATE_QUEUE_DEPTH.set(0)
            GENERATE_IN_FLIGHT.set(0)
        except Exception:
            pass

    @property
    def cfg(self) -> GenerateGateConfig:
        return self._state.cfg

    def snapshot(self) -> GenerateGateSnapshot:
        """
        Best-effort instantaneous view.

        Uses semaphore internal counters:
          - in_flight ~= max_concurrent - sem._value
          - queue_depth ~= max_queue - queue_slots._value (only meaningful if max_queue > 0)
        """
        cfg = self._state.cfg
        sem = self._state.sem
        q = self._state.queue_slots

        try:
            sem_avail = int(getattr(sem, "_value", 0))  # type: ignore[attr-defined]
        except Exception:
            sem_avail = 0

        try:
            q_avail = int(getattr(q, "_value", 0))  # type: ignore[attr-defined]
        except Exception:
            q_avail = 0

        in_flight = max(0, int(cfg.max_concurrent) - sem_avail)

        queue_depth = 0
        if cfg.max_queue > 0:
            queue_depth = max(0, int(cfg.max_queue) - q_avail)

        return GenerateGateSnapshot(
            enabled=bool(cfg.enabled),
            max_concurrent=int(cfg.max_concurrent),
            max_queue=int(cfg.max_queue),
            timeout_seconds=float(cfg.timeout_seconds),
            in_flight_estimate=int(in_flight),
            queue_depth_estimate=int(queue_depth),
        )

    def _reject(self, reason: str, *, extra: Optional[dict[str, Any]] = None) -> None:
        try:
            GENERATE_GATE_REJECTS.labels(reason=reason).inc()
        except Exception:
            pass
        raise AppError(
            code="generate_overloaded",
            message="Generate is overloaded. Try again later.",
            status_code=429,
            extra=extra or {"reason": reason},
        )

    async def run(
        self,
        fn: Callable[[], Awaitable[T]],
        *,
        request_id: str | None = None,
        model_id: str | None = None,
    ) -> T:
        cfg = self._state.cfg

        # If disabled, bypass (do not 429).
        if not cfg.enabled:
            return await fn()

        try:
            GENERATE_GATE_ENTERS.inc()
        except Exception:
            pass

        t_enter = time.perf_counter()
        timeout_s = float(cfg.timeout_seconds)

        # Track what we incremented so we can unwind correctly.
        queue_depth_inc = False
        in_flight_inc = False

        # max_queue <= 0 => no queue admission token (no waiting room)
        use_queue = cfg.max_queue > 0
        queue_token: _Token = _SemaphoreToken(self._state.queue_slots) if use_queue else _NullToken()
        exec_token = _SemaphoreToken(self._state.sem)

        try:
            # -------------------------
            # Queue admission
            # -------------------------
            if use_queue:
                ok_q = queue_token.try_acquire() if cfg.fail_fast else await queue_token.acquire_with_timeout(timeout_s)
                if not ok_q:
                    self._reject(
                        "queue_full",
                        extra={
                            "request_id": request_id,
                            "model_id": model_id,
                            "max_queue": cfg.max_queue,
                            "max_concurrent": cfg.max_concurrent,
                            "timeout_seconds": timeout_s,
                        },
                    )

                # we are now in the queue
                try:
                    GENERATE_QUEUE_DEPTH.inc()
                    queue_depth_inc = True
                except Exception:
                    pass

                if cfg.count_queued_as_in_flight:
                    try:
                        GENERATE_IN_FLIGHT.inc()
                        in_flight_inc = True
                    except Exception:
                        pass

            # -------------------------
            # Acquire execution slot
            # -------------------------
            elapsed = time.perf_counter() - t_enter
            remaining = max(0.0, timeout_s - elapsed)

            t_wait = time.perf_counter()

            # fail_fast must apply here too
            ok_exec = exec_token.try_acquire() if cfg.fail_fast else await exec_token.acquire_with_timeout(remaining)
            if not ok_exec:
                if cfg.fail_fast:
                    self._reject(
                        "concurrency_full",
                        extra={
                            "request_id": request_id,
                            "model_id": model_id,
                            "max_concurrent": cfg.max_concurrent,
                            "timeout_seconds": timeout_s,
                        },
                    )
                else:
                    try:
                        GENERATE_GATE_TIMEOUTS.labels(stage="queue_wait").inc()
                    except Exception:
                        pass
                    self._reject(
                        "timeout",
                        extra={
                            "request_id": request_id,
                            "model_id": model_id,
                            "stage": "queue_wait",
                            "timeout_seconds": timeout_s,
                        },
                    )

            # Transition: leaving queue -> executing
            if use_queue and queue_depth_inc:
                try:
                    GENERATE_QUEUE_DEPTH.dec()
                except Exception:
                    pass
                queue_depth_inc = False

            if not cfg.count_queued_as_in_flight:
                try:
                    GENERATE_IN_FLIGHT.inc()
                    in_flight_inc = True
                except Exception:
                    pass

            try:
                GENERATE_GATE_STARTS.inc()
                GENERATE_QUEUE_WAIT_SECONDS.observe(time.perf_counter() - t_wait)
            except Exception:
                pass

            # Free queue slot immediately when execution starts
            await queue_token.release_if_held()

            # -------------------------
            # Execute under remaining time
            # -------------------------
            elapsed2 = time.perf_counter() - t_enter
            remaining2 = max(0.0, timeout_s - elapsed2)

            t_exec = time.perf_counter()
            try:
                if remaining2 <= 0:
                    try:
                        GENERATE_GATE_TIMEOUTS.labels(stage="execution").inc()
                    except Exception:
                        pass
                    self._reject(
                        "timeout",
                        extra={
                            "request_id": request_id,
                            "model_id": model_id,
                            "stage": "execution",
                            "timeout_seconds": timeout_s,
                        },
                    )

                return await asyncio.wait_for(fn(), timeout=remaining2)

            except asyncio.TimeoutError:
                try:
                    GENERATE_GATE_TIMEOUTS.labels(stage="execution").inc()
                except Exception:
                    pass
                self._reject(
                    "timeout",
                    extra={
                        "request_id": request_id,
                        "model_id": model_id,
                        "stage": "execution",
                        "timeout_seconds": timeout_s,
                    },
                )
            finally:
                try:
                    GENERATE_EXECUTION_SECONDS.observe(time.perf_counter() - t_exec)
                except Exception:
                    pass

        finally:
            # Unwind queue gauge if we never started execution
            if use_queue and queue_depth_inc:
                try:
                    GENERATE_QUEUE_DEPTH.dec()
                except Exception:
                    pass
                queue_depth_inc = False

            # Release held tokens (safe to call even if not held)
            try:
                await queue_token.release_if_held()
            except Exception:
                pass
            exec_token.release_if_held_sync()

            # Unwind in-flight exactly if we incremented it
            if in_flight_inc:
                try:
                    GENERATE_IN_FLIGHT.dec()
                except Exception:
                    pass


# ----------------------------
# Internal helpers
# ----------------------------

class _Token:
    held: bool

    def try_acquire(self) -> bool: ...
    async def acquire_with_timeout(self, timeout_s: float) -> bool: ...
    def release_if_held_sync(self) -> None: ...
    async def release_if_held(self) -> None: ...


class _SemaphoreToken(_Token):
    """
    Wrapper to track whether we currently hold a semaphore token.

    try_acquire() is best-effort and uses semaphore private _value.
    That is acceptable here because this gate is protective / approximate by design.
    """

    def __init__(self, sem: asyncio.Semaphore) -> None:
        self._sem = sem
        self.held = False

    def try_acquire(self) -> bool:
        try:
            if getattr(self._sem, "_value", 0) <= 0:  # type: ignore[attr-defined]
                return False
            self._sem._value -= 1  # type: ignore[attr-defined]
            self.held = True
            return True
        except Exception:
            return False

    async def acquire_with_timeout(self, timeout_s: float) -> bool:
        if timeout_s <= 0:
            return False
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=timeout_s)
            self.held = True
            return True
        except asyncio.TimeoutError:
            return False

    def release_if_held_sync(self) -> None:
        if not self.held:
            return
        self.held = False
        try:
            self._sem.release()
        except Exception:
            pass

    async def release_if_held(self) -> None:
        self.release_if_held_sync()


class _NullToken(_Token):
    held: bool = False

    def try_acquire(self) -> bool:
        return True

    async def acquire_with_timeout(self, timeout_s: float) -> bool:
        return True

    def release_if_held_sync(self) -> None:
        return None

    async def release_if_held(self) -> None:
        return None


# ----------------------------
# Singleton accessor
# ----------------------------

_GATE: GenerateGate | None = None


def get_generate_gate() -> GenerateGate:
    global _GATE
    if _GATE is None:
        cfg = load_generate_gate_config(settings=None)
        _GATE = GenerateGate(cfg)
    return _GATE


def reset_generate_gate_for_tests() -> None:
    global _GATE
    _GATE = None