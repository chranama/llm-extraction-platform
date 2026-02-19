# server/src/llm_server/services/limits/metrics.py
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram


# Current number of requests waiting to enter execution
GENERATE_QUEUE_DEPTH = Gauge(
    "llm_generate_queue_depth",
    "Number of /v1/generate requests waiting in the gate queue",
)

# Current number of requests actively executing (i.e., acquired the semaphore)
GENERATE_IN_FLIGHT = Gauge(
    "llm_generate_in_flight",
    "Number of /v1/generate requests currently executing",
)

# Decisions / outcomes
GENERATE_GATE_REJECTS = Counter(
    "llm_generate_gate_rejects_total",
    "Number of /v1/generate requests rejected by the gate",
    ["reason"],  # queue_full | disabled | timeout
)

GENERATE_GATE_ENTERS = Counter(
    "llm_generate_gate_enters_total",
    "Number of /v1/generate requests that entered the gate (queued or immediate)",
)

GENERATE_GATE_STARTS = Counter(
    "llm_generate_gate_starts_total",
    "Number of /v1/generate requests that started execution (semaphore acquired)",
)

GENERATE_GATE_TIMEOUTS = Counter(
    "llm_generate_gate_timeouts_total",
    "Number of /v1/generate requests that timed out waiting or executing in the gate",
    ["stage"],  # queue_wait | execution
)

# Latency histograms (seconds)
GENERATE_QUEUE_WAIT_SECONDS = Histogram(
    "llm_generate_queue_wait_seconds",
    "Time spent waiting in /v1/generate gate queue before execution",
    buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)

GENERATE_EXECUTION_SECONDS = Histogram(
    "llm_generate_execution_seconds",
    "Time spent executing /v1/generate while holding concurrency slot",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
)