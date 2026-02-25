# server/src/llm_server/services/llm_runtime/metrics.py
from __future__ import annotations

from typing import Optional

try:
    from prometheus_client import Counter, Gauge, Histogram
except Exception:  # pragma: no cover
    Counter = None  # type: ignore
    Gauge = None  # type: ignore
    Histogram = None  # type: ignore


def _counter(name: str, doc: str, labelnames: Optional[list[str]] = None):
    if Counter is None:
        return None
    return Counter(name, doc, labelnames=labelnames or [])


def _gauge(name: str, doc: str, labelnames: Optional[list[str]] = None):
    if Gauge is None:
        return None
    return Gauge(name, doc, labelnames=labelnames or [])


def _hist(name: str, doc: str, labelnames: Optional[list[str]] = None):
    if Histogram is None:
        return None
    return Histogram(name, doc, labelnames=labelnames or [])


# ----------------------------
# Loader operations
# ----------------------------

LLM_LOADER_OPS_TOTAL = _counter(
    "llm_loader_ops_total",
    "Count of llm_loader operations",
    ["op"],
)

LLM_LOADER_OPS_FAIL_TOTAL = _counter(
    "llm_loader_ops_fail_total",
    "Count of failed llm_loader operations",
    ["op", "reason"],
)

LLM_LOADER_OP_LATENCY_SECONDS = _hist(
    "llm_loader_op_latency_seconds",
    "Latency of llm_loader operations (seconds)",
    ["op"],
)

# ----------------------------
# Model state gauges
# ----------------------------

LLM_MODEL_LOADED = _gauge(
    "llm_model_loaded",
    "1 if in-process model weights are loaded (transformers), else 0",
)

LLM_MODEL_ERROR = _gauge(
    "llm_model_error",
    "1 if model_error is set, else 0",
)

LLM_LOADED_MODEL_ID_INFO = _gauge(
    "llm_loaded_model_id_info",
    "Info gauge: 1 for the currently loaded model id label",
    ["model_id"],
)

LLM_RUNTIME_DEFAULT_MODEL_ID_INFO = _gauge(
    "llm_runtime_default_model_id_info",
    "Info gauge: 1 for the current runtime default model id label",
    ["model_id"],
)


def set_state_gauges(*, model_loaded: bool, model_error: bool, loaded_model_id: str | None, runtime_default_model_id: str | None) -> None:
    """
    Best-effort state reporting.
    Uses INFO-style gauges with a label for model ids (common Prometheus pattern).
    """
    try:
        if LLM_MODEL_LOADED is not None:
            LLM_MODEL_LOADED.set(1 if model_loaded else 0)
        if LLM_MODEL_ERROR is not None:
            LLM_MODEL_ERROR.set(1 if model_error else 0)
    except Exception:
        pass

    # Info gauges: clear old label series is not feasible without keeping history.
    # We just set the current label to 1; dashboards should use `max by(model_id)` patterns.
    try:
        if LLM_LOADED_MODEL_ID_INFO is not None and loaded_model_id:
            LLM_LOADED_MODEL_ID_INFO.labels(model_id=loaded_model_id).set(1)
    except Exception:
        pass

    try:
        if LLM_RUNTIME_DEFAULT_MODEL_ID_INFO is not None and runtime_default_model_id:
            LLM_RUNTIME_DEFAULT_MODEL_ID_INFO.labels(model_id=runtime_default_model_id).set(1)
    except Exception:
        pass