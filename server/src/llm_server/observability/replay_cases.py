from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from llm_server.telemetry.types import TraceDetail


def _utc_iso_z(ts: datetime | None = None) -> str:
    now = ts or datetime.now(timezone.utc)
    return now.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _compact_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, item in value.items():
        if item is None:
            continue
        if isinstance(item, Mapping):
            nested = _compact_mapping(item)
            if nested:
                out[key] = nested
            continue
        out[key] = item
    return out


def _request_kind_from_route(route: str) -> str:
    if route.startswith("/v1/extract/jobs"):
        return "async_extract"
    if route.startswith("/v1/extract"):
        return "sync_extract"
    if route.startswith("/v1/generate"):
        return "generate"
    return "unknown"


def _primary_request_id_from_trace(detail: TraceDetail) -> str | None:
    for event in detail.events:
        if isinstance(event.request_id, str) and event.request_id:
            return event.request_id
    return None


def _json_if_possible(output: str | None) -> Any:
    if not isinstance(output, str):
        return None
    try:
        return json.loads(output)
    except Exception:
        return output


def _extract_request_from_log(log: Any) -> dict[str, Any]:
    params = getattr(log, "params_json", None)
    params = params if isinstance(params, Mapping) else {}
    return _compact_mapping(
        {
            "schema_id": params.get("schema_id"),
            "text": getattr(log, "prompt", None),
            "model": params.get("requested_model_id") or getattr(log, "model_id", None),
            "max_new_tokens": params.get("requested_max_new_tokens")
            or params.get("max_new_tokens"),
            "temperature": params.get("temperature"),
            "cache": params.get("cache"),
            "repair": params.get("repair"),
        }
    )


def _generate_request_from_log(log: Any) -> dict[str, Any]:
    params = getattr(log, "params_json", None)
    params = params if isinstance(params, Mapping) else {}
    return _compact_mapping(
        {
            "prompt": getattr(log, "prompt", None),
            "model": params.get("requested_model_id") or getattr(log, "model_id", None),
            "max_new_tokens": params.get("requested_max_new_tokens")
            or params.get("max_new_tokens"),
            "temperature": params.get("temperature"),
            "top_p": params.get("top_p"),
            "top_k": params.get("top_k"),
            "stop": params.get("stop"),
            "cache": params.get("cache"),
        }
    )


def _request_payload_from_log(log: Any) -> dict[str, Any]:
    route = str(getattr(log, "route", "") or "")
    if route.startswith("/v1/extract"):
        return _extract_request_from_log(log)
    if route.startswith("/v1/generate"):
        return _generate_request_from_log(log)
    return _compact_mapping(
        {
            "prompt": getattr(log, "prompt", None),
            "model": getattr(log, "model_id", None),
            "params": getattr(log, "params_json", None),
        }
    )


def _expectation_from_log(log: Any) -> dict[str, Any]:
    status_code = getattr(log, "status_code", None)
    cached = getattr(log, "cached", None)
    output = _json_if_possible(getattr(log, "output", None))
    is_failed = bool(
        (isinstance(status_code, int) and status_code >= 400)
        or getattr(log, "error_code", None)
        or getattr(log, "error_stage", None)
    )
    return _compact_mapping(
        {
            "status": "failed" if is_failed else "succeeded",
            "status_code": status_code,
            "output": output,
            "cached": cached,
            "error_code": getattr(log, "error_code", None),
            "error_stage": getattr(log, "error_stage", None),
        }
    )


def _trace_request_details(detail: TraceDetail) -> Mapping[str, Any]:
    for event in detail.events:
        if event.event_name in {"extract.accepted", "extract_job.submitted"}:
            if isinstance(event.details, Mapping):
                return event.details
    return {}


def _request_payload_from_trace(detail: TraceDetail) -> dict[str, Any]:
    details = _trace_request_details(detail)
    if detail.request_kind in {"sync_extract", "async_extract"}:
        return _compact_mapping(
            {
                "schema_id": details.get("schema_id"),
                "text": details.get("text"),
                "model": details.get("requested_model_id"),
                "max_new_tokens": details.get("requested_max_new_tokens"),
                "temperature": details.get("temperature"),
                "cache": details.get("cache"),
                "repair": details.get("repair"),
            }
        )
    return _compact_mapping(
        {
            "prompt": details.get("prompt"),
            "model": details.get("requested_model_id"),
            "max_new_tokens": details.get("requested_max_new_tokens"),
        }
    )


def _expectation_from_trace(detail: TraceDetail) -> dict[str, Any]:
    for event in reversed(detail.events):
        if event.event_name.endswith(".failed") or event.status == "failed":
            return _compact_mapping(
                {
                    "status": "failed",
                    "error_code": (
                        event.details.get("error_code")
                        if isinstance(event.details, Mapping)
                        else None
                    ),
                    "error_stage": (
                        event.details.get("error_stage")
                        if isinstance(event.details, Mapping)
                        else None
                    ),
                }
            )
        if event.event_name.endswith(".completed") or event.status in {"completed", "succeeded"}:
            return _compact_mapping(
                {
                    "status": "succeeded",
                    "cached": (
                        event.details.get("cached") if isinstance(event.details, Mapping) else None
                    ),
                    "repair_attempted": (
                        event.details.get("repair_attempted")
                        if isinstance(event.details, Mapping)
                        else None
                    ),
                }
            )

    return {"status": "in_progress"}


def _missing_fields_for_request(*, route: str, request_payload: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    if route.startswith("/v1/extract"):
        if "text" not in request_payload:
            missing.append("text")
        if "schema_id" not in request_payload:
            missing.append("schema_id")
    elif route.startswith("/v1/generate"):
        if "prompt" not in request_payload:
            missing.append("prompt")
    else:
        if "prompt" not in request_payload and "text" not in request_payload:
            missing.append("input")
    return missing


def _trace_observability(detail: TraceDetail, *, log_count: int) -> dict[str, Any]:
    return _compact_mapping(
        {
            "event_count": len(detail.events),
            "event_names": [event.event_name for event in detail.events],
            "started_at": _utc_iso_z(detail.started_at),
            "finished_at": _utc_iso_z(detail.finished_at) if detail.finished_at else None,
            "trace_status": detail.status,
            "log_count": log_count,
        }
    )


def build_replay_case_from_inference_log(
    log: Any,
    *,
    trace_detail: TraceDetail | None = None,
) -> dict[str, Any]:
    route = str(getattr(log, "route", "") or "")
    request_kind = (
        trace_detail.request_kind if trace_detail is not None else _request_kind_from_route(route)
    )
    request_payload = _request_payload_from_log(log)
    missing_fields = _missing_fields_for_request(route=route, request_payload=request_payload)

    observability = _compact_mapping(
        {
            "log_id": getattr(log, "id", None),
            "created_at": _utc_iso_z(getattr(log, "created_at", None)),
            "latency_ms": getattr(log, "latency_ms", None),
            "prompt_tokens": getattr(log, "prompt_tokens", None),
            "completion_tokens": getattr(log, "completion_tokens", None),
            "trace_status": trace_detail.status if trace_detail is not None else None,
            "event_count": len(trace_detail.events) if trace_detail is not None else None,
        }
    )

    return {
        "case_id": f"log:{getattr(log, 'id', 'unknown')}",
        "route": route,
        "request_kind": request_kind,
        "replay_ready": not missing_fields,
        "missing_fields": missing_fields,
        "correlation": _compact_mapping(
            {
                "request_id": getattr(log, "request_id", None),
                "trace_id": getattr(log, "trace_id", None),
                "job_id": getattr(log, "job_id", None),
            }
        ),
        "request": request_payload,
        "expectation": _expectation_from_log(log),
        "observability": observability,
    }


def build_replay_case_from_trace(
    detail: TraceDetail,
    *,
    inference_logs: Sequence[Any],
) -> dict[str, Any]:
    if inference_logs:
        primary_log = sorted(
            inference_logs,
            key=lambda row: (
                getattr(row, "created_at", None) or datetime.min.replace(tzinfo=timezone.utc),
                getattr(row, "id", 0),
            ),
        )[-1]
        case = build_replay_case_from_inference_log(primary_log, trace_detail=detail)
        case["case_id"] = f"trace:{detail.trace_id}"
        case["correlation"] = _compact_mapping(
            {
                **(case.get("correlation") or {}),
                "trace_id": detail.trace_id,
                "job_id": detail.job_id,
            }
        )
        case["observability"] = _compact_mapping(
            {
                **(case.get("observability") or {}),
                **_trace_observability(detail, log_count=len(inference_logs)),
            }
        )
        return case

    request_payload = _request_payload_from_trace(detail)
    missing_fields = _missing_fields_for_request(
        route=detail.root_route, request_payload=request_payload
    )
    return {
        "case_id": f"trace:{detail.trace_id}",
        "route": detail.root_route,
        "request_kind": detail.request_kind,
        "replay_ready": not missing_fields,
        "missing_fields": missing_fields,
        "correlation": _compact_mapping(
            {
                "request_id": _primary_request_id_from_trace(detail),
                "trace_id": detail.trace_id,
                "job_id": detail.job_id,
            }
        ),
        "request": request_payload,
        "expectation": _expectation_from_trace(detail),
        "observability": _trace_observability(detail, log_count=0),
    }
