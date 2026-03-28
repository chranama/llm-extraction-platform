from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from opentelemetry import propagate, trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Link, Span, SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


@dataclass
class TracingRuntime:
    enabled: bool
    service_name: str
    exporter_endpoint: Optional[str]
    _shutdown: Callable[[], None] = field(default=lambda: None, repr=False)

    def shutdown(self) -> None:
        self._shutdown()


def tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(f"llm-extraction-platform/{name}")


def _install_default_propagator() -> None:
    set_global_textmap(
        CompositePropagator(
            [
                TraceContextTextMapPropagator(),
                W3CBaggagePropagator(),
            ]
        )
    )


def _build_otlp_span_exporter(endpoint: str) -> OTLPSpanExporter:
    return OTLPSpanExporter(endpoint=endpoint)


def _build_tracer_provider(
    *,
    service_name: str,
    component: str,
    exporter: OTLPSpanExporter,
) -> TracerProvider:
    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": service_name,
                "llm.component": component,
            }
        )
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def _install_tracing_globals(provider: TracerProvider) -> None:
    trace.set_tracer_provider(provider)


def setup_tracing(
    settings: Any,
    logger: logging.Logger,
    *,
    component: str = "backend",
) -> TracingRuntime:
    runtime = TracingRuntime(
        enabled=False,
        service_name=str(getattr(settings, "otel_service_name", "llm-extraction-platform")),
        exporter_endpoint=getattr(settings, "otel_exporter_otlp_endpoint", None),
    )

    _install_default_propagator()

    if not bool(getattr(settings, "otel_enabled", False)):
        logger.info("otel bootstrap disabled", extra={"otel_service_name": runtime.service_name})
        return runtime

    exporter_endpoint = str(getattr(settings, "otel_exporter_otlp_endpoint", "") or "").strip()
    if not exporter_endpoint:
        logger.warning(
            "otel bootstrap enabled but exporter endpoint is empty; running without exporter",
            extra={"otel_service_name": runtime.service_name},
        )
        return runtime

    exporter = _build_otlp_span_exporter(exporter_endpoint)
    provider = _build_tracer_provider(
        service_name=runtime.service_name,
        component=component,
        exporter=exporter,
    )
    _install_tracing_globals(provider)

    runtime.enabled = True
    runtime.exporter_endpoint = exporter_endpoint
    runtime._shutdown = provider.shutdown

    logger.info(
        "otel bootstrap configured",
        extra={
            "otel_service_name": runtime.service_name,
            "otel_exporter_otlp_endpoint": exporter_endpoint,
        },
    )
    return runtime


def _normalize_attributes(raw: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(raw, Mapping):
        return out

    for key, value in raw.items():
        if value is None:
            continue
        if isinstance(value, bool | int | float | str):
            out[str(key)] = value
            continue
        out[str(key)] = str(value)
    return out


def _normalize_carrier(raw: Mapping[str, Any] | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not isinstance(raw, Mapping):
        return out

    for key, value in raw.items():
        if value is None:
            continue
        key_s = str(key).strip()
        value_s = str(value).strip()
        if key_s and value_s:
            out[key_s] = value_s
    return out


def current_trace_carrier() -> dict[str, str] | None:
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    normalized = _normalize_carrier(carrier)
    return normalized or None


def extract_parent_context(carrier: Mapping[str, Any] | None = None) -> Any | None:
    normalized = _normalize_carrier(carrier)
    if not normalized:
        return None
    return propagate.extract(normalized)


def span_link_from_carrier(
    carrier: Mapping[str, Any] | None = None,
    *,
    attributes: Mapping[str, Any] | None = None,
) -> Link | None:
    parent_ctx = extract_parent_context(carrier)
    if parent_ctx is None:
        return None
    span_ctx = trace.get_current_span(parent_ctx).get_span_context()
    if not span_ctx.is_valid:
        return None
    normalized_attrs = _normalize_attributes(attributes)
    return Link(span_ctx, attributes=normalized_attrs or None)


def _request_attr_bundle(request: Any, *, route: str | None = None) -> dict[str, Any]:
    state = getattr(request, "state", None)
    attrs: dict[str, Any] = {
        "llm.component": "backend",
        "http.request.method": getattr(request, "method", "UNKNOWN"),
        "url.path": getattr(getattr(request, "url", None), "path", ""),
    }

    request_id = getattr(state, "request_id", None)
    trace_id = getattr(state, "trace_id", None)
    job_id = getattr(state, "trace_job_id", None)
    route_value = route or getattr(state, "route", None) or attrs["url.path"]

    if isinstance(request_id, str) and request_id.strip():
        attrs["llm.request_id"] = request_id.strip()
    if isinstance(trace_id, str) and trace_id.strip():
        attrs["llm.trace_id"] = trace_id.strip()
    if isinstance(job_id, str) and job_id.strip():
        attrs["llm.job_id"] = job_id.strip()
    if isinstance(route_value, str) and route_value.strip():
        attrs["llm.route"] = route_value.strip()

    return attrs


def bind_request_span(
    request: Any,
    *,
    name: str | None = None,
    route: str | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> Span:
    span = (
        getattr(getattr(request, "state", None), "otel_request_span", None)
        or trace.get_current_span()
    )
    state = getattr(request, "state", None)
    if state is not None and isinstance(route, str) and route.strip():
        state.route = route.strip()
    if name:
        span.update_name(name)
    attrs = _request_attr_bundle(request, route=route)
    attrs.update(_normalize_attributes(attributes))
    if attrs:
        span.set_attributes(attrs)
    return span


def set_http_response(span: Span, status_code: int) -> None:
    span.set_attribute("http.response.status_code", int(status_code))
    if status_code >= 400:
        span.set_status(Status(StatusCode.ERROR, str(status_code)))
        return
    span.set_status(Status(StatusCode.OK))


def record_error(span: Span, err: BaseException) -> None:
    span.record_exception(err)
    span.set_status(Status(StatusCode.ERROR, str(err)))


@contextmanager
def start_child_span(
    name: str,
    *,
    request: Any | None = None,
    attributes: Mapping[str, Any] | None = None,
    context: Any | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    links: Sequence[Link] | None = None,
) -> Iterator[Span]:
    span_attrs: dict[str, Any] = {}
    if request is not None:
        span_attrs.update(_request_attr_bundle(request))
    span_attrs.update(_normalize_attributes(attributes))

    with tracer("application").start_as_current_span(
        name,
        context=context,
        kind=kind,
        attributes=span_attrs,
        links=list(links or []),
    ) as span:
        try:
            yield span
        except Exception as err:
            record_error(span, err)
            raise


@contextmanager
def start_consumer_span(
    name: str,
    *,
    carrier: Mapping[str, Any] | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[Span]:
    span_attrs = {"llm.component": "backend"}
    span_attrs.update(_normalize_attributes(attributes))

    with tracer("worker").start_as_current_span(
        name,
        context=extract_parent_context(carrier),
        kind=SpanKind.CONSUMER,
        attributes=span_attrs,
    ) as span:
        try:
            yield span
        except Exception as err:
            record_error(span, err)
            raise


@contextmanager
def start_request_span(request: Any, *, name: str = "backend.request") -> Iterator[Span]:
    headers = getattr(request, "headers", None)
    parent_ctx = propagate.extract(headers) if headers is not None else None

    with tracer("http.server").start_as_current_span(
        name,
        context=parent_ctx,
        kind=SpanKind.SERVER,
        attributes=_request_attr_bundle(request),
    ) as span:
        state = getattr(request, "state", None)
        if state is not None:
            state.otel_request_span = span
        try:
            yield span
        except Exception as err:
            record_error(span, err)
            raise
