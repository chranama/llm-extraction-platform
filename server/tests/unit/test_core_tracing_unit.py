from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from llm_server.core.config import Settings
from llm_server.core.tracing import current_trace_carrier, setup_tracing, span_link_from_carrier


def test_setup_tracing_disabled_is_noop():
    runtime = setup_tracing(
        SimpleNamespace(
            otel_enabled=False,
            otel_service_name="llm-extraction-platform",
            otel_exporter_otlp_endpoint=None,
        ),
        logging.getLogger("test.tracing"),
    )

    assert runtime.enabled is False
    runtime.shutdown()


def test_setup_tracing_missing_endpoint_is_noop():
    runtime = setup_tracing(
        SimpleNamespace(
            otel_enabled=True,
            otel_service_name="llm-extraction-platform",
            otel_exporter_otlp_endpoint=None,
        ),
        logging.getLogger("test.tracing"),
    )

    assert runtime.enabled is False


def test_setup_tracing_enabled_builds_runtime(monkeypatch):
    calls: dict[str, object] = {}

    class _Provider:
        def __init__(self) -> None:
            self.shutdown_called = False

        def shutdown(self) -> None:
            self.shutdown_called = True

    provider = _Provider()

    monkeypatch.setattr(
        "llm_server.core.tracing._build_otlp_span_exporter",
        lambda endpoint: {"endpoint": endpoint},
        raising=True,
    )
    monkeypatch.setattr(
        "llm_server.core.tracing._build_tracer_provider",
        lambda *, service_name, component, exporter: calls.update(
            {
                "service_name": service_name,
                "component": component,
                "exporter": exporter,
            }
        )
        or provider,
        raising=True,
    )
    monkeypatch.setattr(
        "llm_server.core.tracing._install_tracing_globals",
        lambda installed_provider: calls.update({"installed_provider": installed_provider}),
        raising=True,
    )

    runtime = setup_tracing(
        SimpleNamespace(
            otel_enabled=True,
            otel_service_name="llmep-dev",
            otel_exporter_otlp_endpoint="http://127.0.0.1:4318/v1/traces",
        ),
        logging.getLogger("test.tracing"),
        component="backend",
    )

    assert runtime.enabled is True
    assert runtime.service_name == "llmep-dev"
    assert runtime.exporter_endpoint == "http://127.0.0.1:4318/v1/traces"
    assert calls["service_name"] == "llmep-dev"
    assert calls["component"] == "backend"
    assert calls["exporter"] == {"endpoint": "http://127.0.0.1:4318/v1/traces"}
    assert calls["installed_provider"] is provider

    runtime.shutdown()
    assert provider.shutdown_called is True


def test_settings_rejects_empty_otel_service_name(monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "   ")
    with pytest.raises(ValidationError):
        Settings()


def test_settings_rejects_invalid_otel_exporter_endpoint(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "not-a-url")
    with pytest.raises(ValidationError):
        Settings()


def test_current_trace_carrier_returns_normalized_mapping(monkeypatch):
    monkeypatch.setattr(
        "llm_server.core.tracing.propagate.inject",
        lambda carrier: carrier.update(
            {
                "traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
                "tracestate": "vendor=value",
            }
        ),
        raising=True,
    )

    assert current_trace_carrier() == {
        "traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
        "tracestate": "vendor=value",
    }


def test_span_link_from_carrier_returns_none_without_valid_parent(monkeypatch):
    monkeypatch.setattr(
        "llm_server.core.tracing.extract_parent_context",
        lambda carrier: None,
        raising=True,
    )

    assert span_link_from_carrier({"traceparent": "ignored"}) is None
