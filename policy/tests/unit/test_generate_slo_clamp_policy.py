from __future__ import annotations

from types import SimpleNamespace

from llm_policy.policies.generate_slo_clamp import (
    GenerateSloClampThresholds,
    decide_generate_slo_clamp,
    thresholds_from_mapping,
)
from llm_policy.types.decision import Decision, DecisionStatus


def _base_decision() -> Decision:
    return Decision(
        policy="p",
        pipeline="extract_plus_generate_clamp",
        status=DecisionStatus.allow,
        enable_extract=True,
        reasons=[],
        warnings=[],
        metrics={},
    )


def _thresholds() -> GenerateSloClampThresholds:
    return GenerateSloClampThresholds(
        min_requests=10,
        error_rate_threshold=0.02,
        error_rate_cap=128,
        latency_p95_steps={1500: 128, 1000: 256, 800: 384},
    )


def test_thresholds_from_mapping_parses_and_ignores_invalid_steps() -> None:
    th = thresholds_from_mapping(
        {
            "min_requests": 7,
            "safe_cap_on_invalid_snapshot": 64,
            "error_rate": {"threshold": 0.05, "cap": 111},
            "latency_p95_ms": {"steps": {"1000": 200, "bad": "x"}},
        }
    )

    assert th.min_requests == 7
    assert th.safe_cap_on_invalid_snapshot == 64
    assert th.error_rate_threshold == 0.05
    assert th.error_rate_cap == 111
    assert th.latency_p95_steps == {1000: 200}


def test_disabled_returns_base_unchanged() -> None:
    base = _base_decision()
    out = decide_generate_slo_clamp(base=base, slo=None, thresholds=_thresholds(), enabled=False)
    assert out is base


def test_missing_snapshot_adds_warning_and_no_clamp() -> None:
    base = _base_decision()
    out = decide_generate_slo_clamp(base=base, slo=None, thresholds=_thresholds())

    assert out.generate_max_new_tokens_cap is None
    assert out.status == base.status
    assert out.enable_extract == base.enable_extract
    assert any(r.code == "generate_slo_no_snapshot" for r in out.reasons)
    assert any(w.code == "generate_slo_snapshot_missing" for w in out.warnings)


def test_invalid_snapshot_adds_warning_and_no_clamp() -> None:
    base = _base_decision()
    slo = SimpleNamespace(
        total_requests=100,
        error_rate=0.1,
        latency_p95_ms=900,
        completion_tokens_p95=100,
        source_path="/tmp/slo.json",
        error="parse error",
    )

    out = decide_generate_slo_clamp(base=base, slo=slo, thresholds=_thresholds())

    assert out.generate_max_new_tokens_cap is None
    assert any(r.code == "generate_slo_invalid_snapshot_no_clamp" for r in out.reasons)
    assert any(w.code == "generate_slo_snapshot_invalid" for w in out.warnings)


def test_incomplete_snapshot_no_clamp() -> None:
    base = _base_decision()
    slo = SimpleNamespace(
        total_requests=100,
        error_rate=None,
        latency_p95_ms=900,
        completion_tokens_p95=100,
        source_path="/tmp/slo.json",
        error=None,
    )

    out = decide_generate_slo_clamp(base=base, slo=slo, thresholds=_thresholds())

    assert out.generate_max_new_tokens_cap is None
    assert any(r.code == "generate_slo_incomplete_snapshot_no_clamp" for r in out.reasons)
    assert any(w.code == "generate_slo_snapshot_incomplete" for w in out.warnings)


def test_insufficient_traffic_no_clamp() -> None:
    base = _base_decision()
    slo = SimpleNamespace(
        total_requests=3,
        error_rate=0.0,
        latency_p95_ms=400,
        completion_tokens_p95=90,
        source_path="/tmp/slo.json",
        error=None,
    )

    out = decide_generate_slo_clamp(base=base, slo=slo, thresholds=_thresholds())

    assert out.generate_max_new_tokens_cap is None
    assert any(r.code == "generate_slo_insufficient_traffic" for r in out.reasons)


def test_error_rate_clamp_takes_precedence_over_latency() -> None:
    base = _base_decision()
    slo = SimpleNamespace(
        total_requests=100,
        error_rate=0.5,
        latency_p95_ms=2000,
        completion_tokens_p95=300,
        source_path="/tmp/slo.json",
        error=None,
    )

    out = decide_generate_slo_clamp(base=base, slo=slo, thresholds=_thresholds())

    assert out.generate_max_new_tokens_cap == 128
    assert any(r.code == "generate_slo_error_rate_high" for r in out.reasons)


def test_latency_clamp_uses_descending_step_match() -> None:
    base = _base_decision()
    slo = SimpleNamespace(
        total_requests=100,
        error_rate=0.01,
        latency_p95_ms=1200,
        completion_tokens_p95=120,
        source_path="/tmp/slo.json",
        error=None,
    )

    out = decide_generate_slo_clamp(base=base, slo=slo, thresholds=_thresholds())

    assert out.generate_max_new_tokens_cap == 256
    assert any(r.code == "generate_slo_latency_high" for r in out.reasons)


def test_no_clamp_when_within_thresholds() -> None:
    base = _base_decision()
    slo = SimpleNamespace(
        total_requests=100,
        error_rate=0.01,
        latency_p95_ms=500,
        completion_tokens_p95=80,
        source_path="/tmp/slo.json",
        error=None,
    )

    out = decide_generate_slo_clamp(base=base, slo=slo, thresholds=_thresholds())

    assert out.generate_max_new_tokens_cap is None
    assert any(r.code == "generate_slo_no_clamp" for r in out.reasons)
