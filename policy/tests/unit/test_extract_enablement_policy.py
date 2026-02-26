from __future__ import annotations

from llm_policy.policies.extract_enablement import decide_extract_enablement
from llm_policy.types.eval_artifact import EvalArtifact, EvalSummary
from llm_policy.types.extract_thresholds import ExtractThresholds


def _thresholds() -> ExtractThresholds:
    return ExtractThresholds(
        version="v1",
        metrics={
            "schema_validity_rate": {"min": 95.0},
            "required_present_rate": {"min": 95.0},
            "doc_required_exact_match_rate": {"min": 80.0},
            "latency_p95_ms": {"max": 500.0},
        },
        params={
            "min_n_total": 10,
            "min_n_for_point_estimate": 200,
            "min_field_exact_match_rate": {"total": 90.0},
        },
    )


def _artifact(**overrides) -> EvalArtifact:
    payload = {
        "task": "extraction_sroie",
        "run_id": "r1",
        "run_dir": "/tmp/run",
        "deployment_key": "dep1",
        "deployment": {"provider": "openai", "model": "gpt"},
        "n_total": 250,
        "n_ok": 245,
        "schema_validity_rate": 99.0,
        "required_present_rate": 99.0,
        "doc_required_exact_match_rate": 95.0,
        "field_exact_match_rate": {"total": 95.0},
        "latency_p95_ms": 120.0,
    }
    payload.update(overrides)
    return EvalArtifact(summary=EvalSummary.model_validate(payload), results=None)


def test_all_thresholds_pass_enables_extract() -> None:
    d = decide_extract_enablement(
        _artifact(), thresholds=_thresholds(), thresholds_profile="extract/sroie"
    )

    assert d.pipeline == "extract_only"
    assert d.enable_extract is True
    assert d.ok() is True
    assert d.thresholds_profile == "extract/sroie"
    assert d.eval_task == "extraction_sroie"
    assert d.metrics["schema_validity_rate__gate_source"] == "point"


def test_missing_schema_validity_blocks_extract() -> None:
    d = decide_extract_enablement(
        _artifact(schema_validity_rate=None),
        thresholds=_thresholds(),
        thresholds_profile="extract/sroie",
    )

    assert d.enable_extract is False
    assert any(r.code == "missing_metric" for r in d.reasons)


def test_ci95_low_used_for_small_samples() -> None:
    d = decide_extract_enablement(
        _artifact(
            n_total=50,
            n_ok=45,
            schema_validity_rate=90.0,
            schema_validity_ci95_low=96.0,
            required_present_rate=95.0,
            required_present_ci95_low=95.0,
            doc_required_exact_match_rate=80.0,
            doc_required_exact_match_ci95_low=80.0,
        ),
        thresholds=_thresholds(),
        thresholds_profile="extract/sroie",
    )

    assert d.enable_extract is True
    assert d.metrics["schema_validity_rate__gate_source"] == "ci95_low"
    assert d.metrics["schema_validity_rate__gate_value"] == 96.0


def test_field_exact_match_missing_field_blocks() -> None:
    d = decide_extract_enablement(
        _artifact(field_exact_match_rate={"invoice_no": 100.0}),
        thresholds=_thresholds(),
        thresholds_profile="extract/sroie",
    )

    assert d.enable_extract is False
    assert any(
        r.code == "missing_metric" and "field_exact_match_rate.total" in r.message
        for r in d.reasons
    )


def test_latency_threshold_blocks_when_exceeded() -> None:
    d = decide_extract_enablement(
        _artifact(latency_p95_ms=900.0),
        thresholds=_thresholds(),
        thresholds_profile="extract/sroie",
    )

    assert d.enable_extract is False
    assert any(r.code == "latency_p95_ms_too_high" for r in d.reasons)


def test_missing_deployment_provenance_fails_closed() -> None:
    d = decide_extract_enablement(
        _artifact(deployment_key="", deployment={}),
        thresholds=_thresholds(),
        thresholds_profile="extract/sroie",
    )

    assert d.enable_extract is False
    assert any(r.code == "deployment_contract_error" for r in d.reasons)


def test_system_health_thresholds_can_block_extract() -> None:
    th = _thresholds().model_copy(
        update={
            "params": {
                "min_n_total": 1,
                "min_n_for_point_estimate": 1,
                "max_http_5xx_rate": 1.0,
                "max_timeout_rate": 1.0,
                "max_non_200_rate": 1.0,
                "min_field_exact_match_rate": {"total": 90.0},
            }
        }
    )
    d = decide_extract_enablement(
        _artifact(http_5xx_rate=5.0, timeout_rate=2.0, non_200_rate=3.0),
        thresholds=th,
        thresholds_profile="extract/sroie",
    )
    assert d.enable_extract is False
    assert sum(1 for r in d.reasons if r.code == "system_unhealthy") >= 1


def test_field_exact_match_non_numeric_is_parse_error() -> None:
    class Artifact:
        def __init__(self):
            self.summary = _artifact().summary.model_copy(
                update={"field_exact_match_rate": {"total": "bad"}}
            )

        def contract_issues(self):
            return []

    d = decide_extract_enablement(
        Artifact(),
        thresholds=_thresholds(),
        thresholds_profile="extract/sroie",
    )
    assert d.enable_extract is False
    assert any(r.code == "metric_parse_error" for r in d.reasons)


def test_missing_latency_metric_is_reason_when_threshold_present() -> None:
    d = decide_extract_enablement(
        _artifact(latency_p95_ms=None),
        thresholds=_thresholds(),
        thresholds_profile="extract/sroie",
    )
    assert d.enable_extract is False
    assert any(r.code == "missing_metric" and "latency_p95_ms" in r.message for r in d.reasons)


def test_contract_warn_and_info_become_warnings() -> None:
    class Issue:
        def __init__(self, severity: str, code: str):
            self.severity = severity
            self.code = code
            self.message = "m"
            self.context = {"k": "v"}

    class Artifact:
        def __init__(self):
            self.summary = _artifact().summary

        def contract_issues(self):
            return [Issue("warn", "c_warn"), Issue("info", "c_info")]

    d = decide_extract_enablement(
        Artifact(), thresholds=_thresholds(), thresholds_profile="extract/sroie"
    )
    assert any(w.code == "artifact_contract_warn" for w in d.warnings)
    assert any(w.code == "artifact_contract_info" for w in d.warnings)
