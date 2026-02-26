from __future__ import annotations

from llm_policy.reports.writer import render_decision_md, render_decision_text
from llm_policy.types.decision import (
    Decision,
    DecisionReason,
    DecisionStatus,
    DecisionWarning,
)


def test_render_decision_text_includes_core_fields():
    d = Decision(
        policy="extract_enablement",
        pipeline="extract_only",
        status=DecisionStatus.allow,
        enable_extract=True,
        thresholds_profile="extract/sroie",
        reasons=[],
        warnings=[],
        metrics={"n_total": 10, "schema_validity_rate": 99.0},
    )

    out = render_decision_text(d)

    assert "policy=extract_enablement" in out
    assert "thresholds_profile=extract/sroie" in out
    assert "enable_extract=True" in out
    assert "ok=True" in out


def test_render_decision_text_and_md_reasons_warnings_and_provenance():
    d = Decision(
        policy="extract_enablement",
        pipeline="extract_only",
        status=DecisionStatus.deny,
        enable_extract=False,
        reasons=[
            DecisionReason(code="schema_validity_too_low", message="too low", context={"cur": 0.8}),
            {
                "code": "missing_metric",
                "message": "required_present_rate missing",
                "context": {"k": "v"},
            },
        ],
        warnings=[
            DecisionWarning(code="insufficient_sample_size", message="low N", context={"n": 3}),
        ],
        metrics={"deployment_key": "dep1", "deployment": {"region": "us"}, "n_total": 3},
    )

    out = render_decision_text(d)
    assert "REASONS:" in out
    assert "schema_validity_too_low" in out
    assert "missing_metric" in out
    assert "WARNINGS:" in out
    assert "insufficient_sample_size" in out
    assert "PROVENANCE:" in out

    md = render_decision_md(d)
    assert "## Reasons" in md
    assert "**schema_validity_too_low**" in md
    assert "## Warnings" in md
    assert "**insufficient_sample_size**" in md
    assert "## Provenance" in md
