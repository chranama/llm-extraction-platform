from __future__ import annotations

from types import SimpleNamespace

from llm_policy.onboarding.runner import (
    _deployment_from_artifact,
    _print_decision,
    _short_ctx,
    evaluate_model_onboarding,
)
from llm_policy.types.decision import Decision, DecisionReason, DecisionStatus, DecisionWarning
from llm_policy.types.eval_artifact import EvalArtifact, EvalSummary


def _artifact() -> EvalArtifact:
    summary = EvalSummary.model_validate(
        {
            "task": "extraction_sroie",
            "run_id": "r1",
            "run_dir": "/tmp/run",
            "n_total": 10,
            "n_ok": 9,
            "deployment_key": "dep1",
            "deployment": {"provider": "openai", "model": "gpt-4o-mini"},
        }
    )
    return EvalArtifact(summary=summary, results=None)


def test_short_ctx_truncates_long_context() -> None:
    s = _short_ctx({"a": "x" * 2000}, max_len=40)
    assert s.endswith("...")
    assert len(s) == 40


def test_short_ctx_non_dict_and_unserializable_dict() -> None:
    assert _short_ctx(None) == ""
    s = _short_ctx({"x": object()})
    assert "object" in s


def test_deployment_from_artifact_extracts_values() -> None:
    dep, dep_key = _deployment_from_artifact(_artifact())
    assert dep_key == "dep1"
    assert dep == {"provider": "openai", "model": "gpt-4o-mini"}


def test_deployment_from_artifact_without_summary_returns_none() -> None:
    dep, dep_key = _deployment_from_artifact(SimpleNamespace(summary=None))
    assert dep is None
    assert dep_key is None


def test_print_decision_renders_reason_warning_and_provenance(capsys) -> None:
    d = Decision(
        policy="p",
        pipeline="extract_only",
        status=DecisionStatus.allow,
        enable_extract=True,
        reasons=[DecisionReason(code="r1", message="reason", context={"k": "v"})],
        warnings=[DecisionWarning(code="w1", message="warn", context={"k2": "v2"})],
        metrics={"deployment_key": "dep1", "deployment": {"provider": "openai"}},
    )
    _print_decision(d)
    out = capsys.readouterr().out
    assert "MODEL ONBOARDING" in out
    assert "[r1] reason" in out
    assert "[w1] warn" in out
    assert "deployment_key: dep1" in out


def test_print_decision_tolerates_metrics_property_errors(capsys) -> None:
    class BadDecision:
        reasons = []
        warnings = []
        enable_extract = True
        status = "allow"
        pipeline = "extract_only"

        @property
        def metrics(self):
            raise RuntimeError("boom")

    _print_decision(BadDecision())  # no exception
    assert "MODEL ONBOARDING" in capsys.readouterr().out


def test_evaluate_model_onboarding_wires_decision_and_provenance(monkeypatch) -> None:
    artifact = _artifact()

    base_decision = Decision(
        policy="extract_enablement",
        pipeline="extract_only",
        status=DecisionStatus.allow,
        enable_extract=True,
        metrics={"existing": 1},
    )

    monkeypatch.setattr(
        "llm_policy.onboarding.runner.load_eval_artifact", lambda *_a, **_k: artifact
    )
    monkeypatch.setattr(
        "llm_policy.onboarding.runner.load_extract_thresholds",
        lambda cfg, profile: ("extract/default", object()),
    )
    monkeypatch.setattr(
        "llm_policy.onboarding.runner.decide_extract_enablement",
        lambda *_a, **_k: base_decision,
    )

    printed = {"called": False}
    monkeypatch.setattr(
        "llm_policy.onboarding.runner._print_decision",
        lambda d: printed.__setitem__("called", True),
    )

    res = evaluate_model_onboarding(
        eval_run_dir="latest",
        threshold_profile="extract/default",
        thresholds_root="/tmp/thresholds",
        policy_name="model_onboarding",
        pipeline="extract_plus_generate_clamp",
        verbose=True,
    )

    assert printed["called"] is True
    assert res.thresholds_profile == "extract/default"
    assert res.deployment_key == "dep1"
    assert res.deployment == {"provider": "openai", "model": "gpt-4o-mini"}

    assert res.decision.policy == "model_onboarding"
    assert res.decision.pipeline == "extract_plus_generate_clamp"
    assert res.decision.metrics["existing"] == 1
    assert res.decision.metrics["deployment_key"] == "dep1"
    assert res.decision.metrics["deployment"] == {"provider": "openai", "model": "gpt-4o-mini"}


def test_evaluate_model_onboarding_verbose_false_skips_print(monkeypatch) -> None:
    artifact = _artifact()
    base_decision = Decision(
        policy="extract_enablement",
        pipeline="extract_only",
        status=DecisionStatus.allow,
        enable_extract=True,
        metrics={},
    )

    monkeypatch.setattr(
        "llm_policy.onboarding.runner.load_eval_artifact", lambda *_a, **_k: artifact
    )
    monkeypatch.setattr(
        "llm_policy.onboarding.runner.load_extract_thresholds",
        lambda cfg, profile: ("extract/default", object()),
    )
    monkeypatch.setattr(
        "llm_policy.onboarding.runner.decide_extract_enablement",
        lambda *_a, **_k: base_decision,
    )

    printed = {"called": False}
    monkeypatch.setattr(
        "llm_policy.onboarding.runner._print_decision",
        lambda d: printed.__setitem__("called", True),
    )

    evaluate_model_onboarding(verbose=False)
    assert printed["called"] is False


def test_evaluate_model_onboarding_tolerates_metrics_copy_error(monkeypatch) -> None:
    artifact = _artifact()

    class FakeDecision:
        def __init__(self):
            self.policy = "extract_enablement"
            self.pipeline = "extract_only"
            self.metrics = object()

        def model_copy(self, update=None):
            if update and "metrics" in update:
                raise RuntimeError("cannot copy metrics")
            if update:
                for k, v in update.items():
                    setattr(self, k, v)
            return self

    monkeypatch.setattr(
        "llm_policy.onboarding.runner.load_eval_artifact", lambda *_a, **_k: artifact
    )
    monkeypatch.setattr(
        "llm_policy.onboarding.runner.load_extract_thresholds",
        lambda cfg, profile: ("extract/default", object()),
    )
    monkeypatch.setattr(
        "llm_policy.onboarding.runner.decide_extract_enablement",
        lambda *_a, **_k: FakeDecision(),
    )

    res = evaluate_model_onboarding(verbose=False)
    assert res.deployment_key == "dep1"
