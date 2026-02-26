from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from llm_policy.io.models_yaml_patch import PatchChange, PatchResult
from llm_policy.onboarding.demo import apply_model_onboarding


class _DecisionLike:
    def __init__(
        self, *, enable_extract, ok: bool, policy: str = "p1", pipeline: str = "extract_only"
    ):
        self.enable_extract = enable_extract
        self.policy = policy
        self.pipeline = pipeline
        self._ok = ok

    def ok(self) -> bool:
        return self._ok


def test_apply_model_onboarding_refuses_non_boolean_enable(monkeypatch, tmp_path: Path) -> None:
    decision = _DecisionLike(enable_extract=None, ok=False)
    res_stub = SimpleNamespace(
        decision=decision,
        thresholds_profile="extract/default",
        eval_run_dir="/tmp/run",
        deployment={"provider": "openai"},
        deployment_key="dep1",
    )

    monkeypatch.setattr(
        "llm_policy.onboarding.demo.evaluate_model_onboarding", lambda **_k: res_stub
    )

    called = {"patch": False}

    def _never_called(**_kwargs):
        called["patch"] = True
        raise AssertionError("patch should not be called")

    monkeypatch.setattr(
        "llm_policy.onboarding.demo.patch_models_yaml_extract_capability", _never_called
    )

    out = apply_model_onboarding(models_yaml=tmp_path / "models.yaml", model_id="m1", verbose=False)

    assert out.ok is False
    assert out.patch_ok is False
    assert out.enable_extract is None
    assert called["patch"] is False


def test_apply_model_onboarding_passes_provenance_to_patch(monkeypatch, tmp_path: Path) -> None:
    decision = _DecisionLike(
        enable_extract=True,
        ok=True,
        policy="model_onboarding",
        pipeline="extract_plus_generate_clamp",
    )
    res_stub = SimpleNamespace(
        decision=decision,
        thresholds_profile="extract/default",
        eval_run_dir="/tmp/run",
        deployment={"provider": "openai", "model": "gpt-4o-mini"},
        deployment_key="dep1",
    )
    monkeypatch.setattr(
        "llm_policy.onboarding.demo.evaluate_model_onboarding", lambda **_k: res_stub
    )

    captured = {}

    def _patch(path, model_id, enable, **kwargs):
        captured["path"] = str(path)
        captured["model_id"] = model_id
        captured["enable"] = enable
        captured.update(kwargs)
        return PatchResult(
            ok=True,
            changed=True,
            message="patched",
            path=str(path),
            model_id=model_id,
            enable=enable,
            profile=kwargs.get("profile"),
            changes=(PatchChange(scope="base", model_id=model_id, before=False, after=True),),
        )

    monkeypatch.setattr("llm_policy.onboarding.demo.patch_models_yaml_extract_capability", _patch)

    out = apply_model_onboarding(
        models_yaml=tmp_path / "models.yaml",
        model_id="m1",
        eval_run_dir="latest",
        threshold_profile="extract/default",
        verbose=False,
    )

    assert out.ok is True
    assert out.patch_ok is True
    assert out.changed is True
    assert out.deployment_key == "dep1"
    assert captured["deployment_key"] == "dep1"
    assert captured["deployment"] == {"provider": "openai", "model": "gpt-4o-mini"}
    assert captured["assessed_by"] == "model_onboarding"
    assert captured["assessed_pipeline"] == "extract_plus_generate_clamp"


def test_apply_model_onboarding_verbose_prints_changes(monkeypatch, tmp_path: Path, capsys) -> None:
    decision = _DecisionLike(enable_extract=False, ok=True, policy="p1", pipeline="extract_only")
    res_stub = SimpleNamespace(
        decision=decision,
        thresholds_profile="extract/default",
        eval_run_dir="/tmp/run",
        deployment={"provider": "openai"},
        deployment_key="dep1",
    )
    monkeypatch.setattr(
        "llm_policy.onboarding.demo.evaluate_model_onboarding", lambda **_k: res_stub
    )

    monkeypatch.setattr(
        "llm_policy.onboarding.demo.patch_models_yaml_extract_capability",
        lambda path, model_id, enable, **kwargs: PatchResult(
            ok=True,
            changed=False,
            message="noop",
            path=str(path),
            model_id=model_id,
            enable=enable,
            profile=kwargs.get("profile"),
            changes=(),
        ),
    )

    _ = apply_model_onboarding(models_yaml=tmp_path / "models.yaml", model_id="m1", verbose=True)
    out = capsys.readouterr().out
    assert "MODEL ONBOARDING" in out
    assert "deployment_key: dep1" in out
    assert "changes: (none)" in out
