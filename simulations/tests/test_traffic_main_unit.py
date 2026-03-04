from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import simulations.traffic.main as tm


class _Resp:
    def __init__(self, *, status: int, json_payload=None, text: str = ""):
        self.status = int(status)
        self.json = json_payload
        self.text = text
        self.elapsed_ms = 1.0


def _args(**overrides):
    base = dict(
        model_id="demo-model",
        schema_id="sroie_receipt_v1",
        text="Vendor: ACME\nTotal: 42.13",
        max_new_tokens=128,
        temperature=0.0,
        cache=True,
        repair=True,
        expect="any",
        expect_code=None,
        expect_model_extract="any",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _rt():
    return SimpleNamespace(
        api_key="k",
        base_url="http://127.0.0.1:8000",
        timeout_s=20.0,
        dry_run=False,
    )


def _proof_args(**overrides):
    base = dict(
        model_id="demo-model",
        artifact_models_yaml=None,
        artifact_models_profile=None,
        expect_policy_source="any",
        expect_policy_enable_extract="any",
        expect_model_extract="any",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_extract_gate_check_allow_success(monkeypatch, capsys):
    class _Client:
        cfg = SimpleNamespace(base_url="http://127.0.0.1:8000")

        def get_models(self):
            return _Resp(
                status=200,
                json_payload={
                    "default_model": "demo-model",
                    "deployment_capabilities": {"extract": True, "generate": True},
                    "models": [{"id": "demo-model", "capabilities": {"extract": True, "generate": True}}],
                },
            )

        def post_extract(self, **kwargs):
            assert kwargs["model"] == "demo-model"
            return _Resp(status=200, json_payload={"ok": True, "model": "demo-model"})

    monkeypatch.setattr(tm, "_client_from_rt", lambda rt, model=None: _Client())

    rc = tm._extract_gate_check(
        _rt(),
        _args(expect="allow", expect_model_extract="true"),
    )
    assert rc == 0

    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["extract_probe"]["status"] == 200
    assert out["models"]["model_extract_capability"] is True


def test_extract_gate_check_block_expectation_failure(monkeypatch):
    class _Client:
        cfg = SimpleNamespace(base_url="http://127.0.0.1:8000")

        def get_models(self):
            return _Resp(
                status=200,
                json_payload={
                    "default_model": "demo-model",
                    "deployment_capabilities": {"extract": True, "generate": True},
                    "models": [{"id": "demo-model", "capabilities": {"extract": True, "generate": True}}],
                },
            )

        def post_extract(self, **kwargs):
            return _Resp(status=200, json_payload={"ok": True})

    monkeypatch.setattr(tm, "_client_from_rt", lambda rt, model=None: _Client())

    with pytest.raises(tm.SimError):
        tm._extract_gate_check(
            _rt(),
            _args(expect="block"),
        )


def test_runtime_proof_offline_policy_and_caps_success(monkeypatch, capsys):
    class _Client:
        cfg = SimpleNamespace(base_url="http://127.0.0.1:8000")

        def _request(self, method, path):
            assert method == "GET"
            assert path == "/modelz"
            return _Resp(
                status=200,
                json_payload={
                    "policy": {"source_path": None, "enable_extract": None},
                    "deployment": {"profiles": {"models_profile_selected": "host-transformers"}, "deployment_key": "dk-1"},
                    "assessed_gate": {"snapshot": {"selected_deployment_key": "dk-1"}},
                },
            )

        def get_models(self):
            return _Resp(
                status=200,
                json_payload={
                    "default_model": "demo-model",
                    "deployment_capabilities": {"extract": True, "generate": True},
                    "models": [{"id": "demo-model", "capabilities": {"extract": True, "generate": True}}],
                },
            )

    monkeypatch.setattr(tm, "_client_from_rt", lambda rt, model=None: _Client())

    rc = tm._runtime_proof(
        _rt(),
        _proof_args(
            expect_policy_source="none",
            expect_policy_enable_extract="none",
            expect_model_extract="true",
        ),
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["runtime"]["policy_source_path"] is None


def test_runtime_proof_artifact_extract_mismatch_fails(monkeypatch, tmp_path: Path):
    artifact = tmp_path / "models.yaml"
    artifact.write_text(
        """
profiles:
  host-transformers:
    models:
      - id: demo-model
        capabilities:
          extract: false
        assessment:
          deployment_key: dk-1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    class _Client:
        cfg = SimpleNamespace(base_url="http://127.0.0.1:8000")

        def _request(self, method, path):
            return _Resp(
                status=200,
                json_payload={
                    "policy": {"source_path": None, "enable_extract": None},
                    "deployment": {"profiles": {"models_profile_selected": "host-transformers"}, "deployment_key": "dk-1"},
                    "assessed_gate": {"snapshot": {"selected_deployment_key": "dk-1"}},
                },
            )

        def get_models(self):
            return _Resp(
                status=200,
                json_payload={
                    "default_model": "demo-model",
                    "deployment_capabilities": {"extract": True, "generate": True},
                    "models": [{"id": "demo-model", "capabilities": {"extract": True, "generate": True}}],
                },
            )

    monkeypatch.setattr(tm, "_client_from_rt", lambda rt, model=None: _Client())

    with pytest.raises(tm.SimError):
        tm._runtime_proof(
            _rt(),
            _proof_args(
                artifact_models_yaml=str(artifact),
                artifact_models_profile="host-transformers",
            ),
        )
