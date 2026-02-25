from __future__ import annotations

from pathlib import Path

import pytest

import llm_eval.io.run_pointers as rp


def test_env_flag_parsing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("X", "true")
    assert rp._env_flag("X", default=False) is True

    monkeypatch.setenv("X", "0")
    assert rp._env_flag("X", default=True) is False

    monkeypatch.setenv("X", "invalid")
    assert rp._env_flag("X", default=True) is True

    monkeypatch.delenv("X", raising=False)
    assert rp._env_flag("X", default=False) is False


def test_pointer_out_path_for_task_default_and_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EVAL_LATEST_PATH", raising=False)
    p = rp.pointer_out_path_for_task("task")
    assert p.as_posix().endswith("eval_out/latest.json")

    monkeypatch.setenv("EVAL_LATEST_PATH", "~/custom/latest.json")
    p2 = rp.pointer_out_path_for_task("task")
    assert p2.as_posix().endswith("custom/latest.json")


def test_should_write_eval_latest_pointer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EVAL_WRITE_LATEST", "1")
    assert rp.should_write_eval_latest_pointer(default=False) is True

    monkeypatch.setenv("EVAL_WRITE_LATEST", "off")
    assert rp.should_write_eval_latest_pointer(default=True) is False


def test_build_eval_run_pointer_delegates(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def _fake_builder(**kwargs):
        captured.update(kwargs)
        return {"ok": True, **kwargs}

    monkeypatch.setattr(rp, "build_eval_run_pointer_payload_v1", _fake_builder)

    out = rp.build_eval_run_pointer(
        task="t",
        run_id="r",
        run_dir="rd",
        summary_path="sp",
        base_url="http://svc",
        model_override="m",
        schema_id="sid",
        max_examples=3,
        notes={"n": 1},
        store="fs",
    )

    assert out["ok"] is True
    assert captured["task"] == "t"
    assert captured["summary_path"] == "sp"


def test_write_eval_latest_pointer_with_and_without_out_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rp, "build_eval_run_pointer", lambda **kwargs: {"payload": kwargs})

    written = {}

    def _fake_write(path: Path, payload: dict):
        written["path"] = path
        written["payload"] = payload
        return path

    monkeypatch.setattr(rp, "write_eval_run_pointer", _fake_write)

    p = rp.write_eval_latest_pointer(
        task="t", run_id="r", run_dir="rd", summary_path="sp", out_path="/tmp/p.json"
    )
    assert str(p).endswith("/tmp/p.json")
    assert written["payload"]["payload"]["task"] == "t"

    monkeypatch.setattr(rp, "pointer_out_path_for_task", lambda _task: Path("/tmp/default.json"))
    p2 = rp.write_eval_latest_pointer(task="t2", run_id="r2", run_dir="rd2", summary_path="sp2")
    assert str(p2) == "/tmp/default.json"


def test_read_eval_latest_pointer_delegates(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rp, "read_eval_run_pointer", lambda p: {"ok": True, "path": str(p)})
    monkeypatch.setattr(rp, "pointer_out_path_for_task", lambda _task: Path("/tmp/default.json"))

    out1 = rp.read_eval_latest_pointer(task="t")
    assert out1["ok"] is True
    assert out1["path"] == "/tmp/default.json"

    out2 = rp.read_eval_latest_pointer(task="t", path="/tmp/explicit.json")
    assert out2["path"].endswith("/tmp/explicit.json")
