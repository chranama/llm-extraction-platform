from __future__ import annotations

import re
from typing import Any

import pytest

from llm_eval.runners.base import BaseEvalRunner, EvalConfig, RunnerDeps


class _Runner(BaseEvalRunner):
    task_name = "r"

    async def _run_impl(self) -> Any:
        return {"ok": True, "snap": self.server_snapshot()}


def _deps(client: Any) -> RunnerDeps:
    return RunnerDeps(client_factory=lambda base_url, api_key: client)


def test_default_run_id_and_default_ensure_dir(tmp_path):
    rid = BaseEvalRunner.__dict__["new_run_id"]
    # sanity only: default function yields UTC-ish id
    from llm_eval.runners import base as b

    s = b._default_run_id()
    assert re.match(r"^\d{8}T\d{6}Z$", s)

    p = tmp_path / "a" / "b"
    b._default_ensure_dir(str(p))
    assert p.exists()


def test_helper_branches_for_string_and_safe_get():
    assert BaseEvalRunner._as_str(" ") is None
    assert BaseEvalRunner._safe_get({"a": 1}, "a", "b") is None

    assert BaseEvalRunner._compute_effective_model_id({"runtime_default_model_id": "m1"}) == "m1"
    assert BaseEvalRunner._compute_effective_model_id({"default_model_id": "m2"}) == "m2"


@pytest.mark.asyncio
async def test_preflight_modelz_missing_exception_bad_shape_and_model_block():
    class _NoModelz:
        pass

    r1 = _Runner(base_url="http://svc", api_key="k", deps=_deps(_NoModelz()))
    out1 = await r1._preflight_modelz(r1.make_client())
    assert out1["stage"] == "client_missing_modelz"

    class _Raises:
        async def modelz(self):
            raise RuntimeError("boom")

    r2 = _Runner(base_url="http://svc", api_key="k", deps=_deps(_Raises()))
    out2 = await r2._preflight_modelz(r2.make_client())
    assert out2["stage"] == "modelz_exception"

    class _Bad:
        async def modelz(self):
            return 123

    r3 = _Runner(base_url="http://svc", api_key="k", deps=_deps(_Bad()))
    out3 = await r3._preflight_modelz(r3.make_client())
    assert out3["stage"] == "modelz_bad_shape"

    class _Good:
        async def modelz(self):
            return {
                "status": "ready",
                "deployment": {"deployment_key": "dep"},
                "model": {"required": True, "status": "ok", "ok": True},
            }

    r4 = _Runner(base_url="http://svc", api_key="k", deps=_deps(_Good()))
    out4 = await r4._preflight_modelz(r4.make_client())
    assert out4["stage"] == "modelz_ok"
    assert out4["deployment"]["deployment_key"] == "dep"
    assert out4["model"]["required"] is True


@pytest.mark.asyncio
async def test_preflight_modelz_getattr_dict_raises_branch():
    class _Weird:
        async def modelz(self):
            class _Obj:
                def __getattribute__(self, name: str):
                    if name == "__dict__":
                        raise RuntimeError("no dict")
                    return object.__getattribute__(self, name)

            return _Obj()

    r = _Runner(base_url="http://svc", api_key="k", deps=_deps(_Weird()))
    out = await r._preflight_modelz(r.make_client())
    assert out["stage"] == "modelz_bad_shape"


@pytest.mark.asyncio
async def test_run_sets_preflight_failed_when_make_client_raises():
    r = _Runner(base_url="http://svc", api_key="k", deps=_deps(object()))
    r.make_client = lambda: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[assignment]

    out = await r.run(max_examples=1, model_override="m")
    assert out["ok"] is True
    assert out["snap"]["stage"] == "modelz_preflight_failed"
    assert r.config == EvalConfig(max_examples=1, model_override="m")
