from __future__ import annotations

import contextlib
import types

import pytest

from llm_server.core.errors import AppError
from llm_server.services.backends import transformers_backend as mod


class _FakeTensor:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (1, len(self.values))

    def to(self, _device):
        return self

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.values[idx]
        return self.values[idx]


class _FakeTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _FakeTokenizer()

    def __call__(self, prompt, return_tensors="pt", add_special_tokens=False):
        assert isinstance(prompt, str)
        return {"input_ids": _FakeTensor([10, 20])}

    def decode(self, token_ids, skip_special_tokens=True):
        return "decoded:" + ",".join(str(x) for x in token_ids)


class _FakeModelConfig:
    _name_or_path = "hf/path"


class _FakeModel:
    def __init__(self):
        self.device = "cpu"
        self.dtype = "float32"
        self.config = _FakeModelConfig()
        self.eval_called = False
        self.to_calls = []

    def eval(self):
        self.eval_called = True

    def to(self, dev):
        self.to_calls.append(dev)
        self.device = dev

    def generate(self, **kwargs):
        # input length is 2, generated ids append [30, 31]
        return [_FakeTensor([10, 20, 30, 31])]


class _FakeAutoModel:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _FakeModel()


def _install_fake_deps(monkeypatch, *, cuda_available=False, mps_available=False):
    class _MPS:
        @staticmethod
        def is_available():
            return mps_available

    fake_torch = types.SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        float32="float32",
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
        backends=types.SimpleNamespace(mps=_MPS()),
        no_grad=contextlib.nullcontext,
        __version__="2.fake",
    )
    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=_FakeAutoModel,
        AutoTokenizer=_FakeTokenizer,
        __version__="5.fake",
    )
    monkeypatch.setitem(mod.__dict__, "torch", fake_torch)
    monkeypatch.setitem(mod.__dict__, "transformers", fake_transformers)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)


def test_ensure_loaded_requires_hf_id(monkeypatch):
    _install_fake_deps(monkeypatch)
    b = mod.TransformersBackend(model_id="m1", cfg=mod.TransformersBackendConfig(hf_id=""))
    with pytest.raises(AppError) as e:
        b.ensure_loaded()
    assert e.value.code == "backend_config_invalid"


def test_ensure_loaded_and_model_info(monkeypatch):
    _install_fake_deps(monkeypatch, cuda_available=False, mps_available=False)
    b = mod.TransformersBackend(model_id="m1", cfg=mod.TransformersBackendConfig(hf_id="hf/test", device="auto"))
    b.ensure_loaded()
    assert b.is_loaded() is True

    info = b.model_info()
    assert info["loaded"] is True
    assert info["torch_version"] == "2.fake"
    assert info["transformers_version"] == "5.fake"
    assert info["hf_name_or_path"] == "hf/path"


def test_generate_rich_requires_loaded_and_non_empty_prompt():
    b = mod.TransformersBackend(model_id="m1", cfg=mod.TransformersBackendConfig(hf_id="hf/test"))
    with pytest.raises(AppError) as e1:
        b.generate_rich(prompt="")
    assert e1.value.code == "invalid_request"

    with pytest.raises(AppError) as e2:
        b.generate_rich(prompt="hello")
    assert e2.value.code == "backend_not_ready"


def test_generate_rich_success(monkeypatch):
    _install_fake_deps(monkeypatch)
    b = mod.TransformersBackend(model_id="m1", cfg=mod.TransformersBackendConfig(hf_id="hf/test"))
    b.ensure_loaded()

    out = b.generate_rich(prompt="hello", max_new_tokens=4, temperature=0.5, top_p=0.8, top_k=20)
    assert out.text == "decoded:30,31"
    assert out.timings.total_ms is not None
