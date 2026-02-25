from __future__ import annotations

import os
import sys
import time
import types
from dataclasses import dataclass
from typing import Any

import pytest

import llm_eval.datasets.voxel51_scanned_receipts as ds


@dataclass
class _Det:
    label: str
    bounding_box: list[float] | None = None


@dataclass
class _Poly:
    label: str
    points: list[list[list[float]]] | None = None


@dataclass
class _Sample:
    filepath: str | None = None
    text_detections: Any = None
    text_polygons: Any = None
    company: str | None = None
    address: str | None = None
    date: str | None = None
    total: str | None = None


def test_safe_str_and_bbox_sort_key_det():
    assert ds._safe_str(None) is None
    assert ds._safe_str("  ") is None
    assert ds._safe_str(" x ") == "x"
    assert ds._safe_str(123) == "123"

    d1 = _Det(label="a", bounding_box=[0.2, 0.1, 0.0, 0.0])
    d2 = _Det(label="b", bounding_box=[0.1, 0.1, 0.0, 0.0])
    d3 = _Det(label="c", bounding_box=None)
    assert ds._bbox_sort_key_det(d1) > ds._bbox_sort_key_det(d2)
    assert ds._bbox_sort_key_det(d3) == (1e9, 1e9)


def test_build_ocr_text_prefers_detections_sorted():
    dets = types.SimpleNamespace(
        detections=[
            _Det(label="second", bounding_box=[0.6, 0.5, 0.1, 0.1]),
            _Det(label="first", bounding_box=[0.1, 0.2, 0.1, 0.1]),
            _Det(label="  ", bounding_box=[0.2, 0.3, 0.1, 0.1]),
        ]
    )
    sample = _Sample(text_detections=dets)
    assert ds.build_ocr_text_from_sample(sample) == "first\nsecond"


def test_build_ocr_text_falls_back_to_polylines_sorted():
    polys = types.SimpleNamespace(
        polylines=[
            _Poly(label="p2", points=[[[0.5, 0.6], [0.8, 0.6]]]),
            _Poly(label="p1", points=[[[0.1, 0.2], [0.2, 0.2]]]),
            _Poly(label="", points=[[[0.0, 0.0], [0.0, 0.0]]]),
        ]
    )
    sample = _Sample(text_detections=types.SimpleNamespace(detections=[]), text_polygons=polys)
    assert ds.build_ocr_text_from_sample(sample) == "p1\np2"


def test_build_expected_from_sample():
    sample = _Sample(company=" ACME ", address=" 123 Main ", date="2024-01-01", total=" 10.00 ")
    out = ds.build_expected_from_sample(sample)
    assert out == {
        "company": "ACME",
        "address": "123 Main",
        "date": "2024-01-01",
        "total": "10.00",
    }


def test_fiftyone_import_probe_uses_importable_modules(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "fiftyone", types.ModuleType("fiftyone"))
    monkeypatch.setitem(sys.modules, "fiftyone.utils", types.ModuleType("fiftyone.utils"))
    hf = types.ModuleType("fiftyone.utils.huggingface")
    hf.load_from_hub = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "fiftyone.utils.huggingface", hf)

    ds._fiftyone_import_probe()  # should not raise


class _FakeProcess:
    def __init__(self, *, alive_after_join: bool, exitcode: int | None):
        self._alive = alive_after_join
        self.exitcode = exitcode
        self.killed = False

    def start(self):
        return None

    def join(self, _timeout: int):
        return None

    def is_alive(self) -> bool:
        return self._alive

    def kill(self):
        self.killed = True
        self._alive = False


class _FakeContext:
    def __init__(self, proc: _FakeProcess):
        self.proc = proc

    def Process(self, target, daemon=True):  # noqa: N802
        assert callable(target)
        assert daemon is True
        return self.proc


class _FakeMP:
    def __init__(self, proc: _FakeProcess):
        self.proc = proc

    def get_context(self, mode: str):
        assert mode == "spawn"
        return _FakeContext(self.proc)


def test_ensure_fiftyone_ready_success(monkeypatch: pytest.MonkeyPatch):
    proc = _FakeProcess(alive_after_join=False, exitcode=0)
    monkeypatch.setattr(ds, "mp", _FakeMP(proc), raising=False)
    monkeypatch.setattr(ds, "os", os, raising=False)
    monkeypatch.setattr(ds, "time", time, raising=False)
    monkeypatch.delenv("LLM_EVAL_FIFTYONE_TIMEOUT", raising=False)

    ds.ensure_fiftyone_ready(timeout_s=1)


def test_ensure_fiftyone_ready_timeout(monkeypatch: pytest.MonkeyPatch):
    proc = _FakeProcess(alive_after_join=True, exitcode=None)
    monkeypatch.setattr(ds, "mp", _FakeMP(proc), raising=False)
    monkeypatch.setattr(ds, "os", os, raising=False)
    monkeypatch.setattr(ds, "time", time, raising=False)

    with pytest.raises(RuntimeError, match="did not complete"):
        ds.ensure_fiftyone_ready(timeout_s=1)
    assert proc.killed is True


def test_ensure_fiftyone_ready_exitcode_failure(monkeypatch: pytest.MonkeyPatch):
    proc = _FakeProcess(alive_after_join=False, exitcode=2)
    monkeypatch.setattr(ds, "mp", _FakeMP(proc), raising=False)
    monkeypatch.setattr(ds, "os", os, raising=False)
    monkeypatch.setattr(ds, "time", time, raising=False)

    with pytest.raises(RuntimeError, match="import probe failed"):
        ds.ensure_fiftyone_ready(timeout_s=1)


def test_iter_voxel51_scanned_receipts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ds, "ensure_fiftyone_ready", lambda: None)

    samples = [
        _Sample(
            filepath="/tmp/a.png",
            text_detections=types.SimpleNamespace(
                detections=[_Det(label="A", bounding_box=[0.1, 0.1, 0, 0])]
            ),
            company="ACME",
            address="123 Main",
            date="2024-01-01",
            total="10.00",
        ),
        _Sample(
            filepath=None,
            text_detections=types.SimpleNamespace(
                detections=[_Det(label="B", bounding_box=[0.1, 0.1, 0, 0])]
            ),
            company="BIZ",
            address="456 State",
            date="2024-01-02",
            total="20.00",
        ),
    ]

    hf = types.ModuleType("fiftyone.utils.huggingface")
    hf.load_from_hub = lambda _name: samples
    monkeypatch.setitem(sys.modules, "fiftyone.utils.huggingface", hf)

    out = list(ds.iter_voxel51_scanned_receipts(split="train", schema_id="sid", max_samples=2))

    assert len(out) == 2
    assert out[0].id == "/tmp/a.png"
    assert out[0].schema_id == "sid"
    assert out[0].text == "A"
    assert out[1].id == "train:1"
    assert out[1].expected["company"] == "BIZ"
