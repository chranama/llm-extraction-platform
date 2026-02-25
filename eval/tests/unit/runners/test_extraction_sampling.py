from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pytest

from llm_eval.client.http_client import ExtractOk
from llm_eval.runners.base import EvalConfig, RunnerDeps
from llm_eval.runners.extraction_runner import ExtractionEvalRunner


@dataclass
class _Example:
    id: str
    schema_id: str
    text: str
    expected: dict[str, Any]


class _OneShotClient:
    def __init__(self) -> None:
        self.calls = 0

    async def extract(
        self,
        *,
        schema_id: str,
        text: str,
        model: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        cache: bool = True,
        repair: bool = True,
    ) -> ExtractOk:
        self.calls += 1
        return ExtractOk(
            schema_id=schema_id,
            model=model or "m",
            data={"company": "ACME", "date": "2024-01-01", "total": "10.00"},
            cached=False,
            repair_attempted=False,
            latency_ms=1.0,
        )


def _runner() -> ExtractionEvalRunner:
    deps = RunnerDeps(client_factory=lambda base_url, api_key: _OneShotClient())
    return ExtractionEvalRunner(
        base_url="http://svc",
        api_key="KEY",
        config=EvalConfig(max_examples=3),
        schema_id="sroie_receipt_v1",
        split="train",
        deps=deps,
    )


def test_select_examples_non_stratified_honors_limit():
    r = _runner()

    def _iter(*, split: str, schema_id: str, max_samples: int):
        assert split == "train"
        assert schema_id == "sroie_receipt_v1"
        xs = [
            _Example("a", schema_id, "x" * 10, {}),
            _Example("b", schema_id, "y" * 20, {}),
            _Example("c", schema_id, "z" * 30, {}),
        ]
        return xs[:max_samples]

    out = r._select_examples_1p4(
        iter_fn=_iter,
        max_examples=2,
        split="train",
        schema_id="sroie_receipt_v1",
        stratify=False,
    )
    assert [x.id for x in out] == ["a", "b"]


def test_select_examples_stratified_balances_bins_then_fills():
    r = _runner()

    def _iter(*, split: str, schema_id: str, max_samples: int):
        xs = [
            _Example("s1", schema_id, "a" * 100, {}),
            _Example("s2", schema_id, "b" * 200, {}),
            _Example("m1", schema_id, "c" * 1200, {}),
            _Example("l1", schema_id, "d" * 2600, {}),
        ]
        return xs[:max_samples]

    out = r._select_examples_1p4(
        iter_fn=_iter,
        max_examples=4,
        split="train",
        schema_id="sroie_receipt_v1",
        stratify=True,
        prefetch_multiplier=2,
    )
    ids = [x.id for x in out]
    assert len(ids) == 4
    assert "s1" in ids or "s2" in ids
    assert "m1" in ids
    assert "l1" in ids


@pytest.mark.asyncio
async def test_run_impl_env_stratify_and_bad_server_snapshot(monkeypatch: pytest.MonkeyPatch):
    r = _runner()
    fake_client = _OneShotClient()

    def _iter(*, split: str, schema_id: str, max_samples: int):
        assert max_samples >= 1
        return [
            _Example(
                "r1", schema_id, "text", {"company": "ACME", "date": "2024-01-01", "total": "10.00"}
            )
        ]

    class _BadDict(dict):
        def get(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setenv("LLM_EVAL_STRATIFY_TEXTLEN", "1")
    monkeypatch.setenv("LLM_EVAL_STRATIFY_PREFETCH_MULT", "bad")

    monkeypatch.setattr(r, "make_client", lambda: fake_client)
    monkeypatch.setattr(r, "new_run_id", lambda: "RID")
    monkeypatch.setattr(r, "get_dataset_callable", lambda _k, _d: _iter)
    monkeypatch.setattr(r, "server_snapshot", lambda: _BadDict())

    out = await r._run_impl()

    assert out["summary"]["run_id"] == "RID"
    assert out["summary"]["sampling"]["stratified_by"] == "text_length_bin"
    assert out["summary"]["sampling"]["prefetch_multiplier"] == 5
    assert fake_client.calls == 1
