from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Optional

import pytest

from llm_eval.client.http_client import ExtractOk
from llm_eval.runners.base import EvalConfig, RunnerDeps
from llm_eval.runners.extraction_runner import ExtractionEvalRunner

from tests.fakes.fake_examples import ReceiptExample
from tests.fakes.fake_http_client import FakeHttpClient


def _mk_deps_with_overrides(
    base: RunnerDeps,
    overrides: dict[str, Callable[..., Iterable[Any]]],
) -> RunnerDeps:
    return RunnerDeps(
        client_factory=base.client_factory,
        run_id_factory=base.run_id_factory,
        ensure_dir=base.ensure_dir,
        open_fn=base.open_fn,
        dataset_overrides=overrides,
    )


@pytest.mark.asyncio
async def test_extraction_runner_extract_call_shape(deps: RunnerDeps, fake_client: FakeHttpClient):
    def _iter_receipts(*, split: str, schema_id: str, max_samples: Optional[int] = None):
        assert split == "train"
        xs = [
            ReceiptExample(
                id="r1",
                schema_id=schema_id,
                text="Company: ACME\\nDate: 2024-01-01\\nTotal: 10.00",
                expected={"company": "ACME", "date": "2024-01-01", "total": "10.00"},
            )
        ]
        return xs[: (max_samples or len(xs))]

    deps2 = _mk_deps_with_overrides(deps, {"iter_voxel51_scanned_receipts": _iter_receipts})

    fake_client.extract_queue.append(
        ExtractOk(
            schema_id="sroie_receipt_v1",
            model="mX",
            data={"company": "ACME", "date": "2024-01-01", "total": "10.00"},
            cached=False,
            repair_attempted=True,
            latency_ms=1.0,
        )
    )

    r = ExtractionEvalRunner(
        base_url="http://svc",
        api_key="KEY",
        config=EvalConfig(max_examples=1, model_override="mX"),
        schema_id="sroie_receipt_v1",
        split="train",
        deps=deps2,
    )
    await r.run()

    assert len(fake_client.extract_calls) == 1
    call = fake_client.extract_calls[0]

    assert call["schema_id"] == "sroie_receipt_v1"
    assert isinstance(call.get("text"), str)
    assert call["model"] == "mX"
    assert call["max_new_tokens"] == 512
    assert call["temperature"] == 0.0
    assert call["cache"] is False
    assert call["repair"] is True
