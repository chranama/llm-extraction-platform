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
async def test_receipts_uses_dataset_override_and_passes_schema_id(
    deps: RunnerDeps, fake_client: FakeHttpClient
):
    calls: dict[str, Any] = {}

    def _iter_receipts(*, split: str, schema_id: str, max_samples: Optional[int] = None):
        calls["split"] = split
        calls["schema_id"] = schema_id
        calls["max_samples"] = max_samples
        return [
            ReceiptExample(
                id="r1",
                schema_id=schema_id,
                text="Company: ACME\\nDate: 2024-01-01\\nTotal: 10.00",
                expected={"company": "ACME", "date": "2024-01-01", "total": "10.00"},
            )
        ]

    deps2 = _mk_deps_with_overrides(deps, {"iter_voxel51_scanned_receipts": _iter_receipts})

    fake_client.extract_queue.append(
        ExtractOk(
            schema_id="sroie_receipt_v1",
            model="m",
            data={"company": "ACME", "date": "2024-01-01", "total": "10.00"},
            cached=False,
            repair_attempted=False,
            latency_ms=1.0,
        )
    )

    r = ExtractionEvalRunner(
        base_url="http://x",
        api_key="k",
        config=EvalConfig(max_examples=1),
        schema_id="sroie_receipt_v1",
        split="train",
        deps=deps2,
    )
    payload = await r.run()

    assert calls["split"] == "train"
    assert calls["schema_id"] == "sroie_receipt_v1"
    assert calls["max_samples"] == 1
    assert payload["summary"]["task"] == "extraction_sroie"
    assert len(payload["results"]) == 1
