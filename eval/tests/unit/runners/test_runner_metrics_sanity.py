from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import pytest

from llm_eval.client.http_client import ExtractOk
from llm_eval.runners.base import EvalConfig, RunnerDeps
from llm_eval.runners.extraction_runner import ExtractionEvalRunner

from tests.fakes.fake_examples import ReceiptExample
from tests.fakes.fake_http_client import FakeHttpClient


def _deps_with_datasets(
    base_deps: RunnerDeps,
    dataset_overrides: dict[str, Callable[..., Iterable[Any]]],
) -> RunnerDeps:
    return RunnerDeps(
        client_factory=base_deps.client_factory,
        run_id_factory=base_deps.run_id_factory,
        ensure_dir=base_deps.ensure_dir,
        open_fn=base_deps.open_fn,
        dataset_overrides=dataset_overrides,
    )


@pytest.mark.asyncio
async def test_extraction_metrics_sanity(deps: RunnerDeps, fake_client: FakeHttpClient):
    def _iter(split: str, schema_id: str, max_samples=None):
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

    deps2 = _deps_with_datasets(deps, {"iter_voxel51_scanned_receipts": _iter})

    fake_client.extract_queue.append(
        ExtractOk(
            schema_id="sroie_receipt_v1",
            model="m",
            data={"company": "ACME", "date": "2024-01-01", "total": "10.00"},
            cached=False,
            repair_attempted=False,
            latency_ms=11.0,
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
    summary = payload["summary"]

    for k in (
        "schema_validity_rate",
        "doc_required_exact_match_rate",
        "required_present_rate",
    ):
        if k in summary and summary[k] is not None:
            assert isinstance(summary[k], (int, float))
            assert not math.isnan(float(summary[k]))
            assert float(summary[k]) >= 0.0

    assert summary.get("task") == "extraction_sroie"
    assert summary.get("dataset") == "Voxel51/scanned_receipts"
    assert summary.get("split") == "train"
