from __future__ import annotations

from typing import Any, Callable, Iterable

import pytest

from llm_eval.client.http_client import ExtractOk
from llm_eval.runners.base import EvalConfig, RunnerDeps
from llm_eval.runners.extraction_runner import ExtractionEvalRunner

from tests.fakes.fake_examples import ReceiptExample
from tests.fakes.fake_http_client import FakeHttpClient

RECEIPTS_KEY = "iter_voxel51_scanned_receipts"


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
async def test_extraction_runner_nested_contract_via_dataset_overrides(
    deps, fake_client: FakeHttpClient
):
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

    deps2 = _deps_with_datasets(deps, {RECEIPTS_KEY: _iter})

    fake_client.extract_queue.append(
        ExtractOk(
            schema_id="sroie_receipt_v1",
            model="m",
            data={"company": "ACME", "date": "2024-01-01", "total": "10.00"},
            cached=False,
            repair_attempted=False,
            latency_ms=15.0,
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

    assert isinstance(payload, dict)
    assert set(payload.keys()) >= {"summary", "results", "report_text", "config"}
    assert payload["summary"]["task"] == "extraction_sroie"
    assert payload["summary"]["run_id"] == "RUNID_TEST_0001"
    assert len(payload["results"]) == 1
    assert payload["results"][0]["model"] == "m"
