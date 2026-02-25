from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Optional

import pytest

from llm_eval.client.http_client import ExtractErr
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
async def test_extraction_runner_error_row_includes_stage_and_latency(
    deps: RunnerDeps, fake_client: FakeHttpClient
):
    def _iter_receipts(*, split: str, schema_id: str, max_samples: Optional[int] = None):
        assert split == "train"
        xs = [
            ReceiptExample(
                id="r_err",
                schema_id=schema_id,
                text="Company: ACME\\nDate: 2024-01-01\\nTotal: 10.00",
                expected={"company": "ACME", "date": "2024-01-01", "total": "10.00"},
            )
        ]
        return xs[: (max_samples or len(xs))]

    deps2 = _mk_deps_with_overrides(deps, {"iter_voxel51_scanned_receipts": _iter_receipts})

    fake_client.extract_queue.append(
        ExtractErr(
            status_code=422,
            error_code="validation_error",
            message="bad json",
            extra={"stage": "schema_validate"},
            latency_ms=66.0,
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
    payload = await r.run()

    assert payload["summary"]["task"] == "extraction_sroie"
    assert len(payload["results"]) == 1
    row = payload["results"][0]

    assert row["doc_id"] == "r_err"
    assert row["schema_id"] == "sroie_receipt_v1"
    assert row["ok"] is False
    assert row["status_code"] == 422
    assert row["error_code"] == "validation_error"
    assert row["latency_ms"] == 66.0
    assert row.get("model") is None
    assert row.get("error_stage") == "schema_validate"
    assert row.get("predicted") is None
    assert isinstance(row.get("extra"), (dict, type(None)))
