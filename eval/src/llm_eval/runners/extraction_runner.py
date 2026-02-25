# eval/src/llm_eval/runners/extraction_runner.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, cast

from llm_eval.client.http_client import ExtractErr, ExtractOk
from llm_eval.datasets.voxel51_scanned_receipts import (
    DEFAULT_SCHEMA_ID,
    iter_voxel51_scanned_receipts,
    ensure_fiftyone_ready,  # ✅ from your A change
)
from llm_eval.metrics.extraction_scoring import (
    ExtractAttempt,
    format_summary,
    score_document,
    summarize_extraction,
)
from llm_eval.runners.base import BaseEvalRunner, EvalConfig


class ExtractionEvalRunner(BaseEvalRunner):
    """
    Evaluates /v1/extract on Voxel51/scanned_receipts (SROIE-like receipts).

    Pure runner: no filesystem writes.
    Returns nested payload:
      - summary: aggregate metrics
      - results: per-doc rows
      - report_text: human-readable report
      - config: runner config snapshot

    """

    task_name = "extraction_sroie"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        config: Optional[EvalConfig] = None,
        *,
        schema_id: str = DEFAULT_SCHEMA_ID,
        split: str = "train",
        deps=None,
    ) -> None:
        super().__init__(base_url=base_url, api_key=api_key, config=config, deps=deps)
        self.schema_id = schema_id
        self.split = split

    # -------------------------
    # stratified sampling
    # -------------------------

    @staticmethod
    def _truthy(x: Any) -> bool:
        if isinstance(x, bool):
            return x
        if x is None:
            return False
        s = str(x).strip().lower()
        return s in ("1", "true", "yes", "y", "on")

    @staticmethod
    def _len_bin(text_len: int) -> str:
        """
        Simple fixed bins by text length.
        Chosen to be stable and understandable in a demo (not "academic").
        """
        if text_len <= 800:
            return "short"
        if text_len <= 2000:
            return "medium"
        return "long"

    def _select_examples_1p4(
        self,
        *,
        iter_fn: Any,
        max_examples: int,
        split: str,
        schema_id: str,
        stratify: bool,
        prefetch_multiplier: int = 5,
    ) -> List[Any]:
        """
        Returns a list of examples to evaluate.

        If stratify=False: stream-first behavior (up to max_examples).
        If stratify=True:
          - prefetch up to (max_examples * prefetch_multiplier)
          - bucket by text length bin
          - take equal share per bin, deterministically by doc id then length
        """
        max_examples = int(max_examples or 0)
        if max_examples <= 0:
            max_examples = 1

        if not stratify:
            out: List[Any] = []
            for ex in iter_fn(split=split, schema_id=schema_id, max_samples=max_examples):
                out.append(ex)
            return out

        cap = int(max_examples) * int(prefetch_multiplier)
        if cap < max_examples:
            cap = max_examples

        # Prefetch (deterministic: take first cap from iterator)
        pool: List[Any] = []
        for ex in iter_fn(split=split, schema_id=schema_id, max_samples=cap):
            pool.append(ex)

        if not pool:
            return []

        # Bucket by text length bin
        buckets: Dict[str, List[Any]] = {"short": [], "medium": [], "long": []}
        for ex in pool:
            t = getattr(ex, "text", "") or ""
            b = self._len_bin(len(t))
            buckets.setdefault(b, []).append(ex)

        # Stable ordering inside buckets (id first, then length)
        def _key(ex: Any) -> tuple[str, int]:
            ex_id = str(getattr(ex, "id", "") or "")
            ex_len = len(getattr(ex, "text", "") or "")
            return (ex_id, ex_len)

        for k in list(buckets.keys()):
            buckets[k].sort(key=_key)

        # Allocate roughly equal samples per non-empty bucket
        non_empty_bins = [k for k, v in buckets.items() if v]
        if not non_empty_bins:
            return []

        target = int(max_examples)
        per = max(1, target // len(non_empty_bins))
        remainder = target - (per * len(non_empty_bins))

        selected: List[Any] = []
        # First pass: take per from each bucket
        for b in non_empty_bins:
            take = min(per, len(buckets[b]))
            selected.extend(buckets[b][:take])

        # Second pass: distribute remainder by cycling bins, taking next unseen
        if remainder > 0:
            # Track next index per bin
            idx: Dict[str, int] = {b: min(per, len(buckets[b])) for b in non_empty_bins}
            bi = 0
            while remainder > 0 and len(selected) < target:
                b = non_empty_bins[bi % len(non_empty_bins)]
                bi += 1
                i = idx[b]
                if i < len(buckets[b]):
                    selected.append(buckets[b][i])
                    idx[b] = i + 1
                    remainder -= 1
                else:
                    # bin exhausted; if all bins exhausted we break
                    if all(idx[x] >= len(buckets[x]) for x in non_empty_bins):
                        break

        # If still short (some bins too small), fill from remaining pool (stable)
        if len(selected) < target:
            selected_ids = {str(getattr(ex, "id", "") or "") for ex in selected}
            pool_sorted = sorted(pool, key=_key)
            for ex in pool_sorted:
                ex_id = str(getattr(ex, "id", "") or "")
                if ex_id in selected_ids:
                    continue
                selected.append(ex)
                selected_ids.add(ex_id)
                if len(selected) >= target:
                    break

        return selected[:target]

    async def _run_impl(self) -> Dict[str, Any]:
        client = self.make_client()
        run_id = self.new_run_id()

        # Contract for sroie_receipt_v1.json
        fields = ["company", "address", "date", "total"]
        required = ["company", "date", "total"]

        attempts: List[ExtractAttempt] = []
        results: List[Dict[str, Any]] = []

        # --- dataset seam (patchable in tests) ---
        # Stable key: "iter_voxel51_scanned_receipts"
        iter_fn = self.get_dataset_callable("iter_voxel51_scanned_receipts", iter_voxel51_scanned_receipts)
        iter_fn = cast(Any, iter_fn)

        # ✅ Deterministic preflight: only do the heavy FiftyOne preload
        # when we are actually using the real Voxel51 iterator (not a test override).
        if iter_fn is iter_voxel51_scanned_receipts:
            ensure_fiftyone_ready()

        # We avoid touching EvalConfig contract here: use deps override (preferred),
        # or environment variable as a fallback, so this is a drop-in change.
        stratify = False
        prefetch_multiplier = 5

        # deps-based knobs (best: your BaseEvalRunner already carries deps)
        if isinstance(getattr(self, "deps", None), dict):
            stratify = self._truthy(self.deps.get("stratify_by_text_length", False))
            pm = self.deps.get("stratify_prefetch_multiplier")
            try:
                if pm is not None:
                    prefetch_multiplier = max(1, int(pm))
            except Exception:
                pass

        # env fallback (useful for demos without wiring)
        # LLM_EVAL_STRATIFY_TEXTLEN=1
        # LLM_EVAL_STRATIFY_PREFETCH_MULT=5
        if not stratify:
            import os

            stratify = self._truthy(os.getenv("LLM_EVAL_STRATIFY_TEXTLEN", "0"))
            try:
                prefetch_multiplier = max(
                    1, int(os.getenv("LLM_EVAL_STRATIFY_PREFETCH_MULT", str(prefetch_multiplier)))
                )
            except Exception:
                pass

        selected_examples = self._select_examples_1p4(
            iter_fn=iter_fn,
            max_examples=int(self.config.max_examples or 0),
            split=self.split,
            schema_id=self.schema_id,
            stratify=bool(stratify),
            prefetch_multiplier=int(prefetch_multiplier),
        )

        # Evaluate the selected examples
        for ex in selected_examples:
            resp = await client.extract(
                schema_id=ex.schema_id,
                text=ex.text,
                model=self.config.model_override,
                temperature=0.0,
                max_new_tokens=512,
                cache=False,
                repair=True,
            )

            if isinstance(resp, ExtractOk):
                attempt = ExtractAttempt(
                    doc_id=ex.id,
                    schema_id=ex.schema_id,
                    expected=ex.expected,
                    predicted=resp.data,
                    ok=True,
                    status_code=200,
                    error_code=None,
                    error_stage=None,
                    repair_attempted=resp.repair_attempted,
                    cached=resp.cached,
                    cache_layer=None,
                    latency_ms=resp.latency_ms,
                )
                model_id: Optional[str] = resp.model
                extra: Optional[Dict[str, Any]] = None
            else:
                assert isinstance(resp, ExtractErr)

                stage: Optional[str] = None
                if isinstance(resp.extra, dict):
                    stage_val = resp.extra.get("stage") or resp.extra.get("error_stage")
                    if stage_val is not None:
                        stage = str(stage_val)

                attempt = ExtractAttempt(
                    doc_id=ex.id,
                    schema_id=ex.schema_id,
                    expected=ex.expected,
                    predicted=None,
                    ok=False,
                    status_code=resp.status_code,
                    error_code=resp.error_code,
                    error_stage=stage,
                    repair_attempted=False,
                    cached=False,
                    cache_layer=None,
                    latency_ms=resp.latency_ms,
                )
                model_id = None
                extra = resp.extra if isinstance(resp.extra, dict) else None

            doc_score = score_document(
                attempt,
                fields=fields,
                required_fields=required,
                ignore_if_expected_missing=True,
            )

            results.append(
                {
                    "doc_id": ex.id,
                    "schema_id": ex.schema_id,
                    "ok": attempt.ok,
                    "status_code": attempt.status_code,
                    "error_code": attempt.error_code,
                    "error_stage": attempt.error_stage,
                    "extra": extra,
                    "repair_attempted": attempt.repair_attempted,
                    "cached": attempt.cached,
                    "latency_ms": attempt.latency_ms,
                    "model": model_id,
                    "expected": ex.expected,
                    "predicted": attempt.predicted,
                    "field_correct": doc_score.field_correct,
                    "required_present_non_null": doc_score.required_present_non_null,
                    "required_all_correct": doc_score.required_all_correct,
                }
            )

            attempts.append(attempt)

        summary_obj = summarize_extraction(
            attempts,
            fields=fields,
            required_fields=required,
            ignore_if_expected_missing=True,
        )

        summary = asdict(summary_obj)

        # NEW: attach server snapshot (deployment_key + loaded/default/effective model ids)
        server_snap = self.server_snapshot() or {"ok": False, "stage": "no_server_snapshot"}

        summary.update(
            {
                "task": self.task_name,
                "run_id": run_id,
                "dataset": "Voxel51/scanned_receipts",
                "split": self.split,
                "schema_id": self.schema_id,
                "base_url": self.base_url,
                "model_override": self.config.model_override,
                "max_examples": self.config.max_examples,
                "server": server_snap,
                "sampling": {
                    "stratified_by": "text_length_bin" if stratify else "none",
                    "bins": ["short", "medium", "long"] if stratify else None,
                    "prefetch_multiplier": int(prefetch_multiplier) if stratify else None,
                    "selected_n": int(len(selected_examples)),
                    "selected_ids": [str(getattr(ex, "id", "") or "") for ex in selected_examples],
                },
            }
        )

        # Put key correlation facts at the very top of the report
        deployment_key = None
        try:
            deployment_key = (
                server_snap.get("deployment", {}).get("deployment_key")
                if isinstance(server_snap, dict)
                else None
            )
        except Exception:
            deployment_key = None

        effective_mid = server_snap.get("effective_model_id") if isinstance(server_snap, dict) else None
        loaded_mid = server_snap.get("loaded_model_id") if isinstance(server_snap, dict) else None
        default_mid = server_snap.get("default_model_id") if isinstance(server_snap, dict) else None

        report_lines = [
            f"task={self.task_name}",
            f"run_id={run_id}",
            "dataset=Voxel51/scanned_receipts",
            f"split={self.split}",
            f"schema_id={self.schema_id}",
            f"base_url={self.base_url}",
            f"model_override={self.config.model_override}",
            f"max_examples={self.config.max_examples}",
            f"deployment_key={deployment_key}",
            f"effective_model_id={effective_mid}",
            f"loaded_model_id={loaded_mid}",
            f"default_model_id={default_mid}",
            f"sampling={'text_length_bin' if stratify else 'none'}",
            "",
            format_summary(summary_obj),
        ]
        report_text = "\n".join(report_lines)

        return {
            "summary": summary,
            "results": results,
            "report_text": report_text,
            "config": asdict(self.config),
        }


def make_extraction_runner(
    base_url: str,
    api_key: str,
    config: Optional[EvalConfig] = None,
) -> BaseEvalRunner:
    return ExtractionEvalRunner(
        base_url=base_url,
        api_key=api_key,
        config=config or EvalConfig(),
    )