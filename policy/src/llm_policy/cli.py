# policy/src/llm_policy/cli.py
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

from llm_contracts.schema import validate_internal  # type: ignore

from llm_policy.config import PolicyConfig, load_extract_thresholds, load_generate_thresholds
from llm_policy.io.eval_runs import load_eval_artifact
from llm_policy.io.generate_slo import read_generate_slo_snapshot_result
from llm_policy.io.policy_decisions import (
    POLICY_DECISION_SCHEMA,
    default_policy_out_path,
    write_policy_decision_artifact,
)
from llm_policy.policies.extract_enablement import decide_extract_enablement
from llm_policy.policies.generate_slo_clamp import decide_generate_slo_clamp, thresholds_from_mapping
from llm_policy.reports.writer import render_decision_md, render_decision_text
from llm_policy.types.decision import (
    Decision,
    DecisionReason,
    DecisionWarning,
    PipelineType,
)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="policy",
        description="Policy engine for gating and shaping LLM capabilities.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def _add_common(subp: argparse.ArgumentParser) -> None:
        subp.add_argument(
            "--pipeline",
            required=True,
            choices=[
                "extract_only",
                "generate_clamp_only",
                "extract_plus_generate_clamp",
            ],
            help="Policy pipeline to execute",
        )

        subp.add_argument(
            "--run-dir",
            type=str,
            default=os.getenv("POLICY_RUN_DIR", "latest"),
            help=(
                "Path to eval run directory (contains summary.json), or 'latest' "
                "(default: $POLICY_RUN_DIR or 'latest')."
            ),
        )

        subp.add_argument(
            "--threshold-profile",
            type=str,
            default=None,
            help="Extract threshold profile, e.g. extract/sroie",
        )

        subp.add_argument(
            "--thresholds-root",
            type=str,
            default=None,
            help="Override thresholds root directory",
        )

        # Generate clamp (Phase 2)
        subp.add_argument(
            "--generate-threshold-profile",
            type=str,
            default=os.getenv("POLICY_GENERATE_PROFILE", "generate/portable"),
            help="Generate clamp threshold profile",
        )

        subp.add_argument(
            "--no-generate-clamp",
            action="store_true",
            help="Disable generate SLO clamp enrichment (debug only)",
        )

        subp.add_argument(
            "--generate-slo-path",
            type=str,
            default=os.getenv("POLICY_GENERATE_SLO_PATH", "").strip() or None,
            help="Override path to generate SLO snapshot JSON",
        )

        # Reporting / artifacts
        subp.add_argument(
            "--report",
            type=str,
            default="text",
            choices=["text", "md"],
            help="Human report format",
        )

        subp.add_argument(
            "--report-out",
            type=str,
            default=None,
            help="Write human report to file (optional)",
        )

        subp.add_argument(
            "--artifact-out",
            type=str,
            default=None,
            help="Write runtime policy decision artifact JSON to this path",
        )

        subp.add_argument(
            "--no-write-artifact",
            action="store_true",
            help="Do not write runtime policy artifact (debug only)",
        )

    # decide-extract
    d = sub.add_parser(
        "decide-extract",
        help="Run policy decision pipeline and emit runtime policy artifact",
    )
    _add_common(d)

    # run (high-level)
    r = sub.add_parser(
        "run",
        help="High-level entrypoint: decide pipeline, write artifact, optionally patch models.yaml",
    )
    _add_common(r)

    # patching controls (run only)
    r.add_argument(
        "--patch-models",
        action="store_true",
        help="If pipeline includes extract, patch models.yaml capabilities for --model-id based on decision.",
    )
    r.add_argument("--models-yaml", type=str, default=os.getenv("POLICY_MODELS_YAML", "").strip() or None)
    r.add_argument("--model-id", type=str, default=os.getenv("POLICY_MODEL_ID", "").strip() or None)

    # patch-models (unchanged)
    pm = sub.add_parser(
        "patch-models",
        help="Apply a decision to models.yaml by editing capabilities",
    )
    pm.add_argument("--models-yaml", type=str, required=True)
    pm.add_argument("--model-id", type=str, required=True)
    pm.add_argument("--enable-extract", action="store_true")
    pm.add_argument("--disable-extract", action="store_true")
    pm.add_argument("--dry-run", action="store_true")

    return p


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def _emit(s: str, out: Optional[str]) -> None:
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(s)
    else:
        print(s, end="")


def _render_human(decision: Decision, fmt: str) -> str:
    # For generate-only, "enable_extract" is not meaningful; display as None
    # even if the internal Decision model defaults it to False.
    if str(getattr(decision, "pipeline", "") or "") == "generate_clamp_only":
        try:
            decision_for_report = decision.model_copy(update={"enable_extract": None})
        except Exception:
            decision_for_report = decision
    else:
        decision_for_report = decision

    if fmt == "md":
        return render_decision_md(decision_for_report)
    return render_decision_text(decision_for_report)


def _artifact_path_or_default(raw: Optional[str]) -> str:
    if raw and raw.strip():
        return raw.strip()
    env = os.getenv("POLICY_OUT_PATH", "").strip()
    if env:
        return env
    return str(default_policy_out_path())


def _validate_outfile_path(p: str) -> None:
    pp = Path(p)
    if pp.name in ("", ".", ".."):
        raise ValueError(f"artifact path must be a file path, got: {p!r}")


def _unwrap_slo_read_result(res: Any) -> Tuple[bool, Any, str, Optional[str]]:
    ok = bool(getattr(res, "ok", False))
    snap = getattr(res, "artifact", None) or getattr(res, "snapshot", None)
    path = getattr(res, "resolved_path", None) or getattr(res, "path", "") or ""
    err = getattr(res, "error", None)
    return ok, snap, str(path), str(err) if err is not None else None


def _is_generate_only(decision: Decision) -> bool:
    pipeline = str(getattr(decision, "pipeline", "") or "")
    return pipeline == "generate_clamp_only"


def _normalize_generate_only(decision: Decision) -> Decision:
    """
    generate_clamp_only is SHAPING ONLY (never gating).
    Contract requirements for policy_decision_v2:
      - status = "allow"
      - ok = true
      - enable_extract = null
    """
    if not _is_generate_only(decision):
        return decision

    # Force allow + ok semantics (ok() is derived; we rely on allow status)
    decision = decision.model_copy(update={"status": "allow"})
    # Even if Decision.enable_extract is typed as bool in the model, we do NOT rely on it
    # for artifact writing in generate-only mode (we force null at payload level).
    return decision


def _write_generate_only_artifact(decision: Decision, out_path: str) -> None:
    """
    Write a policy_decision_v2 artifact for generate_clamp_only while forcibly
    satisfying the contract: enable_extract must be null.

    We bypass write_policy_decision_artifact() here because the Decision model
    may coerce enable_extract to False, violating the schema.
    """
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    reasons = [r.model_dump() if hasattr(r, "model_dump") else dict(r) for r in (decision.reasons or [])]
    warnings = [w.model_dump() if hasattr(w, "model_dump") else dict(w) for w in (decision.warnings or [])]

    payload = {
        "schema_version": "policy_decision_v2",
        "generated_at": now,
        "policy": getattr(decision, "policy", "llm_policy"),
        "pipeline": getattr(decision, "pipeline", "generate_clamp_only"),
        "status": "allow",
        "ok": True,
        # 🔒 Contract: must be null in generate_clamp_only
        "enable_extract": None,
        "generate_max_new_tokens_cap": getattr(decision, "generate_max_new_tokens_cap", None),
        "contract_errors": int(getattr(decision, "contract_errors", 0) or 0),
        "contract_warnings": int(getattr(decision, "contract_warnings", 0) or 0),
        "thresholds_profile": getattr(decision, "thresholds_profile", None),
        "generate_thresholds_profile": getattr(decision, "generate_thresholds_profile", None),
        "eval_run_dir": getattr(decision, "eval_run_dir", None),
        "reasons": reasons,
        "warnings": warnings,
        "metrics": getattr(decision, "metrics", {}) or {},
    }

    # Validate against the internal schema contract
    validate_internal(POLICY_DECISION_SCHEMA, payload)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=False)


# ------------------------------------------------------------------------------
# Pipeline execution
# ------------------------------------------------------------------------------


def _apply_generate_clamp(args, decision: Decision) -> Decision:
    if args.no_generate_clamp:
        return decision

    pcfg = PolicyConfig.default()
    if args.thresholds_root:
        pcfg = PolicyConfig(thresholds_root=args.thresholds_root)

    slo_res = read_generate_slo_snapshot_result(args.generate_slo_path)
    slo_ok, slo_snap, slo_path, slo_err = _unwrap_slo_read_result(slo_res)

    resolved_prof, gen_th = load_generate_thresholds(cfg=pcfg, profile=args.generate_threshold_profile)
    th = thresholds_from_mapping(gen_th.model_dump())

    updated = decide_generate_slo_clamp(
        base=decision,
        slo=slo_snap,
        thresholds=th,
        policy_name="generate_slo_clamp",
        enabled=True,
    )

    # If snapshot is missing/unreadable, do not clamp; keep provenance as warning.
    if not slo_ok:
        updated = updated.model_copy(
            update={
                "warnings": list(updated.warnings)
                + [
                    DecisionWarning(
                        code="generate_slo_snapshot_read_failed",
                        message="Generate SLO snapshot unreadable; no clamp applied (Option A).",
                        context={"path": slo_path, "error": slo_err},
                    )
                ]
            }
        )

    updated = updated.model_copy(
        update={
            "reasons": list(updated.reasons)
            + [
                DecisionReason(
                    code="generate_slo_profile",
                    message="Generate SLO clamp thresholds applied",
                    context={"profile": resolved_prof},
                )
            ],
            "generate_thresholds_profile": resolved_prof,
        }
    )

    return updated


def _build_decision(args) -> Decision:
    pipeline: PipelineType = args.pipeline  # validated by argparse

    # Base decision
    decision = Decision(
        policy="llm_policy",
        pipeline=pipeline,
    )

    # Extract enablement
    if pipeline in ("extract_only", "extract_plus_generate_clamp"):
        artifact = load_eval_artifact(args.run_dir or "latest")

        pcfg = PolicyConfig.default()
        if args.thresholds_root:
            pcfg = PolicyConfig(thresholds_root=args.thresholds_root)

        profile, th = load_extract_thresholds(cfg=pcfg, profile=args.threshold_profile)

        decision = decide_extract_enablement(
            artifact,
            thresholds=th,
            thresholds_profile=profile,
        ).model_copy(update={"pipeline": pipeline})

    # Generate clamp
    if pipeline in ("generate_clamp_only", "extract_plus_generate_clamp"):
        decision = _apply_generate_clamp(args, decision)

    # Enforce shaping-only semantics at the end (after clamp logic may have mutated status)
    decision = _normalize_generate_only(decision)

    return decision


def _maybe_patch_models(args, decision: Decision) -> None:
    """
    If requested and pipeline includes extract:
      - patch models.yaml capabilities for model_id based on decision.enable_extract.
    """
    if not getattr(args, "patch_models", False):
        return

    pipeline = str(getattr(decision, "pipeline", "") or "")
    if "extract" not in pipeline:
        return

    models_yaml = (getattr(args, "models_yaml", None) or "").strip()
    model_id = (getattr(args, "model_id", None) or "").strip()
    if not models_yaml:
        raise ValueError("--models-yaml is required when --patch-models is set")
    if not model_id:
        raise ValueError("--model-id is required when --patch-models is set")

    enable_extract = getattr(decision, "enable_extract", None)
    if not isinstance(enable_extract, bool):
        return

    argv = ["patch-models", "--models-yaml", models_yaml, "--model-id", model_id]
    argv += ["--enable-extract"] if enable_extract else ["--disable-extract"]

    rc = main(argv)
    if rc != 0:
        raise RuntimeError(f"patch-models failed (rc={rc})")


def _write_artifact(decision: Decision, out_path: str) -> None:
    """
    Use the standard writer for non-generate-only pipelines.
    For generate_clamp_only, force enable_extract=null via a schema-valid payload.
    """
    if _is_generate_only(decision):
        _write_generate_only_artifact(decision, out_path)
        return
    write_policy_decision_artifact(decision, out_path)


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "decide-extract":
        decision = _build_decision(args)

        if not args.no_write_artifact:
            out_path = _artifact_path_or_default(args.artifact_out)
            _validate_outfile_path(out_path)
            _write_artifact(decision, out_path)

        rendered = _render_human(decision, args.report)
        _emit(rendered, args.report_out)

        # generate-only is always allow/ok by design; ok() remains meaningful for extract pipelines
        return 0 if _is_generate_only(decision) or decision.ok() else 2

    if args.cmd == "run":
        decision = _build_decision(args)

        if not args.no_write_artifact:
            out_path = _artifact_path_or_default(args.artifact_out)
            _validate_outfile_path(out_path)
            _write_artifact(decision, out_path)

        try:
            _maybe_patch_models(args, decision)
        except Exception as e:
            print(f"❌ models.yaml patch failed: {e}", flush=True)
            return 2

        rendered = _render_human(decision, args.report)
        _emit(rendered, args.report_out)

        return 0 if _is_generate_only(decision) or decision.ok() else 2

    if args.cmd == "patch-models":
        from llm_policy.tools.patch_models import patch_models_yaml  # type: ignore

        return patch_models_yaml(
            models_yaml=args.models_yaml,
            model_id=args.model_id,
            enable_extract=bool(args.enable_extract),
            disable_extract=bool(args.disable_extract),
            dry_run=bool(args.dry_run),
        )

    print("Unknown command")
    return 2