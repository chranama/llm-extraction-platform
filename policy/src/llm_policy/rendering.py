from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Mapping, Optional

from pydantic import BaseModel

from llm_policy.types.decision import Decision


def _to_mapping(x: Any) -> Mapping[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, BaseModel):
        try:
            return x.model_dump()
        except Exception:
            pass
    out: Dict[str, Any] = {}
    for k in ("code", "message", "context", "extra"):
        if hasattr(x, k):
            out[k] = getattr(x, k)
    return out


def _iter_issues(items: Optional[Iterable[Any]]) -> Iterable[Mapping[str, Any]]:
    for it in items or []:
        yield _to_mapping(it)


def _safe_json_one_line(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(x)


def _extract_provenance(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    dep = metrics.get("deployment")
    dep_key = metrics.get("deployment_key")

    if dep is None:
        dep = metrics.get("deployment_info") or metrics.get("eval_deployment")
    if dep_key is None:
        dep_key = metrics.get("deployment_id") or metrics.get("eval_deployment_key")

    return {
        "deployment_key": dep_key,
        "deployment": dep,
        "eval_run_dir": metrics.get("run_dir") or metrics.get("eval_run_dir"),
        "eval_task": metrics.get("task") or metrics.get("eval_task"),
        "eval_run_id": metrics.get("run_id") or metrics.get("eval_run_id"),
        "model_id": metrics.get("model_id") or metrics.get("model") or metrics.get("eval_model_id"),
        "base_url": metrics.get("base_url") or metrics.get("eval_base_url"),
    }


def _render_provenance_text(metrics: Mapping[str, Any]) -> str:
    p = _extract_provenance(metrics)

    lines: list[str] = []
    lines.append("PROVENANCE:")

    def add(k: str, v: Any) -> None:
        if v is None:
            lines.append(f"- {k}: (missing)")
        else:
            if isinstance(v, (dict, list)):
                lines.append(f"- {k}: {_safe_json_one_line(v)}")
            else:
                lines.append(f"- {k}: {v}")

    add("deployment_key", p.get("deployment_key"))
    add("deployment", p.get("deployment"))
    add("task", p.get("eval_task"))
    add("run_id", p.get("eval_run_id"))
    add("run_dir", p.get("eval_run_dir"))

    if p.get("base_url") is not None:
        add("base_url", p.get("base_url"))
    if p.get("model_id") is not None:
        add("model_id", p.get("model_id"))

    return "\n".join(lines)


def _render_provenance_md(metrics: Mapping[str, Any]) -> str:
    p = _extract_provenance(metrics)

    def cell(v: Any) -> str:
        if v is None:
            return "`(missing)`"
        if isinstance(v, (dict, list)):
            return f"`{_safe_json_one_line(v)}`"
        return f"`{v}`"

    lines: list[str] = []
    lines.append("## Provenance")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| deployment_key | {cell(p.get('deployment_key'))} |")
    lines.append(f"| deployment | {cell(p.get('deployment'))} |")
    lines.append(f"| task | {cell(p.get('eval_task'))} |")
    lines.append(f"| run_id | {cell(p.get('eval_run_id'))} |")
    lines.append(f"| run_dir | {cell(p.get('eval_run_dir'))} |")

    if p.get("base_url") is not None:
        lines.append(f"| base_url | {cell(p.get('base_url'))} |")
    if p.get("model_id") is not None:
        lines.append(f"| model_id | {cell(p.get('model_id'))} |")

    return "\n".join(lines)


def render_decision_text(decision: Decision) -> str:
    lines: list[str] = []
    lines.append(f"policy={decision.policy}")

    if getattr(decision, "thresholds_profile", None):
        lines.append(f"thresholds_profile={decision.thresholds_profile}")
    if getattr(decision, "enable_extract", None) is not None:
        lines.append(f"enable_extract={bool(decision.enable_extract)}")
    if getattr(decision, "status", None) is not None:
        lines.append(f"status={decision.status}")

    lines.append(f"ok={decision.ok() if hasattr(decision, 'ok') else 'unknown'}")

    ce = int(getattr(decision, "contract_errors", 0) or 0)
    cw = int(getattr(decision, "contract_warnings", 0) or 0)
    if ce or cw:
        lines.append(f"contract_errors={ce}")
        lines.append(f"contract_warnings={cw}")

    metrics_any = getattr(decision, "metrics", None) or {}
    if isinstance(metrics_any, dict) and metrics_any:
        lines.append("")
        lines.append(_render_provenance_text(metrics_any))

    reasons = list(_iter_issues(getattr(decision, "reasons", None)))
    if reasons:
        lines.append("")
        lines.append("REASONS:")
        for r in reasons:
            code = r.get("code", "reason")
            msg = r.get("message", "")
            lines.append(f"- {code}: {msg}")

    warnings = list(_iter_issues(getattr(decision, "warnings", None)))
    if warnings:
        lines.append("")
        lines.append("WARNINGS:")
        for w in warnings:
            code = w.get("code", "warning")
            msg = w.get("message", "")
            lines.append(f"- {code}: {msg}")

    metrics = metrics_any
    if isinstance(metrics, dict) and metrics:
        lines.append("")
        lines.append("METRICS:")
        for k in sorted(metrics.keys()):
            lines.append(f"- {k}: {metrics[k]}")

    return "\n".join(lines) + "\n"


def render_decision_md(decision: Decision) -> str:
    lines: list[str] = []
    lines.append(f"# Policy Decision: `{decision.policy}`")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| ok | `{decision.ok() if hasattr(decision,'ok') else 'unknown'}` |")

    if getattr(decision, "status", None) is not None:
        lines.append(f"| status | `{decision.status}` |")
    if getattr(decision, "thresholds_profile", None):
        lines.append(f"| thresholds_profile | `{decision.thresholds_profile}` |")
    if getattr(decision, "enable_extract", None) is not None:
        lines.append(f"| enable_extract | `{bool(decision.enable_extract)}` |")

    ce = int(getattr(decision, "contract_errors", 0) or 0)
    cw = int(getattr(decision, "contract_warnings", 0) or 0)
    if ce or cw:
        lines.append(f"| contract_errors | `{ce}` |")
        lines.append(f"| contract_warnings | `{cw}` |")

    metrics = getattr(decision, "metrics", None) or {}
    if isinstance(metrics, dict) and metrics:
        lines.append("")
        lines.append(_render_provenance_md(metrics))

    reasons = list(_iter_issues(getattr(decision, "reasons", None)))
    if reasons:
        lines.append("")
        lines.append("## Reasons")
        for r in reasons:
            code = r.get("code", "reason")
            msg = r.get("message", "")
            lines.append(f"- **{code}** - {msg}")

    warnings = list(_iter_issues(getattr(decision, "warnings", None)))
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        for w in warnings:
            code = w.get("code", "warning")
            msg = w.get("message", "")
            lines.append(f"- **{code}** - {msg}")

    if isinstance(metrics, dict) and metrics:
        lines.append("")
        lines.append("## Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for k in sorted(metrics.keys()):
            lines.append(f"| `{k}` | `{metrics[k]}` |")

    return "\n".join(lines).strip() + "\n"
