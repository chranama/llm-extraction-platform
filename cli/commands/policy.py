# cli/commands/policy.py
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from cli.errors import CLIError
from cli.types import GlobalConfig
from cli.utils.proc import ensure_bins, run

JOB_NAME_DEFAULT = "policy"
NAMESPACE_DEFAULT = "llm"

# NOTE: keep these paths relative to repo_root (cfg.repo_root)
K8S_POLICY_JOB_YAML = Path("deploy/k8s/base/policy/job.yaml")


# -------------------------
# Small helpers
# -------------------------


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CLIError(f"Missing policy decision file: {path}", code=2)
    except Exception as e:
        raise CLIError(f"Failed to read policy decision file {path}: {e}", code=2)


def _http_post_json(
    url: str,
    *,
    headers: Dict[str, str],
    body: Optional[Dict[str, Any]] = None,
    timeout_s: int = 10,
) -> Dict[str, Any]:
    payload = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(url, method="POST", data=payload)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        if v is not None and str(v).strip():
            req.add_header(k, v)

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read().decode("utf-8", errors="replace").strip()
            if not data:
                return {}
            return json.loads(data)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise CLIError(f"POST {url} failed ({e.code}): {raw}", code=2)
    except urllib.error.URLError as e:
        raise CLIError(f"POST {url} failed: {e}", code=2)
    except json.JSONDecodeError:
        raise CLIError(f"POST {url} returned non-JSON response", code=2)


def _compact_kv(k: str, v: Any) -> str:
    if v is None:
        return f"{k}=null"
    if isinstance(v, bool):
        return f"{k}={'true' if v else 'false'}"
    return f"{k}={v}"


def _print_apply_summary(*, decision: Dict[str, Any], reload_resp: Dict[str, Any]) -> None:
    # Decision summary (best-effort; tolerate schema drift)
    # contracts artifact: { ok, model_id, enable_extract, ... }
    decision_ok = bool(decision.get("ok") if "ok" in decision else decision.get("decision_ok"))
    enable_extract = decision.get("enable_extract", decision.get("extract_enabled"))
    decision_model = decision.get("model_id")

    # Reload response summary (per your AdminReloadResponse)
    models = (reload_resp.get("models") or {}) if isinstance(reload_resp, dict) else {}
    policy = (reload_resp.get("policy") or {}) if isinstance(reload_resp, dict) else {}
    eff = (reload_resp.get("effective") or {}) if isinstance(reload_resp, dict) else {}

    default_model = models.get("default_model")
    model_ids = models.get("models") or []
    snapshot_ok = policy.get("snapshot_ok")
    source_path = policy.get("source_path")
    eff_extract = eff.get("extract_enabled")

    lines = [
        "policy.apply:",
        "  decision:",
        f"    {_compact_kv('ok', decision_ok)}",
        f"    {_compact_kv('model_id', decision_model)}",
        f"    {_compact_kv('enable_extract', enable_extract)}",
        "  api.reload:",
        f"    {_compact_kv('default_model', default_model)}",
        f"    {_compact_kv('models', ','.join(model_ids) if isinstance(model_ids, list) else model_ids)}",
        f"    {_compact_kv('policy_snapshot_ok', snapshot_ok)}",
        f"    {_compact_kv('source_path', source_path)}",
        f"    {_compact_kv('extract_enabled', eff_extract)}",
    ]
    print("\n".join(lines))


# -------------------------
# Argparse wiring
# -------------------------


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("policy", help="Policy helpers (compose + k8s runners).")
    p.set_defaults(_handler=_handle)

    sp = p.add_subparsers(dest="policy_cmd", required=True)

    # -------------------------
    # Phase 1: golden-path apply
    # -------------------------
    a = sp.add_parser(
        "apply",
        help="Run policy decision -> write policy_out/latest.json -> POST /v1/admin/reload (golden path).",
    )
    a.add_argument("--run-dir", required=True, help="Eval run directory (or 'latest' pointer).")
    a.add_argument("--model-id", required=True, help="Model id the decision is intended for (recorded in artifact if policy emits it).")
    a.add_argument("--profile", required=True, help="Threshold profile (e.g. extract/sroie).")
    a.add_argument("--api-base-url", default=None, help="API base URL (default: http://localhost:<api_port>).")
    a.add_argument("--api-key", default=os.getenv("API_KEY", ""), help="Admin API key (default: env API_KEY).")
    a.add_argument("--decision-path", default=None, help="Host path to policy decision JSON (default: <repo>/policy_out/latest.json).")
    a.add_argument(
        "--container-artifact-out",
        default="/work/policy_out/latest.json",
        help="Artifact out path INSIDE the policy container (default: /work/policy_out/latest.json).",
    )
    a.add_argument("--timeout-seconds", type=int, default=15, help="HTTP timeout seconds for /v1/admin/reload (default: 15).")
    a.add_argument("--no-reload", action="store_true", help="Only write latest.json; do not call /v1/admin/reload.")
    a.add_argument(
        "--report",
        default="text",
        choices=["text", "md"],
        help="Policy human report format printed by llm-policy (default: text).",
    )

    # -------------------------
    # Compose runner
    # -------------------------
    c = sp.add_parser("compose", help="Run llm-policy inside docker-compose (ephemeral).")
    csp = c.add_subparsers(dest="compose_cmd", required=True)

    _compose_add_decide_args(csp.add_parser("decide-extract", help="decide-extract (compose)"))
    _compose_add_patch_args(csp.add_parser("patch-models", help="patch-models (compose)"))
    _compose_add_decide_and_patch_args(csp.add_parser("decide-and-patch", help="decide-and-patch (compose)"))

    # -------------------------
    # K8s runner (Job-based)
    # -------------------------
    k = sp.add_parser("k8s", help="Run llm-policy as a Kubernetes Job (apply/wait/logs).")
    k.add_argument("--namespace", default=NAMESPACE_DEFAULT, help="Kubernetes namespace (default: llm)")
    k.add_argument("--job-name", default=JOB_NAME_DEFAULT, help="Policy job name (default: policy)")

    ksp = k.add_subparsers(dest="k8s_cmd", required=True)

    r = ksp.add_parser("run", help="(Re)apply policy job, wait, print logs, return exit code.")
    r.add_argument("--timeout-seconds", type=int, default=600, help="Wait timeout in seconds for job completion (default: 600).")
    r.add_argument("--no-logs", action="store_true", help="Do not print job logs after completion.")

    ksp.add_parser("delete", help="Delete the policy job if it exists.")
    ksp.add_parser("logs", help="Show policy job logs (best effort).")
    ksp.add_parser("status", help="Show job status (best effort).")


def _compose_add_decide_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run-dir", required=True, help="Eval run directory (contains summary.json), or 'latest'.")
    p.add_argument("--threshold-profile", default=None, help="Threshold profile (e.g. extract/sroie).")
    p.add_argument("--thresholds-root", default=None, help="Override thresholds root directory (optional).")

    # Human report (policy CLI uses --report)
    p.add_argument("--report", default="text", choices=["text", "md"], help="Human report format.")

    # Runtime artifact JSON
    p.add_argument("--artifact-out", default=None, help="Write decision artifact JSON to this path (inside container).")
    p.add_argument("--no-write-artifact", action="store_true", help="Do not write runtime decision artifact JSON.")


def _compose_add_patch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models-yaml", required=True, help="Path to models.yaml (inside container).")
    p.add_argument("--model-id", required=True, help="Model id to patch.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--enable-extract", action="store_true", help="Enable extract capability.")
    g.add_argument("--disable-extract", action="store_true", help="Disable extract capability.")
    p.add_argument("--dry-run", action="store_true", help="Do not write; just show changes.")


def _compose_add_decide_and_patch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run-dir", required=True, help="Eval run directory (contains summary.json), or 'latest'.")
    p.add_argument("--models-yaml", required=True, help="Path to models.yaml (inside container).")
    p.add_argument("--model-id", required=True, help="Model id to patch.")
    p.add_argument("--threshold-profile", default=None, help="Threshold profile (e.g. extract/sroie).")
    p.add_argument("--thresholds-root", default=None, help="Override thresholds root directory (optional).")
    p.add_argument("--dry-run", action="store_true")

    # Human report
    p.add_argument("--report", default="text", choices=["text", "md"], help="Human report format.")
    p.add_argument("--report-out", default=None, help="Write human report to file (optional).")

    # Runtime artifact JSON
    p.add_argument("--artifact-out", default=None, help="Write decision artifact JSON to this path (inside container).")
    p.add_argument("--no-write-artifact", action="store_true", help="Do not write runtime decision artifact JSON.")


def _handle(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    ensure_bins("docker", "bash")

    if args.policy_cmd == "apply":
        return _handle_apply(cfg, args)

    if args.policy_cmd == "compose":
        return _handle_compose(cfg, args)

    if args.policy_cmd == "k8s":
        ensure_bins("kubectl")
        return _handle_k8s(cfg, args)

    raise CLIError(f"Unknown policy command: {args.policy_cmd}", code=2)


# -------------------------
#  Apply implementation
# -------------------------


def _handle_apply(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    """
    Phase 1 golden path:
      1) Run policy decision in compose policy container
      2) Write policy_out/latest.json on host via bind-mount
      3) POST /v1/admin/reload so runtime picks it up
      4) Print concise merged summary
    """
    api_base = args.api_base_url or f"http://localhost:{cfg.api_port}"
    api_key = (args.api_key or "").strip()
    if not api_key and not args.no_reload:
        raise CLIError("Missing --api-key (or env API_KEY) for /v1/admin/reload", code=2)

    decision_host_path = Path(args.decision_path) if args.decision_path else (cfg.repo_root / "policy_out" / "latest.json")

    # 1) Run policy decision (compose), forcing artifact-out to canonical path inside container.
    # IMPORTANT: policy CLI now uses:
    #   --report (text|md)
    #   --artifact-out for runtime JSON
    # and does NOT accept --format or --model-id for decide-extract.
    base = _compose_base(cfg, args)
    policy_argv: List[str] = [
        "decide-extract",
        "--run-dir",
        args.run_dir,
        "--threshold-profile",
        args.profile,
        "--artifact-out",
        args.container_artifact_out,
        "--report",
        args.report,
    ]
    full = base + ["run", "--rm", "policy"] + policy_argv
    r = run(full, cwd=str(cfg.repo_root), verbose=args.verbose, check=False)
    if r.code != 0:
        raise CLIError("Policy decision failed (see logs above).", code=int(r.code or 1))

    # 2) Read decision artifact from host bind-mount
    decision = _read_json_file(decision_host_path)

    # Optional: sanity check (do not fail if policy doesn't include model_id)
    decision_model = decision.get("model_id")
    if decision_model and str(decision_model) != str(args.model_id):
        # Not fatal, but loud.
        print(f"warning: decision model_id={decision_model!r} != --model-id {args.model_id!r}")

    # 3) Reload runtime (unless disabled)
    reload_resp: Dict[str, Any] = {}
    if not args.no_reload:
        reload_url = f"{api_base.rstrip('/')}/v1/admin/reload"
        reload_resp = _http_post_json(
            reload_url,
            headers={"X-API-Key": api_key},
            body={},
            timeout_s=int(args.timeout_seconds),
        )

    # 4) Print summary
    _print_apply_summary(decision=decision, reload_resp=reload_resp or {})
    return 0


# -------------------------
# Compose implementation
# -------------------------


def _compose_base(cfg: GlobalConfig, args: argparse.Namespace) -> List[str]:
    return [
        "docker",
        "compose",
        "-f",
        str(cfg.compose_yml),
        "--project-name",
        cfg.project_name,
        "--profile",
        "policy",
    ]


def _handle_compose(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    cmd = args.compose_cmd
    base = _compose_base(cfg, args)

    policy_argv: List[str] = []

    if cmd == "decide-extract":
        policy_argv = [
            "decide-extract",
            "--run-dir",
            args.run_dir,
        ]
        if args.threshold_profile:
            policy_argv += ["--threshold-profile", args.threshold_profile]
        if args.thresholds_root:
            policy_argv += ["--thresholds-root", args.thresholds_root]
        if args.report:
            policy_argv += ["--report", args.report]
        if args.artifact_out:
            policy_argv += ["--artifact-out", args.artifact_out]
        if args.no_write_artifact:
            policy_argv.append("--no-write-artifact")

    elif cmd == "patch-models":
        enable = True if args.enable_extract else False
        policy_argv = [
            "patch-models",
            "--models-yaml",
            args.models_yaml,
            "--model-id",
            args.model_id,
            "--enable-extract" if enable else "--disable-extract",
        ]
        if args.dry_run:
            policy_argv.append("--dry-run")

    elif cmd == "decide-and-patch":
        policy_argv = [
            "decide-and-patch",
            "--run-dir",
            args.run_dir,
            "--models-yaml",
            args.models_yaml,
            "--model-id",
            args.model_id,
        ]
        if args.threshold_profile:
            policy_argv += ["--threshold-profile", args.threshold_profile]
        if args.thresholds_root:
            policy_argv += ["--thresholds-root", args.thresholds_root]
        if args.report:
            policy_argv += ["--report", args.report]
        if getattr(args, "report_out", None):
            policy_argv += ["--report-out", args.report_out]
        if args.artifact_out:
            policy_argv += ["--artifact-out", args.artifact_out]
        if args.no_write_artifact:
            policy_argv.append("--no-write-artifact")
        if args.dry_run:
            policy_argv.append("--dry-run")

    else:
        raise CLIError(f"Unknown policy compose command: {cmd}", code=2)

    full = base + ["run", "--rm", "policy"] + policy_argv
    r = run(full, cwd=str(cfg.repo_root), verbose=args.verbose, check=False)
    return int(r.code)


# -------------------------
# K8s implementation
# -------------------------


def _job_yaml_path(cfg: GlobalConfig) -> Path:
    p = cfg.repo_root / K8S_POLICY_JOB_YAML
    if not p.exists():
        raise CLIError(f"Missing k8s policy job yaml: {p}", code=2)
    return p


def _kubectl(ns: str, *argv: str) -> List[str]:
    return ["kubectl", "-n", ns, *argv]


def _handle_k8s(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    ns = args.namespace
    job_name = args.job_name
    cmd = args.k8s_cmd

    job_yaml = _job_yaml_path(cfg)

    if cmd == "delete":
        run(_kubectl(ns, "delete", "job", job_name, "--ignore-not-found=true"), verbose=args.verbose, check=False)
        return 0

    if cmd == "logs":
        r = run(_kubectl(ns, "logs", f"job/{job_name}", "--all-containers=true"), verbose=args.verbose, check=False)
        return int(r.code)

    if cmd == "status":
        r = run(_kubectl(ns, "get", "job", job_name, "-o", "wide"), verbose=args.verbose, check=False)
        return int(r.code)

    if cmd == "run":
        run(_kubectl(ns, "delete", "job", job_name, "--ignore-not-found=true"), verbose=args.verbose, check=False)
        run(_kubectl(ns, "apply", "-f", str(job_yaml)), verbose=args.verbose)

        timeout = int(getattr(args, "timeout_seconds", 600))
        wait_ok = run(
            _kubectl(ns, "wait", "--for=condition=complete", f"job/{job_name}", f"--timeout={timeout}s"),
            verbose=args.verbose,
            check=False,
        )

        if wait_ok.code != 0:
            _ = run(
                _kubectl(ns, "wait", "--for=condition=failed", f"job/{job_name}", f"--timeout=1s"),
                verbose=args.verbose,
                check=False,
            )

        if not getattr(args, "no_logs", False):
            run(_kubectl(ns, "logs", f"job/{job_name}", "--all-containers=true"), verbose=args.verbose, check=False)

        return 0 if wait_ok.code == 0 else int(wait_ok.code)

    raise CLIError(f"Unknown policy k8s command: {cmd}", code=2)