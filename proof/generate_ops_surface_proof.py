#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "proof" / "artifacts" / "phase10_ops_surface"
PROJECT_NAME = "llmep-ops-surface"
API_PORT = "18085"
UI_PORT = "18086"
PROM_PORT = "18087"
GRAFANA_PORT = "18088"
PROXY_PORT = "18089"
API_BASE = f"http://127.0.0.1:{API_PORT}"
UI_BASE = f"http://127.0.0.1:{UI_PORT}"
PROM_BASE = f"http://127.0.0.1:{PROM_PORT}"
GRAFANA_BASE = f"http://127.0.0.1:{GRAFANA_PORT}"
PROXY_BASE = f"http://127.0.0.1:{PROXY_PORT}"
ENV_FILE = ROOT / ".tmp" / "phase10-ops-surface.env"
API_KEY = "proof-ops-key"
ADMIN_API_KEY = "proof-ops-admin"


def fail(message: str) -> None:
    raise RuntimeError(message)


def run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        fail(
            "command failed: {cmd}\nreturncode={code}\nstdout:\n{stdout}\nstderr:\n{stderr}".format(
                cmd=" ".join(args),
                code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )
    return result


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def http_get(
    url: str,
    *,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 10,
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET")
    if api_key:
        req.add_header("X-API-Key", api_key)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError:
        return 0, b""
    except (ConnectionResetError, TimeoutError, socket.timeout, OSError):
        return 0, b""


def wait_http(
    url: str,
    *,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 120.0,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        code, _ = http_get(url, api_key=api_key, headers=headers)
        if code == 200:
            return
        time.sleep(1)
    fail(f"timed out waiting for {url}")


def save_http(
    path: Path,
    url: str,
    *,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
) -> dict:
    status, body = http_get(url, api_key=api_key, headers=headers)
    text = body.decode("utf-8", errors="replace")
    parsed: dict
    try:
        parsed = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
        parsed = {"raw": text[:2000]}
    payload = {"url": url, "status_code": status, "body": parsed}
    write_json(path, payload)
    return payload


def read_json_url(
    url: str,
    *,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
) -> dict:
    status, body = http_get(url, api_key=api_key, headers=headers)
    text = body.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
        parsed = {"raw": text[:2000]}
    return {"url": url, "status_code": status, "body": parsed}


def grafana_headers() -> dict[str, str]:
    token = base64.b64encode(b"admin:admin").decode("ascii")
    return {"Authorization": f"Basic {token}"}


def prometheus_query_url(query: str) -> str:
    return f"{PROM_BASE}/api/v1/query?query={urllib.parse.quote(query)}"


def grafana_proxy_query_url(query: str) -> str:
    quoted = urllib.parse.quote(query)
    return f"{GRAFANA_BASE}/api/datasources/proxy/uid/Prometheus/api/v1/query?query={quoted}"


def _prometheus_result_has_data(payload: dict) -> bool:
    body = payload.get("body") or {}
    data = body.get("data") if isinstance(body, dict) else None
    result = data.get("result") if isinstance(data, dict) else None
    return isinstance(result, list) and bool(result)


def _api_target_up(targets_payload: dict) -> bool:
    body = targets_payload.get("body") or {}
    data = body.get("data") if isinstance(body, dict) else None
    active = data.get("activeTargets") if isinstance(data, dict) else None
    if not isinstance(active, list):
        return False
    for target in active:
        if not isinstance(target, dict):
            continue
        labels = target.get("labels") or {}
        if labels.get("job") == "llm_server_container" and target.get("health") == "up":
            return True
    return False


def wait_prometheus_scrape(timeout_seconds: float = 90.0) -> None:
    targets_url = f"{PROM_BASE}/api/v1/targets"
    query_url = prometheus_query_url('up{job="llm_server_container"}')
    deadline = time.time() + timeout_seconds
    last_targets: dict | None = None
    last_query: dict | None = None
    while time.time() < deadline:
        last_targets = read_json_url(targets_url)
        last_query = read_json_url(query_url)
        if _api_target_up(last_targets) and _prometheus_result_has_data(last_query):
            return
        time.sleep(2)
    fail(
        "timed out waiting for Prometheus API scrape target; "
        f"last_targets={last_targets} last_query={last_query}"
    )


def _panel_exprs(dashboard_payload: dict) -> list[str]:
    body = dashboard_payload.get("body") or {}
    dashboard = body.get("dashboard") if isinstance(body, dict) else None
    panels = dashboard.get("panels") if isinstance(dashboard, dict) else None
    exprs: list[str] = []
    if not isinstance(panels, list):
        return exprs
    for panel in panels:
        if not isinstance(panel, dict):
            continue
        for target in panel.get("targets") or []:
            if not isinstance(target, dict):
                continue
            expr = str(target.get("expr") or "").strip()
            if expr:
                exprs.append(expr)
    return exprs


def _dashboard_panel_count(dashboard_payload: dict) -> int:
    body = dashboard_payload.get("body") or {}
    dashboard = body.get("dashboard") if isinstance(body, dict) else None
    panels = dashboard.get("panels") if isinstance(dashboard, dict) else None
    return len(panels) if isinstance(panels, list) else 0


def llmctl_base() -> list[str]:
    return [
        "uv",
        "run",
        "llmctl",
        "--project-name",
        PROJECT_NAME,
        "--env-override-file",
        str(ENV_FILE),
        "--api-port",
        API_PORT,
        "--ui-port",
        UI_PORT,
        "--prom-port",
        PROM_PORT,
        "--grafana-port",
        GRAFANA_PORT,
    ]


def write_env_file() -> None:
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENV_FILE.write_text(
        "\n".join(
            [
                f"API_KEY={API_KEY}",
                f"ADMIN_API_KEY={ADMIN_API_KEY}",
                f"PROXY_PORT={PROXY_PORT}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def remove_temp_envs() -> None:
    for path in (
        ENV_FILE,
        ROOT / ".tmp" / "llmctl" / "compose-effective-docker_reviewer-smoke.env",
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def generate_ops_surface_proof() -> None:
    for binary in ("docker", "uv"):
        if shutil.which(binary) is None:
            fail(f"missing required binary: {binary}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    write_env_file()

    try:
        smoke = run(llmctl_base() + ["smoke"])
        write_text(ARTIFACT_DIR / "llmctl_smoke_stdout.txt", smoke.stdout)
        write_text(ARTIFACT_DIR / "llmctl_smoke_stderr.txt", smoke.stderr)

        ops = run(
            llmctl_base()
            + [
                "compose",
                "--defaults-profile",
                "docker+reviewer-smoke",
                "up",
                "--profiles",
                "infra",
                "server",
                "ui",
                "obs",
                "proxy",
                "-d",
                "--build",
                "--remove-orphans",
            ],
            check=False,
        )
        write_text(ARTIFACT_DIR / "llmctl_ops_up_stdout.txt", ops.stdout)
        write_text(ARTIFACT_DIR / "llmctl_ops_up_stderr.txt", ops.stderr)
        if ops.returncode != 0:
            fail(f"ops surface compose up failed with exit code {ops.returncode}")

        wait_http(f"{API_BASE}/healthz")
        wait_http(f"{UI_BASE}/")
        wait_http(f"{PROM_BASE}/-/ready")
        wait_http(f"{GRAFANA_BASE}/api/health")
        wait_http(f"{PROXY_BASE}/api/healthz")
        wait_http(f"{PROXY_BASE}/ui/")
        wait_http(f"{PROXY_BASE}/prometheus/-/ready")
        wait_http(f"{PROXY_BASE}/grafana/api/health")
        wait_http(f"{GRAFANA_BASE}/api/datasources", headers=grafana_headers())
        wait_prometheus_scrape()

        direct_api = save_http(ARTIFACT_DIR / "direct_api_healthz.json", f"{API_BASE}/healthz")
        direct_ui = save_http(ARTIFACT_DIR / "direct_ui_index.json", f"{UI_BASE}/")
        direct_prom = save_http(
            ARTIFACT_DIR / "direct_prometheus_ready.json", f"{PROM_BASE}/-/ready"
        )
        direct_grafana = save_http(
            ARTIFACT_DIR / "direct_grafana_health.json", f"{GRAFANA_BASE}/api/health"
        )
        proxy_api = save_http(ARTIFACT_DIR / "proxy_api_healthz.json", f"{PROXY_BASE}/api/healthz")
        proxy_ui = save_http(ARTIFACT_DIR / "proxy_ui_index.json", f"{PROXY_BASE}/ui/")
        proxy_prom = save_http(
            ARTIFACT_DIR / "proxy_prometheus_ready.json", f"{PROXY_BASE}/prometheus/-/ready"
        )
        proxy_grafana = save_http(
            ARTIFACT_DIR / "proxy_grafana_health.json", f"{PROXY_BASE}/grafana/api/health"
        )
        prometheus_targets = save_http(
            ARTIFACT_DIR / "prometheus_targets.json", f"{PROM_BASE}/api/v1/targets"
        )
        prometheus_query_up = save_http(
            ARTIFACT_DIR / "prometheus_query_up.json",
            prometheus_query_url('up{job="llm_server_container"}'),
        )
        grafana_datasources = save_http(
            ARTIFACT_DIR / "grafana_datasources.json",
            f"{GRAFANA_BASE}/api/datasources",
            headers=grafana_headers(),
        )
        grafana_dashboards = save_http(
            ARTIFACT_DIR / "grafana_dashboards.json",
            f"{GRAFANA_BASE}/api/search?type=dash-db",
            headers=grafana_headers(),
        )
        grafana_proxy_up = save_http(
            ARTIFACT_DIR / "grafana_prometheus_proxy_query_up.json",
            grafana_proxy_query_url('up{job="llm_server_container"}'),
            headers=grafana_headers(),
        )

        dashboard_search = grafana_dashboards.get("body")
        if not isinstance(dashboard_search, list) or not dashboard_search:
            fail("Grafana did not return any provisioned dashboards")

        dashboard_summaries: list[dict[str, object]] = []
        dashboard_query_artifacts: list[str] = []
        for item in dashboard_search:
            if not isinstance(item, dict):
                continue
            uid = str(item.get("uid") or "").strip()
            if not uid:
                continue
            dashboard_payload = save_http(
                ARTIFACT_DIR / f"grafana_dashboard_{uid}.json",
                f"{GRAFANA_BASE}/api/dashboards/uid/{urllib.parse.quote(uid)}",
                headers=grafana_headers(),
            )
            exprs = _panel_exprs(dashboard_payload)
            expr_results = []
            for idx, expr in enumerate(exprs[:8], start=1):
                query_payload = save_http(
                    ARTIFACT_DIR / f"grafana_dashboard_{uid}_query_{idx}.json",
                    prometheus_query_url(expr),
                )
                artifact_rel = str(
                    (ARTIFACT_DIR / f"grafana_dashboard_{uid}_query_{idx}.json").relative_to(ROOT)
                )
                dashboard_query_artifacts.append(artifact_rel)
                expr_results.append(
                    {
                        "expr": expr,
                        "artifact": artifact_rel,
                        "has_data": _prometheus_result_has_data(query_payload),
                    }
                )
            dashboard_summaries.append(
                {
                    "uid": uid,
                    "title": item.get("title"),
                    "panels_count": _dashboard_panel_count(dashboard_payload),
                    "queries_checked": len(expr_results),
                    "queries_with_data": sum(1 for row in expr_results if row["has_data"]),
                }
            )

        datasource_body = grafana_datasources.get("body")
        prometheus_datasource_present = isinstance(datasource_body, list) and any(
            isinstance(item, dict)
            and item.get("type") == "prometheus"
            and item.get("uid") == "Prometheus"
            for item in datasource_body
        )
        dashboards_with_panels = sum(
            1 for row in dashboard_summaries if int(row["panels_count"]) > 0
        )
        dashboards_with_data = sum(
            1 for row in dashboard_summaries if int(row["queries_with_data"]) > 0
        )
        dashboard_population = {
            "grafana_reachable": direct_grafana["status_code"] == 200,
            "prometheus_datasource_present": prometheus_datasource_present,
            "dashboards_count": len(dashboard_summaries),
            "dashboards_with_panels": dashboards_with_panels,
            "dashboards_with_query_data": dashboards_with_data,
            "panels_count": sum(int(row["panels_count"]) for row in dashboard_summaries),
            "api_prometheus_target_up": _api_target_up(prometheus_targets),
            "prometheus_query_up_has_data": _prometheus_result_has_data(prometheus_query_up),
            "grafana_datasource_proxy_query_has_data": _prometheus_result_has_data(
                grafana_proxy_up
            ),
            "dashboards": dashboard_summaries,
            "query_artifacts": dashboard_query_artifacts,
        }
        write_json(
            ARTIFACT_DIR / "dashboard_population_summary.json",
            dashboard_population,
        )

        ps = run(
            llmctl_base()
            + [
                "compose",
                "--defaults-profile",
                "docker+reviewer-smoke",
                "ps",
                "--profiles",
                "infra",
                "server",
                "ui",
                "obs",
                "proxy",
            ],
            check=False,
        )
        write_text(ARTIFACT_DIR / "compose_ps.txt", ps.stdout + ps.stderr)

        summary = {
            "proof_phase": "phase10_ops_surface",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "pass",
            "compose_project": PROJECT_NAME,
            "ports": {
                "api": API_PORT,
                "ui": UI_PORT,
                "prometheus": PROM_PORT,
                "grafana": GRAFANA_PORT,
                "proxy": PROXY_PORT,
            },
            "checks": {
                "direct_api": direct_api["status_code"] == 200,
                "direct_ui": direct_ui["status_code"] == 200,
                "direct_prometheus": direct_prom["status_code"] == 200,
                "direct_grafana": direct_grafana["status_code"] == 200,
                "proxy_api": proxy_api["status_code"] == 200,
                "proxy_ui": proxy_ui["status_code"] == 200,
                "proxy_prometheus": proxy_prom["status_code"] == 200,
                "proxy_grafana": proxy_grafana["status_code"] == 200,
                "prometheus_api_target_up": dashboard_population["api_prometheus_target_up"],
                "prometheus_query_up": dashboard_population["prometheus_query_up_has_data"],
                "grafana_datasource": dashboard_population["prometheus_datasource_present"],
                "grafana_dashboard_population": (
                    dashboard_population["dashboards_count"] > 0
                    and dashboard_population["dashboards_with_panels"]
                    == dashboard_population["dashboards_count"]
                    and dashboard_population["dashboards_with_query_data"]
                    == dashboard_population["dashboards_count"]
                    and dashboard_population["grafana_datasource_proxy_query_has_data"]
                ),
            },
        }
        if not all(summary["checks"].values()):
            fail(f"ops surface checks failed: {summary}")
        write_json(ARTIFACT_DIR / "ops_surface_summary.json", summary)
    finally:
        stop = run(llmctl_base() + ["stop", "--volumes"], check=False)
        write_text(ARTIFACT_DIR / "llmctl_stop_stdout.txt", stop.stdout)
        write_text(ARTIFACT_DIR / "llmctl_stop_stderr.txt", stop.stderr)
        remove_temp_envs()


if __name__ == "__main__":
    generate_ops_surface_proof()
