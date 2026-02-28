from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from integrations.lib.eval_job import read_json, read_jsonl, run_eval_cli
from integrations.lib.policy_job import run_policy_cli

pytestmark = [pytest.mark.e2e_loop]


def _write_thresholds_root(root: Path) -> Path:
    extract = root / "extract"
    extract.mkdir(parents=True, exist_ok=True)
    (extract / "default.yaml").write_text(
        "\n".join(
            [
                "version: v1",
                "metrics:",
                "  schema_validity_rate:",
                "    min: 95.0",
                "  required_present_rate:",
                "    min: 95.0",
                "  doc_required_exact_match_rate:",
                "    min: 80.0",
                "params:",
                "  min_n_total: 1",
                "  min_n_for_point_estimate: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return root


def test_e2e_eval_to_policy_extract_only_contract_linkage(tmp_path: Path):
    eval_outdir = tmp_path / "eval_results"
    eval_pointer = tmp_path / "eval_out" / "latest.json"

    eval_res = run_eval_cli(
        args=[
            "--task",
            "extraction_sroie",
            "--base-url",
            "http://127.0.0.1:9",
            "--api-key",
            "k",
            "--max-examples",
            "1",
            "--no-print-summary",
            "--save",
            "--outdir",
            str(eval_outdir),
        ],
        env={"EVAL_LATEST_PATH": str(eval_pointer)},
    )
    if eval_res.returncode != 0 and (
        "fiftyone.core.service.ServiceListenTimeout" in eval_res.stderr
        or "Subprocess" in eval_res.stderr and "mongod" in eval_res.stderr
    ):
        pytest.skip("local FiftyOne/mongod is unavailable in this environment")
    assert eval_res.returncode == 0, f"stdout:\n{eval_res.stdout}\n\nstderr:\n{eval_res.stderr}"

    summary = read_json(eval_outdir / "summary.json")
    assert summary["schema_version"] == "eval_run_summary_v2"
    assert summary["task"] == "extraction_sroie"
    assert isinstance(summary.get("run_id"), str) and summary["run_id"]

    rows = read_jsonl(eval_outdir / "results.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row["schema_version"] == "eval_result_row_v2"
    assert row["task"] == summary["task"]
    assert row["run_id"] == summary["run_id"]
    assert isinstance(row["ok"], bool)
    assert row["status_code"] != 200

    thresholds_root = _write_thresholds_root(tmp_path / "thresholds")
    policy_artifact = tmp_path / "policy_out" / "latest.json"

    policy_res = run_policy_cli(
        args=[
            "runtime-decision",
            "--pipeline",
            "extract_only",
            "--run-dir",
            str(eval_outdir),
            "--threshold-profile",
            "extract/default",
            "--thresholds-root",
            str(thresholds_root),
            "--artifact-out",
            str(policy_artifact),
            "--report",
            "text",
        ]
    )
    assert policy_res.returncode == 2, (
        f"stdout:\n{policy_res.stdout}\n\nstderr:\n{policy_res.stderr}"
    )

    decision = read_json(policy_artifact)
    assert decision["schema_version"] == "policy_decision_v2"
    assert decision["pipeline"] == "extract_only"
    assert decision["ok"] is False
    assert decision["enable_extract"] is False
    assert decision["status"] in {"deny", "unknown"}
    assert decision["eval_run_dir"] == str(eval_outdir)
    assert decision["eval_task"] == summary["task"]
    assert decision["eval_run_id"] == summary["run_id"]


def test_e2e_policy_generate_clamp_only_contract_linkage(tmp_path: Path):
    thresholds_root = _write_thresholds_root(tmp_path / "thresholds")
    policy_artifact = tmp_path / "policy_out" / "generate_loop.json"

    policy_res = run_policy_cli(
        args=[
            "runtime-decision",
            "--pipeline",
            "generate_clamp_only",
            "--thresholds-root",
            str(thresholds_root),
            "--artifact-out",
            str(policy_artifact),
            "--report",
            "md",
            "--no-generate-clamp",
        ]
    )
    assert policy_res.returncode == 0, (
        f"stdout:\n{policy_res.stdout}\n\nstderr:\n{policy_res.stderr}"
    )

    decision = read_json(policy_artifact)
    assert decision["schema_version"] == "policy_decision_v2"
    assert decision["pipeline"] == "generate_clamp_only"
    assert decision["status"] == "allow"
    assert decision["ok"] is True
    assert decision["enable_extract"] is None
    assert decision["thresholds_profile"] is None
    assert decision["eval_run_dir"] is None
    assert decision["generate_thresholds_profile"] is None


@pytest.mark.requires_api_key
@pytest.mark.server_live
def test_e2e_live_eval_to_policy_extract_only_contract_linkage(
    require_api_key: bool,
    base_url: str,
    api_key: str,
    sync_probe,
    tmp_path: Path,
):
    if int(sync_probe("/healthz")) == 0:
        pytest.skip("live server not reachable")

    eval_outdir = tmp_path / "eval_results_live"
    eval_pointer = tmp_path / "eval_out" / "latest.json"
    eval_res = run_eval_cli(
        args=[
            "--task",
            "extraction_sroie",
            "--base-url",
            base_url,
            "--api-key",
            api_key,
            "--max-examples",
            "1",
            "--no-print-summary",
            "--save",
            "--outdir",
            str(eval_outdir),
        ],
        env={"EVAL_LATEST_PATH": str(eval_pointer)},
    )
    assert eval_res.returncode == 0, f"stdout:\n{eval_res.stdout}\n\nstderr:\n{eval_res.stderr}"

    summary = read_json(eval_outdir / "summary.json")
    assert summary["schema_version"] == "eval_run_summary_v2"
    assert summary["task"] == "extraction_sroie"
    assert isinstance(summary.get("run_id"), str) and summary["run_id"]
    assert summary.get("metrics", {}).get("base_url") == base_url

    rows = read_jsonl(eval_outdir / "results.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row["schema_version"] == "eval_result_row_v2"
    assert row["task"] == summary["task"]
    assert row["run_id"] == summary["run_id"]
    assert isinstance(row["ok"], bool)
    assert row.get("status_code") == 200

    thresholds_root = _write_thresholds_root(tmp_path / "thresholds")
    policy_artifact = tmp_path / "policy_out" / "live_latest.json"
    policy_res = run_policy_cli(
        args=[
            "runtime-decision",
            "--pipeline",
            "extract_only",
            "--run-dir",
            str(eval_outdir),
            "--threshold-profile",
            "extract/default",
            "--thresholds-root",
            str(thresholds_root),
            "--artifact-out",
            str(policy_artifact),
            "--report",
            "text",
        ]
    )
    assert policy_res.returncode in (0, 2), (
        f"stdout:\n{policy_res.stdout}\n\nstderr:\n{policy_res.stderr}"
    )

    decision = read_json(policy_artifact)
    assert decision["schema_version"] == "policy_decision_v2"
    assert decision["pipeline"] == "extract_only"
    assert isinstance(decision["ok"], bool)
    assert decision["status"] in {"allow", "deny", "unknown"}
    assert decision["eval_run_dir"] == str(eval_outdir)
    assert decision["eval_task"] == summary["task"]
    assert decision["eval_run_id"] == summary["run_id"]

    # Post-policy server verification hook:
    # - health endpoint remains reachable
    # - authenticated models endpoint remains healthy
    # - admin policy endpoint is checked opportunistically if exposed to this key
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(20.0)) as c:
        health = c.get("/healthz")
        health.raise_for_status()
        assert health.status_code == 200

        models = c.get("/v1/models", headers={"X-API-Key": api_key})
        models.raise_for_status()
        models_body = models.json()
        assert isinstance(models_body, dict)
        assert isinstance(models_body.get("models"), list)

        admin_policy = c.get("/v1/admin/policy", headers={"X-API-Key": api_key})
        if admin_policy.status_code == 200:
            body = admin_policy.json()
            assert isinstance(body, dict)
            assert "effective" in body
        elif admin_policy.status_code in (401, 403, 404):
            # 401/403: key is not admin; 404: endpoint not exposed in this deployment
            pass
        else:
            raise AssertionError(
                f"Unexpected /v1/admin/policy status: {admin_policy.status_code} "
                f"body={admin_policy.text[:300]}"
            )
