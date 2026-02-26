from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _schemas_root(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_schemas = Path(__file__).resolve().parents[3] / "schemas"
    monkeypatch.setenv("SCHEMAS_ROOT", str(repo_schemas))


@pytest.fixture
def write_summary():
    def _write(run_dir: Path, payload: dict) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return run_dir

    return _write


@pytest.fixture
def write_models_yaml():
    def _write(path: Path) -> Path:
        path.write_text(
            "\n".join(
                [
                    "base:",
                    "  models:",
                    "    - id: m1",
                    "      capabilities:",
                    "        extract: false",
                    "profiles:",
                    "  host-transformers:",
                    "    models:",
                    "      - id: m1",
                    "        capabilities:",
                    "          extract: false",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return path

    return _write


@pytest.fixture
def write_thresholds_root():
    def _write(root: Path) -> Path:
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

        generate = root / "generate"
        generate.mkdir(parents=True, exist_ok=True)
        (generate / "portable.yaml").write_text(
            "\n".join(
                [
                    "min_requests: 10",
                    "error_rate:",
                    "  threshold: 0.02",
                    "  cap: 128",
                    "latency_p95_ms:",
                    "  steps:",
                    "    1000: 256",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return root

    return _write


@pytest.fixture
def allow_payload() -> dict:
    return {
        "task": "extraction_sroie",
        "run_id": "run_allow",
        "deployment_key": "dep1",
        "deployment": {"k": "v"},
        "n_total": 100,
        "n_ok": 95,
        "schema_validity_rate": 99.0,
        "required_present_rate": 99.0,
        "doc_required_exact_match_rate": 95.0,
        "field_exact_match_rate": {"total": 100.0},
    }


@pytest.fixture
def deny_payload() -> dict:
    return {
        "task": "extraction_sroie",
        "run_id": "run_deny",
        "deployment_key": "dep1",
        "deployment": {"k": "v"},
        "n_total": 100,
        "n_ok": 5,
        "schema_validity_rate": 10.0,
        "required_present_rate": 10.0,
        "doc_required_exact_match_rate": 10.0,
    }
