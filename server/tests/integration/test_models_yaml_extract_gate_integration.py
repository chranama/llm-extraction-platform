from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_MODEL_ID = "sshleifer/tiny-gpt2"


def _write_models_yaml(tmp_path: Path, *, extract_enabled: bool) -> Path:
    src = REPO_ROOT / "config" / "models.yaml"
    data = yaml.safe_load(src.read_text(encoding="utf-8"))

    profiles = data.get("profiles") or {}
    host = profiles.get("host-transformers") or {}
    host["default_model"] = DEMO_MODEL_ID

    models = host.get("models") or []
    found = False
    for model in models:
        if str(model.get("id", "")) != DEMO_MODEL_ID:
            continue
        caps = model.setdefault("capabilities", {})
        caps["extract"] = bool(extract_enabled)
        assess = model.setdefault("assessment", {})
        assess["assessed"] = True
        found = True
        break

    assert found, f"{DEMO_MODEL_ID} not found in host-transformers profile"

    out = tmp_path / f"models.host_transformers.extract_{int(extract_enabled)}.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("extract_enabled", "expect_capability_block"),
    [
        (True, False),
        (False, True),
    ],
)
async def test_models_yaml_controls_extract_capability(
    monkeypatch,
    tmp_path: Path,
    app_client,
    auth_headers,
    extract_enabled: bool,
    expect_capability_block: bool,
):
    models_yaml = _write_models_yaml(tmp_path, extract_enabled=extract_enabled)
    monkeypatch.setenv("MODELS_PROFILE", "host-transformers")
    monkeypatch.setenv("MODELS_YAML", str(models_yaml))
    monkeypatch.setenv("ENABLE_EXTRACT", "1")

    async with await app_client() as client:
        r_models = await client.get("/v1/models", headers=auth_headers)
        assert r_models.status_code == 200, r_models.text
        body = r_models.json()
        assert body.get("default_model") == DEMO_MODEL_ID
        row = (body.get("models") or [{}])[0]
        assert row.get("id") == DEMO_MODEL_ID
        assert (row.get("capabilities") or {}).get("extract") is bool(extract_enabled), body

        r_extract = await client.post(
            "/v1/extract",
            headers=auth_headers,
            json={
                "schema_id": "sroie_receipt_v1",
                "text": "Vendor: ACME\nTotal: 10.00",
                "model": DEMO_MODEL_ID,
                "cache": False,
                "repair": False,
            },
        )

        if expect_capability_block:
            assert r_extract.status_code in (400, 501), r_extract.text
            payload = r_extract.json()
            assert payload.get("code") in {
                "capability_not_supported",
                "capability_disabled",
            }, payload
        else:
            # Allow path may still fail later (e.g., parsing/validation), but
            # it must not be blocked by capability gating.
            if r_extract.status_code in (400, 501):
                payload = r_extract.json()
                assert payload.get("code") not in {
                    "capability_not_supported",
                    "capability_disabled",
                }, payload
