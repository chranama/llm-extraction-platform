from __future__ import annotations

import json
import pytest

pytestmark = pytest.mark.unit


def test_schema_registry_load_schema_from_schemas_dir(monkeypatch, tmp_path):
    from llm_server.core.schema_registry import load_schema, list_schemas, SchemaNotFoundError

    schema = {
        "title": "UnitSchema",
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
    }

    (tmp_path / "unit_v1.json").write_text(json.dumps(schema), encoding="utf-8")
    monkeypatch.setenv("SCHEMAS_DIR", str(tmp_path))

    idx = list_schemas()
    assert any(s.schema_id == "unit_v1" for s in idx)

    loaded = load_schema("unit_v1")
    assert loaded["title"] == "UnitSchema"

    with pytest.raises(SchemaNotFoundError):
        load_schema("does_not_exist")