from __future__ import annotations

import json
from pathlib import Path

from llm_policy.utils.fs import (
    atomic_write_text,
    read_jsonl,
    read_yaml,
    write_json,
    write_text,
    write_yaml,
)


def test_atomic_write_text_overwrites(tmp_path: Path) -> None:
    p = tmp_path / "a" / "b" / "out.txt"
    atomic_write_text(p, "first")
    atomic_write_text(p, "second")
    assert p.read_text(encoding="utf-8") == "second"


def test_read_jsonl_skips_invalid_lines_and_non_objects(tmp_path: Path) -> None:
    p = tmp_path / "rows.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"a": 1}),
                "not-json",
                json.dumps([1, 2, 3]),
                json.dumps({"b": 2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = read_jsonl(p)
    assert rows == [{"a": 1}, {"b": 2}]


def test_write_json_creates_parent(tmp_path: Path) -> None:
    p = tmp_path / "nested" / "obj.json"
    write_json(p, {"x": 1})
    assert json.loads(p.read_text(encoding="utf-8")) == {"x": 1}


def test_read_yaml_non_mapping_returns_empty_dict(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    assert read_yaml(p) == {}


def test_write_yaml_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "cfg" / "v.yaml"
    obj = {"a": 1, "b": {"c": True}}
    write_yaml(p, obj)
    assert read_yaml(p) == obj


def test_write_text_creates_parent(tmp_path: Path) -> None:
    p = tmp_path / "x" / "y" / "z.txt"
    write_text(p, "hello")
    assert p.read_text(encoding="utf-8") == "hello"
