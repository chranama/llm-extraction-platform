from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


# -------------------------
# Data structures
# -------------------------

@dataclass(frozen=True)
class GoldenFixture:
    name: str
    kind: str  # invoice | ticket | sroie
    schema_id: str
    text: str
    expected: Dict[str, Any]
    contract: Dict[str, Any]


# -------------------------
# Loading helpers
# -------------------------


def _data_dir(base_dir: Optional[Path] = None) -> Path:
    if base_dir is not None:
        return base_dir
    return Path(__file__).resolve().parents[1] / "data"


def load_prompt(name: str, *, base_dir: Optional[Path] = None) -> str:
    p = _data_dir(base_dir) / "prompt" / name
    return p.read_text(encoding="utf-8")


def load_golden_fixture(
    kind: str,
    name: str,
    base_dir: Optional[Path] = None,
) -> GoldenFixture:
    """
    Load a single golden fixture by name.
    """
    d = _data_dir(base_dir) / "golden" / kind

    text = (d / f"{name}.txt").read_text(encoding="utf-8")
    expected = json.loads((d / f"{name}.expected.json").read_text(encoding="utf-8"))
    contract = yaml.safe_load((d / f"{name}.contract.yaml").read_text(encoding="utf-8"))

    schema_id = str(contract.get("schema_id") or "").strip()
    if not schema_id:
        raise ValueError(f"{name}: contract.yaml missing schema_id")

    return GoldenFixture(
        name=name,
        kind=kind,
        schema_id=schema_id,
        text=text,
        expected=expected,
        contract=contract,
    )


def iter_golden_fixtures(
    kind: str,
    base_dir: Optional[Path] = None,
) -> Iterable[GoldenFixture]:
    """
    Iterate all golden fixtures for a given kind.
    """
    d = _data_dir(base_dir) / "golden" / kind
    for contract_path in sorted(d.glob("*.contract.yaml")):
        name = contract_path.name.replace(".contract.yaml", "")
        yield load_golden_fixture(kind=kind, name=name, base_dir=base_dir)


# -------------------------
# Contract evaluation
# -------------------------

def _assert_required_keys(obj: Dict[str, Any], keys: List[str], ctx: str) -> None:
    for k in keys:
        assert k in obj, f"{ctx}: missing required key '{k}'"


def _assert_non_empty_if_present(obj: Dict[str, Any], keys: List[str], ctx: str) -> None:
    for k in keys:
        if k in obj:
            v = obj.get(k)
            assert isinstance(v, str), f"{ctx}: '{k}' must be a string"
            assert v.strip(), f"{ctx}: '{k}' must be non-empty"


def _assert_regex_any_of_if_present(
    obj: Dict[str, Any],
    rules: Dict[str, List[str]],
    ctx: str,
) -> None:
    for field, patterns in rules.items():
        if field not in obj:
            continue
        value = str(obj.get(field) or "")
        for pat in patterns:
            if re.match(pat, value):
                break
        else:
            raise AssertionError(
                f"{ctx}: field '{field}' value '{value}' "
                f"did not match any regex: {patterns}"
            )


def evaluate_contract(
    *,
    extracted: Dict[str, Any],
    contract: Dict[str, Any],
    ctx: str,
) -> None:
    """
    Evaluate extracted output against contract.yaml.
    """
    assertions = contract.get("assertions") or {}

    if "required_keys" in assertions:
        _assert_required_keys(extracted, assertions["required_keys"], ctx)

    if "non_empty_if_present" in assertions:
        _assert_non_empty_if_present(extracted, assertions["non_empty_if_present"], ctx)

    if "regex_any_of_if_present" in assertions:
        _assert_regex_any_of_if_present(
            extracted,
            assertions["regex_any_of_if_present"],
            ctx,
        )
