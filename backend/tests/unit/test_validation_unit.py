from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_parse_json_strict_accepts_object():
    from llm_server.core.validation import parse_json_strict
    assert parse_json_strict('{"a": 1}') == {"a": 1}


def test_parse_json_strict_rejects_empty():
    from llm_server.core.validation import parse_json_strict, StrictJSONError
    with pytest.raises(StrictJSONError) as e:
        parse_json_strict("   ")
    assert e.value.code == "invalid_json"


def test_parse_json_strict_rejects_code_fence():
    from llm_server.core.validation import parse_json_strict, StrictJSONError
    with pytest.raises(StrictJSONError) as e:
        parse_json_strict("```json\n{\"a\":1}\n```")
    assert e.value.code == "invalid_json"


def test_parse_json_strict_rejects_trailing_garbage():
    from llm_server.core.validation import parse_json_strict, StrictJSONError
    with pytest.raises(StrictJSONError) as e:
        parse_json_strict('{"a": 1} trailing')
    assert e.value.code == "invalid_json"


def test_parse_json_strict_rejects_nan_infinity():
    from llm_server.core.validation import parse_json_strict, StrictJSONError
    with pytest.raises(StrictJSONError):
        parse_json_strict('{"x": NaN}')
    with pytest.raises(StrictJSONError):
        parse_json_strict('{"x": Infinity}')
    with pytest.raises(StrictJSONError):
        parse_json_strict('{"x": -Infinity}')