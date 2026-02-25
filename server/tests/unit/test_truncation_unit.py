from __future__ import annotations

import pytest

from llm_server.core.errors import AppError
from llm_server.services.api_deps.extract import truncation as trunc


def test_truncation_guard_ignores_non_applicable_cases():
    trunc.maybe_raise_truncation_error(raw_output="x", effective_max_new_tokens=10, applied_cap=None, stage="extract")
    trunc.maybe_raise_truncation_error(raw_output="x", effective_max_new_tokens=None, applied_cap=5, stage="extract")
    trunc.maybe_raise_truncation_error(raw_output="   ", effective_max_new_tokens=10, applied_cap=5, stage="extract")


def test_truncation_guard_no_error_for_complete_json_like_output():
    text = f'{trunc._JSON_BEGIN} {{"a":1}} {trunc._JSON_END}'
    trunc.maybe_raise_truncation_error(
        raw_output=text,
        effective_max_new_tokens=128,
        applied_cap=64,
        stage="extract_validate",
    )


def test_truncation_guard_raises_on_missing_end_marker():
    text = f'{trunc._JSON_BEGIN} {{"a":1}}'
    with pytest.raises(AppError) as e:
        trunc.maybe_raise_truncation_error(
            raw_output=text,
            effective_max_new_tokens=50,
            applied_cap=25,
            stage="extract_parse",
        )
    err = e.value
    assert err.code == "possible_truncation"
    assert err.status_code == 422
    assert err.extra["stage"] == "extract_parse"
    assert err.extra["has_json_begin"] is True
    assert err.extra["has_json_end"] is False


def test_truncation_guard_raises_on_brace_delta():
    with pytest.raises(AppError) as e:
        trunc.maybe_raise_truncation_error(
            raw_output='{"a":{"b":1}',
            effective_max_new_tokens=90,
            applied_cap=40,
            stage="extract",
        )
    assert e.value.extra["brace_delta"] > 0
