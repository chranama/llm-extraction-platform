from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class DummyState:
    def __init__(self, request_id):
        self.request_id = request_id


class DummyRequest:
    def __init__(self, request_id="rid123"):
        self.state = DummyState(request_id)


def test_to_json_error_shape_and_request_id_header():
    from llm_server.core.errors import _to_json_error

    req = DummyRequest("abc")
    resp = _to_json_error(
        req,  # type: ignore[arg-type]
        status_code=418,
        code="teapot",
        message="nope",
        extra={"foo": "bar"},
    )

    assert resp.status_code == 418
    assert resp.headers.get("X-Request-ID") == "abc"

    body = resp.body.decode("utf-8")
    assert '"code":"teapot"' in body
    assert '"message":"nope"' in body
    assert '"foo":"bar"' in body
    assert '"request_id":"abc"' in body