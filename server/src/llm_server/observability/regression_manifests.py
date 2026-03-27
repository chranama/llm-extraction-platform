from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

REGRESSION_REPLAY_MANIFEST_VERSION = "regression_replay_manifest_v1"


def _utc_iso_z(ts: datetime | None = None) -> str:
    now = ts or datetime.now(timezone.utc)
    return now.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_regression_replay_manifest(
    *,
    source: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": REGRESSION_REPLAY_MANIFEST_VERSION,
        "generated_at": _utc_iso_z(),
        "source": dict(source),
        "cases": [dict(case) for case in cases],
    }
