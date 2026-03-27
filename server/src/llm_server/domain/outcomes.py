from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RunStatus(str, Enum):
    ACCEPTED = "accepted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RunOutcome:
    status: RunStatus
    error_code: str | None = None
    error_stage: str | None = None
    cached: bool | None = None
    repair_attempted: bool | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @classmethod
    def accepted(cls) -> "RunOutcome":
        return cls(status=RunStatus.ACCEPTED)

    @classmethod
    def succeeded(
        cls,
        *,
        cached: bool,
        repair_attempted: bool,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> "RunOutcome":
        return cls(
            status=RunStatus.SUCCEEDED,
            cached=cached,
            repair_attempted=repair_attempted,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @classmethod
    def failed(
        cls,
        *,
        error_code: str,
        error_stage: str | None,
    ) -> "RunOutcome":
        return cls(
            status=RunStatus.FAILED,
            error_code=error_code,
            error_stage=error_stage,
        )
