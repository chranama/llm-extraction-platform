from __future__ import annotations

from dataclasses import dataclass, field, replace

from llm_server.domain.outcomes import RunOutcome


@dataclass(frozen=True, slots=True)
class RunIdentity:
    request_id: str | None
    trace_id: str | None
    job_id: str | None = None


@dataclass(frozen=True, slots=True)
class RunPolicySnapshot:
    generate_max_new_tokens_cap: int | None = None


@dataclass(frozen=True, slots=True)
class ExtractionRun:
    identity: RunIdentity
    route: str
    schema_id: str
    requested_model_id: str | None = None
    resolved_model_id: str | None = None
    cache_enabled: bool = True
    repair_enabled: bool = True
    requested_max_new_tokens: int | None = None
    effective_max_new_tokens: int | None = None
    policy: RunPolicySnapshot | None = None
    outcome: RunOutcome = field(default_factory=RunOutcome.accepted)

    @property
    def request_id(self) -> str | None:
        return self.identity.request_id

    @property
    def trace_id(self) -> str | None:
        return self.identity.trace_id

    @property
    def job_id(self) -> str | None:
        return self.identity.job_id

    def with_resolution(
        self,
        *,
        resolved_model_id: str | None,
        effective_max_new_tokens: int | None,
    ) -> "ExtractionRun":
        return replace(
            self,
            resolved_model_id=resolved_model_id,
            effective_max_new_tokens=effective_max_new_tokens,
        )

    def with_policy(self, policy: RunPolicySnapshot | None) -> "ExtractionRun":
        return replace(self, policy=policy)

    def with_outcome(self, outcome: RunOutcome) -> "ExtractionRun":
        return replace(self, outcome=outcome)
