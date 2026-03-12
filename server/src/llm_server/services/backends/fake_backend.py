from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FakeBackendConfig:
    output_text: str = "ping"


class FakeBackend:
    """
    Deterministic lightweight backend used only for local proof and test-like deployment flows.
    """

    backend_name: str = "fake"

    def __init__(self, *, model_id: str, cfg: FakeBackendConfig | None = None) -> None:
        self.model_id = model_id
        self.cfg = cfg or FakeBackendConfig()

    def generate(self, **_: Any) -> str:
        return self.cfg.output_text

    def is_loaded(self) -> bool:
        return True
