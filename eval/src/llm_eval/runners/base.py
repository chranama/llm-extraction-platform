# eval/src/llm_eval/runners/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol, TYPE_CHECKING


if TYPE_CHECKING:
    # Type-only imports so importing llm_eval.runners.* stays light.
    from llm_eval.client.http_client import ExtractErr, ExtractOk, GenerateErr, GenerateOk


@dataclass
class EvalConfig:
    """
    Shared configuration object for evaluation runs.

    - max_examples: optional cap on number of examples to evaluate
    - model_override: optional model name to pass through to the server
    """
    max_examples: Optional[int] = None
    model_override: Optional[str] = None


# -------------------------
# Runtime deps (DI seam)
# -------------------------


class HttpClient(Protocol):
    """
    Structural protocol for the eval HTTP client.

    Runners must NOT depend on httpx directly.
    They only depend on this protocol + typed results.
    """

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        model: Optional[str] = None,
        cache: Optional[bool] = None,
    ) -> "GenerateOk | GenerateErr":
        ...

    async def extract(
        self,
        *,
        schema_id: str,
        text: str,
        model: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        cache: bool = True,
        repair: bool = True,
    ) -> "ExtractOk | ExtractErr":
        ...

    # NEW: /modelz probe (best-effort preflight)
    async def modelz(self) -> Any:
        """
        Returns a parsed JSON dict on success, or an error-shaped object/dict on failure.

        This is intentionally typed as Any in the Protocol to keep runner imports light
        and avoid tight coupling to the client module's typed results.
        """
        ...


ClientFactory = Callable[[str, str], HttpClient]
RunIdFactory = Callable[[], str]
EnsureDirFn = Callable[[str], None]
OpenFn = Callable[..., Any]
DatasetOverrides = dict[str, Any]  # callables keyed by dataset name


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_ensure_dir(path: str) -> None:
    import os
    os.makedirs(path, exist_ok=True)


@dataclass(frozen=True)
class RunnerDeps:
    """
    Injectable runtime dependencies.

    In prod you use defaults.
    In tests you pass fakes to avoid HTTP/filesystem/time nondeterminism.

    dataset_overrides:
      - A dict of callables keyed by dataset name (runner-defined keys).
      - Lets tests inject small fixture iterators without monkeypatching imports.
    """
    client_factory: ClientFactory
    run_id_factory: RunIdFactory = _default_run_id
    ensure_dir: EnsureDirFn = _default_ensure_dir
    open_fn: OpenFn = open  # allows in-memory file capture in tests if you want
    dataset_overrides: DatasetOverrides = field(default_factory=dict)


def default_deps() -> RunnerDeps:
    # Import inside function so llm_eval package import is light-weight
    from llm_eval.client.http_client import HttpEvalClient

    return RunnerDeps(
        client_factory=lambda base_url, api_key: HttpEvalClient(base_url=base_url, api_key=api_key)
    )


# -------------------------
# Base runner
# -------------------------


class BaseEvalRunner(ABC):
    task_name: str = "base"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        config: Optional[EvalConfig] = None,
        *,
        deps: Optional[RunnerDeps] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.config = config or EvalConfig()
        self.deps = deps or default_deps()

        # NEW: filled once per run, best-effort snapshot of server reality (/modelz)
        self._server_snapshot: Optional[dict[str, Any]] = None

    def make_client(self) -> HttpClient:
        return self.deps.client_factory(self.base_url, self.api_key)

    def new_run_id(self) -> str:
        return self.deps.run_id_factory()

    def ensure_dir(self, path: str) -> None:
        self.deps.ensure_dir(path)

    def open_file(self, *args: Any, **kwargs: Any) -> Any:
        return self.deps.open_fn(*args, **kwargs)

    def get_dataset_callable(self, key: str, default: Any) -> Any:
        """
        Returns an override callable if present, else the provided default.
        """
        overrides = self.deps.dataset_overrides
        if isinstance(overrides, dict) and key in overrides and overrides[key] is not None:
            return overrides[key]
        return default

    @staticmethod
    def _as_str(x: Any) -> Optional[str]:
        if isinstance(x, str):
            s = x.strip()
            return s if s else None
        return None

    @staticmethod
    def _as_dict(x: Any) -> Optional[dict[str, Any]]:
        return x if isinstance(x, dict) else None

    @staticmethod
    def _safe_get(d: Any, *path: str) -> Any:
        cur: Any = d
        for p in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(p)
        return cur

    @classmethod
    def _compute_effective_model_id(cls, snap: dict[str, Any]) -> Optional[str]:
        """
        Prefer the actually-loaded model id if present; else runtime_default; else default_model_id.

        Mirrors the “single loaded model” reality in server/:
          - loaded_model_id is truth
          - runtime_default_model_id is next best (multi-model mode)
          - default_model_id is config default
        """
        loaded = cls._as_str(snap.get("loaded_model_id"))
        if loaded:
            return loaded
        runtime_default = cls._as_str(snap.get("runtime_default_model_id"))
        if runtime_default:
            return runtime_default
        default_mid = cls._as_str(snap.get("default_model_id"))
        return default_mid

    async def _preflight_modelz(self, client: HttpClient) -> dict[str, Any]:
        """
        Best-effort call to /modelz to capture deployment_key + loaded/default model ids.
        Never raises.
        """
        base: dict[str, Any] = {
            "ok": False,
            "stage": "modelz_unavailable",
            "base_url": self.base_url,
        }

        fn = getattr(client, "modelz", None)
        if not callable(fn):
            base["stage"] = "client_missing_modelz"
            return base

        try:
            resp = await fn()
        except Exception as e:
            base["stage"] = "modelz_exception"
            base["error"] = f"{type(e).__name__}: {e}"
            return base

        # Allow several shapes:
        #  - dict: parsed JSON body
        #  - typed error object with .__dict__
        #  - anything else => wrap
        snap = self._as_dict(resp)
        if snap is None:
            try:
                snap = self._as_dict(getattr(resp, "__dict__", None))
            except Exception:
                snap = None

        if snap is None:
            base["stage"] = "modelz_bad_shape"
            base["raw_type"] = type(resp).__name__
            return base

        # Normalize a compact, correlation-friendly view
        deployment = self._as_dict(snap.get("deployment")) or {}
        deployment_key = self._as_str(deployment.get("deployment_key")) or self._as_str(
            self._safe_get(deployment, "identity", "deployment_key")  # defensive, just in case
        )

        out: dict[str, Any] = {
            "ok": True,
            "stage": "modelz_ok",
            "timestamp_utc": self._as_str(snap.get("timestamp_utc")) or None,
            "status": self._as_str(snap.get("status")) or None,
            "http": {
                # /modelz returns JSONResponse; we don't always get HTTP status here.
                # Client may inject it later; keep placeholder for future.
            },
            "deployment": {
                "deployment_key": deployment_key,
                "profiles": self._as_dict(deployment.get("profiles")) or None,
                "container": deployment.get("container"),
                "platform": self._as_dict(deployment.get("platform")) or None,
                "accelerators": self._as_dict(deployment.get("accelerators")) or None,
                "routing": self._as_dict(deployment.get("routing")) or None,
                "identity": self._as_dict(deployment.get("identity")) or None,
            },
            # top-level mirrors from server/ payload
            "default_model_id": self._as_str(snap.get("default_model_id")),
            "default_backend": self._as_str(snap.get("default_backend")),
            "model_loaded": bool(snap.get("model_loaded", False)),
            "loaded_model_id": self._as_str(snap.get("loaded_model_id")),
            "runtime_default_model_id": self._as_str(snap.get("runtime_default_model_id")),
            "model_readiness_mode": self._as_str(snap.get("model_readiness_mode")),
            "model_load_mode": self._as_str(snap.get("model_load_mode")),
        }

        out["effective_model_id"] = self._compute_effective_model_id(out)

        # Keep a small slice of the model section (if present) for debugging correlation
        model_block = self._as_dict(snap.get("model"))
        if model_block is not None:
            out["model"] = {
                "required": bool(model_block.get("required", False)),
                "status": self._as_str(model_block.get("status")),
                "ok": bool(model_block.get("ok", False)),
            }

        return out

    def server_snapshot(self) -> Optional[dict[str, Any]]:
        """
        Accessor for runners: returns the last captured /modelz snapshot (if any).
        """
        return self._server_snapshot

    async def run(
        self,
        max_examples: Optional[int] = None,
        model_override: Optional[str] = None,
    ) -> Any:
        if max_examples is not None:
            self.config.max_examples = max_examples
        if model_override is not None:
            self.config.model_override = model_override

        # NEW: capture modelz once per run (best-effort).
        # This enables eval artifacts to correlate via deployment_key and to reflect the loaded model.
        try:
            client = self.make_client()
            self._server_snapshot = await self._preflight_modelz(client)
            # Intentionally DO NOT close client here; runners may reuse it.
            # The concrete HttpEvalClient will be closed by callers if needed.
        except Exception:
            self._server_snapshot = {"ok": False, "stage": "modelz_preflight_failed"}

        return await self._run_impl()

    @abstractmethod
    async def _run_impl(self) -> Any:
        raise NotImplementedError