# simulations/traffic/scenarios/demo_clamp.py
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from simulations.traffic.models import RequestSpec, Scenario, TrafficConfig
from simulations.traffic.prompts import generate_prompt


def _mk_prompt(i: int, *, seed: int, size: str | None = None) -> str:
    """
    Back-compat wrapper because generate_prompt may or may not accept `size`.
    """
    try:
        if size is not None:
            return generate_prompt(i, seed=seed, size=size)  # type: ignore[call-arg]
        return generate_prompt(i, seed=seed)  # type: ignore[call-arg]
    except TypeError:
        return generate_prompt(i, seed=seed)  # type: ignore[call-arg]


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


# Minimal mapping to avoid "model_not_allowed" in Demo A llama.cpp profile.
# Add more aliases as needed.
_MODEL_ALIASES: dict[str, str] = {
    "smollm2-360m-instruct": "llama.cpp/SmolLM2-360M-Instruct-Q8_0-GGUF",
    "SmolLM2-360M-Instruct": "llama.cpp/SmolLM2-360M-Instruct-Q8_0-GGUF",
}


def _normalize_model_id(mid: Optional[str]) -> Optional[str]:
    if mid is None:
        return None
    s = str(mid).strip()
    if not s:
        return None
    # If user already passed a canonical id, keep it.
    if s.startswith("llama.cpp/"):
        return s
    # Alias mapping for demo ergonomics
    return _MODEL_ALIASES.get(s, s)


def _resolve_request_model(cfg: TrafficConfig) -> Optional[str]:
    """
    Model id used in request payloads.
    """
    return _normalize_model_id(cfg.model)


def _force_env_var_in_file(text: str, key: str, value: str) -> str:
    """
    Ensure KEY=VALUE exists in a dotenv-style file content, replacing any existing KEY=... line.
    Preserves comments/blank lines.
    """
    lines = text.splitlines()
    out: list[str] = []
    found = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out.append(line)
            continue
        if stripped.startswith(f"{key}="):
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        if out and out[-1].strip() != "":
            out.append("")
        out.append(f"{key}={value}")
    out.append("")  # newline at EOF
    return "\n".join(out)


@dataclass(frozen=True)
class PolicyClampHook:
    """
    Midstream action (docker compose server):
      1) Ask server to write generate SLO snapshot to /app/slo_out (container path)
      2) Run policy in the compose "policy" service so it reads /work/slo_out (host-mounted)
      3) POST /v1/admin/policy/reload so server applies /app/policy_out/latest.json

    Fixes included:
      - Ensures POLICY_MODEL_ID is non-empty and normalized.
      - Ensures LLAMA_MODEL_FILE is injected into the env-file passed to llmctl so
        docker-compose interpolation cannot fail, even if llama profile isn't enabled.
    """
    name: str = "policy_clamp"
    at_s: float = 5.0

    # Keep for back-compat with older call sites
    write_slo_first: bool = True
    wait_for_host_slo: bool = True
    host_wait_s: float = 8.0

    pipeline: str = "generate_clamp_only"
    generate_threshold_profile: str = "generate/portable"
    window_seconds: int = 300

    # REQUIRED by compose policy entrypoint (even if patching disabled)
    model_id: Optional[str] = None

    # Known-good runner defaults
    env_file: str = ".env.docker"
    compose_defaults_profile: str = "docker+jobs"

    # Where we expect artifacts (HOST paths; mounted into containers)
    host_slo_rel: str = "slo_out/generate/latest.json"
    host_policy_out_rel: str = "policy_out/latest.json"

    # Dummy value to satisfy compose interpolation for llama_server.LLAMA_MODEL_FILE
    # (We are NOT starting llama_server in the policy command.)
    llama_model_file_fallback: str = "/models/dummy.gguf"

    def _resolve_policy_model_id(self, cfg: Any) -> str:
        raw = (
            (self.model_id or "").strip()
            or (getattr(cfg, "model", None) or "").strip()
            or (os.getenv("POLICY_MODEL_ID", "") or "").strip()
            or (os.getenv("SIM_MODEL", "") or "").strip()
        )
        mid = _normalize_model_id(raw)
        if not mid:
            raise RuntimeError(
                "PolicyClampHook requires a model id because deploy/compose 'policy' service enforces POLICY_MODEL_ID.\n"
                "Fix: pass --model-id to sim traffic demo-clamp OR set POLICY_MODEL_ID OR set SIM_MODEL."
            )
        return mid

    def _wait_for_file(self, p: Path, *, seconds: float) -> None:
        deadline = time.time() + float(seconds)
        last_err: str | None = None
        while time.time() < deadline:
            try:
                if p.exists() and p.stat().st_size > 0:
                    return
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(0.1)
        raise RuntimeError(
            f"SLO snapshot not visible on host: {p}"
            + (f" (last_err={last_err})" if last_err else "")
        )

    def run(self, client: Any, cfg: Any, repo_root: Any = None) -> None:
        rr = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()

        policy_model_id = self._resolve_policy_model_id(cfg)

        host_slo_path = (rr / self.host_slo_rel).resolve()
        host_policy_out = (rr / self.host_policy_out_rel).resolve()

        # 1) Ensure the server writes SLO snapshot where it is guaranteed to be volume-mounted.
        if self.write_slo_first:
            client.post_admin_write_generate_slo(
                window_seconds=int(self.window_seconds),
                out_path="/app/slo_out/generate/latest.json",
            )

        # 1b) Wait until the host can see it (volume sync / write latency).
        if self.wait_for_host_slo:
            self._wait_for_file(host_slo_path, seconds=float(self.host_wait_s))

        # ----
        # Compose interpolation fix:
        # Compose reads variables from the env-file(s) it is given. In your flow, llmctl
        # generates an effective env-file that may leave LLAMA_MODEL_FILE empty, which
        # causes interpolation to fail even if our Python subprocess env sets it.
        #
        # Solution: create a temp env-file that *explicitly* sets LLAMA_MODEL_FILE.
        # ----
        base_env_file = (rr / self.env_file).resolve()
        effective_env_file = base_env_file
        tmp_env_file: Path | None = None

        try:
            base_lines: list[str] = []
            if base_env_file.exists():
                base_lines = base_env_file.read_text(encoding="utf-8").splitlines()

            def _line_key(line: str) -> str:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    return ""
                return s.split("=", 1)[0].strip()

            # Remove any existing LLAMA_MODEL_FILE lines (including empty ones)
            filtered = [ln for ln in base_lines if _line_key(ln) != "LLAMA_MODEL_FILE"]

            tmp_env_file = base_env_file.with_name(base_env_file.name + ".policyhook.tmp")
            filtered.append(f"LLAMA_MODEL_FILE={self.llama_model_file_fallback}")
            tmp_env_file.write_text("\n".join(filtered) + "\n", encoding="utf-8")
            effective_env_file = tmp_env_file

            # 2) Run policy inside compose service (reads /work/slo_out; writes /work/policy_out)
            cmd = [
                "uv",
                "run",
                "llmctl",
                "--env-file",
                str(effective_env_file),
                "--compose-defaults-profile",
                self.compose_defaults_profile,
                "compose",
                "dc",
                "policy",
                "jobs",
                "--",
                "run",
                "--rm",
                "-e",
                f"POLICY_PIPELINE={self.pipeline}",
                "-e",
                "POLICY_PATCH_MODELS=0",
                "-e",
                f"POLICY_MODEL_ID={policy_model_id}",
                "-e",
                "POLICY_GENERATE_SLO_PATH=/work/slo_out/generate/latest.json",
                "-e",
                "POLICY_OUT_PATH=/work/policy_out/latest.json",
                "--entrypoint",
                "policy",
                "policy",
                "run",
                "--pipeline",
                self.pipeline,
                "--run-dir",
                "latest",
                "--thresholds-root",
                "/app/policy/src/llm_policy/thresholds",
                "--generate-threshold-profile",
                self.generate_threshold_profile,
                "--generate-slo-path",
                "/work/slo_out/generate/latest.json",
                "--artifact-out",
                "/work/policy_out/latest.json",
                "--report",
                "text",
            ]

            # Backstop: also set in process env (in case llmctl/compose merges env sources)
            env = dict(os.environ)
            if not str(env.get("LLAMA_MODEL_FILE", "")).strip():
                env["LLAMA_MODEL_FILE"] = str(self.llama_model_file_fallback)

            r = subprocess.run(
                cmd,
                cwd=str(rr),
                env=env,
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                msg = (r.stderr or r.stdout or "").strip()
                raise RuntimeError(f"policy compose run failed (rc={r.returncode}): {msg}")

            if not host_policy_out.exists():
                raise RuntimeError(f"policy_out artifact not created on host: {host_policy_out}")

            # 3) Reload policy in the server
            client.post_admin_policy_reload()

        finally:
            if tmp_env_file is not None:
                try:
                    tmp_env_file.unlink(missing_ok=True)  # py3.8+ on mac should support this
                except Exception:
                    pass


def _build_requests(cfg: TrafficConfig) -> Iterable[RequestSpec]:
    """
    Goal: show baseline->clamp transition within ONE scenario run.

    Plan:
      - Phase 0: baseline chunk (moderate requests) to generate DB logs
      - Hook fires ~t=5s to compute SLO + run policy + reload
      - Phase 1: clamp-proof burst AFTER hook (requests huge max_new_tokens)
      - Phase 2: steady traffic continues; clamp should remain visible in responses
    """
    seed = int(cfg.seed)
    prompt_size = str(cfg.prompt_size)
    rps = _safe_float(cfg.rps, 2.0)

    model = _resolve_request_model(cfg)

    # Phase 0 baseline chunk length (~6 seconds of traffic)
    baseline_n = max(3, int(rps * 6.0))

    baseline_max_new = _safe_int(cfg.max_new_tokens, 512 if prompt_size != "long" else 1024)
    baseline_temp = _safe_float(cfg.temperature, 0.2)

    # -------------------------
    # Phase 0: baseline chunk
    # -------------------------
    for i in range(0, baseline_n):
        prompt = _mk_prompt(i, seed=seed, size=prompt_size)
        yield RequestSpec(
            idx=i,
            endpoint="generate",
            payload={
                "prompt": prompt,
                "cache": bool(cfg.cache),
                "model": model,
                "max_new_tokens": int(baseline_max_new),
                "temperature": float(baseline_temp),
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "stop": cfg.stop,
            },
            tags={"demo": "CLAMP", "phase": "baseline", "prompt_size": prompt_size},
        )

    # -------------------------
    # Phase 1: clamp-proof burst
    # -------------------------
    heavy_prompt = _mk_prompt(baseline_n, seed=seed, size="long")
    small_prompt = _mk_prompt(baseline_n + 1, seed=seed, size="short")

    heavy_max_new = 4096
    small_max_new = 8

    yield RequestSpec(
        idx=baseline_n,
        endpoint="generate",
        payload={
            "prompt": heavy_prompt,
            "cache": False,
            "model": model,
            "max_new_tokens": heavy_max_new,
            "temperature": float(baseline_temp),
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "stop": cfg.stop,
        },
        tags={"demo": "CLAMP", "phase": "burst", "kind": "H1", "prompt_size": "long"},
    )

    yield RequestSpec(
        idx=baseline_n + 1,
        endpoint="generate",
        payload={
            "prompt": heavy_prompt,
            "cache": False,
            "model": model,
            "max_new_tokens": heavy_max_new,
            "temperature": float(baseline_temp),
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "stop": cfg.stop,
        },
        tags={"demo": "CLAMP", "phase": "burst", "kind": "H2", "prompt_size": "long"},
    )

    yield RequestSpec(
        idx=baseline_n + 2,
        endpoint="generate",
        payload={
            "prompt": small_prompt,
            "cache": False,
            "model": model,
            "max_new_tokens": small_max_new,
            "temperature": 0.0,
        },
        tags={"demo": "CLAMP", "phase": "burst", "kind": "S1", "prompt_size": "short"},
    )

    # -------------------------
    # Phase 2: steady traffic
    # -------------------------
    post_max_new = _safe_int(cfg.max_new_tokens, 2048 if prompt_size == "long" else 512)
    post_temp = _safe_float(cfg.temperature, 0.2)

    for i in range(baseline_n + 3, 10_000):
        prompt = _mk_prompt(i, seed=seed, size=prompt_size)
        yield RequestSpec(
            idx=i,
            endpoint="generate",
            payload={
                "prompt": prompt,
                "cache": bool(cfg.cache),
                "model": model,
                "max_new_tokens": int(post_max_new),
                "temperature": float(post_temp),
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "stop": cfg.stop,
            },
            tags={"demo": "CLAMP", "phase": "steady", "prompt_size": prompt_size},
        )


def _setup(client: Any, cfg: TrafficConfig) -> None:
    # Load whatever current policy_out/latest.json is before starting.
    client.post_admin_policy_reload()


def build_demo_clamp() -> Scenario:
    hook = PolicyClampHook(
        at_s=5.0,
        write_slo_first=True,
        wait_for_host_slo=True,
        host_wait_s=8.0,
        window_seconds=300,
        pipeline="generate_clamp_only",
        generate_threshold_profile="generate/portable",
        model_id=None,  # derived from cfg.model / env
    )

    return Scenario(
        name="demo_clamp",
        endpoint="generate",
        build_requests=_build_requests,
        setup=_setup,
        hooks=[hook],
    )