# simulations/traffic/hooks/policy_compose.py
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simulations.traffic.models import ScenarioHook, TrafficConfig


def _coerce_repo_root(repo_root: Any) -> Path:
    if isinstance(repo_root, Path):
        return repo_root
    if isinstance(repo_root, str) and repo_root.strip():
        return Path(repo_root).resolve()
    # runner passes repo_root; if it didn't, fail loudly
    raise RuntimeError("Policy hook requires repo_root (runner should pass repo_root=...).")


@dataclass(frozen=True)
class PolicyComposeRunHook(ScenarioHook):
    """
    Midstream hook:
      - runs: uv run llmctl ... compose dc policy jobs -- run --rm ... policy run ...
      - then POSTs /v1/admin/policy/reload to load policy_out/latest.json into server

    This uses the repo's tooling (uv + llmctl) and assumes sim is run from repo root.
    """
    name: str = "policy_compose_run"
    at_s: float = 3.0

    # llmctl / compose knobs
    env_file: str = ".env.docker"
    compose_defaults_profile: str = "docker+jobs"
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"

    # policy knobs
    pipeline: str = "generate_clamp_only"
    run_dir: str = "latest"
    thresholds_root: str = "/app/policy/src/llm_policy/thresholds"
    generate_threshold_profile: str = "generate/portable"
    generate_slo_path: str = "/work/slo_out/generate/latest.json"
    artifact_out: str = "/work/policy_out/latest.json"

    def run(self, client: Any, cfg: TrafficConfig, repo_root: Any = None) -> None:
        rr = _coerce_repo_root(repo_root)

        cmd = [
            "uv",
            "run",
            "llmctl",
            "--env-file",
            self.env_file,
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
            f'POLICY_MODEL_ID={self.model_id}',
            "--entrypoint",
            "policy",
            "policy",
            "run",
            "--pipeline",
            self.pipeline,
            "--run-dir",
            self.run_dir,
            "--thresholds-root",
            self.thresholds_root,
            "--generate-threshold-profile",
            self.generate_threshold_profile,
            "--generate-slo-path",
            self.generate_slo_path,
            "--artifact-out",
            self.artifact_out,
        ]

        # Keep environment stable; inherit current env (API keys etc) but don't mutate it.
        env = os.environ.copy()

        try:
            res = subprocess.run(
                cmd,
                cwd=str(rr),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            out = (e.stdout or "").strip()
            err = (e.stderr or "").strip()
            msg = f"policy compose run failed (rc={e.returncode})"
            if out:
                msg += f"\n--- stdout ---\n{out}"
            if err:
                msg += f"\n--- stderr ---\n{err}"
            raise RuntimeError(msg) from e

        # Now load what policy wrote into the running server process
        client.post_admin_policy_reload()