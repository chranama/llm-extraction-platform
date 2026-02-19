# simulations/traffic/scenarios/demo_baseline.py
from __future__ import annotations

from typing import Any, Iterable

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


def _baseline_max_new(cfg: TrafficConfig, *, prompt_size: str) -> int:
    """
    Baseline is a CONTROL run: choose values the system can handle and
    avoid clamp-visible territory.

    Given generate/portable can cap as low as 128 under load, keep baseline <= 128
    so it's effectively "non-clampable" even if clamp is active.
    """
    v: int
    if cfg.max_new_tokens is not None:
        try:
            v = int(cfg.max_new_tokens)
        except Exception:
            v = 128
    else:
        # Conservative defaults for portable demos
        v = 128 if prompt_size == "long" else 64

    return max(1, min(int(v), 128))


def _build_requests(cfg: TrafficConfig) -> Iterable[RequestSpec]:
    seed = int(cfg.seed)

    # ------------------------------------------------------------
    # Phase 0: Burst trio to validate baseline behavior
    # ------------------------------------------------------------
    heavy_prompt = _mk_prompt(0, seed=seed, size="long")
    small_prompt = _mk_prompt(1, seed=seed, size="short")

    heavy_max_new = _baseline_max_new(cfg, prompt_size="long")
    small_max_new = 8

    temp = cfg.temperature if cfg.temperature is not None else 0.2

    # H1
    yield RequestSpec(
        idx=0,
        endpoint="generate",
        payload={
            "prompt": heavy_prompt,
            "cache": False,  # don't let cache hide behavior
            "model": cfg.model,
            "max_new_tokens": heavy_max_new,
            "temperature": temp,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "stop": cfg.stop,
        },
        tags={"demo": "BASELINE", "phase": "burst", "kind": "H1", "prompt_size": "long"},
    )

    # H2
    yield RequestSpec(
        idx=1,
        endpoint="generate",
        payload={
            "prompt": heavy_prompt,
            "cache": False,
            "model": cfg.model,
            "max_new_tokens": heavy_max_new,
            "temperature": temp,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "stop": cfg.stop,
        },
        tags={"demo": "BASELINE", "phase": "burst", "kind": "H2", "prompt_size": "long"},
    )

    # S1
    yield RequestSpec(
        idx=2,
        endpoint="generate",
        payload={
            "prompt": small_prompt,
            "cache": False,
            "model": cfg.model,
            "max_new_tokens": small_max_new,
            "temperature": 0.0,
        },
        tags={"demo": "BASELINE", "phase": "burst", "kind": "S1", "prompt_size": "short"},
    )

    # ------------------------------------------------------------
    # Phase 1: Normal open-loop traffic
    # ------------------------------------------------------------
    prompt_size = str(cfg.prompt_size)
    max_new = _baseline_max_new(cfg, prompt_size=prompt_size)
    temperature = cfg.temperature if cfg.temperature is not None else 0.2

    for i in range(3, 10_000):
        prompt = _mk_prompt(i, seed=seed, size=prompt_size)
        yield RequestSpec(
            idx=i,
            endpoint="generate",
            payload={
                "prompt": prompt,
                "cache": bool(cfg.cache),
                "model": cfg.model,
                "max_new_tokens": int(max_new),
                "temperature": float(temperature),
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "stop": cfg.stop,
            },
            tags={"demo": "BASELINE", "phase": "steady", "prompt_size": prompt_size},
        )


def _setup(client: Any, cfg: TrafficConfig) -> None:
    # Baseline runs under whatever policy is currently present; just ensure
    # the server has loaded policy_out/latest.json before traffic starts.
    client.post_admin_policy_reload()


def build_demo_baseline() -> Scenario:
    return Scenario(
        name="demo_baseline",
        endpoint="generate",
        build_requests=_build_requests,
        setup=_setup,
        hooks=None,
    )