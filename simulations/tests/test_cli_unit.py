from __future__ import annotations

import argparse
from pathlib import Path

import simulations.cli as sim_cli


def test_normalize_base_url_defaults_and_trims() -> None:
    assert sim_cli._normalize_base_url("") == "http://127.0.0.1:8000"
    assert sim_cli._normalize_base_url("http://x:8000/") == "http://x:8000"


def test_build_runtime_uses_defaults_and_overrides(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sim_cli, "find_repo_root", lambda _p: tmp_path)
    args = argparse.Namespace(
        base_url="http://localhost:9000/",
        api_key="k",
        timeout=12.5,
        dry_run=True,
        policy_out="policy_out/custom.json",
        slo_out=None,
        eval_out="eval_out/custom/latest.json",
    )

    rt = sim_cli._build_runtime(args)
    assert rt.repo_root == tmp_path
    assert rt.base_url == "http://localhost:9000"
    assert rt.api_key == "k"
    assert rt.timeout_s == 12.5
    assert rt.dry_run is True
    assert str(rt.policy_out_path).endswith("policy_out/custom.json")
    assert str(rt.eval_out_path).endswith("eval_out/custom/latest.json")


def test_main_dispatches_handler(monkeypatch) -> None:
    class _Parser:
        def parse_args(self, _argv):
            ns = argparse.Namespace()
            ns._handler = lambda _rt, _args: 7
            return ns

    monkeypatch.setattr(sim_cli, "build_parser", lambda: _Parser())
    monkeypatch.setattr(sim_cli, "_build_runtime", lambda _args: object())
    assert sim_cli.main([]) == 7


def test_main_returns_simerror_code(monkeypatch) -> None:
    class _Parser:
        def parse_args(self, _argv):
            ns = argparse.Namespace()
            ns._handler = lambda _rt, _args: (_ for _ in ()).throw(sim_cli.SimError("bad", code=5))
            return ns

    monkeypatch.setattr(sim_cli, "build_parser", lambda: _Parser())
    monkeypatch.setattr(sim_cli, "_build_runtime", lambda _args: object())
    assert sim_cli.main([]) == 5
