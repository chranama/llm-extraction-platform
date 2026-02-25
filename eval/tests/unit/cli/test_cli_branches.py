from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import llm_eval.cli as cli
from llm_eval.runners.base import BaseEvalRunner, EvalConfig


class _FakeRunner(BaseEvalRunner):
    task_name = "fake_task"

    def __init__(self, payload: dict[str, Any]):
        super().__init__(base_url="http://fake", api_key="fake", config=EvalConfig())
        self._payload = payload

    async def _run_impl(self) -> Any:
        return self._payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args,cfg,get_key",
    [
        (
            dict(task=None, list_tasks=False, base_url=None, api_key=None),
            {"service": {"base_url": "http://svc"}, "datasets": {"fake_task": {"enabled": True}}},
            "APIKEY",
        ),
        (
            dict(task="unknown", list_tasks=False, base_url=None, api_key=None),
            {"service": {"base_url": "http://svc"}, "datasets": {"unknown": {"enabled": True}}},
            "APIKEY",
        ),
        (
            dict(task="fake_task", list_tasks=False, base_url=None, api_key=None),
            {"datasets": {"fake_task": {"enabled": True}}},
            "APIKEY",
        ),
        (
            dict(task="fake_task", list_tasks=False, base_url=None, api_key=None),
            {"service": {"base_url": "http://svc"}, "datasets": {"fake_task": {"enabled": True}}},
            None,
        ),
    ],
)
async def test_cli_parser_error_branches(monkeypatch, args, cfg, get_key):
    monkeypatch.setattr(cli, "load_eval_yaml", lambda _p: cfg)
    monkeypatch.setattr(cli, "get_api_key", lambda _c: get_key)
    monkeypatch.setattr(cli, "TASK_FACTORIES", {"fake_task": lambda base_url, api_key, cfg: None})

    monkeypatch.setattr(
        cli.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: cli.argparse.Namespace(
            config="ignored.yaml",
            task=args["task"],
            list_tasks=args["list_tasks"],
            base_url=args["base_url"],
            api_key=args["api_key"],
            max_examples=None,
            model=None,
            print_summary=False,
            no_print_summary=True,
            save=False,
            no_save=True,
            outdir=None,
            debug_n=0,
            debug_fields=None,
        ),
    )

    with pytest.raises(SystemExit) as e:
        await cli.amain()
    assert e.value.code == 2


@pytest.mark.asyncio
async def test_cli_print_summary_overrides_no_print_and_debug_defaults(
    monkeypatch, tmp_path: Path, capsys
):
    monkeypatch.setattr(
        cli,
        "load_eval_yaml",
        lambda _: {
            "service": {"base_url": "http://svc"},
            "run": {"outdir_root": 123},  # invalid -> fallback branch
            "datasets": {"fake_task": {"enabled": True}},
        },
    )
    monkeypatch.setattr(cli, "get_api_key", lambda _cfg: "APIKEY")

    payload = {
        "summary": {"task": "fake_task", "run_id": "RID", "base_url": "http://svc"},
        "results": [],
        "report_text": None,
        "config": {},
    }
    monkeypatch.setattr(
        cli,
        "TASK_FACTORIES",
        {"fake_task": lambda base_url, api_key, cfg: _FakeRunner(payload)},
    )

    monkeypatch.setattr(
        cli.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: cli.argparse.Namespace(
            config="ignored.yaml",
            task="fake_task",
            list_tasks=False,
            base_url=None,
            api_key=None,
            max_examples=None,
            model=None,
            print_summary=True,
            no_print_summary=True,
            save=False,
            no_save=True,
            outdir=None,
            debug_n=1,
            debug_fields=None,
        ),
    )

    await cli.amain()
    out = capsys.readouterr().out
    assert "DEBUG (first 0 results)" in out
    assert "(no per-example results returned)" in out
    assert '"task": "fake_task"' in out  # summary printed despite no_print_summary=True


@pytest.mark.asyncio
async def test_cli_debug_skips_non_dict_when_coerce_is_patched(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "load_eval_yaml",
        lambda _: {
            "service": {"base_url": "http://svc"},
            "datasets": {"fake_task": {"enabled": True}},
        },
    )
    monkeypatch.setattr(cli, "get_api_key", lambda _cfg: "APIKEY")

    payload = {"summary": {}, "results": [], "report_text": None, "config": {}}
    monkeypatch.setattr(
        cli,
        "TASK_FACTORIES",
        {"fake_task": lambda base_url, api_key, cfg: _FakeRunner(payload)},
    )

    monkeypatch.setattr(
        cli,
        "_coerce_nested_payload",
        lambda _p: ({"task": "fake_task", "run_id": "RID"}, ["not-a-dict"], None, {}),
    )

    monkeypatch.setattr(
        cli.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: cli.argparse.Namespace(
            config="ignored.yaml",
            task="fake_task",
            list_tasks=False,
            base_url=None,
            api_key=None,
            max_examples=None,
            model=None,
            print_summary=False,
            no_print_summary=True,
            save=False,
            no_save=True,
            outdir=None,
            debug_n=1,
            debug_fields=None,
        ),
    )

    await cli.amain()
    out = capsys.readouterr().out
    assert "DEBUG (first 1 results)" in out


def test_cli_main_calls_asyncio_run(monkeypatch):
    called = {"argv": None}

    def _fake_run(coro):
        called["argv"] = coro
        try:
            coro.close()
        except Exception:
            pass

    monkeypatch.setattr(cli.asyncio, "run", _fake_run)
    cli.main()
    assert called["argv"] is not None
