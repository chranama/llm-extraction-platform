from __future__ import annotations

import simulations.cli as sim_cli


def test_sim_parser_registers_paths_artifacts_and_traffic() -> None:
    parser = sim_cli.build_parser()
    subparsers_action = next(a for a in parser._actions if a.dest == "cmd")
    choices = set(subparsers_action.choices.keys())
    assert {"paths", "artifacts", "traffic"} <= choices


def test_paths_command_uses_defaults() -> None:
    parser = sim_cli.build_parser()
    args = parser.parse_args(["paths"])
    assert args.base_url == "http://127.0.0.1:8000"
    assert args.timeout == 20.0
    assert args.dry_run is False
