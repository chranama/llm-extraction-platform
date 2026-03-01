from __future__ import annotations

import cli.main as main_mod


def test_root_parser_registers_expected_commands() -> None:
    parser = main_mod.build_parser()
    subparsers_action = next(a for a in parser._actions if a.dest == "cmd")
    choices = set(subparsers_action.choices.keys())
    assert {"compose", "dev", "k8s"} <= choices


def test_env_file_alias_maps_to_env_override_file() -> None:
    parser = main_mod.build_parser()
    args = parser.parse_args(["--env-file", "envs/local.env", "dev", "doctor"])
    assert args.env_override_file == "envs/local.env"
