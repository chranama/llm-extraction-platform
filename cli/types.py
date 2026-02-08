# cli/types.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GlobalConfig:
    repo_root: Path

    # core paths
    env_file: Path
    compose_yml: Path
    tools_dir: Path
    compose_doctor: Path

    server_dir: Path

    # models config paths (host paths)
    models_yaml: Path

    # runtime ports
    project_name: str
    api_port: str
    ui_port: str
    pgadmin_port: str
    prom_port: str
    grafana_port: str
    prom_host_port: str

    # db defaults (used by tooling)
    pg_user: str
    pg_db: str