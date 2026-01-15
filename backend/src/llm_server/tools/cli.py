from __future__ import annotations

import argparse
import asyncio

from llm_server.db.models import Role
from llm_server.db.session import async_session_maker
from llm_server.tools.api_keys import CreateKeyInput, create_api_key, list_api_keys
from llm_server.tools.db_migrate import migrate_from_env


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llm-tools",
        description="Operational tools for llm-server backend",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # -------------------------
    # migrate-data (existing)
    # -------------------------
    m = sub.add_parser("migrate-data", help="Copy data between DBs (SOURCE_DB_URL -> TARGET_DB_URL)")
    m.add_argument("--batch-size", type=int, default=None, help="Batch size (default: env BATCH_SIZE or 500)")
    m.add_argument(
        "--no-skip-on-conflict",
        action="store_true",
        help="Fail on duplicates instead of skipping (default: skip)",
    )

    # -------------------------
    # api-keys (new)
    # -------------------------
    k = sub.add_parser("api-keys", help="Manage API keys (direct DB access)")
    ksub = k.add_subparsers(dest="api_keys_cmd", required=True)

    k_list = ksub.add_parser("list", help="List API keys")
    k_list.add_argument("--show-secret", action="store_true", help="Print full API key value (dangerous)")

    k_create = ksub.add_parser("create", help="Create an API key")
    k_create.add_argument("--role", type=str, default=Role.standard.value, choices=[r.value for r in Role])
    k_create.add_argument("--label", type=str, default="dev")
    q = k_create.add_mutually_exclusive_group()
    q.add_argument("--quota", type=int, default=None, help="Monthly quota (omit for unlimited)")
    q.add_argument("--unlimited", action="store_true", help="Force unlimited usage")

    return p


async def _cmd_api_keys_list(show_secret: bool) -> int:
    async with async_session_maker() as session:
        rows = await list_api_keys(session)

    if not rows:
        print("No API keys found.")
        return 0

    print("API Keys:")
    for key, role_name in rows:
        print("-" * 60)
        if show_secret:
            print(f"Key:            {key.key}")
        else:
            tail = key.key[-8:] if key.key else ""
            print(f"Key:            ****{tail}")
        print(f"Label:          {key.label}")
        print(f"Active:         {key.active}")
        print(f"Role:           {role_name}")
        print(f"Quota monthly:  {key.quota_monthly}")
        print(f"Quota used:     {key.quota_used}")

    return 0


async def _cmd_api_keys_create(role: str, label: str, quota: int | None, unlimited: bool) -> int:
    quota_monthly = None if unlimited else quota

    async with async_session_maker() as session:
        obj = await create_api_key(
            session,
            CreateKeyInput(role=role, label=label, quota_monthly=quota_monthly),
        )

    print("\n✅ API key created\n")
    print("Key:", obj.key)
    print("Role:", role)
    if quota_monthly is None:
        print("Quota: UNLIMITED")
    else:
        print(f"Quota: {quota_monthly} / month")
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    if args.cmd == "migrate-data":
        migrate_from_env(
            batch_size=args.batch_size,
            skip_on_conflict=(not args.no_skip_on_conflict),
        )
        return 0

    if args.cmd == "api-keys" and args.api_keys_cmd == "list":
        return asyncio.run(_cmd_api_keys_list(args.show_secret))

    if args.cmd == "api-keys" and args.api_keys_cmd == "create":
        return asyncio.run(_cmd_api_keys_create(args.role, args.label, args.quota, args.unlimited))

    p.error(f"Unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())