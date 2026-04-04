from __future__ import annotations

import argparse
import asyncio
import logging

from llm_server.application.process_extract_job import process_extract_job_once
from llm_server.main import create_app
from llm_server.services.extract_jobs import queue_from_app

logger = logging.getLogger("llm_server.worker.extract_jobs")


async def run_worker(*, once: bool = False, poll_timeout_seconds: int = 5) -> None:
    app = create_app()
    async with app.router.lifespan_context(app):
        queue = queue_from_app(app)
        if queue is None:
            raise RuntimeError("extract job queue unavailable; enable Redis for worker mode")

        sessionmaker = app.state.db_sessionmaker if hasattr(app.state, "db_sessionmaker") else None
        if sessionmaker is None:
            import llm_server.db.session as db_session

            sessionmaker = db_session.get_sessionmaker()

        while True:
            result = await process_extract_job_once(
                app=app,
                sessionmaker=sessionmaker,
                queue=queue,
                timeout_seconds=poll_timeout_seconds,
            )
            if result is None and once:
                return
            if once and result is not None:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extract job worker")
    parser.add_argument("--once", action="store_true", help="Process at most one queued job")
    parser.add_argument(
        "--poll-timeout-seconds",
        type=int,
        default=5,
        help="Redis blocking pop timeout",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(
        run_worker(once=bool(args.once), poll_timeout_seconds=int(args.poll_timeout_seconds))
    )


if __name__ == "__main__":
    main()
