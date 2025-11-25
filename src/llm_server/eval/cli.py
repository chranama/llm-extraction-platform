# src/llm_server/eval/cli.py

from __future__ import annotations

import argparse
import asyncio
from typing import Callable

from llm_server.eval.runners.base import BaseEvalRunner
from llm_server.eval.runners.gsm8k_runner import make_gsm8k_runner
from llm_server.eval.runners.mmlu_runner import make_mmlu_runner
from llm_server.eval.runners.mbpp_runner import make_mbpp_runner
from llm_server.eval.runners.summarization_runner import make_summarization_runner
from llm_server.eval.runners.toxicity_runner import make_toxicity_runner


# Map task names to runner factory functions
TASK_FACTORIES: dict[str, Callable[[str, str], BaseEvalRunner]] = {
    "gsm8k": make_gsm8k_runner,
    "mmlu": make_mmlu_runner,
    "mbpp": make_mbpp_runner,
    "summarization": make_summarization_runner,
    "toxicity": make_toxicity_runner,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run evaluation tasks against an llm-server instance.",
    )

    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(TASK_FACTORIES.keys()),
        help="Which evaluation task to run.",
    )

    parser.add_argument(
        "--base-url",
        required=True,
        help="Base URL of the llm-server API (e.g. http://localhost:8000).",
    )

    parser.add_argument(
        "--api-key",
        required=True,
        help="API key used to call /v1/generate.",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (default: all available).",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model override identifier (if your server supports it).",
    )

    return parser


async def amain() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.task not in TASK_FACTORIES:
        parser.error(f"Unknown task '{args.task}'. Valid options: {', '.join(TASK_FACTORIES.keys())}")

    factory = TASK_FACTORIES[args.task]
    runner = factory(base_url=args.base_url, api_key=args.api_key)

    await runner.run(
        max_examples=args.max_examples,
        model_override=args.model,
    )


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()