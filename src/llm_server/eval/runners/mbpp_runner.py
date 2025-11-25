from __future__ import annotations

from typing import Dict, Any

import httpx
from datasets import load_dataset

from llm_server.eval.runners.base import BaseEvalRunner, EvalConfig


def make_mbpp_runner(base_url: str, api_key: str) -> "MBPPRunner":
    """
    Factory used by the CLI.

    The CLI passes in base_url and api_key; we construct a runner with a
    default EvalConfig (which can be overridden per-run).
    """
    return MBPPRunner(
        base_url=base_url,
        api_key=api_key,
        config=EvalConfig(),
    )


class MBPPRunner(BaseEvalRunner):
    """
    Simple MBPP evaluation runner.

    - Loads the MBPP 'sanitized' test split from Hugging Face datasets.
    - Sends each natural language problem description to /v1/generate.
    - Asks the model to write a Python function.
    - Uses a simple string-based correctness heuristic:
      we check whether the reference solution code (with whitespace stripped)
      appears inside the model output (also whitespace-stripped).

    This is intentionally lightweight and does *not* execute model code.
    """

    async def _run_impl(self) -> Dict[str, Any]:
        # 1) Load dataset (config: 'sanitized', split: 'test')
        ds = load_dataset("mbpp", "sanitized", split="test")

        max_examples = self.config.max_examples or len(ds)
        max_examples = min(max_examples, len(ds))

        print(f"Running MBPP eval on {max_examples} examples")

        correct = 0

        async with httpx.AsyncClient(base_url=self.base_url, timeout=60.0) as client:
            for i in range(max_examples):
                ex = ds[i]

                # HF MBPP fields:
                # - "text": natural language description of the task
                # - "code": reference solution
                description: str = ex.get("prompt") or ex.get("text") or ""
                gold_code: str = ex["code"]

                prompt = self._build_prompt(description)
                pred = await self._query_model(client, prompt)

                if self._is_correct(pred, gold_code):
                    correct += 1

                n_done = i + 1
                acc_so_far = correct / n_done
                print(
                    f"[{n_done}/{max_examples}] "
                    f"correct={correct} "
                    f"acc={acc_so_far:.3f}",
                    flush=True,
                )

        accuracy = correct / max_examples if max_examples > 0 else 0.0

        print(
            f"\nFinal MBPP accuracy on {max_examples} examples: {accuracy:.3f}",
            flush=True,
        )

        return {
            "task": "mbpp",
            "num_examples": max_examples,
            "correct": correct,
            "accuracy": accuracy,
            "model_override": self.config.model_override,
        }

    def _build_prompt(self, description: str) -> str:
        """
        Construct the prompt sent to /v1/generate.

        You can later swap this for a template file in prompts/ if desired.
        """
        return (
            "You are a helpful Python coding assistant.\n\n"
            "Write ONLY the Python function that solves the following task.\n"
            "Do not include explanations, comments, or test code.\n\n"
            "Task:\n"
            f"{description.strip()}\n\n"
            "Answer with valid Python code implementing the function:\n"
        )

    async def _query_model(self, client: httpx.AsyncClient, prompt: str) -> str:
        """
        Call your llm-server /v1/generate endpoint.
        Adjust the payload here if you later add model selection, etc.
        """
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "max_new_tokens": 256,
            "temperature": 0.0,
        }

        # If you later wire model selection into your API, you can add:
        # if self.config.model_override:
        #     payload["model"] = self.config.model_override

        headers = {"X-API-Key": self.api_key}

        resp = await client.post("/v1/generate", json=payload, headers=headers)
        resp.raise_for_status()

        data = resp.json()
        # Your /v1/generate returns {"model": ..., "output": ..., "cached": ...}
        return data.get("output", "")

    @staticmethod
    def _normalize_code(code: str) -> str:
        """
        Extremely simple normalization: strip all whitespace.

        This tries to be robust to formatting differences while still
        requiring the model to approximately reproduce the reference.
        """
        return "".join(code.split())

    def _is_correct(self, pred_code: str, gold_code: str) -> bool:
        """
        Heuristic correctness: the normalized gold code must appear somewhere
        in the normalized prediction.

        This is *not* a true functional correctness check, but avoids running
        untrusted model code as part of the eval.
        """
        if not pred_code or not gold_code:
            return False

        pred_norm = self._normalize_code(pred_code)
        gold_norm = self._normalize_code(gold_code)

        return gold_norm in pred_norm