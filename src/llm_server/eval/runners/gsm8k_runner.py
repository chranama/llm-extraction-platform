# src/llm_server/eval/runners/gsm8k_runner.py

from __future__ import annotations

import re
from typing import Optional, Dict, Any

import httpx
from datasets import load_dataset

from llm_server.eval.runners.base import BaseEvalRunner, EvalConfig


def make_gsm8k_runner(base_url: str, api_key: str) -> "GSM8KRunner":
    """
    Factory used by the CLI.

    The CLI passes in base_url and api_key, we construct a runner with a
    default EvalConfig (which can be overridden per-run).
    """
    return GSM8KRunner(
        base_url=base_url,
        api_key=api_key,
        config=EvalConfig(),
    )


class GSM8KRunner(BaseEvalRunner):
    """
    Simple GSM8K evaluation runner.

    - Loads the GSM8K "main" test split via Hugging Face datasets.
    - Sends each question to your /v1/generate endpoint.
    - Extracts a numeric answer from the model output.
    - Compares against the gold answer by numeric equality.
    """

    task_name = "gsm8k"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        config: Optional[EvalConfig] = None,
    ) -> None:
        super().__init__(base_url=base_url, api_key=api_key, config=config)

    async def _run_impl(self) -> Dict[str, Any]:
        # 1) Load dataset
        ds = load_dataset("gsm8k", "main", split="test")

        max_examples = self.config.max_examples or len(ds)
        max_examples = min(max_examples, len(ds))

        print(f"Running {self.task_name} eval on {max_examples} examples")

        correct = 0

        async with httpx.AsyncClient(base_url=self.base_url, timeout=60.0) as client:
            for i in range(max_examples):
                ex = ds[i]
                question: str = ex["question"]
                gold: str = ex["answer"]

                pred = await self._query_model(client, question)

                if self._is_correct(pred, gold):
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
            f"\nFinal {self.task_name} accuracy on {max_examples} examples: {accuracy:.3f}",
            flush=True,
        )

        return {
            "task": self.task_name,
            "num_examples": max_examples,
            "correct": correct,
            "accuracy": accuracy,
            "model_override": self.config.model_override,
        }

    async def _query_model(self, client: httpx.AsyncClient, question: str) -> str:
        """
        Call your llm-server /v1/generate endpoint.
        Adjust the payload here if you later add model selection, etc.
        """
        prompt = question.rstrip() + "\n\nAnswer:"

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
    def _extract_number(text: str) -> Optional[str]:
        """
        Very simple heuristic: take the last integer/float in the string.
        GSM8K answers are usually something like "... The answer is 42."
        """
        matches = re.findall(r"-?\d+\.?\d*", text)
        return matches[-1] if matches else None

    def _is_correct(self, pred: str, gold: str) -> bool:
        """
        Compare numeric answers if possible, falling back to stripped-string match.
        """
        p = self._extract_number(pred)
        g = self._extract_number(gold)

        if p is None or g is None:
            return False

        try:
            return float(p) == float(g)
        except ValueError:
            return p.strip() == g.strip()