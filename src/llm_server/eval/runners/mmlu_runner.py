# src/llm_server/eval/runners/mmlu_runner.py

from __future__ import annotations

from typing import Dict, Any, Optional

import httpx
from datasets import load_dataset

from llm_server.eval.runners.base import BaseEvalRunner, EvalConfig


def make_mmlu_runner(base_url: str, api_key: str) -> "MMLURunner":
    return MMLURunner(
        base_url=base_url,
        api_key=api_key,
        config=EvalConfig(),
    )


class MMLURunner(BaseEvalRunner):
    """
    Multiple choice MMLU evaluation.
    Uses accuracy across questions.
    """

    task_name = "mmlu"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        config: Optional[EvalConfig] = None,
    ):
        super().__init__(base_url=base_url, api_key=api_key, config=config)

    async def _run_impl(self) -> Dict[str, Any]:
        # Use the "all" config of cais/mmlu (merges all subjects)
        ds = load_dataset("cais/mmlu", "all", split="test")

        max_examples = self.config.max_examples or len(ds)
        max_examples = min(max_examples, len(ds))

        print(f"Running {self.task_name} eval on {max_examples} examples")

        correct = 0

        async with httpx.AsyncClient(base_url=self.base_url, timeout=60.0) as client:
            for i in range(max_examples):
                ex = ds[i]

                question = ex["question"]
                choices = ex["choices"]
                answer_idx = ex["answer"]  # integer index

                formatted = self._format_prompt(question, choices)

                pred = await self._query_model(client, formatted)

                predicted_letter = self._extract_choice(pred)

                if predicted_letter is not None:
                    correct_answer = chr(ord("A") + int(answer_idx))
                    if predicted_letter == correct_answer:
                        correct += 1

                n_done = i + 1
                acc = correct / n_done

                print(
                    f"[{n_done}/{max_examples}] correct={correct} acc={acc:.3f}",
                    flush=True,
                )

        accuracy = correct / max_examples if max_examples else 0.0

        print(
            f"\nFinal {self.task_name} accuracy on {max_examples} examples: {accuracy:.3f}"
        )

        return {
            "task": self.task_name,
            "num_examples": max_examples,
            "correct": correct,
            "accuracy": accuracy,
            "model_override": self.config.model_override,
        }

    def _format_prompt(self, question: str, choices: list[str]) -> str:
        letters = ["A", "B", "C", "D"]
        formatted = question.strip() + "\n\n"

        for i, choice in enumerate(choices):
            formatted += f"{letters[i]}) {choice}\n"

        formatted += "\nAnswer:"
        return formatted

    async def _query_model(self, client: httpx.AsyncClient, prompt: str) -> str:
        payload = {
            "prompt": prompt,
            "max_new_tokens": 32,
            "temperature": 0.0,
        }

        # Optional override
        if self.config.model_override:
            payload["model"] = self.config.model_override

        headers = {"X-API-Key": self.api_key}

        resp = await client.post("/v1/generate", json=payload, headers=headers)
        resp.raise_for_status()

        data = resp.json()
        return data.get("output", "")

    @staticmethod
    def _extract_choice(text: str) -> Optional[str]:
        """
        Extracts A/B/C/D from model output
        """
        for c in ["A", "B", "C", "D"]:
            if c in text.upper():
                return c
        return None