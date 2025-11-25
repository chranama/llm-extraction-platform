from __future__ import annotations

from typing import Dict, Any, List

import httpx
from datasets import load_dataset

from llm_server.eval.runners.base import BaseEvalRunner, EvalConfig


def make_summarization_runner(base_url: str, api_key: str) -> "SummarizationRunner":
    """
    Factory used by the CLI.

    The CLI passes in base_url and api_key, we construct a runner with a
    default EvalConfig (which can be overridden per-run).
    """
    return SummarizationRunner(
        base_url=base_url,
        api_key=api_key,
        config=EvalConfig(),
    )


class SummarizationRunner(BaseEvalRunner):
    """
    Simple summarization evaluation runner.

    - Uses CNN/DailyMail ("cnn_dailymail", version "3.0.0") from Hugging Face.
    - For each article, asks the model to produce a short summary.
    - Compares the model summary to the reference highlights using a
      lightweight ROUGE-L–style token overlap metric.
    """

    async def _run_impl(self) -> Dict[str, Any]:
        # 1) Load dataset
        ds = load_dataset("cnn_dailymail", "3.0.0", split="test")

        max_examples = self.config.max_examples or len(ds)
        max_examples = min(max_examples, len(ds))

        print(f"Running summarization eval on {max_examples} examples")

        scores: List[float] = []

        async with httpx.AsyncClient(base_url=self.base_url, timeout=60.0) as client:
            for i in range(max_examples):
                ex = ds[i]
                article: str = ex["article"]
                reference: str = ex["highlights"]

                prompt = self._build_prompt(article)
                pred = await self._query_model(client, prompt)

                score = self._rouge_l_recall(pred, reference)
                scores.append(score)

                n_done = i + 1
                avg_so_far = sum(scores) / n_done
                print(
                    f"[{n_done}/{max_examples}] "
                    f"ROUGE-L (recall-like)={score:.3f} "
                    f"avg={avg_so_far:.3f}",
                    flush=True,
                )

        avg_score = sum(scores) / len(scores) if scores else 0.0

        print(
            f"\nFinal summarization ROUGE-L-like score on {max_examples} examples: {avg_score:.3f}",
            flush=True,
        )

        return {
            "task": "summarization",
            "num_examples": max_examples,
            "avg_rouge_l_recall": avg_score,
            "model_override": self.config.model_override,
        }

    def _build_prompt(self, article: str) -> str:
        """
        Build a simple instruction-style prompt for summarization.
        """
        return (
            "You are a helpful assistant that writes concise news summaries.\n\n"
            "Article:\n"
            f"{article.strip()}\n\n"
            "Write a short summary of the article in 2–3 sentences:\n"
        )

    async def _query_model(self, client: httpx.AsyncClient, prompt: str) -> str:
        """
        Call your llm-server /v1/generate endpoint.
        """
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "max_new_tokens": 256,
            "temperature": 0.2,
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
    def _rouge_l_recall(pred: str, ref: str) -> float:
        """
        Very lightweight ROUGE-L-ish metric based on token-level LCS recall.

        This is not a perfect implementation of ROUGE-L, but it's:
        - deterministic
        - fast
        - dependency-free
        
        It computes:
            recall = LCS(pred_tokens, ref_tokens) / len(ref_tokens)
        """
        pred_tokens = SummarizationRunner._tokenize(pred)
        ref_tokens = SummarizationRunner._tokenize(ref)

        if not ref_tokens:
            return 0.0

        lcs_len = SummarizationRunner._lcs_length(pred_tokens, ref_tokens)
        return lcs_len / len(ref_tokens)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # Super simple whitespace tokenizer; you can swap in something fancier later.
        return text.lower().strip().split()

    @staticmethod
    def _lcs_length(a: list[str], b: list[str]) -> int:
        """
        Classic dynamic programming LCS length for two token sequences.
        """
        len_a, len_b = len(a), len(b)
        dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]

        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[len_a][len_b]