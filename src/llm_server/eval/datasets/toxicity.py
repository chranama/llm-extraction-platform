# ---------- src/llm_server/eval/datasets/toxicity.py ----------

from __future__ import annotations

from datasets import load_dataset


def load_real_toxicity_prompts(split: str = "train"):
    """
    Load RealToxicityPrompts from Hugging Face.

    Fields we expect:
      - prompt: str
      - (plus metadata fields we ignore)
    We only use the prompt text and score model generations via a very
    lightweight keyword-based heuristic (for now).
    """
    return load_dataset("allenai/real-toxicity-prompts", split=split)