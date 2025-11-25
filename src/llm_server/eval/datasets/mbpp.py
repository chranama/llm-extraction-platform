# ---------- src/llm_server/eval/datasets/mbpp.py ----------

from __future__ import annotations

from datasets import load_dataset


def load_mbpp(split: str = "test"):
    """
    Load MBPP from Hugging Face.

    Common configs on HF include:
      - "sanitized"
      - "full"
    Here we default to "sanitized" to avoid any contamination.
    Adjust the config string if needed based on the actual hub schema.
    """
    return load_dataset("mbpp", "sanitized", split=split)