# ---------- src/llm_server/eval/datasets/summarization.py ----------

from __future__ import annotations

from datasets import load_dataset


def load_cnn_dailymail(split: str = "test"):
    """
    Load CNN/DailyMail summarization dataset.

    We use the "3.0.0" config, which is what most examples use.
    Fields we expect:
      - article: str
      - highlights: str (reference summary)
    """
    return load_dataset("cnn_dailymail", "3.0.0", split=split)