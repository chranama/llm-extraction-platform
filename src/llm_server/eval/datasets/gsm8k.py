from __future__ import annotations

from datasets import load_dataset
from typing import Iterator, Dict


def load_gsm8k(split: str = "test", max_examples: int | None = None) -> Iterator[Dict]:
    """
    Returns an iterator of:
    {
        "question": str,
        "answer": str
    }
    """

    ds = load_dataset("gsm8k", "main", split=split)

    if max_examples:
        ds = ds.select(range(max_examples))

    for row in ds:
        yield {
            "question": row["question"],
            "answer": row["answer"],
        }