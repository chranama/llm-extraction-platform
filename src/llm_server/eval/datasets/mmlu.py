# ---------- src/llm_server/eval/datasets/mmlu.py ----------

from __future__ import annotations

from typing import Iterable, Optional

from datasets import load_dataset


def load_mmlu(
    subjects: Optional[Iterable[str]] = None,
    split: str = "test",
):
    """
    Load MMLU from Hugging Face.

    We use the "all" config and (optionally) filter by subject.
    Actual schema on the hub may evolve; expect:
      - question: str
      - choices: list[str]
      - answer: str or int (label)
      - subject: str
    """
    ds = load_dataset("cais/mmlu", "all", split=split)

    if subjects:
        subjects_set = set(subjects)

        def _keep(example):
            s = example.get("subject")
            return s in subjects_set

        ds = ds.filter(_keep)

    return ds
