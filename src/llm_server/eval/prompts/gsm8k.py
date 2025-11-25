def build_prompt(example: dict) -> str:
    return f"""
Solve the following math problem step by step and give the final answer.

Question:
{example["question"]}

Answer:
""".strip()