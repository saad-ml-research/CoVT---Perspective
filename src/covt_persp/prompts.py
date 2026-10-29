"""Prompt templates for the reasoning conditions.

Every condition ends by asking for the letter inside <answer></answer>, so the
same extractor scores all of them.
"""

ANSWER_FORMAT = "Give your final answer as the option letter inside <answer></answer>."

CONDITIONS = {
    # Answer immediately, no reasoning text.
    "direct": "Answer with the option letter only. " + ANSWER_FORMAT,
    # Text chain-of-thought. On a CoVT checkpoint this becomes "visual + text":
    # the model emits its visual tokens, then writes the reasoning.
    "text_cot": (
        "Think step by step. First describe the visual elements that matter, "
        "then the pattern or rule they follow, then deduce the answer. " + ANSWER_FORMAT
    ),
}


def format_question(question: str, options: dict[str, str] | None) -> str:
    text = f"Question: {question.strip()}\n"
    if options:
        text += "Options:\n" + "".join(f"{k}. {v}\n" for k, v in options.items())
    return text


def build_prompt(question: str, options: dict[str, str] | None, condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; choose from {sorted(CONDITIONS)}")
    return format_question(question, options) + CONDITIONS[condition]
