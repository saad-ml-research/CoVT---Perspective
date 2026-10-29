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


def options_are_placeholders(options: dict[str, str]) -> bool:
    """True when option texts are just their own labels (e.g. VERIFY: {"A": "a", ...}),
    meaning the real choices are drawn in the image."""
    return all(str(v).strip().strip("().").lower() == k.lower() for k, v in options.items())


def format_question(question: str, options: dict[str, str] | None) -> str:
    text = f"Question: {question.strip()}\n"
    if options and options_are_placeholders(options):
        text += f"The options {', '.join(options)} are shown in the image.\n"
    elif options:
        text += "Options:\n" + "".join(f"{k}. {v}\n" for k, v in options.items())
    return text


def build_prompt(question: str, options: dict[str, str] | None, condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; choose from {sorted(CONDITIONS)}")
    return format_question(question, options) + CONDITIONS[condition]
