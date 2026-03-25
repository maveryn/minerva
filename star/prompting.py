from __future__ import annotations

from copy import deepcopy
from typing import Dict, List


RATIONALIZATION_BLOCK = """

Correct answer:
{gold_answer}

Use the correct answer above to write a coherent reasoning trace that leads to it.
Do not mention that the answer was provided.
End with the final answer clearly inside \\boxed{{}}.
""".strip()


def build_star_original_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return deepcopy(messages)


def build_star_rationalization_messages(messages: List[Dict[str, str]], gold_answer: str) -> List[Dict[str, str]]:
    built = deepcopy(messages)
    for message in range(len(built) - 1, -1, -1):
        if built[message].get("role") != "user":
            continue
        user_content = built[message].get("content", "").rstrip()
        built[message]["content"] = (
            f"{user_content}\n\n{RATIONALIZATION_BLOCK.format(gold_answer=gold_answer)}"
        )
        return built
    raise ValueError("Could not find a user message to append the STaR rationalization block")

