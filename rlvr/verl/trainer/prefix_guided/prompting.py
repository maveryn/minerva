# --- in verl/trainer/prefix_guided/prompting.py ---

import re
from typing import List, Dict

BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_ROLE_HDR = re.compile(r"^\s*(system|user|assistant)\s*$", re.IGNORECASE)
_STEP_PREFIX_RE = re.compile(
    r"^\s*(?:step\s+)?(?:\(?\d+|\(?[a-zA-Z])[\)\.:\-](?:\s+|(?=[A-Za-z]))",
    re.IGNORECASE,
)
_BULLET_PREFIX_RE = re.compile(r"^\s*[-*•]\s+")
_STEP_BOUNDARY_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:step\s+)?(?:\(?\d+|\(?[a-zA-Z])[\)\.:\-](?:\s+|(?=[A-Za-z]))",
    re.IGNORECASE,
)
_BLANK_LINE_SPLIT_RE = re.compile(r"\n\s*\n+")


def _strip_step_prefix(text: str) -> str:
    """Remove common numbering/bullet prefixes from a step description."""

    stripped = _BULLET_PREFIX_RE.sub("", text, count=1)
    stripped = _STEP_PREFIX_RE.sub("", stripped, count=1)
    return stripped.strip()


def _normalize_step_lines(lines: List[str]) -> str:
    if not lines:
        return ""

    first, *rest = lines
    normalized_lines = [_strip_step_prefix(first)]
    normalized_lines.extend(line.strip() for line in rest if line.strip())

    return "\n".join(l for l in normalized_lines if l).strip()

def _extract_user_text(question: str) -> str:
    parts = {"system": [], "user": [], "assistant": []}
    role = None
    for line in question.splitlines():
        m = _ROLE_HDR.match(line)
        if m:
            role = m.group(1).lower()
            continue
        (parts["user"] if role is None else parts[role]).append(line)
    user_text = "\n".join(parts["user"]).strip()
    return user_text if user_text else question.strip()

def build_answer_guided_prompt(question: str, answer: str, template: str) -> List[Dict[str, str]]:
    """
    RETURN CHAT MESSAGES (no assistant turn). The model will generate the next assistant turn.
    """
    user_question = _extract_user_text(question)
    return [
        {"role": "system", "content": template.strip()},
        {"role": "user",
         "content": (
             f"Problem:\n{user_question}\n\n"
             f"Final Answer (do not change): {answer.strip()}\n\n"
             "Now write the reasoning as numbered steps, separated by blank lines. "
             "The reasoning must be logical, build directly toward the Correct Answer, "
             "and fully agree with the given Final Answer without contradiction or alternatives. "
             "Present the final answer in \\boxed{}."
         )},
    ]

def split_cot_to_steps(cot_text: str) -> List[str]:
    text = cot_text.strip()
    if not text:
        return []

    # Preferred case: the model separates steps with blank lines.
    parts = [p.strip() for p in _BLANK_LINE_SPLIT_RE.split(text) if p.strip()]
    if len(parts) > 1:
        return [_strip_step_prefix(p) for p in parts]

    # Fallback: detect numbered/bulleted steps separated by single newlines.
    lines = cot_text.splitlines()
    steps: List[str] = []
    current: List[str] = []

    for line in lines:
        if not line.strip():
            continue

        if current and _STEP_BOUNDARY_RE.match(line):
            normalized = _normalize_step_lines(current)
            if normalized:
                steps.append(normalized)
            current = []

        current.append(line)

    if current:
        normalized = _normalize_step_lines(current)
        if normalized:
            steps.append(normalized)

    if steps:
        return steps

    return [text]

def extract_boxed_answer(text: str):
    m = BOXED_RE.findall(text)
    return None if not m else m[-1].strip()

def build_prefix_hint_messages(question: str, prefix_steps: List[str], template: str) -> List[Dict[str, str]]:
    """
    RETURN CHAT MESSAGES (no assistant turn). System carries the instruction; user carries Q + prefix.
    """
    user_question = _extract_user_text(question)
    prefix = "\n\n".join(f"{i+1}. {s.strip()}" for i, s in enumerate(prefix_steps))
    return [
        {"role": "system", "content": template.strip()},
        {"role": "user",
         "content": (
             f"Problem:\n{user_question}\n\n"
             f"Partial Steps:\n{prefix}\n\n"
             "Continue the given steps logically to reach the conclusion, and present the final answer in \\boxed{}."
         )},
    ]
