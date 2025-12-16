# verl/trainer/instruction_guided/prompting.py
from __future__ import annotations
import re
from typing import Dict, List

# Headers and chat block detectors
_ROLE_HDR = re.compile(r"^\s*(system|user|assistant)\s*$", re.IGNORECASE)
_IM_USER_BLOCK = re.compile(r"<\|im_start\|>\s*user\s*(.*?)<\|im_end\|>", re.IGNORECASE | re.DOTALL)
_LLAMA3_USER_BLOCK = re.compile(r"<\|start_header_id\|>\s*user\s*<\|end_header_id\|>\s*(.*?)<\|eot_id\|>", re.IGNORECASE | re.DOTALL)

# System templates
_TPL_INSTRUCTION = (
  "You are given a math problem. Propose a high-level plan to solve it "
  "without performing calculations or symbolic manipulations. "
  "Use a single approach for the plan; do not mention or list alternatives. "
  "Write a short bullet list; start each bullet with an imperative verb "
  "(e.g., 'Define', 'Set up', 'Apply', 'Use', 'Conclude'). "
  "Keep it abstract: you may name variables, define expressions, or cite theorems/lemmas, "
  "but do not plug in values, estimate, or state the final answer. "
  "Limit each bullet to one sentence. "
  "End with a final 'Check' bullet describing how one would verify the result "
  "(units, bounds, substitution, special cases) without carrying it out. "
  "Output only the bullet list—no preamble or conclusion."
)
_TPL_INSTRUCTION_GUIDED = (
  "You are given a math problem. Your goal is to solve it by faithfully executing the provided high-level plan. "
  "Follow the instructions step by step, expanding each bullet into the necessary reasoning and computations. "
  "Carry out all steps carefully and completely, justifying each transition as needed. "
  "If the plan includes a 'Check' step, perform a concise verification immediately before stating the final answer. "
  "Present the final answer clearly inside \\boxed{}."
)

def configure_templates(instruction_template: str, instruction_guided_template: str) -> None:
    global _TPL_INST, _TPL_GUIDED
    _TPL_INST = str(instruction_template or "").strip()
    _TPL_GUIDED = str(instruction_guided_template or "").strip()

def _strip_fences(s: str) -> str:
    # Remove ``` blocks and leading/trailing whitespace; keep plain text bullets.
    s = str(s or "")
    if "```" in s:
        parts = []
        keep = True
        for line in s.splitlines():
            if line.strip().startswith("```"):
                keep = not keep
                continue
            if keep:
                parts.append(line)
        s = "\n".join(parts)
    return s.strip()

def _extract_user_text(question: str) -> str:
  """Extract user-visible problem text from various chat encodings."""
  m = list(_LLAMA3_USER_BLOCK.finditer(question))
  if m:
    return m[-1].group(1).strip()
  m = list(_IM_USER_BLOCK.finditer(question))
  if m:
    return m[-1].group(1).strip()
  parts = {"system": [], "user": [], "assistant": []}
  role = None
  for line in question.splitlines():
    hdr = _ROLE_HDR.match(line)
    if hdr:
      role = hdr.group(1).lower()
      continue
    (parts["user"] if role is None else parts[role]).append(line)
  user_text = "\n".join(parts["user"]).strip()
  return user_text if user_text else question.strip()

# --- NEW: small sanitizers ---
_PROBLEM_HDR = re.compile(r"^\s*problem\s*:\s*", re.IGNORECASE)
def _strip_problem_label(text: str) -> str:
  return _PROBLEM_HDR.sub("", text).strip()

def _sanitize_instructions_text(instr: str) -> str:
  # drop obvious headers / meta
  instr = re.sub(r"^\s*(Human|Assistant|Problem|Solution)\s*:\s*", "", instr, flags=re.IGNORECASE | re.MULTILINE).strip()
  # split into candidate lines
  raw_lines = [l.strip() for l in instr.splitlines() if l.strip()]
  # keep only bullets / numbered bullets
  keep = []
  bullet_pat = re.compile(r"^\s*(?:[-*•]\s+|(?:\(?[0-9ivx]+[\)\.]\s+))", re.IGNORECASE)
  bad_line = re.compile(r"(\\boxed|=|≈|≥|≤|⇒|→|Answer|Final|Therefore|Thus|Option\s*\(|[°])", re.IGNORECASE)
  for l in raw_lines:
    if bullet_pat.match(l) and not bad_line.search(l):
      keep.append(l)
  # enforce 3–7 bullets
  if not keep:
    return ""
  keep = keep[:7]
  if len(keep) < 3:
    return ""
  # ensure final "Check" bullet
  if not any(re.search(r"^\s*(?:[-*•]|(?:\(?[0-9ivx]+[\)\.]))\s*Check\b", b, flags=re.IGNORECASE) for b in keep):
    keep.append("- Check: describe how to verify (units/bounds/substitution/special cases) without carrying it out.")
  return "\n".join(keep)

def _render_template(template: str, *, question: str, instructions: str | None = None) -> str:
  rendered = template.replace("{QUESTION}", question)
  if instructions is not None:
    rendered = rendered.replace("{INSTRUCTIONS}", instructions)
  return rendered


def build_instruction_messages(question: str):
    """
    Return chat messages for getting a high-level plan (instructions) from the model.
    System: _TPL_INST (instruction_template)
    User  : Problem text
    """
    assert _TPL_INST, "instruction_template not configured"
    q = _extract_user_text(question)
    return [
        {"role": "system", "content": _TPL_INST},
        {"role": "user",   "content": f"Problem:\n{q}"},
    ]

def build_instruction_guided_messages(question: str, instructions: str):
    """
    Return chat messages for solving the problem by following the provided plan.
    System: _TPL_GUIDED (instruction_guided_template)
    User  : Problem + High-level Instructions (inlined)
    """
    assert _TPL_GUIDED, "instruction_guided_template not configured"
    q = _extract_user_text(question)
    inst = _strip_fences(instructions)
    if not inst:
        inst = "- Outline the main steps.\n- Apply the steps carefully.\n- Check and state the final answer."
    # IMPORTANT: Inline the plan directly after the header so it survives chat templating.
    user = (
        f"Problem:\n{q}\n\n"
        "High-level Instructions:\n"
        f"{inst}\n\n"
        "Follow the above plan step by step to solve the problem, and present the final answer in \\boxed{}."
    )
    return [
        {"role": "system", "content": _TPL_GUIDED},
        {"role": "user",   "content": user},
    ]
