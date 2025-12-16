import importlib.util
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PROMPTING_PATH = REPO_ROOT / "verl" / "trainer" / "prefix_guided" / "prompting.py"
spec = importlib.util.spec_from_file_location("prefix_prompting", PROMPTING_PATH)
prompting = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(prompting)

split_cot_to_steps = prompting.split_cot_to_steps


@pytest.mark.parametrize(
    "cot,expected",
    [
        (
            "1. Compute the discriminant.\n\n2. Solve for the roots.\n\n3. State the final answer.",
            [
                "Compute the discriminant.",
                "Solve for the roots.",
                "State the final answer.",
            ],
        ),
        (
            "1. Expand the expression.\n2. Combine like terms.\n3. Present the result.",
            [
                "Expand the expression.",
                "Combine like terms.",
                "Present the result.",
            ],
        ),
        (
            "Step 1: Substitute the values.\nStep 2: Simplify.\nStep 3: Conclude.",
            ["Substitute the values.", "Simplify.", "Conclude."],
        ),
        (
            "- Step 1: Evaluate the limit.\n- Step 2: Apply L'Hospital's rule.\n- Step 3: Report the value.",
            [
                "Evaluate the limit.",
                "Apply L'Hospital's rule.",
                "Report the value.",
            ],
        ),
        (
            "Identify key terms.\n\nExplain their relationships.\n\nDraw the conclusion.",
            [
                "Identify key terms.",
                "Explain their relationships.",
                "Draw the conclusion.",
            ],
        ),
        (
            "State the assumption.\n   \nDescribe the derivation.\n\n  \nSummarize the result.",
            [
                "State the assumption.",
                "Describe the derivation.",
                "Summarize the result.",
            ],
        ),
    ],
)
def test_split_cot_to_steps_handles_common_formats(cot, expected):
    assert split_cot_to_steps(cot) == expected


def test_split_cot_to_steps_returns_single_block_when_no_structure():
    cot = "This response does not follow the expected step formatting."
    assert split_cot_to_steps(cot) == [cot]
