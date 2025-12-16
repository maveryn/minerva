# verl/trainer/paraphrase_ppo/paraphrase_prompts.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prompt templates for answer and question paraphrasing."""

from typing import Dict, List, Optional


# Default answer rewrite templates for different styles
ANSWER_REWRITE_TEMPLATES = {
    "condense": """You are revising a correct solution to the following problem.
Problem:
{question_plain}

Original correct solution (may be verbose):
{solution_text}

Rewrite the solution in the requested style, keeping the same final boxed answer.
Style: condense; keep only decisive steps
Requirements: (1) preserve the final answer; (2) remove extraneous chatter; (3) show only the decisive reasoning.
Return just the revised solution with a single \\boxed{{final}} at the end.""",

    "numbered": """You are revising a correct solution to the following problem.
Problem:
{question_plain}

Original correct solution (may be verbose):
{solution_text}

Rewrite the solution in the requested style, keeping the same final boxed answer.
Style: number key steps and final check
Requirements: (1) preserve the final answer; (2) remove extraneous chatter; (3) show only the decisive reasoning.
Return just the revised solution with a single \\boxed{{final}} at the end.""",

    "self_verify": """You are revising a correct solution to the following problem.
Problem:
{question_plain}

Original correct solution (may be verbose):
{solution_text}

Rewrite the solution in the requested style, keeping the same final boxed answer.
Style: self-verify then present final concise solution
Requirements: (1) preserve the final answer; (2) remove extraneous chatter; (3) show only the decisive reasoning.
Return just the revised solution with a single \\boxed{{final}} at the end.""",
}


# Default question paraphrase template
QUESTION_PARAPHRASE_TEMPLATE = """Rephrase the problem below without changing its meaning or its final numeric/symbolic answer.
Keep all constraints intact; avoid introducing new entities.
Problem:
{question_plain}

Paraphrase the *problem statement only* (no solution). Do not include any answer or hints."""


def build_answer_rewrite_prompt(
    question_plain: str,
    solution_text: str,
    style_instruction: str,
    template: Optional[str] = None
) -> str:
    """Build prompt for answer rewriting with given style.
    
    Args:
        question_plain: Original problem text
        solution_text: Original correct solution 
        style_instruction: Style description (e.g., "condense; keep only decisive steps")
        template: Optional custom template. If None, uses default template.
    
    Returns:
        Formatted prompt string
    """
    if template is None:
        template = """You are revising a correct solution to the following problem.
Problem:
{question_plain}

Original correct solution (may be verbose):
{solution_text}

Rewrite the solution in the requested style, keeping the same final boxed answer.
Style: {style_instruction}
Requirements: (1) preserve the final answer; (2) remove extraneous chatter; (3) show only the decisive reasoning.
Return just the revised solution with a single \\boxed{{final}} at the end."""
    
    return template.format(
        question_plain=question_plain,
        solution_text=solution_text,
        style_instruction=style_instruction
    )


def build_question_paraphrase_prompt(
    question_plain: str,
    template: Optional[str] = None
) -> str:
    """Build prompt for question paraphrasing.
    
    Args:
        question_plain: Original problem text
        template: Optional custom template. If None, uses default template.
    
    Returns:
        Formatted prompt string
    """
    if template is None:
        template = QUESTION_PARAPHRASE_TEMPLATE
    
    return template.format(question_plain=question_plain)


def extract_paraphrased_question(response: str) -> str:
    """Extract paraphrased question from model response.
    
    Simple extraction that handles common patterns.
    Can be extended to handle more complex formats.
    
    Args:
        response: Model's paraphrase response
        
    Returns:
        Extracted question text
    """
    # Remove common prefixes
    lines = response.strip().split('\n')
    result_lines = []
    skip_prefixes = [
        "Here's a paraphrase",
        "Here is a paraphrase", 
        "Paraphrase:",
        "Rephrased problem:",
        "Problem:",
    ]
    
    for line in lines:
        # Skip lines that are just prefixes
        if any(line.strip().lower().startswith(prefix.lower()) for prefix in skip_prefixes):
            continue
        result_lines.append(line)
    
    return '\n'.join(result_lines).strip()


def format_answer_style_context(question: str, style: str) -> str:
    """Format the context prompt for answer-style generation.
    
    This creates the combined (x,s) context used during answer paraphrasing.
    
    Args:
        question: Original question x
        style: Style instruction s
        
    Returns:
        Formatted context string
    """
    return f"{question}\n\nGenerate solution in style: {style}"