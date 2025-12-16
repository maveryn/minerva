"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union

from deepscaler.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END

from deepscaler.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
import multiprocessing as mp

from math_verify import parse, verify

import os
import re
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout



# Use fork on Linux to avoid spawn re-import overhead; fall back on spawn elsewhere.
def _mp_ctx():
    methods = mp.get_all_start_methods()
    return mp.get_context("fork") if "fork" in methods else mp.get_context("spawn")

def _mv_init():
    # Preload heavy modules inside worker once (amortizes import cost).
    import math_verify  # noqa: F401

_CTX = _mp_ctx()
_EXEC = ProcessPoolExecutor(
    max_workers=int(os.getenv("MATHV_POOL", "16")),
    mp_context=_CTX,
    initializer=_mv_init,
)


# Prefer content after </think>, then the last \boxed{...}; else fallback to raw.
_THINK_END = THOUGHT_DELIMITER_END  # you already import this
_BOXED_OPEN = re.compile(r"\\boxed\s*\{", re.DOTALL)

def _extract_candidate_quick(text: str) -> str:
    # split after last </think>
    i = text.rfind(_THINK_END)
    seg = text[i + len(_THINK_END):] if i != -1 else text

    # find last \boxed{...} with simple brace counting
    last = None
    for m in _BOXED_OPEN.finditer(seg):
        last = m.end()
    if last is not None:
        depth, out = 1, []
        for ch in seg[last:]:
            if ch == "{": depth += 1; out.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0: break
                out.append(ch)
            else: out.append(ch)
        if depth == 0:
            return "".join(out).strip()

    return seg.strip()

_LATEX_FRAC = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")

def _strip_tex(s: str) -> str:
    s = s.strip()
    # drop \boxed{...}
    s = re.sub(r"\\boxed\s*\{(.*)\}$", r"\1", s, flags=re.DOTALL)
    # remove inline $...$ (keep inner)
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1]
    # collapse small LaTeX spacing
    s = s.replace(r"\,", "").replace(r"\!", "")
    s = s.strip()
    return s

def _to_simple_number(s: str):
    """Try quick numeric parse for integers/decimals and simple \\frac{a}{b}."""
    s0 = _strip_tex(s)
    s0 = _LATEX_FRAC.sub(r"\1/\2", s0)  # \frac{a}{b} -> a/b
    s0 = s0.replace(" ", "")
    # trivial cleanups
    s0 = s0.rstrip(".")
    try:
        return Fraction(s0)  # handles "a/b" and integers/decimals losslessly
    except Exception:
        try:
            return float(s0)
        except Exception:
            return None

def _wrap_tex(ans: str) -> str:
    ans = (ans or "").strip()
    if not ans: return ans
    if ans.startswith(r"\boxed{") or "$" in ans: return ans
    return f"${ans}$"



def _verify_once_worker(resp: str, gold: Union[str, List[str]], q: "mp.Queue") -> None:
    """
    Run parse/verify in a separate process so we can hard-timeout safely.
    This prevents the well-known 'signal only works in main thread' / stuck-SymPy problem.
    """
    try:
        # Import inside the child so killing the process also kills SymPy work.
        from math_verify import parse as _parse, verify as _verify

        pred = _parse(resp)

        if isinstance(gold, str):
            gold_list: List[str] = [_wrap_tex(gold)]
        else:
            gold_list = [_wrap_tex(g) for g in gold]

        for g in gold_list:
            if bool(_verify(_parse(g), pred)):  # order: verify(gold, pred)
                q.put(True)
                return
        q.put(False)
    except Exception:
        q.put(False)

def _verify_task(pred_tex: str, gold_list_tex: list[str]) -> bool:
    try:
        from math_verify import parse as _parse, verify as _verify
        pred = _parse(pred_tex)
        for g in gold_list_tex:
            if bool(_verify(_parse(g), pred)):  # verify(gold, pred)
                return True
        return False
    except Exception:
        return False


def labeling_responses(
    responses: list[str],
    golden_answer: Union[str, List[str]],
    timeout_s: float = 1.0,   # slightly tighter default
):
    """
    Faster, no-hang labeling:
      1) Extract final-answer candidate from each response.
      2) Fast-path: exact/number equality -> True without math_verify.
      3) Otherwise, reuse a persistent ProcessPoolExecutor and hard-timeout
         each verification Future.
    """
    # Prepare gold answers (both raw for fast-path and $...$ for verifier)
    gold_raw_list: List[str] = [golden_answer] if isinstance(golden_answer, str) else list(golden_answer)
    gold_raw_list = [g.strip() for g in gold_raw_list]
    gold_tex_list = [_wrap_tex(g) for g in gold_raw_list]

    labels: list[bool] = []
    futures = []

    # 1) extract + fast-path
    cands_raw: list[str] = []
    for resp in responses:
        cand = _extract_candidate_quick(resp)
        cands_raw.append(cand)

        # string-normalized equality (cheap)
        norm_cand = _strip_tex(cand).replace(" ", "")
        if any(_strip_tex(g).replace(" ", "") == norm_cand for g in gold_raw_list):
            labels.append(True)
            futures.append(None)
            continue

        # numeric fast path
        nc = _to_simple_number(cand)
        if nc is not None:
            for g in gold_raw_list:
                ng = _to_simple_number(g)
                if ng is not None and nc == ng:
                    labels.append(True)
                    futures.append(None)
                    break
            else:
                # need verifier
                futures.append(_EXEC.submit(_verify_task, _wrap_tex(cand), gold_tex_list))
                labels.append(None)  # filled later
        else:
            # need verifier
            futures.append(_EXEC.submit(_verify_task, _wrap_tex(cand), gold_tex_list))
            labels.append(None)

    # 2) collect verifier results with a hard timeout
    for i, fut in enumerate(futures):
        if fut is None:
            continue
        try:
            ok = fut.result(timeout=timeout_s)
        except FuturesTimeout:
            ok = False
        except Exception:
            ok = False
        labels[i] = bool(ok)

    return labels




class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)

        try:
        
            problem = input.problem
            model_response = input.model_response
            
            # print("think_format", self.config.think_format)
            if self.config.think_format:
                # Extract solution.
                if THOUGHT_DELIMITER_START in model_response and THOUGHT_DELIMITER_END in model_response:
                    model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
                else:
                    return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
            else:
                model_solution = model_response
            # print(model_solution)
    
            labels = labeling_responses([model_solution,], input.ground_truth["answer"])
            if labels[0] is True:
                return RewardOutput(reward=self.config.correct_reward, is_correct=True)
            else:
                return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)
        except:
            return False

def reward_fn_math_verify(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
    return reward_response.is_correct

def reward_fn_math_verify_no_think(data_source: str, solution_str: str, ground_truth: str | list[str], extra_info=None):
    try:
        enable_llm = False
        reward_config = RewardConfig()
        reward_config.think_format = False
        reward_config.use_math_orm = enable_llm
        reward_fn = RewardMathFn(reward_config)
        reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
        return reward_response.is_correct
    except:
        return False

if __name__ == "__main__":
    # reward = RewardMathFn(RewardConfig)
    # import pandas as pd
    # df = pd.read_parquet('/mnt/petrelfs/share_data/yanjianhao/v9/openr1.parquet')
    # cnt = 0
    # import tqdm
    # for i in tqdm.tqdm(range(len(df))):
    #     row = df.iloc[i]
    #     ground_truth = row['reward_model']['ground_truth']
    #     solution = row['target'][0]['content']
    #     # input = RewardInput(problem="problem", problem_type=RewardType.MATH, model_response=solution, ground_truth={"answer": ground_truth})
    #     solution = solution.split(THOUGHT_DELIMITER_END)[1]
    #     output = labeling_responses([solution], ground_truth)
    #     # output = reward(input)
    #     if output[0] is False:
    #         cnt += 1
    #     print(cnt)
    # print(cnt)
    solution = """Let's break down the problem step-by-step:\n\n1. **Understand the Tournament Structure**: In a round-robin tournament where each player plays against every other player exactly once, if there are \\( n \\) players, the total number of games played is given by \\( \\binom{n}{2} = \\frac{n(n-1)}{2} \\). Each game results in either a win (1 point), a draw (0.5 points each), or a loss (0 points).\n\n2. **Player Points Calculation**: If the winner \\( W \\) wins half of his games and draws the other half, and he ends up scoring points that are 9 times less than the combined points of all other players, we need to model this mathematically. Suppose the winner \\( W \\) has \\( k \\) games, then he has \\( \\frac{k}{2} \\) wins and \\( \\frac{k}{2} \\) draws, scoring \\( \\frac{3k}{4} \\) points in total.\n\n3. **Combined Points of Other Players**: Let \\( T \\) be the total points accumulated by all players including \\( W \\). Since \\( W \\)'s points are \\( \\frac{3k}{4} \\), the rest of the players collectively have \\( T - \\frac{3k}{4} \\) points. Given that \\( W \\)'s points are 9 times less than the others' combined points, we get:\n\\[ \\frac{3k}{4} = \\frac{1}{9}(T - \\frac{3k}{4}). \\]\n\n4. **Solving the Equation**: We now solve the equation above for \\( k \\) (the number of games played by the winner). First, let's express \\( T \\) in terms of \\( k \\):\n\\[ \\frac{3k}{4} * 9 = T - \\frac{3k}{4}, \\]\n\\[ 27k / 4 + 3k / 4 = T, \\]\n\\[ 30k / 4 = T, \\]\n\\[ 15k = 2T. \\]\n\n5. **Total Points Distribution**: Since \\( W \\) is playing a round-robin tournament, \\( k = n - 1 \\), as each player plays every other player once. Substituting \\( k = n - 1 \\) into our equation:\n\\[ 15(n - 1) = 2T. \\]\n\n6. **Points Calculation for Each Player**: The maximum possible points a player can score is if they win all their games, i.e., 1 point per game. So if \\( W \\) has \\( n-1 \\) games, their maximum score is \\( n-1 \\) points. However, since \\( W \\) only wins half and draws half, they receive \\( \\frac{3(n-1)}{4} \\) points.\n\n7. **Final Equations**: Given that the remaining players' total points are 15 times the remaining games played by \\( W \\), we equate this to \\( 2T \\):\n\\[ \\text{(remaining players' total points)} = 2 * \\left(\\frac{15(n-1)}{2}\\right), \\]\n\\[ \\text{Remaining players' total points} = 15(n-1). \\]\n\n8. **Final Simplification**: We know that the sum of points from all rounds played equals \\( \\frac{n(n-1)}{2} \\), as each game contributes 1 point towards the total. Therefore:\n\\[ \\frac{n(n-1)}{2} = \\frac{3(n-1)^2}{4} + 15(n-1). \\]\n\n9. **Solve for \\( n \\)**: Simplifying the right side of the equation:\n\\[ \\frac{n(n-1)}{2} = \\frac{3(n-1)(n-1) + 60(n-1)}{4}, \\]\n\\[ \\frac{n(n-1)}{2} = \\frac{3(n-1)^2 + 60(n-1)}{4}, \\]\n\\[ 2n(n-1) = 3(n-1)^2 + 60(n-1). \\]\n\nFactoring out \\( (n-1) \\):\n\\[ 2n = 3(n-1) + 60, \\]\n\\[ 2n = 3n - 3 + 60, \\]\n\\[ 2n = 3n + 57, \\]\n\\[ 0 = n + 57, \\]\n\\[ n = 57. \\]\n\nTherefore, there must have been \\(\\boxed{57}\\) players in the tournament.<|endoftext|>"""
    
    golden_answer = "$57$"
    from math_verify import parse
    # print(parse(solution))
    # print(parse(golden_answer))
    output = reward_fn_math_verify_no_think(solution, golden_answer, enable_llm=False)
    print(output)
    
        # print(output)
    # input = RewardInput(problem="Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.", problem_type=RewardType.MATH, model_response="<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.", ground_truth={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"]})
    # output = reward(input)
    # print(output)