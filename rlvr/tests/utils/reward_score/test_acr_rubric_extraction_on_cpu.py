from __future__ import annotations

from math import isclose

from verl.utils.reward_score import reward_acr_batch


def _expected_norm(q1: int, q2: int, q3: int, q4: int) -> float:
    weighted = 0.30 * q1 + 0.30 * q2 + 0.20 * q3 + 0.20 * q4
    return (weighted - 1.0) / 3.0


def test_rubric_exact_keys():
    data = {
        "Q1_no_leakage": 4,
        "Q2_clarity": 3,
        "Q3_groundedness": 2,
        "Q4_alignment": 1,
    }
    score = reward_acr_batch._score_from_rubric(data)
    assert isclose(score, _expected_norm(4, 3, 2, 1))


def test_rubric_q_only_keys():
    data = {"Q1": 4, "Q2": 4, "Q3": 4, "Q4": 4}
    score = reward_acr_batch._score_from_rubric(data)
    assert isclose(score, _expected_norm(4, 4, 4, 4))


def test_rubric_keyword_keys():
    data = {
        "leakage": 1,
        "clarity": 2,
        "groundedness": 3,
        "alignment": 4,
    }
    score = reward_acr_batch._score_from_rubric(data)
    assert isclose(score, _expected_norm(1, 2, 3, 4))


def test_rubric_mixed_keys():
    data = {
        "Q1_no_leakage": 2,
        "Q2_clarity_concise": "3",
        "grounded": 4,
        "alignment_score": 1,
    }
    score = reward_acr_batch._score_from_rubric(data)
    assert isclose(score, _expected_norm(2, 3, 4, 1))


def test_rubric_missing_axis_returns_zero():
    data = {"Q1": 4, "Q2": 4, "clarity": 4}
    score = reward_acr_batch._score_from_rubric(data)
    assert score == 0.0


def test_rubric_out_of_range_returns_zero():
    data = {"Q1": 4, "Q2": 5, "Q3": 4, "Q4": 4}
    score = reward_acr_batch._score_from_rubric(data)
    assert score == 0.0
