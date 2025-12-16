import importlib.util
import sys
from pathlib import Path

import pytest

# Manually load myreward_boxed and reward_minerva without importing rlvr package (avoids heavy deps like ray)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
rs_dir = PROJECT_ROOT / "rlvr" / "verl" / "utils" / "reward_score"


def _load_module(name: str, path: Path, package: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


myreward_boxed = _load_module(
    "rlvr.verl.utils.reward_score.myreward_boxed", rs_dir / "myreward_boxed.py", "rlvr.verl.utils.reward_score"
)
reward_minerva = _load_module(
    "rlvr.verl.utils.reward_score.reward_minerva", rs_dir / "reward_minerva.py", "rlvr.verl.utils.reward_score"
)


def test_minerva_technique_full_match():
    pred = "Final answer: \\boxed{T1059.003}"
    truth = {"technique_id": "T1059.003"}
    score = reward_minerva.reward_minerva("reward_technique_id", pred, truth)
    assert score == 1.0


def test_minerva_technique_parent_partial():
    pred = "Result: \\boxed{T1059}"
    truth = {"technique_id": "T1059.003"}
    score = reward_minerva.reward_minerva("reward_technique_id", pred, truth)
    assert score == 0.5


def test_minerva_cwe_ids_f1():
    pred = "Answer: CWE-79, CWE-89"
    truth = {"cwe_ids": ["CWE-79", "CWE-89"]}
    score = reward_minerva.reward_minerva("reward_cwe_ids", pred, truth)
    assert score == 1.0


def test_athena_rcm_exact():
    pred = "Answer: CWE-352"
    truth = "CWE-352"
    score = reward_minerva.reward_minerva("athena-cti-rcm", pred, truth)
    assert score == 1.0


def test_athena_rms_f1(monkeypatch):
    pred = "Mitigations: M1001, M1003"
    truth = "M1001, M1002"
    score = reward_minerva.reward_minerva("athena-cti-rms", pred, truth)
    assert pytest.approx(score, rel=1e-6) == 0.5  # precision=recall=0.5 -> f1=0.5


def test_athena_taa_alias_and_related(monkeypatch):
    # Patch alias/related dictionaries for predictable results
    monkeypatch.setattr(reward_minerva, "_ALIAS_DICT", {"apt1": ["commentcrew"], "commentcrew": ["apt1"]})
    monkeypatch.setattr(reward_minerva, "_RELATED_DICT", {"apt2": ["groupx"], "groupx": ["apt2"]})

    score_alias = reward_minerva.reward_minerva("athena-cti-taa", "APT1", "CommentCrew")
    assert score_alias == 1.0

    score_related = reward_minerva.reward_minerva("athena-cti-taa", "APT2", "GroupX")
    assert score_related == 0.5


def test_minerva_cvss_v31():
    pred = "\\boxed{CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H}"
    truth = {"cvss_v31_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}
    score = reward_minerva.reward_minerva("reward_cvss_v31", pred, truth)
    assert score == 1.0
