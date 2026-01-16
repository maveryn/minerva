import importlib.util
import json
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


def test_athena_taa_alias_only(monkeypatch):
    # Patch alias dictionary for predictable results
    monkeypatch.setattr(reward_minerva, "_ALIAS_DICT", {"apt1": ["commentcrew"], "commentcrew": ["apt1"]})

    score_alias = reward_minerva.reward_minerva("athena-cti-taa", "APT1", "CommentCrew")
    assert score_alias == 1.0

    score_related = reward_minerva.reward_minerva("athena-cti-taa", "APT2", "GroupX")
    assert score_related == 0.0


def test_minerva_cvss_v31():
    pred = "\\boxed{CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H}"
    truth = {"cvss_v31_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}
    score = reward_minerva.reward_minerva("reward_cvss_v31", pred, truth)
    assert score == 1.0


def test_minerva_cvss_v31_invalid_format():
    pred = "Not a CVSS vector"
    truth = {"cvss_v31_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}
    score = reward_minerva.reward_minerva("reward_cvss_v31", pred, truth)
    assert score == 0.0


def test_minerva_cvss_v31_score_distance():
    from cvss import CVSS3

    pred = "CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:H/A:H"
    truth = {"cvss_v31_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}
    pred_score = CVSS3(pred).scores()[0]
    truth_score = CVSS3(truth["cvss_v31_vector"]).scores()[0]
    expected = 1.0 - abs(truth_score - pred_score) / 10.0
    expected = max(0.0, min(1.0, expected))
    score = reward_minerva.reward_minerva("reward_cvss_v31", pred, truth)
    assert pytest.approx(score, rel=1e-6) == expected


def test_threat_actor_alias_lookup(tmp_path, monkeypatch):
    lookup = {
        "version": 1,
        "actors": [
            {
                "name": "APT1",
                "aliases": ["Comment Crew"],
            }
        ],
    }
    lookup_path = tmp_path / "threat_actor_lookup.json"
    lookup_path.write_text(json.dumps(lookup), encoding="utf-8")
    monkeypatch.setenv("MINERVA_THREAT_ACTOR_LOOKUP", str(lookup_path))

    truth = {"threat_actor": "APT1"}
    assert reward_minerva.reward_minerva("reward_threat_actor_name", "Comment Crew", truth) == 1.0
    assert reward_minerva.reward_minerva("reward_threat_actor_name", "OtherGroup", truth) == 0.0


def test_minerva_tactics_extract_list():
    pred = "Answer: TA0004, TA0005"
    truth = {"tactic_ids": ["TA0004", "TA0005"]}
    score = reward_minerva.reward_minerva("reward_tactic_ids", pred, truth)
    assert score == 1.0


def test_minerva_detection_extract():
    pred = "Detection strategy is DET-0005"
    truth = {"detection_id": "DET0005"}
    score = reward_minerva.reward_minerva("reward_detection_id", pred, truth)
    assert score == 1.0
