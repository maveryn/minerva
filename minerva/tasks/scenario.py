import random
from typing import Dict, List, Optional

from minerva.logger import get_logger
from minerva.tasks.common import _balanced_cap
from minerva.utils import write_jsonl


SCENARIO_TECHNIQUE_PROMPT = """Given the adversary procedure description below, provide the single most appropriate MITRE ATT&CK Enterprise technique ID that best represents the behavior.

Requirements:
- Use MITRE ATT&CK Enterprise technique IDs only.
- Return exactly ONE technique ID.
- Prefer a sub-technique ID (e.g., T1059.003) if it is more suitable; otherwise return the parent technique ID (e.g., T1059).

Adversary procedure:
{SCENARIO_TEXT}"""

SCENARIO_TACTIC_PROMPT = """Given the adversary procedure description below, list EXACTLY {COUNT} MITRE ATT&CK Enterprise tactic ID{S_SUFFIX} (TA000x) that apply to this behavior.

Requirements:
- Use MITRE ATT&CK Enterprise tactic IDs (TA000x) only.
- List EXACTLY {COUNT} tactic ID{S_SUFFIX} that are most appropriate.

Adversary procedure:
{SCENARIO_TEXT}"""

SCENARIO_MITIGATION_PROMPT = """Given the adversary procedure description below, list EXACTLY {COUNT} MITRE ATT&CK Enterprise mitigation ID{S_SUFFIX} (M####) that best mitigate this behavior.

Requirements:
- Use MITRE ATT&CK Enterprise mitigation IDs only.
- List EXACTLY {COUNT} mitigation ID{S_SUFFIX} that are most appropriate.

Adversary procedure:
{SCENARIO_TEXT}"""

SCENARIO_DETECTION_PROMPT = """Given the adversary procedure description below, provide the single most appropriate MITRE ATT&CK Enterprise detection strategy ID (DET####) that best detects this behavior.

Requirements:
- Use MITRE ATT&CK Enterprise detection strategy IDs only.
- Return exactly ONE detection strategy ID.

Adversary procedure:
{SCENARIO_TEXT}"""


def build_scenario_to_technique(records: List[Dict[str, any]], max_items: int, output_path: str, seed: int = 1337, logger=None) -> List[Dict[str, any]]:
    if logger is None:
        logger = get_logger("task-scenario-technique")
    rng = random.Random(seed)
    records = _balanced_cap(
        records,
        max_items,
        label_fn=lambda r: r.get("technique_id"),
        rng=rng,
    )
    rows: List[Dict[str, any]] = []
    for r in records:
        scenario_text = r.get("scenario") or ""
        rows.append(
            {
                "task": "scenario_to_attack_technique",
                "input": {
                    "scenario": scenario_text,
                    "platforms": r.get("platforms", []),
                    "prompt": SCENARIO_TECHNIQUE_PROMPT.format(SCENARIO_TEXT=scenario_text),
                },
                "ground_truth": {"technique_id": r.get("technique_id")},
                "reward_fn": "reward_technique_id",
                "metadata": {
                    "tactics": r.get("tactics", []),
                    "mitigations": r.get("mitigations", []),
                    "detection_strategies": r.get("detection_strategies", []),
                },
            }
        )
    write_jsonl(output_path, rows)
    logger.info("Built %d scenario->technique tasks -> %s", len(rows), output_path)
    return rows


def build_scenario_to_tactics(records: List[Dict[str, any]], output_path: str, seed: int = 1337, max_items: Optional[int] = None, logger=None) -> List[Dict[str, any]]:
    if logger is None:
        logger = get_logger("task-tactics")
    rng = random.Random(seed)
    pool: List[Dict[str, any]] = []
    for r in records:
        tactics = r.get("metadata", {}).get("tactics", [])
        if not tactics:
            continue
        scenario_text = r["input"].get("scenario") or ""
        suffix = "" if len(tactics) == 1 else "s"
        pool.append(
            {
                "task": "scenario_to_attack_tactics",
                "input": {
                    **r["input"],
                    "prompt": SCENARIO_TACTIC_PROMPT.format(
                        SCENARIO_TEXT=scenario_text,
                        COUNT=len(tactics),
                        S_SUFFIX=suffix,
                    ),
                },
                "ground_truth": {"tactic_ids": tactics},
                "reward_fn": "reward_tactic_ids",
                "metadata": {"technique_id": r["ground_truth"].get("technique_id")},
            }
        )
    pool = _balanced_cap(pool, max_items or 0, label_fn=lambda r: tuple(sorted(r["ground_truth"]["tactic_ids"])), rng=rng)
    write_jsonl(output_path, pool)
    logger.info("Built %d scenario->tactic tasks -> %s", len(pool), output_path)
    return pool


def build_scenario_to_mitigations(records: List[Dict[str, any]], output_path: str, seed: int = 1337, max_items: Optional[int] = None, logger=None) -> List[Dict[str, any]]:
    if logger is None:
        logger = get_logger("task-mitigations")
    rng = random.Random(seed)
    pool: List[Dict[str, any]] = []
    for r in records:
        mits = r.get("metadata", {}).get("mitigations", [])
        if not mits:
            continue
        scenario_text = r["input"].get("scenario") or ""
        suffix = "" if len(mits) == 1 else "s"
        pool.append(
            {
                "task": "scenario_to_attack_mitigations",
                "input": {
                    **r["input"],
                    "prompt": SCENARIO_MITIGATION_PROMPT.format(
                        SCENARIO_TEXT=scenario_text,
                        COUNT=len(mits),
                        S_SUFFIX=suffix,
                    ),
                },
                "ground_truth": {"mitigation_ids": mits},
                "reward_fn": "reward_mitigation_ids",
                "metadata": {"technique_id": r["ground_truth"].get("technique_id")},
            }
        )
    pool = _balanced_cap(pool, max_items or 0, label_fn=lambda r: tuple(sorted(r["ground_truth"]["mitigation_ids"])), rng=rng)
    write_jsonl(output_path, pool)
    logger.info("Built %d scenario->mitigation tasks -> %s", len(pool), output_path)
    return pool


def build_scenario_to_detections(records: List[Dict[str, any]], output_path: str, seed: int = 1337, max_items: Optional[int] = None, logger=None) -> List[Dict[str, any]]:
    if logger is None:
        logger = get_logger("task-detections")
    rng = random.Random(seed)
    pool: List[Dict[str, any]] = []
    for r in records:
        dets = r.get("metadata", {}).get("detection_strategies", [])
        if not dets:
            continue
        scenario_text = r["input"].get("scenario") or ""
        if len(dets) != 1:
            continue
        det_id = dets[0]
        pool.append(
            {
                "task": "scenario_to_attack_detection",
                "input": {
                    **r["input"],
                    "prompt": SCENARIO_DETECTION_PROMPT.format(SCENARIO_TEXT=scenario_text),
                },
                "ground_truth": {"detection_id": det_id},
                "reward_fn": "reward_detection_id",
                "metadata": {"technique_id": r["ground_truth"].get("technique_id")},
            }
        )
    pool = _balanced_cap(pool, max_items or 0, label_fn=lambda r: r["ground_truth"]["detection_id"], rng=rng)
    write_jsonl(output_path, pool)
    logger.info("Built %d scenario->detection tasks -> %s", len(pool), output_path)
    return pool
