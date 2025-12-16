import argparse
import random
from typing import Dict, List

from minerva.logger import get_logger
from minerva.tasks.common import _balanced_cap
from minerva.utils import write_jsonl


EXAMPLE_TO_CAPEC_PROMPT = """Given the Common Attack Pattern Enumeration and Classification (CAPEC) example below, provide the CAPEC ID (format: CAPEC-<number>) that best matches the described attack pattern.

CAPEC example:
{EXAMPLE_TEXT}"""

EXAMPLE_TO_CWE_PROMPT = """Given the Common Attack Pattern Enumeration and Classification (CAPEC) example below, list EXACTLY {COUNT} Common Weakness Enumeration (CWE) ID{S_SUFFIX} (format: CWE-<number>) associated with this CAPEC entry.

CAPEC example:
{EXAMPLE_TEXT}"""

EXAMPLE_TO_ATTACK_PROMPT = """Given the Common Attack Pattern Enumeration and Classification (CAPEC) example below, provide the single most appropriate MITRE ATT&CK Enterprise technique ID linked to this CAPEC entry.

Requirements:
- Use MITRE ATT&CK Enterprise technique IDs only.
- Return exactly ONE technique ID.
- Prefer a sub-technique ID (e.g., T1059.003) if it is more suitable; otherwise return the parent technique ID (e.g., T1059).

CAPEC example:
{EXAMPLE_TEXT}"""


def _attack_ids_with_parents(patterns: Dict[str, Dict], capec_id: str) -> List[str]:
    atk = patterns[capec_id].get("attack_techniques") or []
    if atk:
        return atk
    agg: List[str] = []
    for pid in patterns[capec_id].get("parent_ids", []):
        agg.extend(patterns.get(pid, {}).get("attack_techniques", []))
    return sorted(set(agg))


def build_capec_example_tasks(
    patterns: Dict[str, Dict],
    *,
    capec_out: str,
    cwe_out: str,
    attack_out: str,
    seed: int = 1337,
    logger=None,
) -> Dict[str, List[Dict]]:
    if logger is None:
        logger = get_logger("capec-examples")
    rng = random.Random(seed)

    capec_rows: List[Dict] = []
    cwe_rows: List[Dict] = []
    attack_rows: List[Dict] = []

    attack_multi = 0
    attack_subtech = 0

    for capec_id, pat in patterns.items():
        examples = pat.get("example_instances") or []
        if not examples:
            continue
        cwes = pat.get("cwe_ids") or []
        atks = _attack_ids_with_parents(patterns, capec_id)
        if len(atks) > 1:
            attack_multi += 1
        attack_subtech += sum(1 for a in atks if "." in a)

        for ex in examples:
            capec_rows.append(
                {
                    "task": "capec_example_to_capec",
                    "input": {"example": ex, "prompt": EXAMPLE_TO_CAPEC_PROMPT.format(EXAMPLE_TEXT=ex)},
                    "ground_truth": {"capec_id": capec_id},
                    "reward_fn": "binary_id",
                    "metadata": {"capec_name": pat.get("name", "")},
                }
            )
            if cwes and len(cwes) <= 3:
                suffix = "" if len(cwes) == 1 else "s"
                cwe_rows.append(
                    {
                        "task": "capec_example_to_cwe",
                        "input": {
                            "example": ex,
                            "prompt": EXAMPLE_TO_CWE_PROMPT.format(
                                EXAMPLE_TEXT=ex, COUNT=len(cwes), S_SUFFIX=suffix
                            ),
                        },
                        "ground_truth": {"cwe_ids": cwes},
                        "reward_fn": "reward_cwe_ids",
                        "metadata": {"capec_id": capec_id},
                    }
                )
            if len(atks) == 1:
                attack_rows.append(
                    {
                        "task": "capec_example_to_attack_technique",
                        "input": {
                            "example": ex,
                            "prompt": EXAMPLE_TO_ATTACK_PROMPT.format(EXAMPLE_TEXT=ex),
                        },
                        "ground_truth": {"technique_id": atks[0]},
                        "reward_fn": "reward_technique_id",
                        "metadata": {"capec_id": capec_id},
                    }
                )

    # No balancing needed; keep all
    write_jsonl(capec_out, capec_rows)
    write_jsonl(cwe_out, cwe_rows)
    write_jsonl(attack_out, attack_rows)

    logger.info(
        "Built CAPEC example tasks -> capec=%d, cwe=%d, attack=%d (attack_multi_patterns=%d, attack_subtech=%d)",
        len(capec_rows),
        len(cwe_rows),
        len(attack_rows),
        attack_multi,
        attack_subtech,
    )
    return {"capec": capec_rows, "cwe": cwe_rows, "attack": attack_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CAPEC example-based tasks")
    parser.add_argument("--config", default="minerva/config.yaml")
    args = parser.parse_args()
    from minerva.data_sources.capec import load_bundle
    from minerva.utils import load_yaml

    cfg = load_yaml(args.config)
    capec_data = load_bundle(cfg.get("CAPEC", {}))
    tasks_cfg = cfg.get("TASKS", {})
    capec_out = tasks_cfg.get("CAPEC_EXAMPLE_CAPEC", {}).get("output_path", "dataset/minerva/capec_example_to_capec.jsonl")
    cwe_out = tasks_cfg.get("CAPEC_EXAMPLE_CWE", {}).get("output_path", "dataset/minerva/capec_example_to_cwe.jsonl")
    atk_out = tasks_cfg.get("CAPEC_EXAMPLE_ATTACK", {}).get("output_path", "dataset/minerva/capec_example_to_attack.jsonl")
    seed = int(tasks_cfg.get("CAPEC_EXAMPLE_CAPEC", {}).get("seed", 1337))
    build_capec_example_tasks(
        capec_data["patterns"],
        capec_out=capec_out,
        cwe_out=cwe_out,
        attack_out=atk_out,
        seed=seed,
    )


if __name__ == "__main__":
    main()
