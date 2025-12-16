import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from minerva.logger import get_logger
from minerva.utils import write_jsonl


TECHNIQUE_TAG_RE = r"^attack\.t\d{4}(?:\.\d{3})?$"

TACTIC_MAP = {
    "reconnaissance": "TA0043",
    "resource-development": "TA0042",
    "initial-access": "TA0001",
    "execution": "TA0002",
    "persistence": "TA0003",
    "privilege-escalation": "TA0004",
    "defense-evasion": "TA0005",
    "credential-access": "TA0006",
    "discovery": "TA0007",
    "lateral-movement": "TA0008",
    "collection": "TA0009",
    "command-and-control": "TA0011",
    "exfiltration": "TA0010",
    "impact": "TA0040",
}

TECHNIQUE_PROMPT = """Given the Sigma rule excerpt below (log source + detection logic), provide the single most appropriate MITRE ATT&CK Enterprise technique ID that best represents the adversary behavior this rule is intended to detect.

Technique - how an adversary achieves a tactical objective by performing an action.

Requirements:
- Use MITRE ATT&CK Enterprise technique IDs only.
- Return exactly ONE technique ID.
- Prefer a sub-technique ID (e.g., T1059.003) if it is more suitable than the parent technique; otherwise return the parent technique ID (e.g., T1059).

Sigma rule excerpt:
{SIGMA_RULE_EXCERPT}"""

TACTIC_PROMPT = """Given the Sigma rule excerpt below (log source + detection logic), enumerate all MITRE ATT&CK Enterprise tactic IDs (TA000x) that are valid for the adversary behavior this rule is intended to detect.

Tactic - why an adversary performs an action (the adversary’s tactical objective).

Requirements:
- Use MITRE ATT&CK Enterprise tactic IDs (TA000x) only.
- List ALL applicable tactic IDs (multiple may apply).

Sigma rule excerpt:
{SIGMA_RULE_EXCERPT}"""


def _load_rule(path: Path) -> Dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _extract_tags(tags: Iterable[str]) -> Tuple[List[str], List[str]]:
    import re

    technique_re = re.compile(TECHNIQUE_TAG_RE, re.IGNORECASE)
    techniques: List[str] = []
    tactics: List[str] = []
    for tag in tags or []:
        if technique_re.match(tag):
            techniques.append(tag.split(".", 1)[1].upper())
            continue
        if not tag.lower().startswith("attack."):
            continue
        slug = tag.split(".", 1)[1].lower()
        if slug.startswith("t"):  # already a technique, handled above
            continue
        tactic_id = TACTIC_MAP.get(slug.replace("_", "-"))
        if tactic_id:
            tactics.append(tactic_id)
    return techniques, sorted(set(tactics))


def _build_excerpt(rule: Dict[str, Any]) -> str:
    title = rule.get("title") or ""
    logsource = yaml.safe_dump(rule.get("logsource", {}), sort_keys=False, allow_unicode=True).strip()
    detection = yaml.safe_dump(rule.get("detection", {}), sort_keys=False, allow_unicode=True).strip()
    parts = [
        f"Title: {title}".strip(),
        "Logsource:",
        logsource or "(none)",
        "Detection:",
        detection or "(none)",
    ]
    return "\n".join(parts)


def _collect_rules(root_dirs: List[str], logger) -> List[Tuple[Path, Dict[str, Any]]]:
    records: List[Tuple[Path, Dict[str, Any]]] = []
    for root in root_dirs:
        for path in Path(root).rglob("*.yml"):
            data = _load_rule(path)
            if data:
                records.append((path, data))
        for path in Path(root).rglob("*.yaml"):
            data = _load_rule(path)
            if data:
                records.append((path, data))
    logger.info("Loaded %d Sigma rule files from %s", len(records), root_dirs)
    return records


def build_sigma_datasets(
    rule_dirs: List[str],
    output_dir: str = "dataset/minerva",
    logger=None,
) -> Dict[str, List[Dict[str, Any]]]:
    if logger is None:
        logger = get_logger("sigma")
    rules = _collect_rules(rule_dirs, logger)

    technique_rows: List[Dict[str, Any]] = []
    tactic_rows: List[Dict[str, Any]] = []

    for path, rule in rules:
        tags = rule.get("tags") or []
        techniques, tactics = _extract_tags(tags)

        excerpt = _build_excerpt(rule)
        meta_common = {
            "rule_path": str(path),
            "title": rule.get("title", ""),
            "status": rule.get("status", ""),
            "level": rule.get("level", ""),
        }

        if len(techniques) == 1:
            tech_prompt = TECHNIQUE_PROMPT.format(SIGMA_RULE_EXCERPT=excerpt)
            technique_rows.append(
                {
                    "task": "sigma_to_attack_technique",
                    "input": {
                        "sigma_rule_excerpt": excerpt,
                        "rule_title": rule.get("title", ""),
                        "prompt": tech_prompt,
                    },
                    "ground_truth": {"technique_id": techniques[0]},
                    "answer": techniques[0],
                    "reward_fn": "reward_technique_id",
                    "metadata": {**meta_common, "tactics": tactics},
                }
            )

        if tactics:
            tac_prompt = TACTIC_PROMPT.format(SIGMA_RULE_EXCERPT=excerpt)
            tactic_rows.append(
                {
                    "task": "sigma_to_attack_tactics",
                    "input": {
                        "sigma_rule_excerpt": excerpt,
                        "rule_title": rule.get("title", ""),
                        "prompt": tac_prompt,
                    },
                    "ground_truth": {"tactic_ids": tactics},
                    "answer": tactics,
                    "reward_fn": "reward_tactic_ids",
                    "metadata": {**meta_common, "techniques": techniques},
                }
            )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tech_path = out_dir / "sigma_to_attack_technique.jsonl"
    tac_path = out_dir / "sigma_to_attack_tactics.jsonl"

    write_jsonl(tech_path, technique_rows)
    write_jsonl(tac_path, tactic_rows)

    logger.info(
        "Wrote %d technique rows -> %s; %d tactic rows -> %s",
        len(technique_rows),
        tech_path,
        len(tactic_rows),
        tac_path,
    )

    return {"technique": technique_rows, "tactic": tactic_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Sigma→ATT&CK datasets")
    parser.add_argument(
        "--rule-dirs",
        nargs="+",
        default=["dataset/sigma/rules", "dataset/sigma/rules-threat-hunting"],
        help="Directories containing Sigma rule YAML files",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/minerva",
        help="Directory where JSONL files will be written",
    )
    args = parser.parse_args()
    logger = get_logger("sigma")
    build_sigma_datasets(args.rule_dirs, output_dir=args.output_dir, logger=logger)


if __name__ == "__main__":
    main()
