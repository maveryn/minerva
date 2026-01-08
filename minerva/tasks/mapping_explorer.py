import argparse
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

import yaml

from minerva.logger import get_logger
from minerva.utils import normalize_text, write_jsonl


EXPLOITATION_PROMPT_ONLY = dedent(
    """
    Given the vulnerability description below, output the single most appropriate MITRE ATT&CK Enterprise technique ID for:

    Exploitation Technique - the method (technique) used to exploit the vulnerability.

    Requirements:
    - Use MITRE ATT&CK Enterprise technique IDs only.
    - Return exactly ONE technique ID (T####).
    - Do not return sub-technique IDs (e.g., T1059.003).

    Vulnerability description:
    {CVE_DESCRIPTION}
    """
).strip()

EXPLOITATION_PROMPT_SUB = dedent(
    """
    Given the vulnerability description below, output the single most appropriate MITRE ATT&CK Enterprise technique ID for:

    Exploitation Technique - the method (technique) used to exploit the vulnerability.

    Requirements:
    - Use MITRE ATT&CK Enterprise technique IDs only.
    - Return exactly ONE technique ID.
    - Prefer a sub-technique ID (e.g., T1059.003) if it is more suitable; otherwise return the parent technique ID (e.g., T1059).

    Vulnerability description:
    {CVE_DESCRIPTION}
    """
).strip()


PRIMARY_IMPACT_PROMPT_ONLY = dedent(
    """
    Given the vulnerability description below, output the single most appropriate MITRE ATT&CK Enterprise technique ID for:

    Primary Impact - the initial benefit (impact) gained through exploitation of the vulnerability.

    Requirements:
    - Use MITRE ATT&CK Enterprise technique IDs only.
    - Return exactly ONE technique ID (T####).
    - Do not return sub-technique IDs (e.g., T1059.003).

    Vulnerability description:
    {CVE_DESCRIPTION}
    """
).strip()

PRIMARY_IMPACT_PROMPT_SUB = dedent(
    """
    Given the vulnerability description below, output the single most appropriate MITRE ATT&CK Enterprise technique ID for:

    Primary Impact - the initial benefit (impact) gained through exploitation of the vulnerability.

    Requirements:
    - Use MITRE ATT&CK Enterprise technique IDs only.
    - Return exactly ONE technique ID.
    - Prefer a sub-technique ID (e.g., T1059.003) if it is more suitable; otherwise return the parent technique ID (e.g., T1059).

    Vulnerability description:
    {CVE_DESCRIPTION}
    """
).strip()


SECONDARY_IMPACT_PROMPT_ONLY = dedent(
    """
    Given the vulnerability description and the primary impact already obtained, output the single most appropriate MITRE ATT&CK Enterprise technique ID for:

    Secondary Impact - what the adversary can do by gaining the benefit of the primary impact.

    Requirements:
    - Use MITRE ATT&CK Enterprise technique IDs only.
    - Return exactly ONE technique ID (T####).
    - Do not return sub-technique IDs (e.g., T1059.003).

    Vulnerability description:
    {CVE_DESCRIPTION}

    Primary impact:
    {PRIMARY_IMPACT_TECHNIQUE_ID}
    """
).strip()

SECONDARY_IMPACT_PROMPT_SUB = dedent(
    """
    Given the vulnerability description and the primary impact already obtained, output the single most appropriate MITRE ATT&CK Enterprise technique ID for:

    Secondary Impact - what the adversary can do by gaining the benefit of the primary impact.

    Requirements:
    - Use MITRE ATT&CK Enterprise technique IDs only.
    - Return exactly ONE technique ID.
    - Prefer a sub-technique ID (e.g., T1059.003) if it is more suitable; otherwise return the parent technique ID (e.g., T1059).

    Vulnerability description:
    {CVE_DESCRIPTION}

    Primary impact:
    {PRIMARY_IMPACT_TECHNIQUE_ID}
    """
).strip()


def _sanitize_description(text: str, capability_id: str) -> str:
    """
    Replace occurrences of the CVE ID in the description with "This vulnerability"
    (capitalized when starting a sentence, lowercase otherwise).
    """
    import re

    if not text or not capability_id:
        return text or ""

    pattern = re.compile(re.escape(capability_id), re.IGNORECASE)

    def _repl(match: re.Match) -> str:
        start = match.start()
        prefix = text[:start].rstrip()
        if not prefix:
            return "This vulnerability"
        if prefix[-1] in ".!?":
            return "This vulnerability"
        return "this vulnerability"

    return pattern.sub(_repl, text)


def _technique_base(technique_id: str) -> str:
    return (technique_id or "").split(".", 1)[0]


def _load_mapping_objects(path: Path, logger) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    objs = data.get("mapping_objects") or []
    logger.info("Loaded %d mapping objects from %s", len(objs), path)
    return objs


def _build_record(
    task: str,
    obj: Dict[str, Any],
    prompt: str,
    *,
    reward_fn: str,
    technique_id: Optional[str] = None,
    extra_input: Optional[Dict[str, Any]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    technique_id = technique_id if technique_id is not None else obj.get("attack_object_id", "")
    input_payload = {
        "cve_id": obj.get("capability_id", ""),
        "cve_description": obj.get("sanitized_comments", obj.get("comments", "")),
        "prompt": prompt,
    }
    if extra_input:
        input_payload.update(extra_input)

    metadata = {
        "attack_object_name": obj.get("attack_object_name", ""),
        "capability_description": obj.get("capability_description", ""),
        "capability_group": obj.get("capability_group", ""),
        "mapping_type": obj.get("mapping_type", ""),
        "references": obj.get("references", []),
    }
    if extra_meta:
        metadata.update(extra_meta)

    return {
        "task": task,
        "input": input_payload,
        "ground_truth": {"technique_id": technique_id},
        "answer": technique_id,
        "reward_fn": reward_fn,
        "metadata": metadata,
    }


def build_cve_attack_datasets(
    mapping_path: str,
    output_dir: str = "dataset/minerva",
    *,
    ask_subtechnique: bool = False,
    logger=None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create JSONL datasets for exploitation technique, primary impact, and secondary impact
    from the mappings-explorer YAML. Uses comments as the CVE description input and
    attack_object_id as the label.
    """
    if logger is None:
        logger = get_logger("mappings-explorer")
    mapping_file = Path(mapping_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    objs = _load_mapping_objects(mapping_file, logger)

    # Index primary impacts by normalized comment and capability_id for lookups.
    primary_by_comment: Dict[str, List[Dict[str, Any]]] = {}
    primary_by_capability: Dict[str, List[Dict[str, Any]]] = {}
    for obj in objs:
        if obj.get("mapping_type") != "primary_impact":
            continue
        comment_key = normalize_text(obj.get("comments", ""))
        if comment_key:
            primary_by_comment.setdefault(comment_key, []).append(obj)
        cap_id = obj.get("capability_id")
        if cap_id:
            primary_by_capability.setdefault(cap_id, []).append(obj)

    exploitation_rows: List[Dict[str, Any]] = []
    primary_rows: List[Dict[str, Any]] = []
    secondary_rows: List[Dict[str, Any]] = []
    missing_primary: List[Dict[str, Any]] = []

    # Filter out comment+mapping_type combos that are not unique
    combo_counts: Dict[tuple[str, str], int] = {}
    for o in objs:
        comment = normalize_text(o.get("comments", ""))
        mtype = o.get("mapping_type")
        combo_counts[(comment, mtype)] = combo_counts.get((comment, mtype), 0) + 1

    for obj in objs:
        mtype = obj.get("mapping_type")
        comment_raw = obj.get("comments", "") or ""
        comment_key = normalize_text(comment_raw)
        if combo_counts.get((comment_key, mtype), 0) != 1:
            continue  # skip non-unique comment/mapping_type combos
        sanitized = _sanitize_description(comment_raw, obj.get("capability_id", ""))
        obj["sanitized_comments"] = sanitized
        technique_id = obj.get("attack_object_id", "")
        if not ask_subtechnique:
            technique_id = _technique_base(technique_id)
        reward_fn = "reward_technique_id" if ask_subtechnique else "reward_technique_id_only"
        if mtype == "exploitation_technique":
            prompt_template = EXPLOITATION_PROMPT_SUB if ask_subtechnique else EXPLOITATION_PROMPT_ONLY
            prompt = prompt_template.format(CVE_DESCRIPTION=sanitized)
            exploitation_rows.append(
                _build_record(
                    "cve_to_attack_exploitation",
                    obj,
                    prompt,
                    reward_fn=reward_fn,
                    technique_id=technique_id,
                )
            )
        elif mtype == "primary_impact":
            prompt_template = PRIMARY_IMPACT_PROMPT_SUB if ask_subtechnique else PRIMARY_IMPACT_PROMPT_ONLY
            prompt = prompt_template.format(CVE_DESCRIPTION=sanitized)
            primary_rows.append(
                _build_record(
                    "cve_to_attack_primary_impact",
                    obj,
                    prompt,
                    reward_fn=reward_fn,
                    technique_id=technique_id,
                )
            )
        elif mtype == "secondary_impact":
            candidates = primary_by_comment.get(comment_key) or primary_by_capability.get(obj.get("capability_id"), [])
            if not candidates:
                logger.warning(
                    "No primary_impact found for secondary mapping: capability_id=%s attack_object_id=%s",
                    obj.get("capability_id"),
                    obj.get("attack_object_id"),
                )
                missing_primary.append(obj)
                continue
            primary_obj = candidates[0]
            if len(candidates) > 1:
                logger.info(
                    "Multiple primary impacts for %s; using %s",
                    obj.get("capability_id"),
                    primary_obj.get("attack_object_id"),
                )
            primary_id = primary_obj.get("attack_object_id", "")
            if not ask_subtechnique:
                primary_id = _technique_base(primary_id)
            prompt_template = SECONDARY_IMPACT_PROMPT_SUB if ask_subtechnique else SECONDARY_IMPACT_PROMPT_ONLY
            prompt = prompt_template.format(
                CVE_DESCRIPTION=sanitized,
                PRIMARY_IMPACT_TECHNIQUE_ID=primary_id,
            )
            secondary_rows.append(
                _build_record(
                    "cve_to_attack_secondary_impact",
                    obj,
                    prompt,
                    reward_fn=reward_fn,
                    technique_id=technique_id,
                    extra_input={"primary_impact_id": primary_id},
                    extra_meta={
                        "primary_candidates": [p.get("attack_object_id", "") for p in candidates]
                    },
                )
            )

    paths = {
        "exploitation": out_dir / "cve_to_attack_exploitation.jsonl",
        "primary_impact": out_dir / "cve_to_attack_primary_impact.jsonl",
        "secondary_impact": out_dir / "cve_to_attack_secondary_impact.jsonl",
    }
    write_jsonl(paths["exploitation"], exploitation_rows)
    write_jsonl(paths["primary_impact"], primary_rows)
    write_jsonl(paths["secondary_impact"], secondary_rows)

    logger.info(
        "Wrote %d exploitation, %d primary, %d secondary records",
        len(exploitation_rows),
        len(primary_rows),
        len(secondary_rows),
    )
    if missing_primary:
        logger.warning("Skipped %d secondary mappings without a primary impact", len(missing_primary))

    return {
        "exploitation": exploitation_rows,
        "primary_impact": primary_rows,
        "secondary_impact": secondary_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CVE→ATT&CK datasets from mappings-explorer YAML")
    parser.add_argument(
        "--mapping-path",
        default="dataset/mappings-explorer/kev-02.13.2025_attack-15.1-enterprise.yaml",
        help="Path to the mappings-explorer YAML file",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/minerva",
        help="Directory where the JSONL files will be written",
    )
    parser.add_argument(
        "--ask-subtechnique",
        action="store_true",
        help="Ask for a sub-technique ID in the prompt (default: technique ID only)",
    )
    args = parser.parse_args()

    logger = get_logger("mappings-explorer")
    build_cve_attack_datasets(
        args.mapping_path,
        args.output_dir,
        ask_subtechnique=bool(args.ask_subtechnique),
        logger=logger,
    )


if __name__ == "__main__":
    main()
