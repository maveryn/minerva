import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from minerva.logger import get_logger
from minerva.reward import binary_id, cvss_reward, f1_set
from minerva.utils import balanced_sample, dedupe_by_text, normalize_text, word_count, write_jsonl


def _select_cvss(rec: Dict[str, Any], prefer_nvd: bool = True, version: str = "v31") -> Optional[Tuple[str, Optional[float]]]:
    key = "v31" if version.lower().startswith("v3") else "v4"
    sources = ["nvd", "cna"]
    if not prefer_nvd:
        sources = ["nvd", "cna"]
    for src in sources:
        vec = rec.get(f"{src}_cvss_{key}_vector") or ""
        score = rec.get(f"{src}_cvss_{key}_score")
        if vec:
            return vec, score
    return None


def _balanced_cap(records: List[Dict[str, Any]], max_items: int, label_fn, rng: random.Random) -> List[Dict[str, Any]]:
    if max_items:
        return balanced_sample(records, label_fn=label_fn, max_items=max_items, rng=rng)
    rng.shuffle(records)
    return records


def build_cve_to_cwe(records: List[Dict[str, Any]], min_words: int, output_path: str, seed: int = 1337, max_items: int = 10000, logger=None) -> List[Dict[str, Any]]:
    if logger is None:
        logger = get_logger("task-cve-cwe")
    rng = random.Random(seed)
    pool = []
    for r in records:
        desc = r.get("description") or ""
        cwes = [c for c in (r.get("cwe_ids") or []) if c]
        if len(cwes) != 1:
            continue
        if word_count(desc) < min_words:
            continue
        pool.append(
            {
                "task": "cve_to_cwe",
                "input": {
                    "cve_id": r.get("cve_id"),
                    "description": desc,
                },
                "ground_truth": {"cwe_id": cwes[0]},
                "reward": {"type": "binary_id", "target": "cwe_id"},
                "metadata": {
                    "published_date": r.get("published_date"),
                },
            }
        )
    pool = _balanced_cap(pool, max_items, label_fn=lambda r: r["ground_truth"]["cwe_id"], rng=rng)
    write_jsonl(output_path, pool)
    logger.info("Built %d CVE->CWE tasks -> %s", len(pool), output_path)
    return pool


def build_cve_to_cvss(records: List[Dict[str, Any]], min_words: int, output_path: str, prefer_nvd: bool = True, seed: int = 1337, max_items: int = 10000, logger=None) -> List[Dict[str, Any]]:
    if logger is None:
        logger = get_logger("task-cvss")
    rng = random.Random(seed)
    pool: List[Dict[str, Any]] = []
    for r in records:
        desc = r.get("description") or ""
        if word_count(desc) < min_words:
            continue
        chosen = _select_cvss(r, prefer_nvd=prefer_nvd, version="v31")
        if not chosen:
            continue
        vec, score = chosen
        pool.append(
            {
                "task": "cve_to_cvss",
                "input": {"cve_id": r.get("cve_id"), "description": desc},
                "ground_truth": {"cvss_v31_vector": vec, "score": score},
                "reward": {"type": "cvss_v31", "target": "cvss_v31_vector"},
                "metadata": {"published_date": r.get("published_date")},
            }
        )
    pool = _balanced_cap(pool, max_items, label_fn=lambda r: r["ground_truth"]["cvss_v31_vector"], rng=rng)
    write_jsonl(output_path, pool)
    logger.info("Built %d CVE->CVSS tasks -> %s", len(pool), output_path)
    return pool


def build_scenario_to_technique(records: List[Dict[str, Any]], max_items: int, output_path: str, seed: int = 1337, logger=None) -> List[Dict[str, Any]]:
    if logger is None:
        logger = get_logger("task-scenario-technique")
    rng = random.Random(seed)
    records = _balanced_cap(
        records,
        max_items,
        label_fn=lambda r: r.get("technique_id"),
        rng=rng,
    )
    rows: List[Dict[str, Any]] = []
    for r in records:
        rows.append(
            {
                "task": "scenario_to_attack_technique",
                "input": {
                    "scenario": r.get("scenario") or "",
                    "platforms": r.get("platforms", []),
                },
                "ground_truth": {"technique_id": r.get("technique_id")},
                "reward": {"type": "binary_id", "target": "technique_id"},
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


def build_scenario_to_tactics(records: List[Dict[str, Any]], output_path: str, seed: int = 1337, max_items: Optional[int] = None, logger=None) -> List[Dict[str, Any]]:
    if logger is None:
        logger = get_logger("task-tactics")
    rng = random.Random(seed)
    pool: List[Dict[str, Any]] = []
    for r in records:
        tactics = r.get("metadata", {}).get("tactics", [])
        if not tactics:
            continue
        pool.append(
            {
                "task": "scenario_to_attack_tactics",
                "input": r["input"],
                "ground_truth": {"tactics": tactics},
                "reward": {"type": "f1_set", "target": "tactics"},
                "metadata": {"technique_id": r["ground_truth"].get("technique_id")},
            }
        )
    pool = _balanced_cap(pool, max_items or 0, label_fn=lambda r: tuple(sorted(r["ground_truth"]["tactics"])), rng=rng)
    write_jsonl(output_path, pool)
    logger.info("Built %d scenario->tactic tasks -> %s", len(pool), output_path)
    return pool


def build_scenario_to_mitigations(records: List[Dict[str, Any]], output_path: str, seed: int = 1337, max_items: Optional[int] = None, logger=None) -> List[Dict[str, Any]]:
    if logger is None:
        logger = get_logger("task-mitigations")
    rng = random.Random(seed)
    pool: List[Dict[str, Any]] = []
    for r in records:
        mits = r.get("metadata", {}).get("mitigations", [])
        if not mits:
            continue
        pool.append(
            {
                "task": "scenario_to_attack_mitigations",
                "input": r["input"],
                "ground_truth": {"mitigations": mits},
                "reward": {"type": "f1_set", "target": "mitigations"},
                "metadata": {"technique_id": r["ground_truth"].get("technique_id")},
            }
        )
    pool = _balanced_cap(pool, max_items or 0, label_fn=lambda r: tuple(sorted(r["ground_truth"]["mitigations"])), rng=rng)
    write_jsonl(output_path, pool)
    logger.info("Built %d scenario->mitigation tasks -> %s", len(pool), output_path)
    return pool


def build_scenario_to_detections(records: List[Dict[str, Any]], output_path: str, seed: int = 1337, max_items: Optional[int] = None, logger=None) -> List[Dict[str, Any]]:
    if logger is None:
        logger = get_logger("task-detections")
    rng = random.Random(seed)
    pool: List[Dict[str, Any]] = []
    for r in records:
        dets = r.get("metadata", {}).get("detection_strategies", [])
        if not dets:
            continue
        pool.append(
            {
                "task": "scenario_to_attack_detection",
                "input": r["input"],
                "ground_truth": {"detection_strategies": dets},
                "reward": {"type": "f1_set", "target": "detection_strategies"},
                "metadata": {"technique_id": r["ground_truth"].get("technique_id")},
            }
        )
    pool = _balanced_cap(pool, max_items or 0, label_fn=lambda r: tuple(sorted(r["ground_truth"]["detection_strategies"])), rng=rng)
    write_jsonl(output_path, pool)
    logger.info("Built %d scenario->detection tasks -> %s", len(pool), output_path)
    return pool


def build_cve_to_capec_attack(
    cve_records: List[Dict[str, Any]],
    cwe_to_capec: Dict[str, List[str]],
    capec_patterns: Dict[str, Dict[str, Any]],
    output_path: str,
    min_words: int,
    seed: int = 1337,
    max_items: Optional[int] = None,
    logger=None,
) -> List[Dict[str, Any]]:
    if logger is None:
        logger = get_logger("task-attack-chain")
    rng = random.Random(seed)
    pool: List[Dict[str, Any]] = []
    for r in cve_records:
        desc = r.get("description") or ""
        if word_count(desc) < min_words:
            continue
        cwes = [c for c in (r.get("cwe_ids") or []) if c]
        capec_ids: List[str] = []
        for cwe in cwes:
            capec_ids.extend(cwe_to_capec.get(cwe, []))
        capec_ids = sorted(set(capec_ids))
        tech_set: List[str] = []
        for capec_id in capec_ids:
            pattern = capec_patterns.get(capec_id, {})
            tech_set.extend(pattern.get("attack_techniques", []))
        tech_set = sorted(set(tech_set))
        if not capec_ids or not tech_set:
            continue
        pool.append(
            {
                "task": "cve_to_attack_chain",
                "input": {"cve_id": r.get("cve_id"), "description": desc},
                "ground_truth": {
                    "cwe_ids": cwes,
                    "capec_ids": capec_ids,
                    "technique_ids": tech_set,
                },
                "reward": {
                    "type": "f1_set",
                    "target": "technique_ids",
                    "partial_credit": ["cwe_ids", "capec_ids"],
                },
                "metadata": {"capec_count": len(capec_ids)},
            }
        )
    pool = _balanced_cap(pool, max_items or 0, label_fn=lambda r: tuple(sorted(r["ground_truth"]["technique_ids"])), rng=rng)
    write_jsonl(output_path, pool)
    logger.info("Built %d CVE->CAPEC->ATT&CK tasks -> %s", len(pool), output_path)
    return pool


def build_cwe_code_to_cwe(output_path: str, logger=None) -> List[Dict[str, Any]]:
    """
    Placeholder for CWE code snippet extraction. Logged as skipped so the pipeline
    continues while leaving a breadcrumb for future implementation.
    """
    if logger is None:
        logger = get_logger("task-cwe-code")
    logger.warning("Skipping CWE code snippet task: no authoritative code snippet source wired yet.")
    write_jsonl(output_path, [])
    return []
