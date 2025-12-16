import random
from typing import Any, Dict, List, Optional, Tuple

from minerva.logger import get_logger
from minerva.tasks.common import _balanced_cap
from minerva.utils import word_count, write_jsonl


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


def build_cve_to_cwe(records: List[Dict[str, Any]], min_words: int, output_path: str, seed: int = 1337, max_items: int = 10000, logger=None) -> List[Dict[str, Any]]:
    if logger is None:
        logger = get_logger("task-cve-cwe")
    rng = random.Random(seed)
    pool = []
    for r in records:
        desc = r.get("description") or ""
        cwes = [c for c in (r.get("cwe_ids") or []) if c]
        if not cwes:
            continue
        if word_count(desc) < min_words:
            continue
        suffix = "" if len(cwes) == 1 else "s"
        prompt = (
            f"Given the Common Vulnerabilities and Exposures (CVE) description below, list EXACTLY {len(cwes)} Common Weakness Enumeration (CWE) ID{suffix} in CWE-<number> format.\n\n"
            f"CVE description:\n{desc}"
        )
        pool.append(
            {
                "task": "cve_to_cwe",
                "input": {
                    "cve_id": r.get("cve_id"),
                    "description": desc,
                    "prompt": prompt,
                },
                "ground_truth": {"cwe_ids": cwes},
                "reward_fn": "reward_cwe_ids",
                "metadata": {
                    "published_date": r.get("published_date"),
                },
            }
        )
    pool = _balanced_cap(pool, max_items, label_fn=lambda r: tuple(sorted(r["ground_truth"]["cwe_ids"])), rng=rng)
    write_jsonl(output_path, pool)
    logger.info("Built %d CVE->CWE tasks -> %s", len(pool), output_path)
    return pool


def build_cve_to_cvss_v31(records: List[Dict[str, Any]], min_words: int, output_path: str, prefer_nvd: bool = True, seed: int = 1337, max_items: int = 10000, logger=None) -> List[Dict[str, Any]]:
    if logger is None:
        logger = get_logger("task-cvss-v31")
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
        prompt = (
            "Given the Common Vulnerabilities and Exposures (CVE) description below, provide the CVSS v3.1 base vector string (format: CVSS:3.1/AV:X/AC:X/PR:X/UI:X/S:X/C:X/I:X/A:X).\n\n"
            f"CVE description:\n{desc}"
        )
        pool.append(
            {
                "task": "cve_to_cvss_v31",
                "input": {"cve_id": r.get("cve_id"), "description": desc, "prompt": prompt},
                "ground_truth": {"cvss_v31_vector": vec, "score": score},
                "reward_fn": "reward_cvss_v31",
                "metadata": {"published_date": r.get("published_date")},
            }
        )
    pool = _balanced_cap(pool, max_items, label_fn=lambda r: r["ground_truth"]["cvss_v31_vector"], rng=rng)
    write_jsonl(output_path, pool)
    logger.info("Built %d CVE->CVSS v3.1 tasks -> %s", len(pool), output_path)
    return pool


def build_cve_to_cvss_v40(records: List[Dict[str, Any]], min_words: int, output_path: str, prefer_nvd: bool = True, seed: int = 1337, max_items: int = 10000, logger=None) -> List[Dict[str, Any]]:
    if logger is None:
        logger = get_logger("task-cvss-v40")
    rng = random.Random(seed)
    pool: List[Dict[str, Any]] = []
    for r in records:
        desc = r.get("description") or ""
        if word_count(desc) < min_words:
            continue
        chosen = _select_cvss(r, prefer_nvd=prefer_nvd, version="v4")
        if not chosen:
            continue
        vec, score = chosen
        prompt = (
            "Given the Common Vulnerabilities and Exposures (CVE) description below, provide the CVSS v4.0 base vector string (format: CVSS:4.0/AV:X/AC:X/AT:X/PR:X/UI:X/VC:X/VI:X/VA:X/SC:X/SI:X/SA:X).\n\n"
            f"CVE description:\n{desc}"
        )
        pool.append(
            {
                "task": "cve_to_cvss_v40",
                "input": {"cve_id": r.get("cve_id"), "description": desc, "prompt": prompt},
                "ground_truth": {"cvss_v4_vector": vec, "score": score},
                "reward_fn": "reward_cvss_v40",
                "metadata": {"published_date": r.get("published_date")},
            }
        )
    pool = _balanced_cap(pool, max_items, label_fn=lambda r: r["ground_truth"]["cvss_v4_vector"], rng=rng)
    write_jsonl(output_path, pool)
    logger.info("Built %d CVE->CVSS v4.0 tasks -> %s", len(pool), output_path)
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
                "reward_fn": "f1_set",
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
