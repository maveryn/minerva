import argparse
import json
from pathlib import Path
from typing import Dict

from minerva.data_sources.capec import load_bundle as load_capec_bundle
from minerva.data_sources.mitre import collect_procedure_scenarios
from minerva.data_sources.nvd import ensure_records as ensure_nvd_records
from minerva.logger import get_logger
from minerva.tasks import (
    build_cve_to_capec_attack,
    build_cve_to_cwe,
    build_cve_to_cvss,
    build_cwe_code_to_cwe,
    build_scenario_to_detections,
    build_scenario_to_mitigations,
    build_scenario_to_tactics,
    build_scenario_to_technique,
)
from minerva.utils import dedupe_by_text, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Minerva RLVR dataset builder")
    parser.add_argument("--config", default="minerva/config.yaml", help="Path to minerva config YAML")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    log_dir = cfg.get("COMMON", {}).get("log_dir", "data/logs")
    logger = get_logger("minerva", log_dir=log_dir)

    logger.info("=== Minerva dataset build starting ===")
    summary: Dict[str, int] = {}

    # Load data sources
    logger.info("Loading NVD CVE records")
    nvd_records = ensure_nvd_records(cfg, logger=logger)
    nvd_records = dedupe_by_text(
        nvd_records,
        text_fn=lambda r: r.get("description", ""),
        prefer_ts_fn=lambda r: r.get("published_date") or r.get("last_modified_date"),
    )
    summary["nvd_records"] = len(nvd_records)

    logger.info("Loading CAPEC bundle")
    capec_data = load_capec_bundle(cfg.get("CAPEC", {}), logger=logger)
    summary["capec_patterns"] = len(capec_data["patterns"])

    logger.info("Loading ATT&CK procedure scenarios")
    scenario_records = collect_procedure_scenarios(cfg.get("MITRE_ATTACK", {}), logger=logger)
    summary["procedure_scenarios"] = len(scenario_records)

    tasks_cfg = cfg.get("TASKS", {})

    # CVE -> CWE
    try:
        cwe_cfg = tasks_cfg.get("CVE_CWE", {})
        cwe_tasks = build_cve_to_cwe(
            nvd_records,
            min_words=int(cwe_cfg.get("min_words", 20)),
            output_path=cwe_cfg.get("output_path", "data/processed/minerva/cve_to_cwe.jsonl"),
            seed=int(cwe_cfg.get("seed", 1337)),
            max_items=int(cwe_cfg.get("max_items", 10000)),
            logger=logger,
        )
        summary["cve_to_cwe"] = len(cwe_tasks)
    except Exception as exc:
        logger.error("Failed CVE->CWE: %s", exc)

    # CVE -> CVSS
    try:
        cvss_cfg = tasks_cfg.get("CVE_CVSS", {})
        cvss_tasks = build_cve_to_cvss(
            nvd_records,
            min_words=int(cvss_cfg.get("min_words", 25)),
            output_path=cvss_cfg.get("output_path", "data/processed/minerva/cve_to_cvss.jsonl"),
            prefer_nvd=bool(cvss_cfg.get("prefer_nvd", True)),
            seed=int(cvss_cfg.get("seed", 1337)),
            max_items=int(cvss_cfg.get("max_items", 10000)),
            logger=logger,
        )
        summary["cve_to_cvss"] = len(cvss_tasks)
    except Exception as exc:
        logger.error("Failed CVE->CVSS: %s", exc)

    # Scenario -> technique (base set reused downstream)
    scenario_tasks = []
    try:
        st_cfg = tasks_cfg.get("SCENARIO_TECHNIQUE", {})
        scenario_tasks = build_scenario_to_technique(
            scenario_records,
            max_items=int(st_cfg.get("max_items", 5000)),
            output_path=st_cfg.get("output_path", "data/processed/minerva/scenario_to_technique.jsonl"),
            seed=int(st_cfg.get("seed", 1337)),
            logger=logger,
        )
        summary["scenario_to_technique"] = len(scenario_tasks)
    except Exception as exc:
        logger.error("Failed scenario->technique: %s", exc)

    # Scenario -> tactics
    try:
        tac_cfg = tasks_cfg.get("SCENARIO_TACTIC", {})
        tactic_tasks = build_scenario_to_tactics(
            scenario_tasks,
            output_path=tac_cfg.get("output_path", "data/processed/minerva/scenario_to_tactics.jsonl"),
            seed=int(tac_cfg.get("seed", 1337)),
            max_items=int(tac_cfg.get("max_items", 0)),
            logger=logger,
        )
        summary["scenario_to_tactics"] = len(tactic_tasks)
    except Exception as exc:
        logger.error("Failed scenario->tactic: %s", exc)

    # Scenario -> mitigations
    try:
        mit_cfg = tasks_cfg.get("SCENARIO_MITIGATION", {})
        mit_tasks = build_scenario_to_mitigations(
            scenario_tasks,
            output_path=mit_cfg.get("output_path", "data/processed/minerva/scenario_to_mitigations.jsonl"),
            seed=int(mit_cfg.get("seed", 1337)),
            max_items=int(mit_cfg.get("max_items", 0)),
            logger=logger,
        )
        summary["scenario_to_mitigations"] = len(mit_tasks)
    except Exception as exc:
        logger.error("Failed scenario->mitigation: %s", exc)

    # Scenario -> detection strategies
    try:
        det_cfg = tasks_cfg.get("SCENARIO_DETECTION", {})
        det_tasks = build_scenario_to_detections(
            scenario_tasks,
            output_path=det_cfg.get("output_path", "data/processed/minerva/scenario_to_detections.jsonl"),
            seed=int(det_cfg.get("seed", 1337)),
            max_items=int(det_cfg.get("max_items", 0)),
            logger=logger,
        )
        summary["scenario_to_detections"] = len(det_tasks)
    except Exception as exc:
        logger.error("Failed scenario->detection: %s", exc)

    # CVE -> CAPEC -> ATT&CK
    try:
        chain_cfg = tasks_cfg.get("CVE_ATTACK_CHAIN", {})
        chain_tasks = build_cve_to_capec_attack(
            nvd_records,
            cwe_to_capec=capec_data["cwe_to_capec"],
            capec_patterns=capec_data["patterns"],
            output_path=chain_cfg.get("output_path", "data/processed/minerva/cve_to_attack_chain.jsonl"),
            min_words=int(chain_cfg.get("min_words", 15)),
            seed=int(chain_cfg.get("seed", 1337)),
            max_items=int(chain_cfg.get("max_items", 0)),
            logger=logger,
        )
        summary["cve_to_attack_chain"] = len(chain_tasks)
    except Exception as exc:
        logger.error("Failed CVE->CAPEC->ATT&CK: %s", exc)

    # CWE code snippet placeholder (logged skip)
    try:
        code_cfg = tasks_cfg.get("CWE_CODE", {})
        code_tasks = build_cwe_code_to_cwe(
            output_path=code_cfg.get("output_path", "data/processed/minerva/cwe_code_to_cwe.jsonl"),
            logger=logger,
        )
        summary["cwe_code_to_cwe"] = len(code_tasks)
    except Exception as exc:
        logger.error("Failed CWE code snippet task: %s", exc)

    # Persist metadata
    meta_path = Path(cfg.get("COMMON", {}).get("metadata_path", "data/processed/minerva/metadata.json"))
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"summary": summary}, indent=2), encoding="utf-8")
    logger.info("Wrote metadata -> %s", meta_path)
    logger.info("=== Finished Minerva build ===")


if __name__ == "__main__":
    main()
