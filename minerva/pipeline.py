import argparse
import copy
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from minerva.data_sources.capec import load_bundle as load_capec_bundle
from minerva.data_sources.mitre import collect_procedure_scenarios
from minerva.data_sources.nvd import ensure_records as ensure_nvd_records
from minerva.logger import get_logger
from minerva.tasks import (
    build_cve_to_capec_attack,
    build_capec_example_tasks,
    build_cve_to_cwe,
    build_cve_to_cvss_v31,
    build_cve_to_cvss_v40,
    build_cve_attack_datasets,
    build_scenario_to_mitigations,
    build_scenario_to_tactics,
    build_scenario_to_technique,
    build_sigma_datasets,
    build_threat_actor_tasks,
)
from minerva.utils import dedupe_by_text, load_yaml


RESAMPLE_COUNTS: Dict[str, int] = {
    "cve_to_attack_exploitation": 265,
    "cve_to_attack_primary_impact": 230,
    "cve_to_attack_secondary_impact": 74,
    "sigma_to_attack_technique": 1500,
    "sigma_to_attack_tactics": 931,
    "scenario_to_technique": 8000,
    "scenario_to_tactics": 2000,
    "scenario_to_mitigations": 8000,
    "cve_to_cwe": 10000,
    "cve_to_cvss_v31": 2000,
    "cve_to_cvss_v40": 0,
    "capec_example_to_capec": 0,
    "capec_example_to_cwe": 0,
    "capec_example_to_attack": 0,
    "threat_actor": 0,
}
RESAMPLE_SEED = 1337


def _pick_random_jsonl(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    chosen = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if random.randint(1, i) == 1:
                chosen = rec
    return chosen


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _reservoir_sample_jsonl(path: Path, n: int, rng: random.Random) -> Tuple[List[Dict], int]:
    """
    Reservoir-sample n rows from JSONL without loading the full file into memory.
    Returns (sampled_rows, total_rows_seen).
    """
    if n <= 0:
        total = 0
        for _ in _iter_jsonl(path):
            total += 1
        return [], total

    sample: List[Dict] = []
    total = 0
    for row in _iter_jsonl(path):
        total += 1
        if len(sample) < n:
            sample.append(row)
            continue
        j = rng.randrange(total)
        if j < n:
            sample[j] = row
    if total <= n:
        rng.shuffle(sample)
    return sample, total


def _resample_jsonl(path: Path, target_n: int, rng: random.Random, logger=None) -> Tuple[int, int]:
    """
    Resample a JSONL file down to target_n rows (if needed).
    Returns (kept_count, total_rows_seen).
    """
    if not path.exists():
        return 0, 0
    sampled, total = _reservoir_sample_jsonl(path, target_n, rng)
    if total <= target_n:
        return total, total
    rng.shuffle(sampled)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in sampled) + "\n", encoding="utf-8")
    if logger is not None:
        logger.info("Resampled %s -> %d/%d", path.name, len(sampled), total)
    return len(sampled), total


def _dedupe_jsonl(path: Path) -> tuple[int, int]:
    """
    Dedupe a JSONL file by the input prompt (falls back to serialized input).
    Returns (kept_count, removed_count).
    """
    if not path.exists():
        return 0, 0
    seen = set()
    rows = []
    removed = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = json.dumps(rec.get("input", {}), sort_keys=True)
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            rows.append(rec)
    if removed > 0:
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return len(rows), removed


def _replace_minerva_path(path: str, output_root: str) -> str:
    if not path:
        return path
    out = str(path)
    for base in ("dataset/minerva", "data/processed/minerva"):
        if base in out:
            out = out.replace(base, output_root)
    return out


def _apply_output_root(cfg: Dict, output_root: str) -> None:
    common_cfg = cfg.setdefault("COMMON", {})
    common_cfg["minerva_out"] = output_root
    common_cfg["metadata_path"] = str(Path(output_root) / "metadata.json")
    tasks_cfg = cfg.setdefault("TASKS", {})
    for task_cfg in tasks_cfg.values():
        if isinstance(task_cfg, dict) and "output_path" in task_cfg:
            task_cfg["output_path"] = _replace_minerva_path(task_cfg["output_path"], output_root)


def _resolve_output_root(base_root: str, suffix: Optional[str]) -> str:
    base = str(base_root).rstrip("/")
    if not suffix:
        return base
    return f"{base}_{suffix}"


def build_minerva_dataset(cfg: Dict, *, logger, detailed_prompts: bool) -> None:
    logger.info("=== Minerva dataset build starting ===")
    summary: Dict[str, int] = {}
    examples: Dict[str, Dict] = {}
    resample_rng = random.Random(RESAMPLE_SEED)

    # Load data sources
    logger.info("Loading NVD CVE records")
    nvd_records = ensure_nvd_records(cfg, logger=logger)
    nvd_records = dedupe_by_text(
        nvd_records,
        text_fn=lambda r: r.get("description", ""),
        prefer_ts_fn=lambda r: r.get("published_date") or r.get("last_modified_date"),
    )

    capec_needed = any(
        RESAMPLE_COUNTS.get(key, 0) > 0
        for key in ("capec_example_to_capec", "capec_example_to_cwe", "capec_example_to_attack")
    )
    capec_data = None
    if capec_needed:
        logger.info("Loading CAPEC bundle")
        capec_data = load_capec_bundle(cfg.get("CAPEC", {}), logger=logger)
    else:
        logger.info("Skipping CAPEC bundle load (CAPEC example tasks disabled)")

    logger.info("Loading ATT&CK procedure scenarios")
    scenario_records = collect_procedure_scenarios(cfg.get("MITRE_ATTACK", {}), logger=logger)

    tactic_id_to_name: Dict[str, str] = {}
    mitigation_id_to_name: Dict[str, str] = {}
    if detailed_prompts:
        from minerva.data_sources.mitre import load_bundle as load_mitre_bundle

        mitre_bundle = load_mitre_bundle(cfg.get("MITRE_ATTACK", {}), logger=logger)
        tactic_id_to_name = {v["id"]: v["name"] for v in (mitre_bundle.get("tactics", {}) or {}).values()}
        mitigation_id_to_name = {
            mid: (obj.get("name", "") if isinstance(obj, dict) else "")
            for mid, obj in (mitre_bundle.get("mitigations", {}) or {}).items()
        }

    # Mapping-explorer CVE -> ATT&CK datasets
    try:
        common_cfg = cfg.get("COMMON", {})
        mappings_path = common_cfg.get(
            "mappings_path", "dataset/mappings-explorer/kev-02.13.2025_attack-15.1-enterprise.yaml"
        )
        minerva_out = common_cfg.get("minerva_out", "dataset/minerva")
        mappings_cfg = cfg.get("MAPPINGS_EXPLORER", {})
        mappings = build_cve_attack_datasets(
            mappings_path,
            output_dir=minerva_out,
            ask_subtechnique=bool(mappings_cfg.get("ask_subtechnique", True)),
            logger=logger,
        )
        summary["cve_to_attack_exploitation"] = len(mappings.get("exploitation", []))
        summary["cve_to_attack_primary_impact"] = len(mappings.get("primary_impact", []))
        summary["cve_to_attack_secondary_impact"] = len(mappings.get("secondary_impact", []))
    except Exception as exc:
        logger.error("Failed mapping-explorer build: %s", exc)

    # CAPEC example tasks
    if capec_needed:
        try:
            cap_cfg = cfg.get("TASKS", {})
            capec_out = cap_cfg.get("CAPEC_EXAMPLE_CAPEC", {}).get(
                "output_path", "dataset/minerva/capec_example_to_capec.jsonl"
            )
            cwe_out = cap_cfg.get("CAPEC_EXAMPLE_CWE", {}).get(
                "output_path", "dataset/minerva/capec_example_to_cwe.jsonl"
            )
            atk_out = cap_cfg.get("CAPEC_EXAMPLE_ATTACK", {}).get(
                "output_path", "dataset/minerva/capec_example_to_attack.jsonl"
            )
            cap_seed = int(cap_cfg.get("CAPEC_EXAMPLE_CAPEC", {}).get("seed", 1337))
            capec_tasks = build_capec_example_tasks(
                capec_data["patterns"],
                capec_out=capec_out,
                cwe_out=cwe_out,
                attack_out=atk_out,
                seed=cap_seed,
                logger=logger,
            )
            summary["capec_example_to_capec"] = len(capec_tasks.get("capec", []))
            summary["capec_example_to_cwe"] = len(capec_tasks.get("cwe", []))
            summary["capec_example_to_attack"] = len(capec_tasks.get("attack", []))
        except Exception as exc:
            logger.error("Failed CAPEC example tasks: %s", exc)
    else:
        logger.info("Skipping CAPEC example tasks (disabled by sampling config)")

    # Sigma rule datasets
    try:
        sigma_dirs = cfg.get("SIGMA", {}).get(
            "rule_dirs", ["dataset/sigma/rules", "dataset/sigma/rules-threat-hunting"]
        )
        minerva_out = cfg.get("COMMON", {}).get("minerva_out", "dataset/minerva")
        sigma_cfg = cfg.get("SIGMA", {})
        sigma_sets = build_sigma_datasets(
            sigma_dirs,
            output_dir=minerva_out,
            ask_subtechnique=bool(sigma_cfg.get("ask_subtechnique", True)),
            include_id_names=bool(detailed_prompts),
            tactic_id_to_name=tactic_id_to_name,
            logger=logger,
        )
        summary["sigma_to_attack_technique"] = len(sigma_sets.get("technique", []))
        summary["sigma_to_attack_tactics"] = len(sigma_sets.get("tactic", []))
    except Exception as exc:
        logger.error("Failed sigma build: %s", exc)

    tasks_cfg = cfg.get("TASKS", {})
    cap_cfg = tasks_cfg

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

    # CVE -> CVSS v3.1
    try:
        cvss31_cfg = tasks_cfg.get("CVE_CVSS_V31", {})
        cvss31_tasks = build_cve_to_cvss_v31(
            nvd_records,
            min_words=int(cvss31_cfg.get("min_words", 25)),
            output_path=cvss31_cfg.get("output_path", "data/processed/minerva/cve_to_cvss_v31.jsonl"),
            prefer_nvd=bool(cvss31_cfg.get("prefer_nvd", True)),
            seed=int(cvss31_cfg.get("seed", 1337)),
            max_items=int(cvss31_cfg.get("max_items", 10000)),
            logger=logger,
        )
        summary["cve_to_cvss_v31"] = len(cvss31_tasks)
    except Exception as exc:
        logger.error("Failed CVE->CVSS v3.1: %s", exc)

    # CVE -> CVSS v4.0
    try:
        cvss40_cfg = tasks_cfg.get("CVE_CVSS_V40", {})
        cvss40_tasks = build_cve_to_cvss_v40(
            nvd_records,
            min_words=int(cvss40_cfg.get("min_words", 25)),
            output_path=cvss40_cfg.get("output_path", "data/processed/minerva/cve_to_cvss_v40.jsonl"),
            prefer_nvd=bool(cvss40_cfg.get("prefer_nvd", True)),
            seed=int(cvss40_cfg.get("seed", 1337)),
            max_items=int(cvss40_cfg.get("max_items", 10000)),
            logger=logger,
        )
        summary["cve_to_cvss_v40"] = len(cvss40_tasks)
    except Exception as exc:
        logger.error("Failed CVE->CVSS v4.0: %s", exc)

    # Scenario -> technique (base set reused downstream)
    scenario_tasks = []
    try:
        st_cfg = tasks_cfg.get("SCENARIO_TECHNIQUE", {})
        scenario_tasks = build_scenario_to_technique(
            scenario_records,
            max_items=int(st_cfg.get("max_items", 5000)),
            output_path=st_cfg.get("output_path", "data/processed/minerva/scenario_to_technique.jsonl"),
            seed=int(st_cfg.get("seed", 1337)),
            ask_subtechnique=bool(st_cfg.get("ask_subtechnique", True)),
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
            include_id_names=bool(detailed_prompts),
            tactic_id_to_name=tactic_id_to_name,
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
            include_id_names=bool(detailed_prompts),
            mitigation_id_to_name=mitigation_id_to_name,
            logger=logger,
        )
        summary["scenario_to_mitigations"] = len(mit_tasks)
    except Exception as exc:
        logger.error("Failed scenario->mitigation: %s", exc)

    # Threat actor (open-ended)
    if RESAMPLE_COUNTS.get("threat_actor", 0) > 0:
        try:
            ta_cfg = tasks_cfg.get("THREAT_ACTOR", {})
            ta_out = ta_cfg.get("output_path", "dataset/minerva/threat_actor.jsonl")
            ta_seed = int(ta_cfg.get("seed", 1337))
            ta_tasks = build_threat_actor_tasks(
                cfg.get("MITRE_ATTACK", {}),
                output_path=ta_out,
                seed=ta_seed,
                task_cfg=ta_cfg,
                logger=logger,
            )
            summary["threat_actor"] = len(ta_tasks)
        except Exception as exc:
            logger.error("Failed threat actor task: %s", exc)
    else:
        logger.info("Skipping threat actor tasks (disabled by sampling config)")

    # Persist metadata
    meta_path = Path(cfg.get("COMMON", {}).get("metadata_path", "data/processed/minerva/metadata.json"))
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect one random example per summarized task and dedupe datasets by prompt
    common_cfg = cfg.get("COMMON", {})
    minerva_out = common_cfg.get("minerva_out", "dataset/minerva")
    task_paths = {
        "cve_to_attack_exploitation": Path(minerva_out) / "cve_to_attack_exploitation.jsonl",
        "cve_to_attack_primary_impact": Path(minerva_out) / "cve_to_attack_primary_impact.jsonl",
        "cve_to_attack_secondary_impact": Path(minerva_out) / "cve_to_attack_secondary_impact.jsonl",
        "cve_to_cwe": Path(tasks_cfg.get("CVE_CWE", {}).get("output_path", "dataset/minerva/cve_to_cwe.jsonl")),
        "cve_to_cvss_v31": Path(tasks_cfg.get("CVE_CVSS_V31", {}).get("output_path", "dataset/minerva/cve_to_cvss_v31.jsonl")),
        "cve_to_cvss_v40": Path(tasks_cfg.get("CVE_CVSS_V40", {}).get("output_path", "dataset/minerva/cve_to_cvss_v40.jsonl")),
        "sigma_to_attack_technique": Path(minerva_out) / "sigma_to_attack_technique.jsonl",
        "sigma_to_attack_tactics": Path(minerva_out) / "sigma_to_attack_tactics.jsonl",
        "capec_example_to_capec": Path(cap_cfg.get("CAPEC_EXAMPLE_CAPEC", {}).get("output_path", "dataset/minerva/capec_example_to_capec.jsonl")),
        "capec_example_to_cwe": Path(cap_cfg.get("CAPEC_EXAMPLE_CWE", {}).get("output_path", "dataset/minerva/capec_example_to_cwe.jsonl")),
        "capec_example_to_attack": Path(cap_cfg.get("CAPEC_EXAMPLE_ATTACK", {}).get("output_path", "dataset/minerva/capec_example_to_attack.jsonl")),
        "threat_actor": Path(
            tasks_cfg.get("THREAT_ACTOR", {}).get("output_path", "dataset/minerva/threat_actor.jsonl")
        ),
        "scenario_to_technique": Path(tasks_cfg.get("SCENARIO_TECHNIQUE", {}).get("output_path", "dataset/minerva/scenario_to_technique.jsonl")),
        "scenario_to_tactics": Path(tasks_cfg.get("SCENARIO_TACTIC", {}).get("output_path", "dataset/minerva/scenario_to_tactics.jsonl")),
        "scenario_to_mitigations": Path(tasks_cfg.get("SCENARIO_MITIGATION", {}).get("output_path", "dataset/minerva/scenario_to_mitigations.jsonl")),
    }
    dedupe_info: Dict[str, int] = {}
    for task_name, path in task_paths.items():
        if task_name in summary:
            kept, removed = _dedupe_jsonl(path)
            if removed:
                dedupe_info[task_name] = removed
            target_n = RESAMPLE_COUNTS.get(task_name)
            if target_n is not None:
                resampled, _ = _resample_jsonl(path, target_n, resample_rng, logger=logger)
                summary[task_name] = resampled
            else:
                summary[task_name] = kept
            examples[task_name] = _pick_random_jsonl(path)

    meta_path.write_text(json.dumps({"summary": summary, "examples": examples, "dedupe_removed": dedupe_info}, indent=2), encoding="utf-8")
    logger.info("Wrote metadata -> %s", meta_path)
    if dedupe_info:
        logger.info("Dedupe removed entries: %s", dedupe_info)
    logger.info("=== Finished Minerva build ===")


def build_lhc_dataset(
    *,
    base_dir: str,
    output_dir: str,
    cfg: Dict,
    selection_path: str,
    k: int = 100,
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    logger,
) -> None:
    from minerva.analysis.retrieval_candidates import TASK_SPECS, augment_task_file, build_retrievers

    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Base dataset not found: {base_path}")

    selection = Path(selection_path)
    if not selection.exists():
        raise FileNotFoundError(f"Retrieval selection not found: {selection}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    retrievers = build_retrievers(
        cfg=cfg,
        selection_path=selection,
        dense_model_name=dense_model_name,
        batch_size=batch_size,
    )

    for path in sorted(base_path.glob("*.jsonl")):
        task_name = path.stem
        dest = out_dir / path.name
        spec = TASK_SPECS.get(task_name)
        retriever = retrievers.get(spec["label_type"]) if spec else None
        if spec and retriever:
            logger.info("LHC candidate pool -> %s", dest)
            augment_task_file(
                input_path=path,
                output_path=dest,
                retriever=retriever,
                spec=spec,
                k=k,
            )
        else:
            shutil.copy2(path, dest)
            if spec and not retriever:
                logger.warning("No retriever for %s; copied base file to LHC output.", task_name)

    meta_src = base_path / "metadata.json"
    if meta_src.exists():
        shutil.copy2(meta_src, out_dir / "metadata.json")
        logger.info("Copied metadata -> %s", out_dir / "metadata.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minerva RLVR dataset builder")
    parser.add_argument("--config", default="minerva/config.yaml", help="Path to minerva config YAML")
    parser.add_argument(
        "--variant",
        choices=["base", "lhc", "both"],
        default="base",
        help="Which dataset variant to build (default: base)",
    )
    parser.add_argument(
        "--output-root",
        default="dataset/minerva_base",
        help="Output root for base dataset (default: dataset/minerva_base)",
    )
    parser.add_argument(
        "--lhc-output-root",
        default="dataset/minerva_lhc",
        help="Output root for LHC dataset (default: dataset/minerva_lhc)",
    )
    parser.add_argument(
        "--lhc-selection",
        default="minerva/analysis/retrieval_selection/retrieval_selection.json",
        help="Path to retrieval selection JSON (default: minerva/analysis/retrieval_selection/retrieval_selection.json)",
    )
    parser.add_argument("--lhc-k", type=int, default=100, help="Top-K candidates to store for LHC (default: 100)")
    parser.add_argument(
        "--lhc-dense-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Dense encoder model for LHC retrieval (default: sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--lhc-batch-size",
        type=int,
        default=64,
        help="Batch size for dense encoder (default: 64)",
    )
    parser.add_argument(
        "--detailed-prompts",
        action="store_true",
        help="Append MITRE Enterprise ID->name catalogs to tactic/mitigation/detection prompts; writes datasets to a suffixed output folder",
    )
    parser.add_argument(
        "--detailed-suffix",
        default="detailed",
        help="Suffix appended to output folders when --detailed-prompts is set (default: detailed)",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    log_dir = cfg.get("COMMON", {}).get("log_dir", "data/logs")
    logger = get_logger("minerva", log_dir=log_dir)

    suffix = None
    if args.detailed_prompts:
        suffix = str(args.detailed_suffix).strip() or "detailed"

    base_output_root = _resolve_output_root(args.output_root, suffix)
    lhc_output_root = _resolve_output_root(args.lhc_output_root, suffix)

    base_cfg = copy.deepcopy(cfg)
    _apply_output_root(base_cfg, base_output_root)

    if args.variant in ("base", "both"):
        build_minerva_dataset(base_cfg, logger=logger, detailed_prompts=bool(args.detailed_prompts))

    if args.variant in ("lhc", "both"):
        if not Path(base_output_root).exists():
            logger.error("Base dataset not found at %s (use --output-root or build base first).", base_output_root)
            return
        build_lhc_dataset(
            base_dir=base_output_root,
            output_dir=lhc_output_root,
            cfg=cfg,
            selection_path=args.lhc_selection,
            k=args.lhc_k,
            dense_model_name=args.lhc_dense_model,
            batch_size=args.lhc_batch_size,
            logger=logger,
        )


if __name__ == "__main__":
    main()
