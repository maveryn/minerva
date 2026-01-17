import json
import random
import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from minerva.logger import get_logger


TARGET_SAMPLES: Dict[str, int] = {
    "cve_to_attack_exploitation": 265,
    "cve_to_attack_primary_impact": 230,
    "cve_to_attack_secondary_impact": 74,
    "sigma_to_attack_technique": 1500,
    "sigma_to_attack_tactics": 931,
    "scenario_to_technique": 8500,
    "scenario_to_tactics": 2000,
    "scenario_to_mitigations": 8500,
    "cve_to_cwe": 9000,
    "cve_to_cvss_v31": 2000,
}

TARGET_VAL_SAMPLES: Dict[str, int] = {
    "cve_to_attack_exploitation": 20,
    "cve_to_attack_primary_impact": 20,
    "cve_to_attack_secondary_impact": 20,
    "sigma_to_attack_technique": 50,
    "sigma_to_attack_tactics": 50,
    "scenario_to_technique": 220,
    "scenario_to_tactics": 50,
    "scenario_to_mitigations": 220,
    "cve_to_cwe": 250,
    "cve_to_cvss_v31": 100,
}

FILE_MAP: Dict[str, str] = {
    "cve_to_attack_exploitation": "cve_to_attack_exploitation.jsonl",
    "cve_to_attack_primary_impact": "cve_to_attack_primary_impact.jsonl",
    "cve_to_attack_secondary_impact": "cve_to_attack_secondary_impact.jsonl",
    "sigma_to_attack_technique": "sigma_to_attack_technique.jsonl",
    "sigma_to_attack_tactics": "sigma_to_attack_tactics.jsonl",
    "scenario_to_technique": "scenario_to_technique.jsonl",
    "scenario_to_tactics": "scenario_to_tactics.jsonl",
    "scenario_to_mitigations": "scenario_to_mitigations.jsonl",
    "cve_to_cwe": "cve_to_cwe.jsonl",
    "cve_to_cvss_v31": "cve_to_cvss_v31.jsonl",
}

LOOKUP_FILENAME = "threat_actor_lookup.json"


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


def _split_rows(rows: List[Dict], train_ratio: float) -> Tuple[List[Dict], List[Dict]]:
    train_n = int(round(len(rows) * train_ratio))
    return rows[:train_n], rows[train_n:]


def _pick_example(rows: List[Dict]) -> Dict:
    return rows[0] if rows else {}


def _project_row(row: Dict) -> Dict:
    if not isinstance(row, dict):
        return {}
    prompt = (row.get("input") or {}).get("prompt", "")
    out = {
        "task": row.get("task"),
        "prompt": prompt,
        "answer": row.get("answer", row.get("ground_truth")),
        "reward_fn": row.get("reward_fn"),
    }
    pool = row.get("candidate_pool_top100")
    if pool is None:
        pool = (row.get("extra_info") or {}).get("candidate_pool_top100")
    if pool is not None:
        out["candidate_pool_top100"] = pool
    return out


def _dedupe_labels(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw in values or []:
        val = str(raw).strip()
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def _extract_label_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        val = value.strip()
        return [val] if val else []
    if isinstance(value, (list, tuple)):
        return _dedupe_labels(value)
    if isinstance(value, dict):
        list_key_order = (
            "technique_ids",
            "tactic_ids",
            "mitigation_ids",
            "detection_ids",
            "cwe_ids",
            "capec_ids",
            "attack_ids",
            "ids",
            "labels",
        )
        key_order = (
            "technique_id",
            "detection_id",
            "mitigation_id",
            "cwe_id",
            "capec_id",
            "attack_id",
            "tactic_id",
            "id",
            "label",
        )
        for key in list_key_order:
            if key in value:
                found = value.get(key)
                if isinstance(found, str):
                    val = found.strip()
                    return [val] if val else []
                if isinstance(found, (list, tuple)):
                    return _dedupe_labels(found)
        for key in key_order:
            if key in value:
                found = value.get(key)
                if isinstance(found, str):
                    val = found.strip()
                    return [val] if val else []
                if isinstance(found, (list, tuple)):
                    return _dedupe_labels(found)
    return []


def _format_candidate_prompt(
    row: Dict,
    *,
    k: int = 5,
    include_descriptions: bool = False,
    desc_max_chars: int = 400,
) -> Optional[str]:
    if not isinstance(row, dict):
        return None
    prompt = str(row.get("prompt") or "")
    pool = row.get("candidate_pool_top100")
    if pool is None:
        pool = (row.get("extra_info") or {}).get("candidate_pool_top100")
    if not isinstance(pool, list) or not pool:
        return None
    gold_ids = _extract_label_list(row.get("answer"))
    required_count = len(gold_ids) if gold_ids else 1
    total_k = k + max(0, required_count - 1)
    if total_k < required_count:
        total_k = required_count
    normalized = []
    seen = set()
    for item in pool:
        if isinstance(item, dict):
            ident = str(item.get("id") or item.get("label") or item.get("value") or item.get("name") or "").strip()
            name = str(item.get("name") or "").strip()
            desc = str(item.get("description") or item.get("desc") or "").strip()
        else:
            ident = str(item).strip()
            name = ""
            desc = ""
        if not ident or ident in seen:
            continue
        seen.add(ident)
        normalized.append({"id": ident, "name": name, "description": desc})
    if not normalized:
        return None
    pool_ids = {item["id"] for item in normalized}
    missing = [gid for gid in gold_ids if gid not in pool_ids]
    if missing:
        normalized = [{"id": gid, "name": "", "description": ""} for gid in missing] + normalized
    options = []
    option_ids = set()
    for gid in gold_ids:
        if not gid or gid in option_ids:
            continue
        match = next((item for item in normalized if item["id"] == gid), None)
        options.append(match or {"id": gid, "name": "", "description": ""})
        option_ids.add(gid)
    if not options:
        options = []
    for item in normalized:
        if len(options) >= total_k:
            break
        if item["id"] in option_ids:
            continue
        options.append(item)
        option_ids.add(item["id"])
    if not options:
        return None
    label = "ID" if required_count == 1 else "IDs"
    lines = [f"Candidate {label} (choose EXACTLY {required_count}):"]
    for idx, item in enumerate(options, start=1):
        ident = str(item.get("id") or "").strip()
        name = str(item.get("name") or "").strip()
        desc = str(item.get("description") or "").strip()
        if not include_descriptions:
            desc = ""
        elif desc and desc_max_chars > 0 and len(desc) > desc_max_chars:
            desc = desc[:desc_max_chars].rsplit(" ", 1)[0] + "..."
        parts = [p for p in (ident, name, desc) if p]
        line = " | ".join(parts) if parts else ident
        lines.append(f"{idx}) {line}")
    if required_count == 1:
        tail = "The correct ID is one of the candidates above. "
    else:
        tail = "The correct IDs are among the candidates above. "
    lines.append(
        f"{tail}You may use them as a reference, but do not mention the list. "
        "Reason step by step to arrive at the answer."
    )
    if prompt:
        return f"{prompt.rstrip()}\n\n" + "\n".join(lines)
    return "\n".join(lines)


def _row_key(row: Dict) -> Tuple[str, str, str]:
    task = ""
    reward_fn = ""
    prompt = ""
    if isinstance(row, dict):
        task = str(row.get("task") or row.get("reward_fn") or "")
        reward_fn = str(row.get("reward_fn") or "")
        if "prompt" in row:
            prompt = str(row.get("prompt") or "")
        else:
            prompt = str((row.get("input") or {}).get("prompt") or "")
    return task, reward_fn, prompt


def _reuse_split_rows(input_dir: Path, reuse_dir: Path, logger) -> Tuple[List[Dict], List[Dict], Dict[str, Dict[str, int]], Dict[str, Dict[str, Dict]]]:
    train_path = reuse_dir / "train.jsonl"
    dev_path = reuse_dir / "dev.jsonl"
    if not train_path.exists() or not dev_path.exists():
        train_candidates = sorted(reuse_dir.glob("*-train.jsonl"))
        dev_candidates = sorted(reuse_dir.glob("*-dev.jsonl"))
        if len(train_candidates) == 1 and len(dev_candidates) == 1:
            train_path = train_candidates[0]
            dev_path = dev_candidates[0]
        else:
            raise FileNotFoundError(f"Reuse split missing train/dev JSONL in {reuse_dir}")

    train_ref = list(_iter_jsonl(train_path))
    dev_ref = list(_iter_jsonl(dev_path))

    index: Dict[Tuple[str, str, str], Dict] = {}
    for task, filename in FILE_MAP.items():
        path = input_dir / filename
        if not path.exists():
            logger.warning("Missing file for task %s: %s", task, path)
            continue
        for row in _iter_jsonl(path):
            key = _row_key(row)
            index[key] = row

    def _map_rows(ref_rows: List[Dict], split_name: str) -> Tuple[List[Dict], int]:
        mapped: List[Dict] = []
        missing = 0
        for row in ref_rows:
            key = _row_key(row)
            match = index.get(key)
            if match is None:
                missing += 1
                continue
            mapped.append(match)
        if missing:
            logger.warning("Missing %d rows while reusing %s split.", missing, split_name)
        return mapped, missing

    train_rows, _ = _map_rows(train_ref, "train")
    dev_rows, _ = _map_rows(dev_ref, "dev")

    stats: Dict[str, Dict[str, int]] = {}
    examples: Dict[str, Dict[str, Dict]] = {}

    def _update_stats(rows: List[Dict], key: str) -> None:
        for row in rows:
            task = str(row.get("task") or row.get("reward_fn") or "unknown")
            stats.setdefault(task, {"train": 0, "dev": 0, "total": 0})
            stats[task][key] += 1
            stats[task]["total"] += 1
            examples.setdefault(task, {})
            examples[task].setdefault(key, row)

    _update_stats(train_rows, "train")
    _update_stats(dev_rows, "dev")

    return train_rows, dev_rows, stats, examples


def build_splits(
    *,
    input_dir: str = "dataset/minerva",
    output_dir: Optional[str] = None,
    file_prefix: Optional[str] = None,
    seed: int = 1337,
    train_ratio: float = 0.8,
    target_train: int = 32000,
    target_dev: int = 1000,
    reuse_split: Optional[str] = None,
) -> None:
    logger = get_logger("split")
    rng = random.Random(seed)

    in_dir = Path(input_dir)
    if output_dir is None:
        output_dir = f"{input_dir}_split"
    out_dir = Path(output_dir)

    if reuse_split:
        reuse_dir = Path(reuse_split)
        train_rows, dev_rows, stats, examples = _reuse_split_rows(in_dir, reuse_dir, logger)
    else:
        train_rows = []
        dev_rows = []
        stats = {}
        examples = {}
        per_task: Dict[str, Dict[str, List[Dict]]] = {}

        use_fixed_val = bool(TARGET_VAL_SAMPLES)

        for task, target_n in TARGET_SAMPLES.items():
            path = in_dir / FILE_MAP[task]
            if not path.exists():
                logger.warning("Missing file for task %s: %s", task, path)
                continue
            sampled, total = _reservoir_sample_jsonl(path, target_n, rng)
            rng.shuffle(sampled)
            if use_fixed_val:
                target_val = TARGET_VAL_SAMPLES.get(task, 0)
                if target_val > len(sampled):
                    logger.warning(
                        "Task %s -> requested dev=%d but only %d rows available; using %d.",
                        task,
                        target_val,
                        len(sampled),
                        len(sampled),
                    )
                    target_val = len(sampled)
                dev_part = sampled[:target_val]
                train_part = sampled[target_val:]
            else:
                train_part, dev_part = _split_rows(sampled, train_ratio)
            per_task[task] = {"train": train_part, "dev": dev_part}
            train_rows.extend(train_part)
            dev_rows.extend(dev_part)
            stats[task] = {
                "train": len(train_part),
                "dev": len(dev_part),
                "total": len(sampled),
            }
            examples[task] = {
                "train": _pick_example(train_part),
                "dev": _pick_example(dev_part),
            }
            logger.info(
                "Task %s -> sampled %d/%d (train %d / dev %d)",
                task,
                len(sampled),
                total,
                len(train_part),
                len(dev_part),
            )

        def _move_one(src_key: str, dst_key: str) -> bool:
            # move one item from src (train/dev) to dst for any task with available rows
            candidates = [t for t, parts in per_task.items() if parts[src_key]]
            if not candidates:
                return False
            task = rng.choice(candidates)
            item = per_task[task][src_key].pop()
            per_task[task][dst_key].append(item)
            if src_key == "train":
                train_rows.remove(item)
                dev_rows.append(item)
                stats[task]["train"] -= 1
                stats[task]["dev"] += 1
            else:
                dev_rows.remove(item)
                train_rows.append(item)
                stats[task]["dev"] -= 1
                stats[task]["train"] += 1
            return True

        if use_fixed_val:
            logger.info(
                "Fixed per-task dev sizes -> train=%d dev=%d (targets train=%d dev=%d)",
                len(train_rows),
                len(dev_rows),
                target_train,
                target_dev,
            )
        else:
            while len(train_rows) < target_train and len(dev_rows) > target_dev:
                if not _move_one("dev", "train"):
                    break
            while len(train_rows) > target_train and len(dev_rows) < target_dev:
                if not _move_one("train", "dev"):
                    break

            logger.info(
                "Adjusted splits to train=%d dev=%d (target train=%d dev=%d)",
                len(train_rows),
                len(dev_rows),
                target_train,
                target_dev,
            )

    # Project rows to the minimal schema
    projected_train = [_project_row(r) for r in train_rows]
    projected_dev = [_project_row(r) for r in dev_rows]

    out_dir.mkdir(parents=True, exist_ok=True)

    if file_prefix:
        train_path = out_dir / f"{file_prefix}-train.jsonl"
        dev_path = out_dir / f"{file_prefix}-dev.jsonl"
    else:
        train_path = out_dir / "train.jsonl"
        dev_path = out_dir / "dev.jsonl"
    train_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in projected_train) + "\n", encoding="utf-8")
    dev_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in projected_dev) + "\n", encoding="utf-8")
    logger.info("Wrote train (%d rows) -> %s", len(projected_train), train_path)
    logger.info("Wrote dev (%d rows) -> %s", len(projected_dev), dev_path)

    projected_examples = {
        k: {"train": _project_row(v.get("train")), "dev": _project_row(v.get("dev"))} for k, v in examples.items()
    }
    candidate_examples: Dict[str, Dict[str, Optional[str]]] = {}
    for task, ex in projected_examples.items():
        cand_train = _format_candidate_prompt(ex.get("train"))
        cand_dev = _format_candidate_prompt(ex.get("dev"))
        if cand_train or cand_dev:
            candidate_examples[task] = {"train": cand_train, "dev": cand_dev}

    meta = {
        "seed": seed,
        "summary": {
            "train": len(train_rows),
            "dev": len(dev_rows),
            "total": len(train_rows) + len(dev_rows),
        },
        "per_task": stats,
        "examples": projected_examples,
    }
    if candidate_examples:
        meta["candidate_examples"] = candidate_examples
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote split metadata -> %s", out_dir / "metadata.json")

    lookup_path = in_dir / LOOKUP_FILENAME
    if lookup_path.exists():
        shutil.copy2(lookup_path, out_dir / LOOKUP_FILENAME)
        logger.info("Copied threat actor lookup -> %s", out_dir / LOOKUP_FILENAME)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/dev splits from Minerva datasets")
    parser.add_argument("--input-dir", default="dataset/minerva", help="Input dataset directory (contains per-task JSONL files)")
    parser.add_argument("--output-dir", default=None, help="Output directory for train/dev JSONL (default: <input-dir>_split)")
    parser.add_argument(
        "--file-prefix",
        default=None,
        help="Optional filename prefix for outputs (writes <prefix>-train.jsonl / <prefix>-dev.jsonl)",
    )
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed (default: 1337)")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio per-task before global adjustment (default: 0.8)")
    parser.add_argument("--target-train", type=int, default=32000, help="Final target train size (default: 32000)")
    parser.add_argument("--target-dev", type=int, default=1000, help="Final target dev size (default: 1000)")
    parser.add_argument(
        "--reuse-split",
        default=None,
        help="Reuse train/dev split from an existing split directory (expects train.jsonl/dev.jsonl)",
    )
    args = parser.parse_args()
    build_splits(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_prefix=args.file_prefix,
        seed=args.seed,
        train_ratio=args.train_ratio,
        target_train=args.target_train,
        target_dev=args.target_dev,
        reuse_split=args.reuse_split,
    )


if __name__ == "__main__":
    main()
