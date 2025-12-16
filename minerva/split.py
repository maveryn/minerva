import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from minerva.logger import get_logger


TARGET_SAMPLES: Dict[str, int] = {
    "cve_to_attack_exploitation": 265,
    "cve_to_attack_primary_impact": 230,
    "cve_to_attack_secondary_impact": 74,
    "sigma_to_attack_technique": 1500,
    "sigma_to_attack_tactics": 1500,
    "scenario_to_technique": 8000,
    "scenario_to_tactics": 3000,
    "scenario_to_detections": 3000,
    "scenario_to_mitigations": 3000,
    "cve_to_cwe": 10000,
    "cve_to_cvss_v31": 4081,
    "cve_to_cvss_v40": 1000,
    "capec_example_to_capec": 380,
    "capec_example_to_cwe": 196,
    "capec_example_to_attack": 144,
    "threat_actor_mcq": 3630,
}

FILE_MAP: Dict[str, str] = {
    "cve_to_attack_exploitation": "dataset/minerva/cve_to_attack_exploitation.jsonl",
    "cve_to_attack_primary_impact": "dataset/minerva/cve_to_attack_primary_impact.jsonl",
    "cve_to_attack_secondary_impact": "dataset/minerva/cve_to_attack_secondary_impact.jsonl",
    "sigma_to_attack_technique": "dataset/minerva/sigma_to_attack_technique.jsonl",
    "sigma_to_attack_tactics": "dataset/minerva/sigma_to_attack_tactics.jsonl",
    "scenario_to_technique": "dataset/minerva/scenario_to_technique.jsonl",
    "scenario_to_tactics": "dataset/minerva/scenario_to_tactics.jsonl",
    "scenario_to_detections": "dataset/minerva/scenario_to_detections.jsonl",
    "scenario_to_mitigations": "dataset/minerva/scenario_to_mitigations.jsonl",
    "cve_to_cwe": "dataset/minerva/cve_to_cwe.jsonl",
    "cve_to_cvss_v31": "dataset/minerva/cve_to_cvss_v31.jsonl",
    "cve_to_cvss_v40": "dataset/minerva/cve_to_cvss_v40.jsonl",
    "capec_example_to_capec": "dataset/minerva/capec_example_to_capec.jsonl",
    "capec_example_to_cwe": "dataset/minerva/capec_example_to_cwe.jsonl",
    "capec_example_to_attack": "dataset/minerva/capec_example_to_attack.jsonl",
    "threat_actor_mcq": "dataset/minerva/threat_actor_mcq.jsonl",
}


def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _sample_rows(rows: List[Dict], n: int, rng: random.Random) -> List[Dict]:
    if n >= len(rows):
        rng.shuffle(rows)
        return rows
    return rng.sample(rows, n)


def _split_rows(rows: List[Dict], train_ratio: float) -> Tuple[List[Dict], List[Dict]]:
    train_n = int(round(len(rows) * train_ratio))
    return rows[:train_n], rows[train_n:]


def _pick_example(rows: List[Dict]) -> Dict:
    return rows[0] if rows else {}


def _project_row(row: Dict) -> Dict:
    prompt = ""
    if isinstance(row, dict):
        prompt = (row.get("input") or {}).get("prompt", "")
    return {
        "task": row.get("task"),
        "prompt": prompt,
        "ground_truth": row.get("ground_truth"),
        "reward_fn": row.get("reward_fn"),
    }


def build_splits(seed: int = 1337) -> None:
    logger = get_logger("split")
    rng = random.Random(seed)

    train_rows: List[Dict] = []
    dev_rows: List[Dict] = []
    stats: Dict[str, Dict[str, int]] = {}
    examples: Dict[str, Dict[str, Dict]] = {}
    per_task: Dict[str, Dict[str, List[Dict]]] = {}

    for task, target_n in TARGET_SAMPLES.items():
        path = Path(FILE_MAP[task])
        if not path.exists():
            logger.warning("Missing file for task %s: %s", task, path)
            continue
        rows = _load_jsonl(path)
        sampled = _sample_rows(rows, target_n, rng)
        rng.shuffle(sampled)
        train_part, dev_part = _split_rows(sampled, 0.8)
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
            "Task %s -> sampled %d (train %d / dev %d)",
            task,
            len(sampled),
            len(train_part),
            len(dev_part),
        )

    TARGET_TRAIN = 32000
    TARGET_DEV = 8000

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

    while len(train_rows) < TARGET_TRAIN and len(dev_rows) > TARGET_DEV:
        if not _move_one("dev", "train"):
            break
    while len(train_rows) > TARGET_TRAIN and len(dev_rows) < TARGET_DEV:
        if not _move_one("train", "dev"):
            break

    logger.info("Adjusted splits to train=%d dev=%d (target train=%d dev=%d)", len(train_rows), len(dev_rows), TARGET_TRAIN, TARGET_DEV)

    # Project rows to the minimal schema
    projected_train = [_project_row(r) for r in train_rows]
    projected_dev = [_project_row(r) for r in dev_rows]

    out_dir = Path("dataset/minerva_split")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.jsonl"
    dev_path = out_dir / "dev.jsonl"
    train_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in projected_train) + "\n", encoding="utf-8")
    dev_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in projected_dev) + "\n", encoding="utf-8")
    logger.info("Wrote train (%d rows) -> %s", len(projected_train), train_path)
    logger.info("Wrote dev (%d rows) -> %s", len(projected_dev), dev_path)

    meta = {
        "seed": seed,
        "summary": {
            "train": len(train_rows),
            "dev": len(dev_rows),
            "total": len(train_rows) + len(dev_rows),
        },
        "per_task": stats,
        "examples": {k: {"train": _project_row(v.get("train")), "dev": _project_row(v.get("dev"))} for k, v in examples.items()},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote split metadata -> %s", out_dir / "metadata.json")


def main() -> None:
    build_splits()


if __name__ == "__main__":
    main()
