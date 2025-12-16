import argparse
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List

from minerva.data_sources.mitre import download_bundle
from minerva.logger import get_logger
from minerva.utils import load_yaml, write_jsonl


THREAT_PROMPT = """Given the observed adversary procedures below, choose the most likely threat actor.

Observed procedures:
{PROCEDURE_LIST}

Select the correct option (A-E) and return the option letter.

Options:
{OPTIONS_TEXT}"""


def _load_attack_bundle(cfg: Dict[str, Any], logger) -> Dict[str, Any]:
    url = cfg.get("attack_url") or "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    cache_path = Path(cfg.get("cache_path") or "dataset/mitre/enterprise-attack.json")
    path = download_bundle(url, cache_path, logger)
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_intrusion_sets(bundle: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    actors: Dict[str, Dict[str, Any]] = {}
    for obj in bundle.get("objects", []) or []:
        if obj.get("type") != "intrusion-set":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        actors[obj["id"]] = {
            "name": obj.get("name", ""),
            "aliases": obj.get("aliases", []) or [],
            "techniques": set(),
            "procedures": [],  # list of (text, tid)
        }
    return actors


def _sanitize(text: str, names: List[str]) -> str:
    if not text:
        return ""
    text = re.sub(r"\(Citation:[^)]+\)", "", text)
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)
    for n in names:
        if not n:
            continue
        pattern = re.escape(n)
        text = re.sub(rf"\b{pattern}'s\b", "the threat actor's", text, flags=re.IGNORECASE)
        text = re.sub(rf"\b{pattern}\b", "the threat actor", text, flags=re.IGNORECASE)
    text = " ".join(text.split()).strip()
    if text and not text.endswith((".", "!", "?")):
        text += "."
    return text


def _collect_actor_data(bundle: Dict[str, Any], actors: Dict[str, Dict[str, Any]]) -> None:
    ap_map: Dict[str, str] = {}
    for obj in bundle.get("objects", []) or []:
        if obj.get("type") == "attack-pattern" and not obj.get("revoked") and not obj.get("x_mitre_deprecated"):
            for ref in obj.get("external_references", []) or []:
                if ref.get("source_name") in ("mitre-attack", "mitre-mobile-attack", "mitre-ics-attack"):
                    eid = ref.get("external_id") or ""
                    if eid.startswith("T"):
                        ap_map[obj["id"]] = eid
                        break
    for rel in bundle.get("objects", []) or []:
        if rel.get("type") != "relationship":
            continue
        if rel.get("relationship_type") != "uses":
            continue
        src = rel.get("source_ref")
        tgt = rel.get("target_ref")
        if src not in actors:
            continue
        tid = ap_map.get(tgt)
        if not tid:
            continue
        actors[src]["techniques"].add(tid)
        desc = rel.get("description", "") or ""
        names = [actors[src]["name"]] + actors[src].get("aliases", [])
        sanitized = _sanitize(desc, names)
        if sanitized:
            actors[src]["procedures"].append((sanitized, tid))


def _sample(items: List[Any], pct_min: float, pct_max: float, rng: random.Random) -> List[Any]:
    if not items:
        return []
    pct = rng.uniform(pct_min, pct_max)
    k = max(1, math.ceil(len(items) * pct))
    items = items[:]
    rng.shuffle(items)
    return items[:k]


def build_threat_actor_tasks(
    mitre_cfg: Dict[str, Any],
    output_path: str,
    seed: int = 1337,
    logger=None,
) -> List[Dict[str, Any]]:
    if logger is None:
        logger = get_logger("task-threat-actor")
    rng = random.Random(seed)
    bundle = _load_attack_bundle(mitre_cfg, logger)
    actors = _collect_intrusion_sets(bundle)
    _collect_actor_data(bundle, actors)

    actor_list = [a for a in actors.values() if a.get("techniques") and a.get("procedures")]

    rows: List[Dict[str, Any]] = []
    for actor in actor_list:
        techniques = sorted(actor["techniques"])
        procedures = actor["procedures"]
        if not techniques or not procedures:
            continue
        labels = [actor["name"]] + actor.get("aliases", [])
        labels = [l for l in labels if l]
        if not labels:
            continue
        for label in labels:
            for variant_idx in range(5):
                sampled_procs = _sample(procedures, 0.6, 1.0, rng)
                if not sampled_procs:
                    continue
                sampled_texts = [p[0] for p in sampled_procs]
                sampled_tids = {p[1] for p in sampled_procs}

                # Build distractors: actors that do NOT cover all sampled technique IDs
                distractor_options: List[str] = []
                other_actors = [a for a in actor_list if a["name"] != actor["name"]]
                rng.shuffle(other_actors)
                for cand in other_actors:
                    if len(distractor_options) >= 4:
                        break
                    cand_label = cand["name"]
                    if cand_label == label or label in (cand.get("aliases") or []):
                        continue
                    if sampled_tids.issubset(cand["techniques"]):
                        continue
                    distractor_options.append(cand_label)
                if len(distractor_options) < 4:
                    continue

                options = [label] + distractor_options[:4]
                rng.shuffle(options)
                option_lines = [f"{chr(ord('A')+i)}. {opt}" for i, opt in enumerate(options)]
                correct_idx = options.index(label)

                prompt = THREAT_PROMPT.format(
                    PROCEDURE_LIST="\n".join(f"- {t}" for t in sampled_texts),
                    OPTIONS_TEXT="\n".join(option_lines),
                )
                rows.append(
                    {
                        "task": "threat_actor_from_procedures_mcq",
                    "input": {
                        "procedures": sampled_texts,
                        "prompt": prompt,
                    },
                    "ground_truth": {"answer": chr(ord("A") + correct_idx)},
                    "answer": chr(ord("A") + correct_idx),
                    "reward_fn": "binary_id",
                    "metadata": {
                        "correct_actor": actor["name"],
                        "alias_used": label,
                        "aliases": actor.get("aliases", []),
                            "options": options,
                            "sampled_techniques": sorted(sampled_tids),
                            "variant_idx": variant_idx,
                        },
                    }
                )

    write_jsonl(output_path, rows)
    logger.info("Built %d threat-actor MCQ tasks -> %s", len(rows), output_path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build threat actor MCQ tasks from ATT&CK bundle")
    parser.add_argument("--config", default="minerva/config.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out = cfg.get("TASKS", {}).get("THREAT_ACTOR", {}).get("output_path", "dataset/minerva/threat_actor_mcq.jsonl")
    seed = int(cfg.get("TASKS", {}).get("THREAT_ACTOR", {}).get("seed", 1337))
    logger = get_logger("task-threat-actor")
    build_threat_actor_tasks(cfg.get("MITRE_ATTACK", {}), output_path=out, seed=seed, logger=logger)


if __name__ == "__main__":
    main()
