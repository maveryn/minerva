import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List

from minerva.data_sources.mitre import download_bundle
from minerva.logger import get_logger
from minerva.utils import load_yaml, write_jsonl


LOOKUP_FILENAME = "threat_actor_lookup.json"
DEFAULT_BINNING = {
    "min_techniques": 3,
    "bin_count": 6,
    "target_questions_per_bin": 100,
    "questions_start": 3,
    "questions_step": 1,
}

THREAT_PROMPT = """Given the observed adversary procedures below, identify the most likely threat actor.

Observed procedures:
{PROCEDURE_LIST}

Return only the threat actor name."""


def _load_attack_bundle(cfg: Dict[str, Any], logger) -> Dict[str, Any]:
    url = cfg.get("attack_url") or "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    cache_path = Path(cfg.get("cache_path") or "dataset/mitre/enterprise-attack.json")
    path = download_bundle(url, cache_path, logger)
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
            "procedures_by_tid": {},
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


def _normalize_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()
    return " ".join(cleaned.split())


def _coerce_int(value: Any, default: int, *, min_value: int = 1) -> int:
    try:
        num = int(value)
    except (TypeError, ValueError):
        num = default
    if num < min_value:
        return min_value
    return num


def _load_alias_csv(path: Path, actor_index: Dict[str, str]) -> Dict[str, List[str]]:
    alias_map: Dict[str, set[str]] = {}
    if not path.exists():
        return {}
    import csv

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            left = row.get("ThreatActor", "").strip()
            right = row.get("Alias", "").strip()
            if not left or not right:
                continue
            canonical_left = actor_index.get(_normalize_name(left))
            canonical_right = actor_index.get(_normalize_name(right))
            if canonical_left and canonical_right and canonical_left != canonical_right:
                continue
            if canonical_left:
                alias_map.setdefault(canonical_left, set()).add(right)
            if canonical_right:
                alias_map.setdefault(canonical_right, set()).add(left)
    return {k: sorted(v) for k, v in alias_map.items()}


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
        desc = rel.get("description", "") or ""
        names = [actors[src]["name"]] + actors[src].get("aliases", [])
        sanitized = _sanitize(desc, names)
        if sanitized:
            actors[src]["techniques"].add(tid)
            actors[src]["procedures"].append((sanitized, tid))
            actors[src]["procedures_by_tid"].setdefault(tid, []).append(sanitized)


def _build_balanced_bins(
    eligible_actors: List[Dict[str, Any]],
    *,
    bin_cfg: Dict[str, Any],
    min_required: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, tuple[int, Dict[str, Any]]]]:
    bin_count = _coerce_int(bin_cfg.get("bin_count", DEFAULT_BINNING["bin_count"]), DEFAULT_BINNING["bin_count"])
    target_questions = _coerce_int(
        bin_cfg.get("target_questions_per_bin", DEFAULT_BINNING["target_questions_per_bin"]),
        DEFAULT_BINNING["target_questions_per_bin"],
    )
    questions_start = _coerce_int(
        bin_cfg.get("questions_start", DEFAULT_BINNING["questions_start"]),
        DEFAULT_BINNING["questions_start"],
    )
    questions_step = _coerce_int(
        bin_cfg.get("questions_step", DEFAULT_BINNING["questions_step"]),
        DEFAULT_BINNING["questions_step"],
        min_value=0,
    )
    if bin_count <= 0:
        bin_count = 1

    questions_per_actor = [questions_start + i * questions_step for i in range(bin_count)]
    actors_sorted = sorted(
        eligible_actors,
        key=lambda a: (len(a.get("techniques", [])), a.get("name", "")),
    )
    total = len(actors_sorted)
    bins: List[Dict[str, Any]] = []
    actor_bin_map: Dict[str, tuple[int, Dict[str, Any]]] = {}
    start_idx = 0

    for idx, q in enumerate(questions_per_actor):
        if start_idx >= total:
            bin_def = {
                "min": None,
                "max": None,
                "questions": int(q),
                "target_questions": int(target_questions),
                "open_ended": bool(idx == bin_count - 1),
                "actor_count": 0,
                "question_total": 0,
            }
            bins.append(bin_def)
            continue

        if idx == bin_count - 1:
            end_idx = total
        else:
            remaining_bins = bin_count - idx - 1
            max_end = total - remaining_bins
            if max_end < start_idx:
                max_end = start_idx
            target_actors = max(1, math.ceil(target_questions / max(1, q)))
            end_idx = min(start_idx + target_actors, max_end)
            if end_idx > start_idx and end_idx < max_end:
                last_count = len(actors_sorted[end_idx - 1].get("techniques", []))
                while end_idx < max_end and len(actors_sorted[end_idx].get("techniques", [])) == last_count:
                    end_idx += 1

        if end_idx == start_idx:
            min_count = None
            max_count = None
        else:
            min_count = len(actors_sorted[start_idx].get("techniques", []))
            max_count = len(actors_sorted[end_idx - 1].get("techniques", []))

        open_ended = idx == bin_count - 1
        actor_count = end_idx - start_idx
        bin_def = {
            "min": int(min_count) if min_count is not None else None,
            "max": None if open_ended else (int(max_count) if max_count is not None else None),
            "questions": int(q),
            "target_questions": int(target_questions),
            "open_ended": bool(open_ended),
            "actor_count": int(actor_count),
            "question_total": int(actor_count * int(q)),
        }
        bins.append(bin_def)

        for j in range(start_idx, end_idx):
            name = actors_sorted[j].get("name", "")
            if name:
                actor_bin_map[name] = (idx, bin_def)

        start_idx = end_idx

    meta = {
        "mode": "balanced",
        "min_techniques": int(min_required),
        "bin_count": int(bin_count),
        "target_questions_per_bin": int(target_questions),
        "questions_start": int(questions_start),
        "questions_step": int(questions_step),
        "questions_per_actor": [int(q) for q in questions_per_actor],
    }
    return bins, meta, actor_bin_map


def build_threat_actor_tasks(
    mitre_cfg: Dict[str, Any],
    output_path: str,
    seed: int = 1337,
    task_cfg: Dict[str, Any] | None = None,
    logger=None,
) -> List[Dict[str, Any]]:
    if logger is None:
        logger = get_logger("task-threat-actor")
    rng = random.Random(seed)
    task_cfg = task_cfg or {}
    bundle = _load_attack_bundle(mitre_cfg, logger)
    actors = _collect_intrusion_sets(bundle)
    _collect_actor_data(bundle, actors)

    actor_list = [a for a in actors.values() if a.get("techniques") and a.get("procedures_by_tid")]
    technique_counts = [len(a["techniques"]) for a in actor_list if a.get("techniques")]
    if not technique_counts:
        logger.warning("No threat actors with techniques found in bundle.")
        write_jsonl(output_path, [])
        return []
    bin_cfg = task_cfg.get("binning") if isinstance(task_cfg.get("binning"), dict) else {}
    if not isinstance(bin_cfg, dict):
        bin_cfg = {}
    min_required = _coerce_int(
        bin_cfg.get("min_techniques", DEFAULT_BINNING["min_techniques"]),
        DEFAULT_BINNING["min_techniques"],
    )

    actor_index: Dict[str, str] = {}
    for actor in actor_list:
        canonical = actor.get("name", "")
        for name in [canonical] + actor.get("aliases", []):
            key = _normalize_name(name)
            if key:
                actor_index[key] = canonical

    alias_csv_path = task_cfg.get("alias_csv_path", "")
    alias_csv_path = Path(alias_csv_path) if alias_csv_path else None
    if alias_csv_path is None:
        alias_csv_path = Path(__file__).resolve().parents[2] / "rlvr" / "verl" / "utils" / "reward_score" / "aliases.csv"
    supplemental_aliases = _load_alias_csv(alias_csv_path, actor_index)

    lookup_rows: List[Dict[str, Any]] = []
    for actor in actor_list:
        name = actor.get("name", "")
        if not name:
            continue
        aliases = [a for a in actor.get("aliases", []) if a]
        if supplemental_aliases:
            aliases.extend(supplemental_aliases.get(name, []))
        lookup_rows.append(
            {
                "name": name,
                "aliases": sorted(set(aliases)),
                "technique_count": int(len(actor.get("techniques", []))),
            }
        )

    eligible_actors = [
        actor
        for actor in actor_list
        if len(actor.get("techniques", [])) >= min_required and len(actor.get("procedures_by_tid", {})) >= min_required
    ]
    if not eligible_actors:
        logger.warning("No threat actors with >=%d techniques found in bundle.", min_required)
        write_jsonl(output_path, [])
        return []
    eligible_counts = [len(actor.get("techniques", [])) for actor in eligible_actors]
    min_tech = min(eligible_counts)
    max_tech = max(eligible_counts)
    question_bins, bin_meta, actor_bin_map = _build_balanced_bins(
        eligible_actors,
        bin_cfg=bin_cfg,
        min_required=min_required,
    )

    lookup_path = Path(output_path).with_name(LOOKUP_FILENAME)
    lookup_payload = {
        "version": 3,
        "min_techniques": int(min_tech),
        "max_techniques": int(max_tech),
        "bin_count": int(len(question_bins)),
        "binning": bin_meta,
        "question_bins": question_bins,
        "actors": lookup_rows,
    }
    lookup_path.write_text(json.dumps(lookup_payload, ensure_ascii=True, indent=2), encoding="utf-8")
    logger.info("Wrote threat actor lookup -> %s", lookup_path)

    rows: List[Dict[str, Any]] = []
    for actor in eligible_actors:
        techniques = sorted(actor["techniques"])
        procedures_by_tid = actor.get("procedures_by_tid", {})
        if not techniques or not procedures_by_tid:
            continue
        if len(techniques) < min_required:
            continue
        bin_info = actor_bin_map.get(actor.get("name", ""))
        if not bin_info:
            continue
        bin_idx, bin_def = bin_info
        num_questions = int(bin_def.get("questions", 1))
        technique_ids = sorted(procedures_by_tid.keys())
        if len(technique_ids) < min_required:
            continue
        for question_idx in range(num_questions):
            pct = rng.uniform(0.6, 1.0)
            k = max(3, math.ceil(len(technique_ids) * pct))
            k = min(k, len(technique_ids))
            sampled_tids = rng.sample(technique_ids, k)
            rng.shuffle(sampled_tids)
            sampled_texts = []
            for tid in sampled_tids:
                candidates = procedures_by_tid.get(tid, [])
                if not candidates:
                    continue
                sampled_texts.append(rng.choice(candidates))
            if len(sampled_texts) < 3:
                continue

            prompt = THREAT_PROMPT.format(
                PROCEDURE_LIST="\n".join(f"- {t}" for t in sampled_texts),
            )
            rows.append(
                {
                    "task": "threat_actor",
                    "input": {
                        "procedures": sampled_texts,
                        "prompt": prompt,
                    },
                    "ground_truth": {"threat_actor": actor["name"]},
                    "answer": actor["name"],
                    "reward_fn": "reward_threat_actor_name",
                    "metadata": {
                        "correct_actor": actor["name"],
                        "aliases": sorted(set((actor.get("aliases") or []) + supplemental_aliases.get(actor["name"], []))),
                        "sampled_techniques": sorted(sampled_tids),
                        "technique_count": len(techniques),
                        "bin_index": int(bin_idx),
                        "bin_range": [bin_def.get("min"), bin_def.get("max")],
                        "question_index": question_idx,
                    },
                }
            )

    write_jsonl(output_path, rows)
    logger.info(
        "Built %d threat-actor tasks (min_tech=%d max_tech=%d bin_count=%d) -> %s",
        len(rows),
        min_tech,
        max_tech,
        len(question_bins),
        output_path,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build threat actor tasks from ATT&CK bundle")
    parser.add_argument("--config", default="minerva/config.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out = cfg.get("TASKS", {}).get("THREAT_ACTOR", {}).get("output_path", "dataset/minerva/threat_actor.jsonl")
    seed = int(cfg.get("TASKS", {}).get("THREAT_ACTOR", {}).get("seed", 1337))
    logger = get_logger("task-threat-actor")
    build_threat_actor_tasks(
        cfg.get("MITRE_ATTACK", {}),
        output_path=out,
        seed=seed,
        task_cfg=cfg.get("TASKS", {}).get("THREAT_ACTOR", {}),
        logger=logger,
    )


if __name__ == "__main__":
    main()
