import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional

from detection.common import (
    build_requirements,
    clean_snippet,
    download_github_repo,
    format_candidate_block,
    iter_files,
    load_yaml,
    normalize_technique_ids,
    select_distractors,
    split_list,
    technique_id_name_map,
    write_jsonl,
)


SENTINEL_PROMPT = """Given the Microsoft Sentinel analytics rule below (title, description, and KQL query), identify the MITRE ATT&CK Enterprise technique ID(s) associated with the rule.

Requirements:
{REQUIREMENTS}

Rule:
{SNIPPET}"""


def _extract_relevant_techniques(rule: Dict) -> List[str]:
    ids: List[str] = []
    fields = [
        rule.get("relevantTechniques"),
        rule.get("relevant_techniques"),
    ]
    tags = rule.get("tags")
    if isinstance(tags, dict):
        fields.append(tags.get("relevantTechniques"))
        fields.append(tags.get("relevant_techniques"))
    elif isinstance(tags, list):
        for item in tags:
            if isinstance(item, dict):
                fields.append(item.get("relevantTechniques"))
                fields.append(item.get("relevant_techniques"))
            elif isinstance(item, str):
                fields.append(item)

    for val in fields:
        ids.extend(normalize_technique_ids(split_list(val)))
    return ids


def _build_snippet(rule: Dict) -> str:
    parts: List[str] = []
    name = rule.get("name") or ""
    desc = rule.get("description") or ""
    query = rule.get("query") or ""
    tactics = rule.get("tactics") or []

    if name:
        parts.append(f"Title: {name}")
    if desc:
        parts.append(f"Description: {desc}")
    if tactics:
        parts.append(f"Tactics: {', '.join([str(t) for t in tactics if t])}")
    if query:
        parts.append("KQL Query:")
        parts.append(str(query))
    return "\n".join(parts).strip()


def build_sentinel_dataset(
    *,
    output_dir: Path,
    cache_dir: Path,
    mitre_path: Path,
    seed: int = 1337,
    with_distractors: bool = False,
    distractor_count: int = 6,
    include_names: bool = True,
    repo: str = "Azure/Azure-Sentinel",
    ref: Optional[str] = None,
) -> Path:
    rng = random.Random(seed)
    repo_root = download_github_repo(repo, cache_dir, ref=ref)
    id_to_name = technique_id_name_map(mitre_path)

    search_dirs = [repo_root / "Detections", repo_root / "Solutions"]
    rows: List[Dict] = []
    for root in search_dirs:
        if not root.exists():
            continue
        for path in iter_files(root, [".yml", ".yaml"]):
            rule = load_yaml(path)
            if not rule:
                continue
            technique_ids = _extract_relevant_techniques(rule)
            if len(technique_ids) != 1:
                continue

            snippet = _build_snippet(rule)
            technique_names = [id_to_name.get(tid, "") for tid in technique_ids]
            snippet = clean_snippet(snippet, technique_names)
            if not snippet:
                continue

            requirements = build_requirements(len(technique_ids), allow_subtechnique=True)
            prompt = SENTINEL_PROMPT.format(REQUIREMENTS=requirements, SNIPPET=snippet)

            candidates: List[str] = []
            if with_distractors and id_to_name:
                distractors = select_distractors(
                    id_to_name=id_to_name,
                    gold_ids=technique_ids,
                    k=distractor_count,
                    rng=rng,
                )
                candidates = technique_ids + distractors
                rng.shuffle(candidates)
                prompt = f"{prompt}\n\n{format_candidate_block(candidates, id_to_name, include_names, len(technique_ids))}"

            technique_id = technique_ids[0]
            rows.append(
                {
                    "task": "sentinel_to_attack_technique",
                    "input": {
                        "rule_snippet": snippet,
                        "prompt": prompt,
                    },
                    "ground_truth": {"technique_id": technique_id},
                    "answer": technique_id,
                    "reward_fn": "reward_technique_detection_sentinel",
                    "metadata": {
                        "source_file": str(path),
                        "title": rule.get("name", ""),
                        "candidate_ids": candidates,
                    },
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "sentinel_to_attack_technique.jsonl"
    write_jsonl(out_path, rows)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Sentinel analytics rule technique dataset.")
    parser.add_argument("--output-dir", default="dataset/detection")
    parser.add_argument("--cache-dir", default="dataset/detection_cache")
    parser.add_argument("--mitre-path", default="dataset/mitre/enterprise-attack.json")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--with-distractors", action="store_true")
    parser.add_argument("--distractor-count", type=int, default=6)
    parser.add_argument("--include-names", action="store_true")
    parser.add_argument("--repo", default="Azure/Azure-Sentinel")
    parser.add_argument("--ref", default=None)
    args = parser.parse_args()

    out_path = build_sentinel_dataset(
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir),
        mitre_path=Path(args.mitre_path),
        seed=args.seed,
        with_distractors=args.with_distractors,
        distractor_count=args.distractor_count,
        include_names=args.include_names,
        repo=args.repo,
        ref=args.ref,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
