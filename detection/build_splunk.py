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


SPLUNK_PROMPT = """Given the Splunk detection below (query, narrative, and tags), identify the MITRE ATT&CK Enterprise technique ID(s) associated with the detection.

Requirements:
{REQUIREMENTS}

Detection:
{SNIPPET}"""


def _extract_mitre_ids(rule: Dict) -> List[str]:
    tags = rule.get("tags") or {}
    ids = tags.get("mitre_attack_id") or tags.get("mitre_attack_ids") or []
    return normalize_technique_ids(split_list(ids))


def _build_snippet(rule: Dict) -> str:
    parts: List[str] = []
    name = rule.get("name") or ""
    desc = rule.get("description") or rule.get("narrative") or ""
    search = rule.get("search") or ""
    tags = rule.get("tags") or {}
    data_source = tags.get("data_source") or tags.get("data_sources") or []

    if name:
        parts.append(f"Title: {name}")
    if desc:
        parts.append(f"Description: {desc}")
    if data_source:
        if isinstance(data_source, (list, tuple)):
            ds_text = ", ".join([str(x) for x in data_source if x])
        else:
            ds_text = str(data_source)
        if ds_text:
            parts.append(f"Data sources: {ds_text}")
    if search:
        parts.append("SPL Query:")
        parts.append(str(search))
    return "\n".join(parts).strip()


def build_splunk_dataset(
    *,
    output_dir: Path,
    cache_dir: Path,
    mitre_path: Path,
    seed: int = 1337,
    with_distractors: bool = False,
    distractor_count: int = 6,
    include_names: bool = True,
    repo: str = "splunk/security_content",
    ref: Optional[str] = None,
) -> Path:
    rng = random.Random(seed)
    repo_root = download_github_repo(repo, cache_dir, ref=ref)
    id_to_name = technique_id_name_map(mitre_path)

    detections_dir = repo_root / "detections"
    rows: List[Dict] = []
    if detections_dir.exists():
        for path in iter_files(detections_dir, [".yml", ".yaml"]):
            rule = load_yaml(path)
            if not rule:
                continue
            technique_ids = _extract_mitre_ids(rule)
            if len(technique_ids) != 1:
                continue

            snippet = _build_snippet(rule)
            technique_names = [id_to_name.get(tid, "") for tid in technique_ids]
            snippet = clean_snippet(snippet, technique_names)
            if not snippet:
                continue

            requirements = build_requirements(len(technique_ids), allow_subtechnique=True)
            prompt = SPLUNK_PROMPT.format(REQUIREMENTS=requirements, SNIPPET=snippet)

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
                    "task": "splunk_to_attack_technique",
                    "input": {
                        "detection_snippet": snippet,
                        "prompt": prompt,
                    },
                    "ground_truth": {"technique_id": technique_id},
                    "answer": technique_id,
                    "reward_fn": "reward_technique_detection_splunk",
                    "metadata": {
                        "source_file": str(path),
                        "title": rule.get("name", ""),
                        "candidate_ids": candidates,
                    },
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "splunk_to_attack_technique.jsonl"
    write_jsonl(out_path, rows)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Splunk Security Content technique dataset.")
    parser.add_argument("--output-dir", default="dataset/detection")
    parser.add_argument("--cache-dir", default="dataset/detection_cache")
    parser.add_argument("--mitre-path", default="dataset/mitre/enterprise-attack.json")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--with-distractors", action="store_true")
    parser.add_argument("--distractor-count", type=int, default=6)
    parser.add_argument("--include-names", action="store_true")
    parser.add_argument("--repo", default="splunk/security_content")
    parser.add_argument("--ref", default=None)
    args = parser.parse_args()

    out_path = build_splunk_dataset(
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
