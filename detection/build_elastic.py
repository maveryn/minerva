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
    load_toml,
    load_yaml,
    normalize_technique_ids,
    select_distractors,
    split_list,
    technique_id_name_map,
    write_jsonl,
)


ELASTIC_PROMPT = """Given the Elastic detection rule below (query and metadata), identify the single MITRE ATT&CK Enterprise technique ID (format: T####) associated with the rule.

Requirements:
{REQUIREMENTS}

Answer format:
- The final line of your response must begin with "Answer:" and contain only the technique ID (e.g., Answer: T1059).

Rule:
{SNIPPET}"""


def _extract_technique_ids(rule: Dict) -> List[str]:
    ids: List[str] = []
    threat = rule.get("threat") or []
    if isinstance(threat, dict):
        threat = [threat]
    for entry in threat:
        if not isinstance(entry, dict):
            continue
        techniques = entry.get("technique") or []
        if isinstance(techniques, dict):
            techniques = [techniques]
        for tech in techniques:
            if not isinstance(tech, dict):
                continue
            ids.extend(split_list(tech.get("id")))
            subtech = tech.get("subtechnique") or []
            if isinstance(subtech, dict):
                subtech = [subtech]
            for sub in subtech:
                if not isinstance(sub, dict):
                    continue
                ids.extend(split_list(sub.get("id")))
    return normalize_technique_ids(ids)


def _collapse_inline(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _build_snippet(rule: Dict, technique_names: List[str]) -> str:
    parts: List[str] = []
    desc = _collapse_inline(rule.get("description") or "")
    query = rule.get("query") or ""
    rule_type = _collapse_inline(rule.get("type") or "")
    language = _collapse_inline(rule.get("language") or "")
    index = rule.get("index") or []

    metadata: List[str] = []
    if desc:
        metadata.append(f"Description: {desc}")
    if rule_type:
        metadata.append(f"Type: {rule_type}")
    if language:
        metadata.append(f"Language: {language}")
    if index:
        if isinstance(index, (list, tuple)):
            idx_text = ", ".join([str(i) for i in index if i])
        else:
            idx_text = str(index)
        idx_text = _collapse_inline(idx_text)
        if idx_text:
            metadata.append(f"Index: {idx_text}")
    if metadata:
        parts.append("Metadata:")
        parts.extend(metadata)
    if query:
        parts.append("Query:")
        parts.append(str(query))
    return "\n".join(parts).strip()


def build_elastic_dataset(
    *,
    output_dir: Path,
    cache_dir: Path,
    mitre_path: Path,
    seed: int = 1337,
    with_distractors: bool = False,
    distractor_count: int = 6,
    include_names: bool = True,
    repo: str = "elastic/detection-rules",
    ref: Optional[str] = None,
) -> Path:
    rng = random.Random(seed)
    repo_root = download_github_repo(repo, cache_dir, ref=ref)
    id_to_name = technique_id_name_map(mitre_path)

    rules_dir = repo_root / "rules"
    rows: List[Dict] = []
    if rules_dir.exists():
        for path in iter_files(rules_dir, [".toml", ".yml", ".yaml"]):
            if path.suffix == ".toml":
                rule = load_toml(path)
            else:
                rule = load_yaml(path)
            if not rule:
                continue
            rule_body = rule.get("rule") if isinstance(rule.get("rule"), dict) else rule
            technique_ids = _extract_technique_ids(rule_body)
            if len(technique_ids) != 1:
                continue

            technique_names = [id_to_name.get(tid, "") for tid in technique_ids]
            snippet = _build_snippet(rule_body, technique_names)
            snippet = clean_snippet(snippet, technique_names)
            if not snippet:
                continue

            requirements = build_requirements(len(technique_ids), allow_subtechnique=False)
            prompt = ELASTIC_PROMPT.format(REQUIREMENTS=requirements, SNIPPET=snippet)

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
                    "task": "elastic_to_attack_technique",
                    "input": {
                        "rule_snippet": snippet,
                        "prompt": prompt,
                    },
                    "ground_truth": {"technique_id": technique_id},
                    "answer": technique_id,
                    "reward_fn": "reward_technique_detection_elastic",
                    "metadata": {
                        "source_file": str(path),
                        "title": rule_body.get("name", ""),
                        "candidate_ids": candidates,
                    },
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "elastic_to_attack_technique.jsonl"
    write_jsonl(out_path, rows)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Elastic detection rule technique dataset.")
    parser.add_argument("--output-dir", default="dataset/detection")
    parser.add_argument("--cache-dir", default="dataset/detection_cache")
    parser.add_argument("--mitre-path", default="dataset/mitre/enterprise-attack.json")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--with-distractors", action="store_true")
    parser.add_argument("--distractor-count", type=int, default=6)
    parser.add_argument("--include-names", action="store_true")
    parser.add_argument("--repo", default="elastic/detection-rules")
    parser.add_argument("--ref", default=None)
    args = parser.parse_args()

    out_path = build_elastic_dataset(
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
