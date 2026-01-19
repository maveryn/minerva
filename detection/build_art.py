import argparse
import random
import re
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
    technique_id_name_map,
    write_jsonl,
)


ART_PROMPT = """Given the Atomic Red Team procedure snippet below (steps, commands, and platform), identify the MITRE ATT&CK Enterprise technique ID(s) that best match the behavior.

Requirements:
{REQUIREMENTS}

Procedure snippet:
{SNIPPET}"""


def _technique_from_path(path: Path) -> Optional[str]:
    match = re.search(r"(T\\d{4}(?:\\.\\d{3})?)", path.as_posix(), re.IGNORECASE)
    return match.group(1).upper() if match else None


def _build_snippet(test: Dict) -> str:
    parts: List[str] = []
    name = test.get("name") or ""
    desc = test.get("description") or ""
    platforms = test.get("supported_platforms") or []
    executor = test.get("executor") or {}
    executor_name = executor.get("name") or ""
    command = executor.get("command") or ""
    steps = executor.get("steps") or []
    cleanup = executor.get("cleanup_command") or ""
    input_args = test.get("input_arguments") or {}
    deps = test.get("dependencies") or []

    if name:
        parts.append(f"Test name: {name}")
    if desc:
        parts.append(f"Description: {desc}")
    if platforms:
        parts.append(f"Platforms: {', '.join([str(p) for p in platforms if p])}")
    if executor_name:
        parts.append(f"Executor: {executor_name}")
    if command:
        parts.append("Command:")
        parts.append(str(command))
    if steps:
        parts.append("Steps:")
        parts.extend([str(s) for s in steps if s])
    if cleanup:
        parts.append("Cleanup:")
        parts.append(str(cleanup))
    if input_args:
        parts.append("Input arguments:")
        for key, val in input_args.items():
            if not key:
                continue
            desc_line = ""
            if isinstance(val, dict):
                desc_line = val.get("description") or ""
                default = val.get("default")
                if default is not None and default != "":
                    desc_line = f"{desc_line} (default: {default})" if desc_line else f"default: {default}"
            line = f"- {key}"
            if desc_line:
                line = f"{line}: {desc_line}"
            parts.append(line)
    if deps:
        parts.append("Prerequisites:")
        for dep in deps:
            if not isinstance(dep, dict):
                continue
            dep_desc = dep.get("description") or ""
            prereq = dep.get("prereq_command") or ""
            if dep_desc:
                parts.append(f"- {dep_desc}")
            if prereq:
                parts.append(f"  prereq_command: {prereq}")
    return "\n".join(parts).strip()


def build_art_dataset(
    *,
    output_dir: Path,
    cache_dir: Path,
    mitre_path: Path,
    seed: int = 1337,
    with_distractors: bool = False,
    distractor_count: int = 6,
    include_names: bool = True,
    repo: str = "redcanaryco/atomic-red-team",
    ref: Optional[str] = None,
) -> Path:
    rng = random.Random(seed)
    repo_root = download_github_repo(repo, cache_dir, ref=ref)
    atomics_dir = repo_root / "atomics"
    id_to_name = technique_id_name_map(mitre_path)

    rows: List[Dict] = []
    for path in iter_files(atomics_dir, [".yml", ".yaml"]):
        data = load_yaml(path)
        if not data:
            continue
        technique_id = _technique_from_path(path) or ""
        if not technique_id:
            technique_id = (data.get("attack_technique") or "").upper()
        if not technique_id:
            continue

        tests = data.get("atomic_tests") or []
        for test in tests:
            if not isinstance(test, dict):
                continue
            snippet = _build_snippet(test)
            if not snippet:
                continue
            technique_names = [id_to_name.get(technique_id, "")]
            snippet = clean_snippet(snippet, technique_names)
            if not snippet:
                continue

            requirements = build_requirements(1, allow_subtechnique=True)
            prompt = ART_PROMPT.format(REQUIREMENTS=requirements, SNIPPET=snippet)

            candidates = []
            if with_distractors and id_to_name:
                distractors = select_distractors(
                    id_to_name=id_to_name,
                    gold_ids=[technique_id],
                    k=distractor_count,
                    rng=rng,
                )
                candidates = [technique_id] + distractors
                rng.shuffle(candidates)
                prompt = f"{prompt}\n\n{format_candidate_block(candidates, id_to_name, include_names, 1)}"

            rows.append(
                {
                    "task": "art_to_attack_technique",
                    "input": {
                        "procedure_snippet": snippet,
                        "prompt": prompt,
                    },
                    "ground_truth": {"technique_id": technique_id},
                    "answer": technique_id,
                    "reward_fn": "reward_technique_detection_art",
                    "metadata": {
                        "source_file": str(path),
                        "test_name": test.get("name", ""),
                        "platforms": test.get("supported_platforms", []),
                        "candidate_ids": candidates,
                    },
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "art_to_attack_technique.jsonl"
    write_jsonl(out_path, rows)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Atomic Red Team technique dataset.")
    parser.add_argument("--output-dir", default="dataset/detection")
    parser.add_argument("--cache-dir", default="dataset/detection_cache")
    parser.add_argument("--mitre-path", default="dataset/mitre/enterprise-attack.json")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--with-distractors", action="store_true")
    parser.add_argument("--distractor-count", type=int, default=6)
    parser.add_argument("--include-names", action="store_true")
    parser.add_argument("--repo", default="redcanaryco/atomic-red-team")
    parser.add_argument("--ref", default=None)
    args = parser.parse_args()

    out_path = build_art_dataset(
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
