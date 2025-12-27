"""
Create a detailed Athena CTI RMS dataset by appending the full MITRE ATT&CK
Enterprise mitigation ID->name catalog to each prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable


CATALOG_HEADING = "Valid MITRE ATT&CK Enterprise mitigation IDs (ID: name):"


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _build_catalog(mitigation_id_to_name: Dict[str, str]) -> str:
    lines = [CATALOG_HEADING]
    for mid in sorted(mitigation_id_to_name.keys()):
        name = (mitigation_id_to_name.get(mid) or "").strip()
        if name:
            lines.append(f"- {mid}: {name}")
        else:
            lines.append(f"- {mid}")
    return "\n".join(lines)


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def build_detailed_prompts(
    input_path: Path,
    output_path: Path,
    mitigation_id_to_name: Dict[str, str],
) -> int:
    catalog = _build_catalog(mitigation_id_to_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in _iter_jsonl(input_path):
            prompt = record.get("prompt", "")
            if CATALOG_HEADING in prompt:
                detailed_prompt = prompt
            else:
                detailed_prompt = f"{prompt.rstrip()}\n\n{catalog}\n"
            record["prompt"] = detailed_prompt
            record["prompt_hash"] = _hash_prompt(detailed_prompt)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    # Ensure repo root is on sys.path to avoid name clashes with local scripts.
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from minerva.data_sources.mitre import load_bundle as load_mitre_bundle

    parser = argparse.ArgumentParser(description="Create a detailed Athena CTI RMS dataset with mitigation catalogs.")
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parent.parent / "cti-in" / "athena-cti-rms.jsonl"),
        help="Path to the athena-cti-rms JSONL file.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent.parent / "cti-in" / "athena-cti-rms-detailed.jsonl"),
        help="Path to write the detailed RMS JSONL file.",
    )
    parser.add_argument(
        "--attack-url",
        default=None,
        help="Optional MITRE ATT&CK enterprise bundle URL (defaults to MITRE CTI enterprise-attack.json).",
    )
    parser.add_argument(
        "--cache-path",
        default=str(Path("dataset/mitre/enterprise-attack.json")),
        help="Cache path for the MITRE ATT&CK enterprise bundle.",
    )
    args = parser.parse_args()

    cfg = {"cache_path": args.cache_path}
    if args.attack_url:
        cfg["attack_url"] = args.attack_url

    bundle = load_mitre_bundle(cfg)
    mitigation_id_to_name = {
        mid: (obj.get("name", "") if isinstance(obj, dict) else "")
        for mid, obj in (bundle.get("mitigations", {}) or {}).items()
    }
    if not mitigation_id_to_name:
        raise RuntimeError("No mitigation IDs found in MITRE ATT&CK bundle.")

    count = build_detailed_prompts(Path(args.input), Path(args.output), mitigation_id_to_name)
    print(f"Wrote {count} rows -> {Path(args.output)}")


if __name__ == "__main__":
    main()
