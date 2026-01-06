import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


CWE_RE = re.compile(r"\bCWE-\d+\b", re.IGNORECASE)


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_cwes(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        ids: List[str] = []
        for item in value:
            ids.extend(extract_cwes(item))
        return ids
    if isinstance(value, dict):
        for key in ("cwe_ids", "cwe_id", "answer"):
            if key in value:
                return extract_cwes(value[key])
        if len(value) == 1:
            return extract_cwes(next(iter(value.values())))
        return []
    text = str(value)
    return [match.upper() for match in CWE_RE.findall(text)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze CWE answer frequencies in athenabench athena-cti-rcm.jsonl."
    )
    parser.add_argument(
        "--dataset",
        default="athenabench/athena-cti-rcm.jsonl",
        help="Path to the athena-cti-rcm JSONL file.",
    )
    parser.add_argument(
        "--out_dir",
        default="minerva/analysis/results",
        help="Directory to write frequency outputs.",
    )
    parser.add_argument(
        "--out_name",
        default="athenabench_rcm_cwe_frequencies.csv",
        help="Output CSV filename.",
    )
    parser.add_argument(
        "--json_out",
        default="athenabench_rcm_cwe_frequencies.json",
        help="Output JSON filename.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    json_path = out_dir / args.json_out

    counts: Dict[str, int] = {}
    total = 0
    for row in load_jsonl(dataset_path):
        total += 1
        answers = extract_cwes(row.get("answer"))
        if not answers:
            continue
        for cwe_id in answers:
            counts[cwe_id] = counts.get(cwe_id, 0) + 1

    rows = [
        {
            "cwe_id": cwe_id,
            "count": count,
            "percentage": (count / total * 100.0) if total else 0.0,
        }
        for cwe_id, count in counts.items()
    ]
    rows.sort(key=lambda r: (-r["count"], r["cwe_id"]))

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cwe_id", "count", "percentage"])
        writer.writeheader()
        for row in rows:
            row["percentage"] = f"{row['percentage']:.4f}"
            writer.writerow(row)

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"Total questions: {total}")
    print(f"Unique CWEs: {len(counts)}")
    print(f"Saved -> {out_path}")
    print(f"Saved -> {json_path}")


if __name__ == "__main__":
    main()
