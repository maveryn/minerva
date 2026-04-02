from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
RLVR_ROOT = REPO_ROOT / "rlvr"
for path in (REPO_ROOT, RLVR_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from verl.utils.reward_score import reward_minerva as reward_minerva_module

from dart.common import iter_jsonl, write_jsonl


def score_completion(
    *,
    data_source: str,
    response_text: str,
    ground_truth: Any,
    extra_info: Dict[str, Any] | None,
    success_threshold: float = 1.0,
) -> Dict[str, Any]:
    reward_info = reward_minerva_module.reward_minerva(
        data_source,
        response_text,
        ground_truth,
        extra_info=extra_info,
        return_dict=True,
    )
    allow_fallback = not reward_minerva_module._is_training_split(extra_info)
    extracted = reward_minerva_module._extract_predicted(
        data_source,
        response_text,
        allow_fallback=allow_fallback,
    )
    score = float(reward_info.get("score", 0.0))
    return {
        "extracted_answer": extracted,
        "verifier_score": score,
        "verifier_success": score >= success_threshold,
        "reward_info": reward_info,
    }


def score_rows(rows: Iterable[Dict[str, Any]], success_threshold: float = 1.0) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for row in rows:
        result = score_completion(
            data_source=str(row.get("data_source") or ""),
            response_text=str(row.get("response_text") or ""),
            ground_truth=row.get("ground_truth"),
            extra_info=row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {},
            success_threshold=success_threshold,
        )
        new_row = dict(row)
        new_row.update(result)
        scored.append(new_row)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Score DART generations with reward_minerva")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--success-threshold", type=float, default=1.0)
    args = parser.parse_args()

    rows = list(iter_jsonl(args.input))
    scored = score_rows(rows, success_threshold=args.success_threshold)
    write_jsonl(args.output, scored)


if __name__ == "__main__":
    main()

