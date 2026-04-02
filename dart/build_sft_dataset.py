from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from transformers import AutoTokenizer

from dart.common import (
    compose_training_response,
    first_present_field,
    get_system_prompt,
    get_user_prompt,
    iter_jsonl,
    normalize_messages,
    split_reasoning_response,
    stringify_ground_truth,
)


def _dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for row in rows:
        key = (
            row.get("uid"),
            row.get("accepted_rank"),
            row.get("trace_source"),
            row.get("response"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _truncate_response(response: str, tokenizer, response_max_tokens: int | None) -> str:
    if not response_max_tokens or response_max_tokens <= 0:
        return response
    token_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if len(token_ids) <= response_max_tokens:
        return response
    return tokenizer.decode(token_ids[:response_max_tokens], skip_special_tokens=True).strip()


def accepted_rows_to_sft_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    tokenizer=None,
    response_max_tokens: int | None = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        base_messages = normalize_messages(
            first_present_field(row, "base_messages", "prompt_nohint_messages", default=[])
        )
        response_raw = str(row.get("response_text_raw") or row.get("response_text") or "").strip()
        response_analysis = str(row.get("response_analysis") or "").strip()
        response_final = str(row.get("response_final") or "").strip()
        if not response_analysis and not response_final:
            response_analysis, response_final = split_reasoning_response(response_raw)
        response = compose_training_response(
            analysis=response_analysis,
            final=response_final,
            text=response_raw,
        )
        if tokenizer is not None:
            response = _truncate_response(response, tokenizer, response_max_tokens)
        if not base_messages or not response:
            continue
        prompt_meta = row.get("prompt_meta")
        if isinstance(prompt_meta, dict) and not prompt_meta:
            prompt_meta = None
        messages = list(base_messages) + [{"role": "assistant", "content": response}]
        out.append(
            {
                "messages": messages,
                "prompt": get_user_prompt(base_messages),
                "system_prompt": get_system_prompt(base_messages),
                "response": response,
                "response_raw": response_raw,
                "response_analysis": response_analysis,
                "response_final": response_final,
                "answer": stringify_ground_truth(row.get("ground_truth")),
                "uid": row.get("uid"),
                "accepted_rank": row.get("accepted_rank"),
                "trace_source": row.get("trace_source"),
                "final_reward": float(row.get("verifier_score", 0.0)),
                "data_source": row.get("data_source"),
                "task": row.get("task"),
                "reward_fn": row.get("reward_fn"),
                "source_file": row.get("source_file"),
                "source_index": row.get("source_index"),
                "attempt_index": row.get("attempt_index"),
                "prompt_meta": prompt_meta,
            }
        )
    return out


def build_sft_dataset(
    *,
    accepted_jsonl: str | Path,
    output_parquet: str | Path,
    model_path: str | None = None,
    response_max_tokens: int | None = None,
    trust_remote_code: bool = False,
) -> int:
    tokenizer = None
    if response_max_tokens and response_max_tokens > 0:
        if not model_path:
            raise ValueError("model_path is required when response_max_tokens is set")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    rows = accepted_rows_to_sft_rows(
        iter_jsonl(accepted_jsonl),
        tokenizer=tokenizer,
        response_max_tokens=response_max_tokens,
    )
    rows = _dedupe_rows(rows)
    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DART SFT parquet from accepted traces")
    parser.add_argument("--accepted-jsonl", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--response-max-tokens", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    count = build_sft_dataset(
        accepted_jsonl=args.accepted_jsonl,
        output_parquet=args.output_parquet,
        model_path=args.model_path,
        response_max_tokens=args.response_max_tokens,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Wrote {count} rows to {args.output_parquet}")


if __name__ == "__main__":
    main()
