from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from transformers import AutoTokenizer

from star.common import (
    get_system_prompt,
    get_user_prompt,
    iter_jsonl,
    normalize_messages,
    stringify_ground_truth,
)


def _dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for row in rows:
        key = (
            row.get("uid"),
            row.get("round_id"),
            row.get("selected_from"),
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


def selected_rows_to_sft_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    tokenizer=None,
    response_max_tokens: int | None = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        base_messages = normalize_messages(row.get("base_messages") or row.get("prompt_nohint_messages") or [])
        response = str(row.get("response_text") or "").strip()
        if tokenizer is not None:
            response = _truncate_response(response, tokenizer, response_max_tokens)
        if not base_messages or not response:
            continue
        messages = list(base_messages) + [{"role": "assistant", "content": response}]
        system_prompt = get_system_prompt(base_messages)
        user_prompt = get_user_prompt(base_messages)
        answer = stringify_ground_truth(row.get("ground_truth"))
        out.append(
            {
                "messages": messages,
                "prompt": user_prompt,
                "system_prompt": system_prompt,
                "response": response,
                "answer": answer,
                "source_index": row.get("source_index"),
                "final_source": row.get("selected_from"),
                "final_reward": float(row.get("verifier_score", 0.0)),
                "data_source": row.get("data_source"),
                "task": row.get("task"),
                "reward_fn": row.get("reward_fn"),
                "source_file": row.get("source_file"),
                "round_id": row.get("round_id"),
                "uid": row.get("uid"),
            }
        )
    return out


def build_sft_dataset(
    *,
    selected_jsonl: str | Path,
    output_parquet: str | Path,
    existing_parquet: str | Path | None = None,
    model_path: str | None = None,
    response_max_tokens: int | None = None,
    trust_remote_code: bool = False,
) -> int:
    tokenizer = None
    if response_max_tokens and response_max_tokens > 0:
        if not model_path:
            raise ValueError("model_path is required when response_max_tokens is set")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    new_rows = selected_rows_to_sft_rows(
        iter_jsonl(selected_jsonl),
        tokenizer=tokenizer,
        response_max_tokens=response_max_tokens,
    )
    combined_rows = list(new_rows)
    if existing_parquet and Path(existing_parquet).exists():
        existing_df = pd.read_parquet(existing_parquet)
        combined_rows = existing_df.to_dict(orient="records") + combined_rows
    combined_rows = _dedupe_rows(combined_rows)
    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(combined_rows).to_parquet(output_path, index=False)
    return len(combined_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cumulative STaR SFT parquet")
    parser.add_argument("--selected-jsonl", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--existing-parquet")
    parser.add_argument("--model-path")
    parser.add_argument("--response-max-tokens", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    count = build_sft_dataset(
        selected_jsonl=args.selected_jsonl,
        output_parquet=args.output_parquet,
        existing_parquet=args.existing_parquet,
        model_path=args.model_path,
        response_max_tokens=args.response_max_tokens,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Wrote {count} rows to {args.output_parquet}")


if __name__ == "__main__":
    main()
