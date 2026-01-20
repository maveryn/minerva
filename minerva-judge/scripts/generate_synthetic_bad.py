#!/usr/bin/env python3
"""Generate synthetic BAD responses from hinted prompts."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from common import (
    ModelConfig,
    build_chat_messages,
    load_cti_system_prompt,
    load_model,
    render_messages_for_model,
    resolve_backend,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACR_MARKER = "You are generating a reasoning trace for training."


def format_answer(answer: Any) -> str:
    if isinstance(answer, (list, tuple, set)):
        return "\n".join(str(item) for item in answer)
    if isinstance(answer, dict):
        return json.dumps(answer, ensure_ascii=False)
    return str(answer)


def iter_jsonl_with_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_num, json.loads(line)


def split_acr_prompt(prompt: str) -> tuple[str, str] | None:
    if ACR_MARKER not in prompt:
        return None
    pre, post = prompt.split(ACR_MARKER, 1)
    question = pre.rstrip()
    reference = extract_reference(post)
    return question, reference


def extract_reference(block: str) -> str:
    lines = block.splitlines()
    ref_lines: list[str] = []
    in_ref = False
    for line in lines:
        if line.strip() == "LABEL_REFERENCE:":
            in_ref = True
            continue
        if in_ref:
            if line.strip() == "Instructions:":
                break
            ref_lines.append(line)
    return "\n".join(ref_lines).strip()


def load_criteria(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    criteria: list[dict[str, Any]] = []
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        crit = {
            "key": key,
            "category_id": int(value.get("category_id", 0)),
            "category_title": str(value.get("category_title", "")).strip(),
            "instruction": str(value.get("instruction", "")).strip(),
        }
        if crit["category_id"] and crit["category_title"] and crit["instruction"]:
            criteria.append(crit)
    if not criteria:
        raise ValueError(f"no criteria found in {path}")
    return criteria


def build_prompt(
    template: str,
    *,
    question: str,
    reference: str,
    correct_answer: str,
    instruction: str,
) -> str:
    return (
        template.replace("{QUESTION}", question)
        .replace("{REFERENCE}", reference)
        .replace("{CORRECT_ANSWER}", correct_answer)
        .replace("{CRITERION_INSTRUCTION}", instruction)
    )


def split_response(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    raw = text.strip()
    analysis = ""
    final = raw

    marker_re = re.compile(
        r"(assistantfinal|<\\|channel\\|>final<\\|message\\|>|<\\|assistant\\|>final)",
        flags=re.IGNORECASE,
    )
    match = marker_re.search(raw)
    if match:
        analysis = raw[: match.start()].strip()
        final = raw[match.end() :].strip()
    else:
        box_matches = list(re.finditer(r"\\boxed\\{[^}]*\\}", raw))
        if box_matches:
            last = box_matches[-1]
            analysis = (raw[: last.start()] + raw[last.end() :]).strip()
            final = last.group(0).strip()

    if analysis:
        analysis = re.sub(
            r"^(?:<\\|channel\\|>analysis<\\|message\\|>)\\s*",
            "",
            analysis,
            flags=re.IGNORECASE,
        )
        analysis = re.sub(r"^analysis\\s*[:\\-]*\\s*", "", analysis, flags=re.IGNORECASE).strip()
    if final:
        final = re.sub(
            r"^(?:<\\|channel\\|>final<\\|message\\|>)\\s*",
            "",
            final,
            flags=re.IGNORECASE,
        )
        final = re.sub(r"^final\\s*[:\\-]*\\s*", "", final, flags=re.IGNORECASE).strip()
    return analysis, final


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic BAD responses using gpt-oss-120b.")
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="Hinted responses JSONL (repeatable). Defaults to minerva-judge/data/responses_*.jsonl",
    )
    parser.add_argument(
        "--output",
        default="minerva-judge/data/synthetic_bad.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of synthetic responses to generate")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--criteria",
        default="minerva-judge/prompts/synthetic_bad_criteria.json",
        help="Criteria JSON path",
    )
    parser.add_argument(
        "--template",
        default="minerva-judge/prompts/synthetic_bad_prompt.txt",
        help="Prompt template path",
    )
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model to generate synthetic responses")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "hf", "vllm", "openai", "gemini"],
        help="Model backend",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Generation batch size")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument(
        "--vllm-args",
        type=str,
        default=None,
        help="Optional JSON dict of vLLM engine args",
    )
    parser.add_argument(
        "--include-system",
        action="store_true",
        help="Prepend the CTI system prompt",
    )
    args = parser.parse_args()

    input_paths: list[Path] = []
    if args.input:
        input_paths = [Path(p) for p in args.input]
    else:
        input_paths = sorted((PROJECT_ROOT / "minerva-judge" / "data").glob("responses_*.jsonl"))
    if not input_paths:
        raise FileNotFoundError("no input files found")

    criteria = load_criteria(PROJECT_ROOT / args.criteria)
    template = (PROJECT_ROOT / args.template).read_text(encoding="utf-8")
    system_prompt = load_cti_system_prompt().strip() if args.include_system else ""

    candidates: list[dict[str, Any]] = []
    seen_uid: set[str] = set()
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"input not found: {path}")
        for line_num, row in iter_jsonl_with_lines(path):
            if not (row.get("hinted") or row.get("prompt_variant") == "hinted"):
                continue
            uid = str(row.get("uid") or "")
            if not uid or uid in seen_uid:
                continue
            prompt = str(row.get("prompt") or row.get("prompt_used") or "")
            if not prompt:
                continue
            split = split_acr_prompt(prompt)
            if split is None:
                continue
            question, reference = split
            if not reference:
                continue
            answer = row.get("answer")
            if answer is None:
                continue
            candidates.append(
                {
                    "uid": uid,
                    "prompt": prompt,
                    "question": question,
                    "reference": reference,
                    "correct_answer": format_answer(answer),
                    "task": row.get("task"),
                    "reward_fn": row.get("reward_fn"),
                    "source_file": str(path.resolve().relative_to(PROJECT_ROOT)),
                    "source_line": line_num,
                }
            )
            seen_uid.add(uid)
    if not candidates:
        raise RuntimeError("no hinted candidates found")

    rng = random.Random(args.seed)
    if args.num_samples <= len(candidates):
        sampled = rng.sample(candidates, args.num_samples)
    else:
        sampled = [rng.choice(candidates) for _ in range(args.num_samples)]

    jobs: list[dict[str, Any]] = []
    for idx, candidate in enumerate(sampled, start=1):
        criterion = rng.choice(criteria)
        synthetic_prompt = build_prompt(
            template,
            question=candidate["question"],
            reference=candidate["reference"],
            correct_answer=candidate["correct_answer"],
            instruction=criterion["instruction"],
        )
        jobs.append(
            {
                "index": idx,
                "candidate": candidate,
                "criterion": criterion,
                "prompt": synthetic_prompt,
                "synthetic_prompt": synthetic_prompt,
            }
        )

    vllm_kwargs = {}
    if args.vllm_args:
        vllm_kwargs = json.loads(args.vllm_args)
        if not isinstance(vllm_kwargs, dict):
            raise ValueError("--vllm-args must be a JSON object")
    backend = resolve_backend(args.model, args.backend)
    model = load_model(
        ModelConfig(
            args.model,
            backend,
            args.max_new_tokens,
            batch_size=args.batch_size or 1,
            vllm_kwargs=vllm_kwargs,
        )
    )
    batch_size = args.batch_size or getattr(model, "batch_size", 1) or 1

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def flush(batch: list[dict[str, Any]], apply_chat_template: bool, out_file) -> None:
        prompts = [item["full_prompt"] for item in batch]
        responses = model.generate_batch(
            prompts,
            temperature=args.temperature,
            apply_chat_template=apply_chat_template,
        )
        ts = int(time.time())
        for item, response in zip(batch, responses, strict=False):
            candidate = item["candidate"]
            criterion = item["criterion"]
            analysis, final = split_response(response)
            record = {
                "uid": f"synthetic-{item['index']}",
                "source_uid": candidate["uid"],
                "source_file": candidate["source_file"],
                "source_line": candidate["source_line"],
                "task": candidate["task"],
                "reward_fn": candidate["reward_fn"],
                "prompt_variant": "hinted",
                "hinted": True,
                "prompt": candidate["prompt"],
                "source_prompt": candidate["prompt"],
                "synthetic_prompt": item.get("synthetic_prompt"),
                "model_prompt": item.get("model_prompt"),
                "correct_answer": candidate["correct_answer"],
                "criterion_key": criterion["key"],
                "judge_label": "BAD",
                "judge_category_id": criterion["category_id"],
                "judge_category_title": criterion["category_title"],
                "model": args.model,
                "system_prompt_included": bool(system_prompt),
                "response": response,
                "response_analysis": analysis,
                "response_final": final,
                "timestamp": ts,
                "synthetic": True,
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    with out_path.open("w", encoding="utf-8") as f:
        batch: list[dict[str, Any]] = []
        batch_apply_chat_template = True
        for job in tqdm(jobs, desc="synthetic", unit="row", dynamic_ncols=True):
            messages = build_chat_messages(system_prompt, job["prompt"])
            full_prompt, used_template = render_messages_for_model(model, messages)
            apply_chat_template = not used_template
            if batch and apply_chat_template != batch_apply_chat_template:
                flush(batch, batch_apply_chat_template, f)
                batch = []
            job["full_prompt"] = full_prompt
            job["model_prompt"] = full_prompt
            batch_apply_chat_template = apply_chat_template
            batch.append(job)
            if len(batch) >= batch_size:
                flush(batch, batch_apply_chat_template, f)
                batch = []
        if batch:
            flush(batch, batch_apply_chat_template, f)

    print(f"Wrote {len(jobs)} synthetic responses to {out_path}")


if __name__ == "__main__":
    main()
