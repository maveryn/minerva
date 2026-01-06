import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minerva.utils import load_yaml
from athena_eval.models import load_model

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


MODEL_ALIASES = {
    "qwen": "qwen3-8B",
    "llama": "llama-3-8B",
}


ID_PATTERNS = {
    "ATTACK_TECHNIQUE": re.compile(r"\bT\d{4}\b(?!\.\d{3})", re.IGNORECASE),
    "ATTACK_SUBTECHNIQUE": re.compile(r"\bT\d{4}\.\d{3}\b", re.IGNORECASE),
    "ATTACK_MITIGATION": re.compile(r"\bM\d{4}\b", re.IGNORECASE),
    "CWE": re.compile(r"\bCWE-\d+\b", re.IGNORECASE),
    "CAPEC": re.compile(r"\bCAPEC-\d+\b", re.IGNORECASE),
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def resolve_model_cfg(model_arg: str, cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str]:
    models_cfg = cfg.get("models", {})
    if model_arg in models_cfg:
        model_cfg = models_cfg[model_arg]
        model_name = model_cfg.get("name") or model_cfg.get("model") or model_arg
        return model_arg, model_cfg, model_name

    alias_key = MODEL_ALIASES.get(model_arg.lower())
    if alias_key and alias_key in models_cfg:
        model_cfg = models_cfg[alias_key]
        model_name = model_cfg.get("name") or model_cfg.get("model") or alias_key
        return alias_key, model_cfg, model_name

    # Fallback: treat as a HuggingFace model name
    model_cfg = {"type": "huggingface", "name": model_arg}
    return model_arg, model_cfg, model_arg


def extract_id(response: str, id_type: str) -> str:
    if not response:
        return ""
    pattern = ID_PATTERNS.get(id_type)
    if not pattern:
        return ""
    match = pattern.search(response)
    if not match:
        return ""
    return match.group(0).upper()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM evaluation on the cyber_id_eval dataset."
    )
    parser.add_argument("--model", required=True, help="Model key or alias (qwen/llama).")
    parser.add_argument(
        "--dataset",
        default="minerva/analysis/cyber_id_eval.jsonl",
        help="Path to the cyber_id_eval JSONL dataset.",
    )
    parser.add_argument(
        "--config",
        default="athena_eval/config.yaml",
        help="Model config YAML (used to resolve aliases).",
    )
    parser.add_argument(
        "--out_dir",
        default="minerva/analysis/runs",
        help="Output directory for run artifacts.",
    )
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    cfg = load_yaml(args.config)
    model_key, model_cfg, model_name = resolve_model_cfg(args.model, cfg)
    model = load_model(model_cfg)

    rows = read_jsonl(dataset_path)
    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        raise SystemExit("Dataset is empty.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_key)
    run_dir = Path(args.out_dir) / f"{run_id}_{safe_model_key}"
    run_dir.mkdir(parents=True, exist_ok=True)

    preds_path = run_dir / "predictions.jsonl"
    metrics_path = run_dir / "metrics.json"

    total = 0
    correct = 0
    category_counts: Dict[str, int] = {}
    category_correct: Dict[str, int] = {}

    iterator: Iterable = rows
    if tqdm is not None:
        iterator = tqdm(rows, desc=f"{model_name}")

    with preds_path.open("w", encoding="utf-8") as handle:
        for row in iterator:
            prompt = row.get("prompt", "")
            answer = str(row.get("answer", "")).upper()
            id_type = row.get("id_type", "")
            category = row.get("category", "")

            response = model.generate(prompt, temperature=args.temperature)
            prediction = extract_id(response, id_type)
            is_correct = prediction == answer if prediction else False

            total += 1
            if is_correct:
                correct += 1
            category_counts[category] = category_counts.get(category, 0) + 1
            if is_correct:
                category_correct[category] = category_correct.get(category, 0) + 1

            out_row = {
                "uid": row.get("uid"),
                "id_type": id_type,
                "category": category,
                "prompt": prompt,
                "answer": answer,
                "response": response,
                "prediction": prediction,
                "correct": is_correct,
            }
            handle.write(json.dumps(out_row, ensure_ascii=True) + "\n")

    overall_accuracy = (correct / total) if total else 0.0
    per_category = {}
    for cat, count in category_counts.items():
        hits = category_correct.get(cat, 0)
        per_category[cat] = {
            "accuracy": hits / count if count else 0.0,
            "n": count,
        }

    metrics = {
        "model_key": model_key,
        "model_name": model_name,
        "dataset_path": str(dataset_path),
        "run_id": run_id,
        "timestamp_utc": utc_timestamp(),
        "n_total": total,
        "accuracy_overall": overall_accuracy,
        "accuracy_by_category": per_category,
    }

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Overall accuracy: {overall_accuracy:.4f} (n={total})")
    for cat in sorted(per_category.keys()):
        cat_acc = per_category[cat]["accuracy"]
        cat_n = per_category[cat]["n"]
        print(f"{cat}: {cat_acc:.4f} (n={cat_n})")
    print(f"Saved predictions -> {preds_path}")
    print(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
