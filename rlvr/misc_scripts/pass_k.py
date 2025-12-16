#!/usr/bin/env python3

import os, json, re, logging, argparse, io, contextlib
from typing import Optional, List, Dict, Any, Tuple
from collections import Counter, defaultdict
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# -------- quiet logs ----------
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"      # vLLM respects this
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # quiet HF tokenizer
os.environ["VLLM_NO_PROGRESS_BAR"] = "1"
os.environ["VLLM_USE_RICH"] = "0"
for name in ("vllm", "vllm.engine", "vllm.worker", "vllm.core", "vllm.tracing"):
    logging.getLogger(name).setLevel(logging.ERROR)

def silent_generate(llm, prompts, sampling_params):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return llm.generate(prompts, sampling_params)

# -------- defaults (overridable by CLI) ----------
DEFAULT_PARQUET_DIR = "mydata/activity"
DEFAULT_MODEL_NAME  = None          # pass via --model
DEFAULT_BATCH_SIZE  = 512
DEFAULT_TEMP        = 0.6
DEFAULT_TOP_P       = 0.95
DEFAULT_MAX_TOKENS  = 2048
DEFAULT_K_SAMPLES   = 512
DEFAULT_SEED_BASE   = 12345
DEFAULT_OUT_DIR     = "passk_out"

# -------- parsing/extraction helpers ----------
_CLIP_CHARS = 1200
ANS_BRACED_RE = re.compile(r"\\answer\{\s*([+-]?\d+)\s*\}", re.IGNORECASE)
INT_RE        = re.compile(r"[+-]?\d+")
IDS_BRACED_DIGITS_RE = re.compile(r"(?mi)^\s*\\ids\{\s*(\d+(?:\s*,\s*\d+)*)\s*\}\s*$")
ID_INT_RE     = re.compile(r"\d+")

def _clip_tail(s: str, k: int = _CLIP_CHARS) -> str:
    return s[-k:] if isinstance(s, str) and len(s) > k else (s or "")

def extract_ids(text: str) -> Optional[List[int]]:
    if not text:
        return None
    matches = IDS_BRACED_DIGITS_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1]  # prefer the last ids{...} line
    return [int(x) for x in re.findall(r"\d+", raw)]

def extract_answer_int(text: str) -> Optional[int]:
    if not text:
        return None
    m = ANS_BRACED_RE.search(text)
    if m:
        return int(m.group(1))
    nums = INT_RE.findall(_clip_tail(text))
    return int(nums[-1]) if nums else None

def extract_gt_ids(gt: str) -> Optional[List[int]]:
    m = IDS_BRACED_DIGITS_RE.search(gt or "")
    if not m:
        return None
    return [int(x) for x in ID_INT_RE.findall(m.group(1))]

def extract_gt_answer(gt: str) -> Optional[int]:
    m = ANS_BRACED_RE.search(gt or "")
    return int(m.group(1)) if m else None

def compute_answer_match(solution_str: str, ground_truth: str) -> bool:
    m = ANS_BRACED_RE.search(ground_truth or "")
    if not m:
        raise ValueError("Ground truth missing \\answer{...}.")
    gt = int(m.group(1))
    pred = extract_answer_int(solution_str)
    return (pred is not None) and (pred == gt)

def compute_ids_match(solution_str: str, ground_truth: str) -> Optional[bool]:
    gt_ids = extract_gt_ids(ground_truth)
    if gt_ids is None:
        return None
    pred_ids = extract_ids(solution_str)
    if pred_ids is None:
        return False
    return pred_ids == gt_ids

def iter_rows(parquet_path: str):
    table = pq.read_table(parquet_path)
    yield from table.to_pylist()

def get_messages_from_row(row: Dict[str, Any]) -> List[Dict[str, str]]:
    msgs = row.get("prompt", None)
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
        return msgs
    return [{"role": "user", "content": json.dumps(msgs)}]

def get_ground_truth(row: Dict[str, Any]) -> str:
    rm = row.get("reward_model", {}) or {}
    return rm.get("ground_truth", "") or ""

def to_text_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = []
        for m in messages:
            role = m.get("role","user")
            content = (m.get("content") or "").strip()
            parts.append(f"{role.upper()}: {content}")
        parts.append("ASSISTANT:")
        return "\n\n".join(parts)

# -------- resume helpers ----------
def safe_json_lines(path: str):
    """Yield parsed json objects from a jsonl file, skipping malformed lines."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # skip partial/corrupt line (e.g., crashed write)
                continue

def load_existing_metrics(metrics_path: str) -> Dict[int, Dict[str, Any]]:
    """Load metrics per k from metrics jsonl. Returns dict k -> metrics_dict."""
    data = {}
    if os.path.exists(metrics_path):
        for rec in safe_json_lines(metrics_path):
            k = int(rec.get("k", 0))
            if k > 0:
                data[k] = rec
    return data

def reconstruct_state_from_preds(pred_path: str, max_k_done: int, N: int, ids_mask: List[bool]):
    """Rebuild ans_seen / ids_seen and self-consistency counters from existing predictions up to max_k_done."""
    ans_seen = [False] * N
    ids_seen = [False] * N
    ans_counters: List[Counter] = [Counter() for _ in range(N)]
    ids_counters: List[Counter] = [Counter() for _ in range(N)]

    if max_k_done <= 0 or not os.path.exists(pred_path):
        return ans_seen, ids_seen, ans_counters, ids_counters

    for rec in safe_json_lines(pred_path):
        k = int(rec.get("k", 0))
        if k <= 0 or k > max_k_done:
            continue
        i = int(rec.get("sample_idx"))
        if not (0 <= i < N):
            continue

        # counters
        pred_ans = rec.get("pred_answer", None)
        pred_ids = rec.get("pred_ids", None)
        if pred_ans is not None:
            try:
                ans_counters[i][int(pred_ans)] += 1
            except Exception:
                pass
        if pred_ids is not None:
            try:
                ids_counters[i][tuple(pred_ids)] += 1
            except Exception:
                pass

        # seen flags
        if bool(rec.get("answer_match", False)):
            ans_seen[i] = True
        ids_match = rec.get("ids_match", None)
        if ids_mask[i] and (ids_match is True):
            ids_seen[i] = True

    return ans_seen, ids_seen, ans_counters, ids_counters

def fill_curves_from_metrics(existing_metrics: Dict[int, Dict[str, Any]], upto_k: int):
    """Produce passk/sc curves lists (1..upto_k) from existing metrics."""
    passk_ans_curve, passk_ids_curve = [], []
    sc_ans_curve, sc_ids_curve = [], []
    for k in range(1, upto_k + 1):
        rec = existing_metrics.get(k)
        if not rec:
            # pad if any gaps (shouldn't happen if metrics were written per k)
            passk_ans_curve.append(None)
            passk_ids_curve.append(None)
            sc_ans_curve.append(None)
            sc_ids_curve.append(None)
            continue
        passk_ans_curve.append(float(rec.get("passk_answer", 0.0)))
        passk_ids_curve.append(rec.get("passk_ids", None))
        sc_ans_curve.append(float(rec.get("sc_answer_acc", 0.0)))
        sc_ids_curve.append(rec.get("sc_ids_acc", None))
    return passk_ans_curve, passk_ids_curve, sc_ans_curve, sc_ids_curve

# -------- main ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["lis", "activity"], required=True,
                    help="Task name; used for default parquet path and output filenames.")
    ap.add_argument("--parquet", dest="parquet_path", default=None,
                    help="Path to test parquet. Default: mydata/{task}_test.parquet")
    ap.add_argument("--model", dest="model_name", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--k", dest="k_samples", type=int, default=DEFAULT_K_SAMPLES)
    ap.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--tp-size", type=int, default=1,
                help="Tensor-parallel GPU count (single-node).")
    return ap.parse_args()

def _mode(counter: Counter, tie_numeric_smallest: bool = True):
    """Return the mode; ties broken deterministically."""
    if not counter:
        return None
    maxc = max(counter.values())
    cands = [v for v, c in counter.items() if c == maxc]
    if tie_numeric_smallest:
        try:
            return sorted(cands)[0]
        except Exception:
            pass
    # generic deterministic fallback
    return sorted(cands, key=lambda x: str(x))[0]

def main():
    args = parse_args()

    task = args.task
    parquet_path = args.parquet_path or os.path.join(DEFAULT_PARQUET_DIR, f"{task}_test.parquet")
    model_name = args.model_name
    batch_size = args.batch_size
    temp       = args.temp
    top_p      = args.top_p
    max_tokens = args.max_tokens
    k_samples  = args.k_samples
    seed_base  = args.seed_base
    out_dir    = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Task-tagged output filenames
    _m = (model_name or "MODEL").split('/')[-1] if model_name else "MODEL"
    out_pred_jsonl    = os.path.join(out_dir, f"{task}_{_m}_predictions_passk.jsonl")
    out_metrics_jsonl = os.path.join(out_dir, f"{task}_{_m}_metrics_per_k.jsonl")
    out_summary_json  = os.path.join(out_dir, f"{task}_{_m}_metrics_summary.json")
    out_plot_png      = os.path.join(out_dir, f"{task}_{_m}_passk_curve.png")
    out_plot_pdf      = os.path.join(out_dir, f"{task}_{_m}_passk_curve.pdf")

    rows = list(iter_rows(parquet_path))
    if not rows:
        print(f"[{task}] No rows found in {parquet_path}.")
        return

    print(f"[{task}] Loaded {len(rows)} samples from {parquet_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    prompts_text, gts, metas, messages_all, data_sources = [], [], [], [], []
    for i, r in enumerate(rows):
        msgs = get_messages_from_row(r)
        gt = get_ground_truth(r)
        prompts_text.append(to_text_prompt(tokenizer, msgs))
        gts.append(gt)
        ei = r.get("extra_info", {}) or {}
        metas.append({"i": i, "split": ei.get("split","?"), "index": ei.get("index", i)})
        messages_all.append(msgs)
        data_sources.append(r.get("data_source"))

    N = len(prompts_text)
    ids_gt_list: List[Optional[List[int]]] = [extract_gt_ids(gt) for gt in gts]
    ids_mask = [gt_ids is not None for gt_ids in ids_gt_list]
    N_ids = sum(ids_mask)
    ans_gt_list: List[Optional[int]] = [extract_gt_answer(gt) for gt in gts]

    # -------- resume: inspect existing files --------
    existing_metrics = load_existing_metrics(out_metrics_jsonl)
    max_k_done = max(existing_metrics.keys()) if existing_metrics else 0

    # Pre-fill curves up to max_k_done
    passk_ans_curve, passk_ids_curve, sc_ans_curve, sc_ids_curve = [], [], [], []
    if max_k_done > 0:
        print(f"[resume] Found existing metrics up to k={max_k_done} in {out_metrics_jsonl}")
        a, b, c, d = fill_curves_from_metrics(existing_metrics, max_k_done)
        passk_ans_curve.extend(a)
        passk_ids_curve.extend(b)
        sc_ans_curve.extend(c)
        sc_ids_curve.extend(d)

    # Reconstruct per-sample seen/counters up to max_k_done from predictions
    ans_seen, ids_seen, ans_counters, ids_counters = reconstruct_state_from_preds(
        out_pred_jsonl, max_k_done, N, ids_mask
    )

    # Output files open mode (append to continue)
    pred_f    = open(out_pred_jsonl, "a", encoding="utf-8")
    metrics_f = open(out_metrics_jsonl, "a", encoding="utf-8")

    # If nothing left to do, still rebuild summary & plots and exit
    k_start = max_k_done + 1
    if k_start > k_samples:
        print(f"[resume] All k=1..{k_samples} already completed. Rebuilding summary/plots...")
    else:
        print(f"[run] Starting generation at k={k_start} (target k={k_samples})")

    # Only instantiate the model if we actually need to generate further
    llm = None
    if k_start <= k_samples:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=args.tp_size,   # <— use multiple GPUs here
            trust_remote_code=True,              # keep if your model needs it
            # optional knobs:
            gpu_memory_utilization=0.95,       # pack KV cache a bit tighter
            # dtype="bfloat16",                  # or "auto"/"float16"
        )

    # Progress bar over remaining ks
    total_remaining_k = max(0, k_samples - (k_start - 1))
    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} k [{elapsed}<{remaining}] {rate_fmt} | {postfix}"
    if total_remaining_k > 0:
        pbar = tqdm(total=total_remaining_k, desc=f"pass@k ({task})", bar_format=bar_format, dynamic_ncols=True)
    else:
        pbar = None

    # ---- generation loop for remaining ks ----
    for k in range(k_start, k_samples + 1):
        sp = SamplingParams(
            temperature=temp,
            top_p=top_p,
            max_tokens=max_tokens,
            n=1,
            seed=seed_base + (k - 1),
        )

        inst_ans_hits = 0
        inst_ids_hits = 0

        for start in range(0, N, batch_size):
            batch_prompts = prompts_text[start:start + batch_size]
            outputs = silent_generate(llm, batch_prompts, sp)

            for j, out in enumerate(outputs):
                i = start + j
                text = (out.outputs[0].text if out.outputs else "").strip()

                ans  = extract_answer_int(text)
                ids  = extract_ids(text)
                try:
                    am = compute_answer_match(text, gts[i])
                except Exception:
                    am = False
                im = compute_ids_match(text, gts[i])

                # instantaneous + pass@k update
                if am:
                    inst_ans_hits += 1
                    ans_seen[i] = True
                if ids_mask[i] and (im is True):
                    inst_ids_hits += 1
                    ids_seen[i] = True

                # self-consistency counters (ignore None)
                if ans is not None:
                    ans_counters[i][ans] += 1
                if ids is not None:
                    ids_counters[i][tuple(ids)] += 1

                rec = {
                    "task": task,
                    "k": k,
                    "sample_idx": i,
                    "split": metas[i]["split"],
                    "index": metas[i]["index"],
                    "data_source": data_sources[i],
                    "messages": messages_all[i],
                    "prompt_text": prompts_text[i],
                    "ground_truth": gts[i],
                    "model_reply": text,
                    "pred_ids": ids,
                    "pred_answer": ans,
                    "answer_match": bool(am),
                    "ids_match": (None if im is None else bool(im)),
                }
                pred_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # metrics for this k
        inst_ans_acc = inst_ans_hits / N if N else 0.0
        inst_ids_acc = (inst_ids_hits / N_ids) if N_ids else 0.0

        cum_ans_acc = sum(1 for s in ans_seen if s) / N if N else 0.0
        cum_ids_acc = (sum(1 for idx_, s in enumerate(ids_seen) if ids_mask[idx_] and s) / N_ids) if N_ids else 0.0

        # self-consistency (mode) accuracy @ k
        sc_ans_hits = 0
        for i in range(N):
            mode_ans = _mode(ans_counters[i], tie_numeric_smallest=True)
            if mode_ans is not None and ans_gt_list[i] is not None and mode_ans == ans_gt_list[i]:
                sc_ans_hits += 1
        sc_ans_acc = sc_ans_hits / N if N else 0.0

        if N_ids:
            sc_ids_hits = 0
            for i in range(N):
                if not ids_mask[i]:
                    continue
                mode_ids = _mode(ids_counters[i], tie_numeric_smallest=False)
                gt_ids = ids_gt_list[i]
                if mode_ids is not None and gt_ids is not None and list(mode_ids) == gt_ids:
                    sc_ids_hits += 1
            sc_ids_acc = sc_ids_hits / N_ids
        else:
            sc_ids_acc = None

        passk_ans_curve.append(cum_ans_acc)
        passk_ids_curve.append(cum_ids_acc if N_ids else None)
        sc_ans_curve.append(sc_ans_acc)
        sc_ids_curve.append(sc_ids_acc if N_ids else None)

        metrics_f.write(json.dumps({
            "task": task,
            "k": k,
            "inst_answer_acc": inst_ans_acc,
            "inst_ids_acc": (None if not N_ids else inst_ids_acc),
            "passk_answer": cum_ans_acc,
            "passk_ids": (None if not N_ids else cum_ids_acc),
            "sc_answer_acc": sc_ans_acc,
            "sc_ids_acc": (None if not N_ids else sc_ids_acc),
        }) + "\n")

        postfix = (
            f"k={k} | inst ans {inst_ans_acc:.2%}"
            + (f" inst ids {inst_ids_acc:.2%}" if N_ids else " inst ids N/A")
            + f" | pass@k ans {cum_ans_acc:.2%}"
            + (f" pass@k ids {cum_ids_acc:.2%}" if N_ids else " pass@k ids N/A")
            + f" | SC ans {sc_ans_acc:.2%}"
            + (f" SC ids {sc_ids_acc:.2%}" if N_ids else " SC ids N/A")
        )
        if pbar is not None:
            pbar.set_postfix_str(postfix)
            pbar.update(1)

    # close files
    pred_f.close()
    metrics_f.close()
    if pbar is not None:
        pbar.close()

    # ----- write summary (full 1..k_samples curves) -----
    # If we resumed but didn't run anything (already finished previously), rebuild from existing metrics map:
    if k_start > k_samples and existing_metrics:
        # rebuild arrays to exact length k_samples
        passk_ans_curve, passk_ids_curve, sc_ans_curve, sc_ids_curve = fill_curves_from_metrics(existing_metrics, k_samples)

    summary = {
        "task": task,
        "num_samples": N,
        "num_ids_countable": N_ids,
        "passk_answer_curve": passk_ans_curve,
        "passk_ids_curve": passk_ids_curve if N_ids else [],
        "sc_answer_curve": sc_ans_curve,
        "sc_ids_curve": sc_ids_curve if N_ids else [],
        "config": {
            "parquet_path": parquet_path,
            "model_name": model_name,
            "batch_size": batch_size,
            "temperature": temp,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "k": k_samples,
            "seed_base": seed_base,
        }
    }
    with open(out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ----- nicer plot with seaborn (falls back to mpl if unavailable) -----
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="talk")
        palette = sns.color_palette()
    except Exception:
        palette = None

    xs = list(range(1, k_samples + 1))
    plt.figure(figsize=(9.5, 5.5))
    if palette:
        me = max(1, k_samples // 12)
        plt.plot(xs, passk_ans_curve, label="Pass@k (answer)", linewidth=2.2, marker="o", markevery=me, color=palette[0])
        if N_ids:
            plt.plot(xs, passk_ids_curve, label="Pass@k (ids)", linewidth=2.2, marker="o", markevery=me, color=palette[1])
        plt.plot(xs, sc_ans_curve, label="Self-consistency (answer)", linewidth=2.2, linestyle="--", marker="s", markevery=me, color=palette[2])
        if N_ids:
            plt.plot(xs, sc_ids_curve, label="Self-consistency (ids)", linewidth=2.2, linestyle="--", marker="s", markevery=me, color=palette[3])
    else:
        plt.plot(xs, passk_ans_curve, label="Pass@k (answer)", linewidth=2.2)
        if N_ids:
            plt.plot(xs, passk_ids_curve, label="Pass@k (ids)", linewidth=2.2)
        plt.plot(xs, sc_ans_curve, label="Self-consistency (answer)", linewidth=2.2, linestyle="--")
        if N_ids:
            plt.plot(xs, sc_ids_curve, label="Self-consistency (ids)", linewidth=2.2, linestyle="--")

    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title(f"Pass@k & Self-Consistency — {task}")
    plt.ylim(0, 1.0)
    plt.legend(frameon=True, ncol=2)
    plt.tight_layout()
    plt.savefig(out_plot_png, dpi=160)
    plt.savefig(out_plot_pdf)

if __name__ == "__main__":
    main()