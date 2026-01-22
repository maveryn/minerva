#!/usr/bin/env bash
set -euo pipefail

# TextCNN sweep (response-only input).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLASSIFIER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$CLASSIFIER_DIR/../.." && pwd)"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="$REPO_ROOT/minerva-judge/classifier/outputs/baselines/textcnn_sweep_response_only_${RUN_TAG}"
LOG_DIR="$REPO_ROOT/minerva-judge/classifier/experiments/logs/textcnn_sweep_response_only_${RUN_TAG}"
RESULTS_PATH="$LOG_DIR/results.jsonl"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
touch "$RESULTS_PATH"

LRS=(3e-4 6e-4 1e-3 2e-3 4e-3)
KERNELS=("3,4,5")
FILTERS=(256 384)
DROPOUTS=(0 0.25)
EMBED_DIMS=(200 300)
MAX_TOKENS=(1024)

RUNS=()
for lr in "${LRS[@]}"; do
  for kernel in "${KERNELS[@]}"; do
    for filters in "${FILTERS[@]}"; do
      for dropout in "${DROPOUTS[@]}"; do
        for embed_dim in "${EMBED_DIMS[@]}"; do
          for max_tokens in "${MAX_TOKENS[@]}"; do
            kernel_label="k${kernel//,/}"
            dropout_label="d${dropout/./p}"
            name="textcnn_response_only_lr${lr}_${kernel_label}_f${filters}_e${embed_dim}_t${max_tokens}_${dropout_label}"
            RUNS+=("${lr}|${kernel}|${filters}|${dropout}|${embed_dim}|${max_tokens}|${name}")
          done
        done
      done
    done
  done
done

GPU_LIST="${GPU_LIST:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No GPUs configured in GPU_LIST."
  exit 1
fi

AVAILABLE_GPUS=("${GPUS[@]}")
declare -A PID_GPU=()
PIDS=()

cleanup() {
  echo "Stopping running jobs..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

pop_gpu() {
  local gpu="${AVAILABLE_GPUS[0]}"
  AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]:1}")
  REPLY="$gpu"
}

push_gpu() {
  AVAILABLE_GPUS+=("$1")
}

wait_for_slot() {
  while true; do
    for idx in "${!PIDS[@]}"; do
      local pid="${PIDS[$idx]}"
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" || exit $?
        local gpu="${PID_GPU[$pid]}"
        unset "PID_GPU[$pid]"
        unset "PIDS[$idx]"
        PIDS=("${PIDS[@]}")
        push_gpu "$gpu"
        return 0
      fi
    done
    sleep 1
  done
}

run_one() {
  local lr="$1"
  local kernel="$2"
  local filters="$3"
  local dropout="$4"
  local embed_dim="$5"
  local max_tokens="$6"
  local name="$7"
  local gpu="$8"
  local output_dir="$OUTPUT_ROOT/$name"
  local log_path="$LOG_DIR/$name.log"

  echo "Running: $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python "$REPO_ROOT/minerva-judge/classifier/baselines/train_textcnn.py" \
    --train-file "$REPO_ROOT/minerva-judge/classifier/data/train.jsonl" \
    --eval-file "$REPO_ROOT/minerva-judge/classifier/data/val.jsonl" \
    --output-dir "$output_dir" \
    --text-mode response \
    --max-tokens "$max_tokens" \
    --max-vocab 100000 \
    --embed-dim "$embed_dim" \
    --kernel-sizes "$kernel" \
    --num-filters "$filters" \
    --dropout "$dropout" \
    --batch-size 128 \
    --epochs 5 \
    --lr "$lr" \
    | tee "$log_path"

  {
    flock -x 200
    python - <<PY
import json
from pathlib import Path

metrics_path = Path(r"$output_dir") / "metrics.json"
payload = {
    "run_tag": r"$RUN_TAG",
    "name": r"$name",
    "lr": r"$lr",
    "kernel_sizes": r"$kernel",
    "num_filters": int(r"$filters"),
    "dropout": float(r"$dropout"),
    "embed_dim": int(r"$embed_dim"),
    "max_tokens": int(r"$max_tokens"),
    "text_mode": "response",
    "output_dir": r"$output_dir",
}
if metrics_path.exists():
    payload.update(json.loads(metrics_path.read_text(encoding="utf-8")))
    payload["status"] = "ok"
else:
    payload["status"] = "missing_metrics"
with open(r"$RESULTS_PATH", "a", encoding="utf-8") as f:
    f.write(json.dumps(payload, ensure_ascii=False) + "\\n")
PY
  } 200>"$RESULTS_PATH.lock"
}

for run in "${RUNS[@]}"; do
  while [[ ${#AVAILABLE_GPUS[@]} -eq 0 ]]; do
    wait_for_slot
  done
  pop_gpu
  gpu="$REPLY"
  IFS='|' read -r lr kernel filters dropout embed_dim max_tokens name <<< "$run"
  run_one "$lr" "$kernel" "$filters" "$dropout" "$embed_dim" "$max_tokens" "$name" "$gpu" &
  pid=$!
  PIDS+=("$pid")
  PID_GPU["$pid"]="$gpu"
done

while [[ ${#PIDS[@]} -gt 0 ]]; do
  wait_for_slot
done

echo "All TextCNN response-only runs completed."
