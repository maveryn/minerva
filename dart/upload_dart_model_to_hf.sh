#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODEL_KEY="${1:-}"
REPO_ID_OVERRIDE="${2:-}"

HF_ACCOUNT="${HF_ACCOUNT:-xashru}"
HF_PRIVATE="${HF_PRIVATE:-true}"

if [ -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HUGGINGFACE_TOKEN:-}}}" ]; then
  echo "Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN / HUGGINGFACE_TOKEN) before uploading." >&2
  exit 1
fi

if [ -z "$MODEL_KEY" ]; then
  cat >&2 <<'EOF'
Usage:
  HF_TOKEN=... bash dart/upload_dart_model_to_hf.sh <model-key> [repo-id]

Model keys:
  llama3b
  llama8b
  qwen4b
  qwen8b
  all
EOF
  exit 1
fi

case "$MODEL_KEY" in
  llama3b)
    LOCAL_DIR="$REPO_ROOT/rlvr/checkpoints/dart/dart_sft_llama_3_2_3b_instruct/best/global_step_430/huggingface"
    DEFAULT_REPO_ID="$HF_ACCOUNT/minerva_dart_llama3b"
    ;;
  llama8b)
    LOCAL_DIR="$REPO_ROOT/rlvr/checkpoints/dart/dart_sft_llama_3_1_8b_instruct/best/global_step_470/huggingface"
    DEFAULT_REPO_ID="$HF_ACCOUNT/minerva_dart_llama8b"
    ;;
  qwen4b)
    LOCAL_DIR="$REPO_ROOT/rlvr/checkpoints/dart/dart_sft_qwen3_4b_base/best/global_step_460/huggingface"
    DEFAULT_REPO_ID="$HF_ACCOUNT/minerva_dart_qwen4b"
    ;;
  qwen8b)
    LOCAL_DIR="$REPO_ROOT/rlvr/checkpoints/dart/dart_sft_qwen3_8b_base/best/global_step_490/huggingface"
    DEFAULT_REPO_ID="$HF_ACCOUNT/minerva_dart_qwen8b"
    ;;
  all)
    for key in llama3b llama8b qwen4b qwen8b; do
      bash "$0" "$key"
    done
    exit 0
    ;;
  *)
    echo "Unknown model key: $MODEL_KEY" >&2
    exit 1
    ;;
esac

REPO_ID="${REPO_ID_OVERRIDE:-$DEFAULT_REPO_ID}"

if [ ! -d "$LOCAL_DIR" ]; then
  echo "Local model folder not found: $LOCAL_DIR" >&2
  exit 1
fi

python - "$REPO_ID" "$LOCAL_DIR" "$HF_PRIVATE" <<'PY'
import sys
from huggingface_hub import HfApi

repo_id = sys.argv[1]
local_dir = sys.argv[2]
private_flag = sys.argv[3].strip().lower() in {"1", "true", "yes", "y"}

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=private_flag, exist_ok=True)
api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")
print(f"Uploaded {local_dir} -> {repo_id}")
PY
