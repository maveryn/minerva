#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${1:-${HF_REPO_ID:-xashru/minerva_dart_llama8b_500_best_470}}"
LOCAL_DIR="${2:-${HF_LOCAL_DIR:-/home/jovyan/work/minerva/rlvr/checkpoints/dart/dart_sft_llama_3_1_8b_instruct/best/global_step_470/huggingface}}"
PRIVATE_FLAG="${HF_PRIVATE:-true}"

if [ ! -d "$LOCAL_DIR" ]; then
  echo "Local model folder not found: $LOCAL_DIR" >&2
  exit 1
fi

python - "$REPO_ID" "$LOCAL_DIR" "$PRIVATE_FLAG" <<'PY'
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
