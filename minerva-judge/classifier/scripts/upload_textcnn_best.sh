#!/usr/bin/env bash
set -euo pipefail

# Upload a TextCNN model directory to Hugging Face Hub.
#
# Usage:
#   huggingface-cli login
#   MODEL_DIR=/path/to/textcnn_run \
#   HF_REPO=xashru/textcnn-lr6e-4-k345-f384-e200-t3072-d0p25 \
#   bash minerva-judge/classifier/scripts/upload_textcnn_best.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ROOT_DIR="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
  ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"
else
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

MODEL_DIR="${MODEL_DIR:-}"
HF_REPO="${HF_REPO:-xashru/textcnn-lr6e-4-k345-f384-e200-t3072-d0p25}"

if [[ -z "$MODEL_DIR" ]]; then
  echo "MODEL_DIR is required."
  exit 1
fi

MODEL_DIR="$(cd "$ROOT_DIR" && realpath "$MODEL_DIR")"
if [[ ! -f "$MODEL_DIR/model.pt" ]]; then
  echo "Missing model.pt in $MODEL_DIR"
  exit 1
fi

export HF_REPO
export MODEL_DIR

python - <<'PY'
import os
from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO"]
model_dir = os.environ["MODEL_DIR"]

api = HfApi()
ignore_patterns = [
    "checkpoint-*",
    "**/checkpoint-*",
    "**/*.log",
    "**/logs/*",
    "**/trainer_state.json",
    "**/rng_state.pth",
]

api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=model_dir,
    path_in_repo=".",
    ignore_patterns=ignore_patterns,
)
print(f"Uploaded {model_dir} -> {repo_id}")
PY
