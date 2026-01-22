#!/usr/bin/env bash
set -euo pipefail

# Upload classifier outputs to Hugging Face Hub.
# Usage:
#   huggingface-cli login
#   HF_ACCOUNT=xashru bash minerva-judge/classifier/scripts/upload_hf_models.sh
#   HF_ACCOUNT=xashru bash minerva-judge/classifier/scripts/upload_hf_models.sh <model_dir> [more_dirs...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ROOT_DIR="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
  ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"
else
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

DEFAULT_MODELS=(
  "minerva-judge/classifier/outputs/modernbert-2k-lr5e-05"
  "minerva-judge/classifier/outputs/modernbert-2k-lr3e-05"
  "minerva-judge/classifier/outputs/modernbert-2k-lr1e-05"
  "minerva-judge/classifier/outputs/cysecbert-2k-lr5e-05"
  "minerva-judge/classifier/outputs/baselines/textcnn_lr5e-4"
)

MODEL_DIRS=("$@")
if [ "${#MODEL_DIRS[@]}" -eq 0 ]; then
  MODEL_DIRS=("${DEFAULT_MODELS[@]}")
fi

export HF_ACCOUNT="${HF_ACCOUNT:-xashru}"
export HF_ROOT="${ROOT_DIR}"
export HF_MODEL_DIRS
HF_MODEL_DIRS="$(printf '%s\n' "${MODEL_DIRS[@]}")"

python - <<'PY'
import os
import pathlib
from huggingface_hub import HfApi

account = os.environ["HF_ACCOUNT"]
root = pathlib.Path(os.environ["HF_ROOT"]).resolve()
model_dirs = [d for d in os.environ["HF_MODEL_DIRS"].splitlines() if d.strip()]

api = HfApi()
ignore_patterns = [
    "checkpoint-*",
    "**/checkpoint-*",
    "**/*.log",
    "**/logs/*",
    "**/trainer_state.json",
    "**/rng_state.pth",
]

for rel_dir in model_dirs:
    path = (root / rel_dir).resolve()
    if not path.exists():
        raise SystemExit(f"Missing model dir: {path}")
    repo_name = path.name.replace("_", "-")
    repo_id = f"{account}/{repo_name}"
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(path),
        path_in_repo=".",
        ignore_patterns=ignore_patterns,
    )
    print(f"Uploaded {path} -> {repo_id}")
PY
