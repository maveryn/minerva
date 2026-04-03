#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LUFFY_ROOT="$REPO_ROOT/luffy"
ENV_DIR="${LUFFY_ENV_DIR:-$REPO_ROOT/.venv-luffy}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
ENV_PY="$ENV_DIR/bin/python"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit("Python 3.10 is required for the prebuilt FlashAttention wheel.")
PY

if [ ! -d "$ENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$ENV_DIR"
fi

"$ENV_PY" -m pip install --upgrade pip setuptools wheel
"$ENV_PY" -m pip install airports-py outlines==0.0.46
"$ENV_PY" -m pip install -r "$LUFFY_ROOT/luffy/requirements.v2.txt"
if ! "$ENV_PY" -m pip show flash-attn >/dev/null 2>&1; then
  "$ENV_PY" -m pip install "$FLASH_ATTN_WHEEL_URL"
fi
"$ENV_PY" -m pip install -e "$LUFFY_ROOT/luffy" --no-deps
"$ENV_PY" -m pip install -e "$LUFFY_ROOT/luffy/verl" --no-deps
"$ENV_PY" "$LUFFY_ROOT/apply_vllm_core_patch.py"
"$ENV_PY" "$LUFFY_ROOT/apply_deepspeed_cuda_home_patch.py"

mkdir -p "${HOME}/.triton/autotune"

"$ENV_PY" - <<'PY'
import importlib.metadata as md
import torch

pkgs = ["torch", "vllm", "ray", "transformers", "datasets", "tensordict", "flash-attn", "wandb", "omegaconf", "xformers"]
for name in pkgs:
    print(f"{name}=={md.version(name)}")
print("cuda_available=", torch.cuda.is_available(), sep="")
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f"device_capability=sm_{major}{minor}")
    print("compiled_arches=", ",".join(torch.cuda.get_arch_list()), sep="")
PY

echo
echo "LUFFY environment is ready at: $ENV_DIR"
echo "Activate with: source $ENV_DIR/bin/activate"
echo "Train with: bash $LUFFY_ROOT/exp_scripts/train_minerva_luffy_llama3b.sh"
