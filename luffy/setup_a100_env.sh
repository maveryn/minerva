#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LUFFY_ROOT="$REPO_ROOT/luffy"
ENV_DIR="${LUFFY_ENV_DIR:-$REPO_ROOT/.venv-luffy}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
ENV_PY="$ENV_DIR/bin/python"
RUNTIME_VERSIONS_FILE="${RUNTIME_VERSIONS_FILE:-$LUFFY_ROOT/runtime_versions.txt}"
FLASH_ATTN_SPEC="${FLASH_ATTN_SPEC:-$(awk -F'==' '$1=="flash_attn"{print "flash-attn=="$2}' "$RUNTIME_VERSIONS_FILE")}"

resolve_cuda_home() {
  if [ -n "${CUDA_HOME:-}" ]; then
    printf '%s\n' "$CUDA_HOME"
    return 0
  fi

  if [ -x /usr/local/cuda/bin/nvcc ]; then
    printf '%s\n' "/usr/local/cuda"
    return 0
  fi

  local nvcc_bin
  nvcc_bin="$(command -v nvcc || true)"
  if [ -z "$nvcc_bin" ]; then
    return 1
  fi

  if [ -f /opt/conda/targets/x86_64-linux/include/cuda.h ]; then
    local shim_dir="$REPO_ROOT/.cuda-home"
    mkdir -p "$shim_dir/bin"
    ln -sfn "$nvcc_bin" "$shim_dir/bin/nvcc"
    ln -sfn /opt/conda/targets/x86_64-linux/include "$shim_dir/include"
    ln -sfn /opt/conda/lib "$shim_dir/lib64"
    printf '%s\n' "$shim_dir"
    return 0
  fi

  printf '%s\n' "$(cd "$(dirname "$nvcc_bin")/.." && pwd)"
}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys

supported = {(3, 10), (3, 12)}
if sys.version_info[:2] not in supported:
    raise SystemExit(
        "Python 3.10 or 3.12 is required. "
        "Use Python 3.12 for the prebuilt Torch 2.9 FlashAttention wheel."
    )
PY

if [ ! -d "$ENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$ENV_DIR"
fi

"$ENV_PY" -m pip install --upgrade pip setuptools wheel
"$ENV_PY" -m pip install airports-py outlines==0.0.46

BASE_REQS="$(mktemp)"
"$ENV_PY" - "$LUFFY_ROOT/luffy/requirements.v2.txt" "$RUNTIME_VERSIONS_FILE" "$BASE_REQS" <<'PY'
from pathlib import Path
import re
import sys

reqs_path = Path(sys.argv[1])
runtime_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])

def normalize(name: str) -> str:
    return name.strip().replace("_", "-").lower()

skip = set()
for raw in runtime_path.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    name = re.split(r"[<>=!~ ]", line, maxsplit=1)[0]
    skip.add(normalize(name))

filtered = []
for raw in reqs_path.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        filtered.append(raw)
        continue
    name = re.split(r"[<>=!~ ]", line, maxsplit=1)[0]
    if normalize(name) in skip:
        continue
    filtered.append(raw)

out_path.write_text("\n".join(filtered) + "\n")
PY
"$ENV_PY" -m pip install -r "$BASE_REQS"
rm -f "$BASE_REQS"

RUNTIME_REQS="$(mktemp)"
grep -vE '^flash_attn==' "$RUNTIME_VERSIONS_FILE" > "$RUNTIME_REQS"
"$ENV_PY" -m pip install -r "$RUNTIME_REQS"
rm -f "$RUNTIME_REQS"

if [ -n "$FLASH_ATTN_SPEC" ]; then
  export CUDA_HOME="$(resolve_cuda_home)"
  export MAX_JOBS="${MAX_JOBS:-8}"
  if [ -x "$CUDA_HOME/nvvm/bin/cicc" ]; then
    export PATH="$CUDA_HOME/nvvm/bin:$PATH"
  elif [ -x /opt/conda/nvvm/bin/cicc ]; then
    export PATH="/opt/conda/nvvm/bin:$PATH"
  fi

  if ! "$ENV_PY" -m pip install --no-build-isolation --upgrade "$FLASH_ATTN_SPEC"; then
    echo "Warning: failed to install $FLASH_ATTN_SPEC; use MINERVA_ATTN_IMPLEMENTATION=sdpa as a fallback." >&2
  fi
fi

"$ENV_PY" -m pip install -e "$LUFFY_ROOT/luffy" --no-deps
"$ENV_PY" -m pip install -e "$LUFFY_ROOT/luffy/verl" --no-deps
"$ENV_PY" "$LUFFY_ROOT/apply_vllm_core_patch.py"
"$ENV_PY" "$LUFFY_ROOT/apply_vllm_flash_attn_fallback_patch.py"
"$ENV_PY" "$LUFFY_ROOT/apply_deepspeed_cuda_home_patch.py"

mkdir -p "${HOME}/.triton/autotune"

"$ENV_PY" - <<'PY'
import importlib.metadata as md
import torch

pkgs = ["torch", "vllm", "ray", "transformers", "datasets", "tensordict", "flash-attn", "wandb", "omegaconf", "xformers"]
for name in pkgs:
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}=<not installed>")
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
