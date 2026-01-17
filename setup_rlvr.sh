#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m pip install -r "${ROOT_DIR}/rlvr/requirements.txt"
python -m pip install --no-build-isolation flash-attn

export NLTK_DATA="${NLTK_DATA:-/home/jovyan/nltk_data}"
python - <<'PY'
import os

import nltk

download_dir = os.environ.get("NLTK_DATA") or "/home/jovyan/nltk_data"
os.makedirs(download_dir, exist_ok=True)
nltk.download("punkt_tab", download_dir=download_dir)
PY
