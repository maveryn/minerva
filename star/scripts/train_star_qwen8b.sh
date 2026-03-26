#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${STAR_CONFIG_PATH:-star/configs/star_cti_qwen8b.yaml}"
FROM_SCRATCH=false
PASS_ARGS=()

while (($#)); do
  case "$1" in
    --from-scratch)
      FROM_SCRATCH=true
      shift
      ;;
    *)
      PASS_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ "$FROM_SCRATCH" = true ]; then
  OUTPUT_ROOT="$(python - <<'PY' "$CONFIG_PATH"
from star.common import load_config
import sys
cfg = load_config(sys.argv[1])
print(cfg["output_root"])
PY
)"
  rm -rf "$OUTPUT_ROOT"
fi

exec python -m star.run_star_cti --config "$CONFIG_PATH" "${PASS_ARGS[@]}"
