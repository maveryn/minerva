#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LLAMA_TOP_LEFT="${ROOT_DIR}/pairwise_preference_3b5models_including_ties.pdf"
LLAMA_TOP_RIGHT="${ROOT_DIR}/pairwise_preference_5models_including_ties.pdf"
QWEN_BOTTOM_LEFT="${ROOT_DIR}/pairwise_preference_qwen4b5models_including_ties.pdf"
QWEN_BOTTOM_RIGHT="${ROOT_DIR}/pairwise_preference_qwen8b5models_including_ties.pdf"
OUT_FILE="${ROOT_DIR}/pairwise_preference_including_ties_2x2.pdf"
OUT_PNG="${ROOT_DIR}/pairwise_preference_including_ties_2x2.png"

for f in \
  "$LLAMA_TOP_LEFT" \
  "$LLAMA_TOP_RIGHT" \
  "$QWEN_BOTTOM_LEFT" \
  "$QWEN_BOTTOM_RIGHT"
do
  if [[ ! -f "$f" ]]; then
    echo "Missing file: $f" >&2
    exit 1
  fi
done

if ! command -v gs >/dev/null 2>&1; then
  echo "Ghostscript (gs) is required but not found." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

render_png() {
  local input_pdf="$1"
  local output_png="$2"
  gs -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m \
    -dTextAlphaBits=4 -dGraphicsAlphaBits=4 \
    -dFirstPage=1 -dLastPage=1 -r320 \
    -sOutputFile="$output_png" "$input_pdf" >/dev/null
}

render_png "$LLAMA_TOP_LEFT" "$TMP_DIR/00_llama_top_left.png"
render_png "$LLAMA_TOP_RIGHT" "$TMP_DIR/01_llama_top_right.png"
render_png "$QWEN_BOTTOM_LEFT" "$TMP_DIR/02_qwen_bottom_left.png"
render_png "$QWEN_BOTTOM_RIGHT" "$TMP_DIR/03_qwen_bottom_right.png"

python - "$TMP_DIR" "$OUT_FILE" "$OUT_PNG" <<'PY'
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

tmp_dir, out_file, out_png = sys.argv[1], sys.argv[2], sys.argv[3]
paths = [
    os.path.join(tmp_dir, "00_llama_top_left.png"),
    os.path.join(tmp_dir, "01_llama_top_right.png"),
    os.path.join(tmp_dir, "02_qwen_bottom_left.png"),
    os.path.join(tmp_dir, "03_qwen_bottom_right.png"),
]


def trim_bottom_caption(
    im: Image.Image,
    threshold: int = 24,
    bottom_margin: int = 22,
) -> Image.Image:
    gray = np.array(im.convert("L"))
    dark_counts = (gray < 245).sum(axis=1)
    rows = [i for i, c in enumerate(dark_counts) if c > threshold]
    if not rows:
        return im

    keep_until = rows[-1] + bottom_margin
    return im.crop((0, 0, im.width, min(im.height, keep_until)))


fig, axes = plt.subplots(2, 2, figsize=(16, 12.4))
for ax, png in zip(axes.flat, paths):
    im = trim_bottom_caption(Image.open(png).convert("RGB"))
    ax.imshow(im)
    ax.axis("off")
    ax.set_aspect("auto")

fig.subplots_adjust(
    left=0.015,
    right=0.985,
    top=0.985,
    bottom=0.07,
    wspace=0.03,
    hspace=0.14,
)
fig.text(
    0.5,
    0.01,
    "Figure: Pairwise CTI Preference (A vs B) using GPT5.2 as judge",
    ha="center",
    va="bottom",
    fontsize=14,
)
fig.savefig(out_file, format="pdf", dpi=200)
fig.savefig(out_png, format="png", dpi=300)
plt.close(fig)
PY

echo "Created $OUT_FILE"
echo "Created $OUT_PNG"
