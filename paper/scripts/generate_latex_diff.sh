#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash paper/scripts/generate_latex_diff.sh <baseline-git-ref> [output-dir]

Example:
  bash paper/scripts/generate_latex_diff.sh tmlr-submitted-baseline

The script compares the current paper/latex tree against paper/latex from the
given git ref, generates a flattened latexdiff source, compiles it, and writes:
  paper/latex_diff/main-diff.tex
  paper/latex_diff/main-diff.pdf
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  usage
  exit $([[ $# -lt 1 ]] && echo 1 || echo 0)
fi

BASE_REF="$1"
ROOT="$(git rev-parse --show-toplevel)"
SRC_DIR="$ROOT/paper/latex"
OUT_DIR="${2:-$ROOT/paper/latex_diff}"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

if ! command -v latexdiff >/dev/null 2>&1; then
  echo "latexdiff is not installed or not on PATH" >&2
  exit 2
fi

if ! command -v latexmk >/dev/null 2>&1; then
  echo "latexmk is not installed or not on PATH" >&2
  exit 2
fi

git -C "$ROOT" rev-parse --verify "$BASE_REF^{commit}" >/dev/null
git -C "$ROOT" archive "$BASE_REF" paper/latex | tar -x -C "$TMP_DIR"

BASE_MAIN="$TMP_DIR/paper/latex/main.tex"
if [[ ! -f "$BASE_MAIN" ]]; then
  echo "Baseline ref '$BASE_REF' does not contain paper/latex/main.tex" >&2
  exit 3
fi

mkdir -p "$OUT_DIR"

cd "$SRC_DIR"
rm -f main-diff.*

latexdiff --flatten "$BASE_MAIN" main.tex > main-diff.tex
latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error main-diff.tex

cp main-diff.tex "$OUT_DIR/main-diff.tex"
cp main-diff.pdf "$OUT_DIR/main-diff.pdf"

latexmk -c main-diff.tex >/dev/null || true
rm -f main-diff.tex main-diff.pdf main-diff.bbl

echo "Diff TeX: $OUT_DIR/main-diff.tex"
echo "Diff PDF: $OUT_DIR/main-diff.pdf"
