#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash paper/scripts/generate_latex_diff.sh <baseline-source-dir-or-git-ref> [output-dir]

Example:
  bash paper/scripts/generate_latex_diff.sh paper/latex_submitted_baseline
  bash paper/scripts/generate_latex_diff.sh tmlr-submitted-baseline

The script compares the current paper/latex tree against either a visible
baseline source directory or paper/latex from the given git ref, generates a
flattened latexdiff source, compiles it, and writes:
  paper/latex_diff/minerva-tmlr-diff.tex
  paper/latex_diff/minerva-tmlr-diff.pdf
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  usage
  exit $([[ $# -lt 1 ]] && echo 1 || echo 0)
fi

BASE_ARG="$1"
ROOT="$(git rev-parse --show-toplevel)"
SRC_DIR="$ROOT/paper/latex"
OUT_DIR="${2:-$ROOT/paper/latex_diff}"
TMP_DIR="$(mktemp -d)"
MOVED_NEW_BBL=""

cleanup() {
  if [[ -n "$MOVED_NEW_BBL" && -f "$MOVED_NEW_BBL" && ! -f "$SRC_DIR/minerva-tmlr.bbl" ]]; then
    mv "$MOVED_NEW_BBL" "$SRC_DIR/minerva-tmlr.bbl"
  fi
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

if [[ "$BASE_ARG" = /* ]]; then
  BASE_PATH="$BASE_ARG"
else
  BASE_PATH="$ROOT/$BASE_ARG"
fi

if [[ -d "$BASE_PATH" ]]; then
  BASE_DIR="$BASE_PATH"
else
  git -C "$ROOT" rev-parse --verify "$BASE_ARG^{commit}" >/dev/null
  git -C "$ROOT" archive "$BASE_ARG" paper/latex | tar -x -C "$TMP_DIR"
  BASE_DIR="$TMP_DIR/paper/latex"
fi

if [[ -f "$BASE_DIR/minerva-tmlr.tex" ]]; then
  BASE_MAIN="$BASE_DIR/minerva-tmlr.tex"
elif [[ -f "$BASE_DIR/main.tex" ]]; then
  BASE_MAIN="$BASE_DIR/main.tex"
else
  echo "Baseline '$BASE_ARG' does not contain minerva-tmlr.tex or main.tex" >&2
  exit 3
fi

NEW_MAIN="$SRC_DIR/minerva-tmlr.tex"
if [[ ! -f "$NEW_MAIN" ]]; then
  echo "Current source '$NEW_MAIN' does not exist" >&2
  exit 4
fi

mkdir -p "$OUT_DIR"

cd "$SRC_DIR"
rm -f minerva-tmlr-diff.*

if [[ -f "$SRC_DIR/minerva-tmlr.bbl" ]]; then
  MOVED_NEW_BBL="$TMP_DIR/minerva-tmlr.bbl"
  mv "$SRC_DIR/minerva-tmlr.bbl" "$MOVED_NEW_BBL"
fi

latexdiff --flatten "$BASE_MAIN" "$NEW_MAIN" > minerva-tmlr-diff.tex

if [[ -n "$MOVED_NEW_BBL" && -f "$MOVED_NEW_BBL" ]]; then
  mv "$MOVED_NEW_BBL" "$SRC_DIR/minerva-tmlr.bbl"
  MOVED_NEW_BBL=""
fi

latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error minerva-tmlr-diff.tex

cp minerva-tmlr-diff.tex "$OUT_DIR/minerva-tmlr-diff.tex"
cp minerva-tmlr-diff.pdf "$OUT_DIR/minerva-tmlr-diff.pdf"

latexmk -c minerva-tmlr-diff.tex >/dev/null || true
rm -f minerva-tmlr-diff.tex minerva-tmlr-diff.pdf minerva-tmlr-diff.bbl

echo "Diff TeX: $OUT_DIR/minerva-tmlr-diff.tex"
echo "Diff PDF: $OUT_DIR/minerva-tmlr-diff.pdf"
