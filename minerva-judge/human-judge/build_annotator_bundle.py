#!/usr/bin/env python3
"""Build a zip bundle that annotators can run locally."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DIST_DIR = ROOT / "dist"
BUNDLE_DIR = DIST_DIR / "human-annotator-pointwise-pilot-50-correct-only"
ZIP_PATH = DIST_DIR / "human-annotator-pointwise-pilot-50-correct-only.zip"

FILES_TO_COPY = [
    ("annotator_app.py", "annotator_app.py"),
    ("ANNOTATOR_README.md", "README.md"),
    ("POINTWISE_RUBRIC.md", "POINTWISE_RUBRIC.md"),
    ("start_annotation.sh", "start_annotation.sh"),
    ("start_annotation.bat", "start_annotation.bat"),
    (
        "pointwise_pilot_50_correct_only/blind_responses.csv",
        "pointwise_pilot_50_correct_only/blind_responses.csv",
    ),
    (
        "pointwise_pilot_50_correct_only/blind_responses.jsonl",
        "pointwise_pilot_50_correct_only/blind_responses.jsonl",
    ),
    (
        "pointwise_pilot_50_correct_only/annotation_template.csv",
        "pointwise_pilot_50_correct_only/annotation_template.csv",
    ),
    (
        "pointwise_pilot_50_correct_only/summary.json",
        "pointwise_pilot_50_correct_only/summary.json",
    ),
]


def write_placeholder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def copy_files() -> None:
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    for src_rel, dst_rel in FILES_TO_COPY:
        src = ROOT / src_rel
        dst = BUNDLE_DIR / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    write_placeholder(BUNDLE_DIR / "annotations" / ".gitkeep")


def make_zip() -> None:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(BUNDLE_DIR.rglob("*")):
            if path.is_dir():
                continue
            arcname = Path(BUNDLE_DIR.name) / path.relative_to(BUNDLE_DIR)
            zf.write(path, arcname)


def main() -> None:
    copy_files()
    make_zip()
    print(f"Built annotator bundle directory: {BUNDLE_DIR}")
    print(f"Built annotator bundle zip: {ZIP_PATH}")


if __name__ == "__main__":
    main()
