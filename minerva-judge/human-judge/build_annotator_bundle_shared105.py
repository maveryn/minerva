#!/usr/bin/env python3
"""Build a standalone annotator bundle for the shared 105-prompt seven-task subset."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DIST_DIR = ROOT / "dist"
SUBSET_NAME = "shared_prompt_105_seven_tasks_sec_reasoning_noctua"
BUNDLE_DIR = DIST_DIR / "human-annotator-shared-prompt-105-seven-tasks"
ZIP_PATH = DIST_DIR / "human-annotator-shared-prompt-105-seven-tasks.zip"

FILES_TO_COPY = [
    ("annotator_app.py", "annotator_app.py"),
    ("POINTWISE_RUBRIC.md", "POINTWISE_RUBRIC.md"),
    ("start_annotation.sh", "start_annotation.sh"),
    ("start_annotation.bat", "start_annotation.bat"),
    (f"{SUBSET_NAME}/blind_responses.jsonl", f"{SUBSET_NAME}/blind_responses.jsonl"),
    (f"{SUBSET_NAME}/annotation_template.csv", f"{SUBSET_NAME}/annotation_template.csv"),
    (f"{SUBSET_NAME}/summary.json", f"{SUBSET_NAME}/summary.json"),
]

README_TEXT = f"""# Human Judge

This bundle is for human annotation only. It does not include hidden judge
labels or model identities.

## Requirements

- Python 3.10 or newer

## Run

On macOS or Linux:

```bash
python3 annotator_app.py
```

Or:

```bash
./start_annotation.sh
```

On Windows:

```bat
python annotator_app.py
```

Or double-click `start_annotation.bat`.

## Use

1. Open `http://127.0.0.1:8787/` in a browser.
2. Choose `{SUBSET_NAME}`.
3. Enter your annotator ID.
4. For each item:
   - read the prompt
   - read the blind model response
   - score all 3 rubric criteria from `1` to `4`
   - click `Save Scores And Next`

The app saves progress automatically. To resume, run the app again and use the
same annotator ID and dataset.

## Return File

When finished, send back the saved annotation file:

```text
annotations/<subset>/<annotator>.jsonl
```
"""


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


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

    write_text(BUNDLE_DIR / "README.md", README_TEXT)
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
    subset_summary = json.loads((ROOT / SUBSET_NAME / "summary.json").read_text(encoding="utf-8"))
    if subset_summary.get("selected_prompt_count") != 105:
        raise ValueError(f"Unexpected prompt count for {SUBSET_NAME}: {subset_summary.get('selected_prompt_count')}")
    copy_files()
    make_zip()
    print(f"Built annotator bundle directory: {BUNDLE_DIR}")
    print(f"Built annotator bundle zip: {ZIP_PATH}")


if __name__ == "__main__":
    main()
