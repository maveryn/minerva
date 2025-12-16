#!/usr/bin/env python3
# passk_patch_all.py
#
# Run passk_patch.py over all "*_predictions_passk.jsonl" files in a folder.
# By default, processes only the top-level of the folder; add --recursive to search subdirs.
# It overwrites each file in place (passk_patch.py already does that) and prints a summary.

import os
import sys
import glob
import argparse
import subprocess
from typing import List

def find_predictions(folder: str, recursive: bool) -> List[str]:
    pattern = "**/*_predictions_passk.jsonl" if recursive else "*_predictions_passk.jsonl"
    return sorted(glob.glob(os.path.join(folder, pattern), recursive=recursive))

def main():
    ap = argparse.ArgumentParser(description="Batch-run passk_patch.py for all predictions_passk files in a folder.")
    ap.add_argument("folder", help="Folder containing *_predictions_passk.jsonl files")
    ap.add_argument("--recursive", action="store_true", help="Search subdirectories recursively")
    ap.add_argument("--patch-script", default=None,
                    help="Path to passk_patch.py (default: resolve next to this script, else 'passk_patch.py' on PATH)")
    ap.add_argument("--no-backup", action="store_true", help="Pass --no-backup to passk_patch.py")
    ap.add_argument("--preserve-order", action="store_true", help="Pass --preserve-order to passk_patch.py")
    args = ap.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Resolve patcher path
    if args.patch_script:
        patcher = os.path.abspath(args.patch_script)
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(here, "passk_patch.py")
        patcher = candidate if os.path.exists(candidate) else "passk_patch.py"

    # Discover files
    files = find_predictions(folder, args.recursive)
    if not files:
        print("No *_predictions_passk.jsonl files found.")
        return

    print(f"Found {len(files)} file(s). Running patcher...\n")

    ok, fail = 0, 0
    for i, fp in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {fp}")
        cmd = [sys.executable, patcher, fp]
        if args.no_backup:
            cmd.append("--no-backup")
        if args.preserve_order:
            cmd.append("--preserve-order")
        try:
            # Stream output live
            subprocess.run(cmd, check=True)
            ok += 1
        except subprocess.CalledProcessError as e:
            print(f"  FAILED with exit code {e.returncode}", file=sys.stderr)
            fail += 1
        except FileNotFoundError:
            print("  ERROR: passk_patch.py not found. Use --patch-script to point to it.", file=sys.stderr)
            fail += 1
            break
        print()  # spacer

    print("=== Batch patch summary ===")
    print(f"Processed : {len(files)}")
    print(f"Succeeded : {ok}")
    print(f"Failed    : {fail}")
    if fail > 0:
        print("Some files failed to patch. Check the logs above for details.")

if __name__ == "__main__":
    main()
