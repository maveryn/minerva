#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility to convert XLSX files under a directory into Parquet format."""

import argparse
from pathlib import Path

import pandas as pd


def convert_dir(data_dir: Path) -> None:
    """Convert every `.xlsx` file in ``data_dir`` to a Parquet file.

    The resulting Parquet files are written next to the original XLSX files
    with the same base name.
    """
    for xlsx_path in data_dir.glob("*.xlsx"):
        df = pd.read_excel(xlsx_path)
        parquet_path = xlsx_path.with_suffix(".parquet")
        df.to_parquet(parquet_path, index=False)
        print(f"Saved {parquet_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert XLSX files in a directory to Parquet for VERL.")
    parser.add_argument("data_dir", type=Path, help="Path to directory containing XLSX files")
    args = parser.parse_args()
    convert_dir(args.data_dir)


if __name__ == "__main__":
    main()
