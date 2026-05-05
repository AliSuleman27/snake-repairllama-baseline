#!/usr/bin/env python3
"""
prepare_data.py
---------------
Copy the IR4/OR2 training dataset (parquet + metadata) from a source location
into train/data/. Useful when the parquet files have been removed locally or
when re-syncing from the master dataset directory.

Usage:
    python train/prepare_data.py
    SOURCE_DATASET_DIR="C:/path/to/dataset" python train/prepare_data.py
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

DEFAULT_SOURCE = r"D:\SnakeRepair-LLAMA\dataset"
SRC = Path(os.environ.get("SOURCE_DATASET_DIR", DEFAULT_SOURCE))
DST = Path(__file__).resolve().parent / "data"
FILES = ["train.parquet", "validation.parquet", "metadata.json"]


def main() -> int:
    if not SRC.exists():
        print(f"Source not found: {SRC}", file=sys.stderr)
        print("Set SOURCE_DATASET_DIR env var or edit DEFAULT_SOURCE.", file=sys.stderr)
        return 1

    DST.mkdir(parents=True, exist_ok=True)
    print(f"Source: {SRC}")
    print(f"Dest:   {DST}")
    print()

    for fn in FILES:
        src = SRC / fn
        dst = DST / fn
        if not src.exists():
            print(f"  [skip] missing in source: {src}")
            continue
        shutil.copy(src, dst)
        size_mb = src.stat().st_size / 1e6
        print(f"  [ok]   {fn}  ({size_mb:.1f} MB)")

    print()
    print(f"Done. Data is in: {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
