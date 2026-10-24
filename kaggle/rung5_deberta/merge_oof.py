"""Merge per-fold OOF files into one prediction parquet, validating full cv coverage.

  uv run python kaggle/rung5_deberta/merge_oof.py --in data/kaggle_out/full --tag deberta_base_t1024

Reads oof_fold*.parquet from --in, checks every cv id appears exactly once (against the split
manifest rebuilt from data/raw/train.csv), asserts no text columns, writes data/predictions/<tag>.parquet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from llm_preference_audit.data import load_train
from llm_preference_audit.deberta_inputs import merge_oof
from llm_preference_audit.split import SHA_PATH, build_manifest, manifest_sha256

RAW = Path("data/raw/train.csv")
PRED_DIR = Path("data/predictions")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--raw", type=Path, default=RAW)
    a = ap.parse_args(argv)
    files = sorted(a.src.glob("oof_fold*.parquet"))
    if not files:
        raise SystemExit(f"no oof_fold*.parquet in {a.src}")
    manifest = build_manifest(load_train(a.raw))
    if manifest_sha256(manifest) != SHA_PATH.read_text().split()[0]:
        raise SystemExit("rebuilt split manifest does not match splits/split_manifest.sha256")
    cv_ids = manifest.loc[manifest["split"] == "cv", "id"]
    merged = merge_oof((pd.read_parquet(f) for f in files), cv_ids)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = PRED_DIR / f"{a.tag}.parquet"
    merged.to_parquet(out, index=False)
    print(f"merged {len(files)} folds -> {out} ({len(merged)} cv rows, each id exactly once)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
