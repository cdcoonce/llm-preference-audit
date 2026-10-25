"""Build a fake /kaggle/input tree for a local dry run of the LightGBM kernel.

The competition dir gets a symlink to the real train.csv and a test.csv made of 500 rows' text
columns (no labels), written wherever --out points (use the scratchpad, never the repo). Source:
  --source test      rows from the locked TEST split (use only to check the pipeline runs;
                     never score them)
  --source cv-fold0  rows from cv fold 0 (in-sample for the final model: a sanity check only)
Also writes ids.json (the chosen ids in file order) next to the tree for later checks.

  uv run python kaggle/lgbm_submission/dry_run_input.py --source cv-fold0 --out SCRATCH/fake-cv
  LPA_INPUT_ROOT=SCRATCH/fake-cv/input LPA_OUTPUT_DIR=SCRATCH/fake-cv/work python lgbm_submission.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from llm_preference_audit.data import load_train
from llm_preference_audit.split import build_manifest

TEXT = ["id", "prompt", "response_a", "response_b"]
COMP_DIR = "llm-classification-finetuning"


def pick_ids(manifest: pd.DataFrame, source: str, n: int, seed: int) -> list[int]:
    if source == "test":
        sel = manifest[manifest["split"] == "test"]
    elif source == "cv-fold0":
        sel = manifest[(manifest["split"] == "cv") & (manifest["fold"] == 0)]
    else:
        raise SystemExit(f"unknown --source {source!r}")
    return sel["id"].sample(n=n, random_state=seed).tolist()  # shuffled: file order != id order


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", choices=["test", "cv-fold0"], required=True)
    ap.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    manifest = build_manifest(load_train(a.train))
    ids = pick_ids(manifest, a.source, a.n, a.seed)
    raw = pd.read_csv(a.train)
    test = raw.set_index("id").loc[ids].reset_index()[TEXT]
    comp = a.out / "input" / COMP_DIR
    comp.mkdir(parents=True, exist_ok=True)
    link = comp / "train.csv"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(a.train.resolve())
    test.to_csv(comp / "test.csv", index=False)
    (a.out / "ids.json").write_text(json.dumps([int(i) for i in ids]))
    print(f"wrote {comp / 'test.csv'} ({len(test)} rows, source {a.source}, no labels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
