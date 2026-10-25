"""Torch-free data-integrity helpers shared by the rung-5 and rung-6 training scripts.

The train.csv sha256 and the split-manifest digest are verified on the FULL file before anything
is restricted, so a training box can never silently run on a different split. The competition
test.csv is never referenced here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from llm_preference_audit.data import load_train
from llm_preference_audit.split import build_manifest, manifest_sha256

# The Hugging Face train.csv (byte-identical to Kaggle's copy) and the committed split digest.
TRAIN_SHA256 = "0692154aaf20fc6649090c3f49b6b5dd1e693a765ecb5f0856ae0da5946f5be2"
MANIFEST_SHA256 = "0a77276dd887690580c1004a7d44224117095f2912947ba9137dd602a4995523"
GIB = 1024**3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def locate_train_csv(data_dir: Path, expect_sha: str) -> Path:
    """data_dir/train.csv, else the train.csv under data_dir's parent whose sha256 matches.

    Kaggle's competition mount point has moved between images, so the hash decides, not the path.
    """
    direct = data_dir / "train.csv"
    if direct.is_file():
        return direct
    candidates = sorted(data_dir.parent.rglob("train.csv"))
    for path in candidates:
        if sha256_file(path) == expect_sha:
            print(f"found train.csv at {path} (searched under {data_dir.parent})", flush=True)
            return path
    raise SystemExit(
        f"no train.csv with sha256 {expect_sha} under {data_dir.parent}; candidates: "
        f"{[str(c) for c in candidates]}"
    )


def load_verified(data_dir: Path, expect_sha: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hash train.csv, build the manifest on the full file and check its digest. Never test.csv."""
    path = locate_train_csv(data_dir, expect_sha)
    got = sha256_file(path)
    if got != expect_sha:
        raise SystemExit(f"train.csv sha256 {got} != expected {expect_sha}")
    df = load_train(path)
    manifest = build_manifest(df)
    digest = manifest_sha256(manifest)
    if digest != MANIFEST_SHA256:
        raise SystemExit(f"split manifest digest {digest} != committed {MANIFEST_SHA256}")
    print(f"verified train.csv sha256 and split manifest digest ({len(df)} rows)", flush=True)
    return df, manifest
