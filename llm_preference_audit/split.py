"""Deterministic grouped test/cv split keyed on the normalized prompt conversation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

MANIFEST_PATH = Path("splits/split_manifest.csv")  # gitignored: ~1.9 MB > leak-check limit
SHA_PATH = Path("splits/split_manifest.sha256")  # committed digest of the manifest bytes
COUNTS_PATH = Path("splits/split_counts.json")  # committed aggregate counts
TEST_BUCKETS = 10  # of 100
N_FOLDS = 5
_WS = re.compile(r"\s+")


def normalize_prompt(turns: list[str]) -> str:
    return "\n".join(_WS.sub(" ", t.casefold()).strip() for t in turns)


def group_hash(turns: list[str]) -> str:
    return hashlib.sha256(normalize_prompt(turns).encode("utf-8")).hexdigest()


def assign(hash_hex: str) -> tuple[str, int | None]:
    n = int(hash_hex, 16)
    if n % 100 < TEST_BUCKETS:
        return "test", None
    return "cv", n // 100 % N_FOLDS


def build_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Return columns id, group (first 16 hex of hash), split, fold (null for test)."""
    hashes = [group_hash(t) for t in df["prompt"]]
    assigned = [assign(h) for h in hashes]
    out = pd.DataFrame(
        {
            "id": df["id"].to_numpy(),
            "group": [h[:16] for h in hashes],
            "split": [a[0] for a in assigned],
            "fold": pd.array([a[1] for a in assigned], dtype="Int64"),
        }
    )
    return out.sort_values("id").reset_index(drop=True)


def manifest_sha256(manifest: pd.DataFrame) -> str:
    """Digest of the canonical CSV bytes (what write_manifest puts on disk)."""
    return hashlib.sha256(manifest.to_csv(index=False).encode("utf-8")).hexdigest()


def split_counts(manifest: pd.DataFrame) -> dict:
    cv = manifest[manifest.split == "cv"]
    return {
        "rows": len(manifest),
        "groups": int(manifest.group.nunique()),
        "test": {
            "rows": int((manifest.split == "test").sum()),
            "groups": int(manifest[manifest.split == "test"].group.nunique()),
        },
        "folds": {
            str(f): {"rows": len(g), "groups": int(g.group.nunique())}
            for f, g in cv.groupby("fold")
        },
    }


def write_manifest(manifest: pd.DataFrame, path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    SHA_PATH.write_text(f"{manifest_sha256(manifest)}  {path.name}\n")
    COUNTS_PATH.write_text(json.dumps(split_counts(manifest), indent=2) + "\n")
