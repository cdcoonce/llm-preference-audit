"""Torch-free and lightgbm-free split guards shared by LightGBM, DeBERTa and LoRA code.

Kept out of `train_lgbm` so the Modal container (which does not install lightgbm's system
library) can import them without importing lightgbm. `train_lgbm` re-exports every name here.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

INNER_BUCKETS = 10  # inner early-stopping holdout is 1 group bucket of 10 (~10%)


class TestSplitLeak(RuntimeError):
    __test__ = False  # not a pytest class


def assert_cv_only(ids, manifest: pd.DataFrame) -> None:
    """Raise if any id is a test-split row (or absent from the manifest)."""
    m = manifest.set_index("id")["split"]
    ids = np.asarray(ids)
    unknown = ~np.isin(ids, m.index.to_numpy())
    if unknown.any():
        raise TestSplitLeak(f"{int(unknown.sum())} ids are not in the split manifest")
    bad = m.loc[ids].to_numpy() != "cv"
    if bad.any():
        raise TestSplitLeak(f"{int(bad.sum())} test-split ids reached training or scoring")


def inner_holdout_mask(groups) -> np.ndarray:
    """Deterministic group-wise ~10% holdout: whole groups, chosen by a hash of the group id."""
    return np.array(
        [
            int(hashlib.sha256(f"inner:{g}".encode()).hexdigest(), 16) % INNER_BUCKETS == 0
            for g in groups
        ]
    )
