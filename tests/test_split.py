from pathlib import Path

import pandas as pd
import pytest

from llm_preference_audit.data import load_train
from llm_preference_audit.split import (
    SHA_PATH,
    build_manifest,
    group_hash,
    manifest_sha256,
    normalize_prompt,
)

RAW = Path("data/raw/train.csv")


def synthetic(n_groups: int = 4000, dup: int = 2) -> pd.DataFrame:
    rows = []
    i = 0
    for g in range(n_groups):
        for _ in range(dup if g % 3 == 0 else 1):
            rows.append({"id": i, "prompt": [f"synthetic prompt number {g}"]})
            i += 1
    return pd.DataFrame(rows)


def test_determinism():
    df = synthetic(500)
    a = build_manifest(df)
    b = build_manifest(df.sample(frac=1, random_state=1).reset_index(drop=True))
    pd.testing.assert_frame_equal(
        a.sort_values("id").reset_index(drop=True), b.sort_values("id").reset_index(drop=True)
    )


def test_group_never_split_across_folds_or_test_cv():
    m = build_manifest(synthetic())
    per_group = m.groupby("group").agg(
        n_split=("split", "nunique"), n_fold=("fold", lambda s: s.fillna(-1).nunique())
    )
    assert (per_group.n_split == 1).all()
    assert (per_group.n_fold == 1).all()
    assert m.groupby("group").size().max() >= 2  # the test saw real duplicate groups


def test_normalization_shares_group():
    assert normalize_prompt(["Hi  there"]) == normalize_prompt([" hi there\n"])
    assert group_hash(["Hi  there"]) == group_hash(["hi there"])
    assert group_hash(["hi there"]) != group_hash(["hi there!"])
    df = pd.DataFrame({"id": [1, 2], "prompt": [["Hi  there"], ["hi there"]]})
    m = build_manifest(df)
    assert m.group.nunique() == 1


def test_test_share_about_ten_percent():
    m = build_manifest(synthetic(5000))
    share = (m.drop_duplicates("group").split == "test").mean()
    assert 0.08 < share < 0.12


def test_every_row_exactly_one_assignment():
    df = synthetic(1000)
    m = build_manifest(df)
    assert len(m) == len(df) and m.id.is_unique and set(m.id) == set(df.id)
    assert set(m.split) == {"test", "cv"}
    assert m[m.split == "test"].fold.isna().all()
    assert m[m.split == "cv"].fold.isin(range(5)).all()


def test_all_five_folds_nonempty():
    m = build_manifest(synthetic(2000))
    assert set(m[m.split == "cv"].fold.astype(int)) == {0, 1, 2, 3, 4}
    assert list(m.columns) == ["id", "group", "split", "fold"]


def test_manifest_sha256_is_stable_and_sensitive():
    df = synthetic(200)
    a = build_manifest(df)
    assert manifest_sha256(a) == manifest_sha256(a.copy())
    b = a.copy()
    b.loc[0, "group"] = "0" * 16
    assert manifest_sha256(a) != manifest_sha256(b)


@pytest.mark.skipif(not RAW.exists(), reason="data/raw absent (CI has no data)")
def test_committed_manifest_digest_matches_real_data():
    committed = SHA_PATH.read_text().split()[0]
    assert manifest_sha256(build_manifest(load_train(RAW))) == committed
