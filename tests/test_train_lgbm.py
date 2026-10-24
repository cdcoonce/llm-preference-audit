import numpy as np
import pandas as pd
import pytest

from llm_preference_audit.features import RESPONSE_FEATURES, swap_pair
from llm_preference_audit.train_lgbm import (
    TestSplitLeak,
    assert_cv_only,
    fit_fold,
    inner_holdout_mask,
    predict_fold,
    split_fold,
)


def manifest(n=20):
    return pd.DataFrame(
        {
            "id": range(n),
            "group": [f"g{i // 2}" for i in range(n)],
            "split": ["test" if i % 5 == 0 else "cv" for i in range(n)],
            "fold": pd.array([None if i % 5 == 0 else i % 5 - 1 for i in range(n)], dtype="Int64"),
        }
    )


def test_guard_raises_on_test_ids_and_unknown_ids():
    m = manifest()
    assert_cv_only([1, 2, 3, 4], m)  # cv ids pass
    with pytest.raises(TestSplitLeak):
        assert_cv_only([1, 2, 5], m)  # id 5 is test
    with pytest.raises(TestSplitLeak):
        assert_cv_only([1, 999], m)


def synthetic_cv(n_groups=400, per=3, seed=0):
    rng = np.random.default_rng(seed)
    n = n_groups * per
    base = pd.DataFrame(
        {"n_words_a": rng.integers(1, 200, n), "n_words_b": rng.integers(1, 200, n)}
    )
    X = pd.DataFrame(index=pd.Index(np.arange(n), name="id"))
    for k in RESPONSE_FEATURES:
        X[f"{k}_a"] = 0.0
        X[f"{k}_b"] = 0.0
        X[f"{k}_diff"] = 0.0
        X[f"{k}_absdiff"] = 0.0
        X[f"{k}_logratio"] = 0.0
    X["n_words_a"] = base["n_words_a"].to_numpy().astype(float)
    X["n_words_b"] = base["n_words_b"].to_numpy().astype(float)
    X["n_words_diff"] = X["n_words_a"] - X["n_words_b"]
    X["n_words_absdiff"] = X["n_words_diff"].abs()
    X["n_words_logratio"] = np.log1p(X["n_words_a"]) - np.log1p(X["n_words_b"])
    X["prompt_words"] = 5.0
    X["prompt_turns"] = 1.0
    longer_a = X["n_words_diff"].to_numpy() > 0
    y = np.where(rng.random(n) < 0.8, np.where(longer_a, 0, 1), 2)
    groups = np.repeat([f"grp{g}" for g in range(n_groups)], per)
    folds = np.repeat(np.arange(n_groups) % 5, per)
    return X, y, groups, folds


def test_inner_holdout_is_groupwise_and_excludes_outer_fold():
    X, y, groups, folds = synthetic_cv()
    for swap in (False, True):
        d = split_fold(X, y, groups, folds, 2, swap)
        fit_ids, val_ids = set(d.X_fit.index), set(d.X_val.index)
        fold2 = set(X.index[folds == 2])
        assert not fit_ids & fold2 and not val_ids & fold2  # fold k never used
        assert fit_ids | val_ids == set(X.index[folds != 2])
        gmap = pd.Series(groups, index=X.index)
        assert not set(gmap[list(fit_ids)]) & set(gmap[list(val_ids)])  # whole groups only
        assert 0.03 < len(val_ids) / (len(fit_ids) + len(val_ids)) < 0.2
    hold = inner_holdout_mask(groups)
    assert (hold == inner_holdout_mask(groups)).all()  # deterministic
    # a group is either entirely held out or entirely kept
    df = pd.DataFrame({"g": groups, "h": hold}).groupby("g")["h"].nunique()
    assert (df == 1).all()


def test_swap_augmentation_doubles_training_rows():
    X, y, groups, folds = synthetic_cv()
    a = split_fold(X, y, groups, folds, 0, False)
    b = split_fold(X, y, groups, folds, 0, True)
    assert len(b.X_fit) == 2 * len(a.X_fit) and len(b.y_val) == 2 * len(a.y_val)


def test_swap_averaged_booster_is_position_symmetric_and_learns():
    X, y, groups, folds = synthetic_cv()
    d = split_fold(X, y, groups, folds, 0, True)
    booster = fit_fold(d)
    te = folds == 0
    p, asym = predict_fold(booster, X[te], True)
    p2, _ = predict_fold(booster, swap_pair(X[te]), True)
    np.testing.assert_allclose(p2, p[:, [1, 0, 2]], atol=1e-9)
    assert asym.max() > 0  # the raw model alone is not exactly symmetric
    acc = (p.argmax(axis=1) == y[te]).mean()
    assert acc > 0.7
