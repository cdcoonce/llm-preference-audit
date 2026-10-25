"""5-fold grouped CV of a LightGBM multiclass model on the cv split only."""

from __future__ import annotations

from typing import NamedTuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from llm_preference_audit.cv_common import (  # noqa: F401  re-exported
    INNER_BUCKETS,
    TestSplitLeak,
    assert_cv_only,
    inner_holdout_mask,
)
from llm_preference_audit.features import swap_labels, swap_pair, swap_probs

PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
    "verbosity": -1,
}
MAX_ROUNDS = 5000
EARLY_STOP = 100


class FoldData(NamedTuple):
    X_fit: pd.DataFrame
    y_fit: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray


def split_fold(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, folds: np.ndarray, k: int, swap: bool
) -> FoldData:
    """Training data for outer fold k: the other folds, minus a group-wise inner holdout.

    Fold k never reaches either side. With swap, each side also gets its swapped copies.
    """
    train = folds != k
    X_tr, y_tr, g_tr = X[train], y[train], groups[train]
    hold = inner_holdout_mask(g_tr)
    X_fit, y_fit, X_val, y_val = X_tr[~hold], y_tr[~hold], X_tr[hold], y_tr[hold]
    if swap:
        X_fit = pd.concat([X_fit, swap_pair(X_fit)])
        y_fit = np.concatenate([y_fit, swap_labels(y_fit)])
        X_val = pd.concat([X_val, swap_pair(X_val)])
        y_val = np.concatenate([y_val, swap_labels(y_val)])
    return FoldData(X_fit, y_fit, X_val, y_val)


def fit_fold(data: FoldData) -> lgb.Booster:
    dtr = lgb.Dataset(data.X_fit, label=data.y_fit)
    dva = lgb.Dataset(data.X_val, label=data.y_val, reference=dtr)
    return lgb.train(
        PARAMS,
        dtr,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )


def predict_fold(
    booster: lgb.Booster, X: pd.DataFrame, swap: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return (final probabilities, raw asymmetry per row).

    asymmetry = max |p(row) - mirror(p(swapped row))|; exactly 0 after swap averaging.
    """
    it = booster.best_iteration
    p = booster.predict(X, num_iteration=it)
    q = swap_probs(booster.predict(swap_pair(X), num_iteration=it))
    final = (p + q) / 2.0 if swap else p
    return final, np.abs(p - q).max(axis=1)


def run_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    manifest: pd.DataFrame,
    swap: bool,
) -> dict:
    """Out-of-fold predictions over all cv rows. X is indexed by id."""
    assert_cv_only(X.index.to_numpy(), manifest)
    oof = np.zeros((len(X), 3))
    asym = np.zeros(len(X))
    best, gains = {}, []
    for k in sorted(set(folds.tolist())):
        te = folds == k
        booster = fit_fold(split_fold(X, y, groups, folds, k, swap))
        oof[te], asym[te] = predict_fold(booster, X[te], swap)
        best[int(k)] = int(booster.best_iteration)
        gains.append(booster.feature_importance("gain"))
    gain = pd.Series(np.mean(gains, axis=0), index=X.columns)
    return {"oof": oof, "asym": asym, "best_iterations": best, "gain": gain}
