import numpy as np
import pytest
from sklearn.metrics import log_loss as sk_log_loss

from llm_preference_audit.scoring import (
    calibration_table,
    fold_scores,
    group_bootstrap_diff,
    log_loss,
)


def random_probs(n, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet([2, 2, 2], n)
    y = rng.integers(0, 3, n)
    return y, P


def test_log_loss_matches_sklearn():
    y, P = random_probs(500)
    assert log_loss(y, P) == pytest.approx(sk_log_loss(y, P, labels=[0, 1, 2]), abs=1e-9)


def test_log_loss_clips_zero_probability():
    y = np.array([0])
    P = np.array([[0.0, 0.5, 0.5]])
    assert np.isfinite(log_loss(y, P)) and log_loss(y, P) > 30


def test_fold_scores_split_by_fold():
    y, P = random_probs(300)
    folds = np.arange(300) % 3
    fs = fold_scores(y, P, folds)
    assert set(fs) == {0, 1, 2}
    assert fs[1] == pytest.approx(log_loss(y[folds == 1], P[folds == 1]))


def test_bootstrap_sign_and_determinism():
    rng = np.random.default_rng(1)
    n_groups = 300
    groups = np.repeat(np.arange(n_groups), 3)
    ref = rng.normal(1.1, 0.1, len(groups))
    better = ref - 0.05 + rng.normal(0, 0.02, len(groups))
    res = group_bootstrap_diff(ref, better, groups)
    assert res["diff"] > 0 and res["lo"] > 0  # new is better: CI excludes 0 on the positive side
    rev = group_bootstrap_diff(better, ref, groups)
    assert rev["diff"] < 0 and rev["hi"] < 0
    noise = group_bootstrap_diff(ref, ref + rng.normal(0, 0.05, len(groups)), groups)
    assert noise["lo"] < 0 < noise["hi"]
    assert group_bootstrap_diff(ref, better, groups) == res


def test_bootstrap_resamples_whole_groups():
    # one group carries all the gain: CI must be wide (include 0) unlike row-level resampling
    groups = np.repeat(np.arange(50), 20)
    diff = np.zeros(len(groups))
    diff[:20] = 1.0
    res = group_bootstrap_diff(np.ones(len(groups)) * 2, np.ones(len(groups)) * 2 - diff, groups)
    assert res["lo"] <= 0 < res["hi"]


def test_calibration_perfect_and_overconfident():
    n = 1000
    P = np.tile([0.8, 0.1, 0.1], (n, 1))
    y = np.zeros(n, dtype=int)
    y[:200] = 1  # accuracy 0.8: perfectly calibrated
    t = calibration_table(y, P)
    assert t["ece"] == pytest.approx(0.0, abs=1e-9)
    assert sum(b["n"] for b in t["bins"]) == n and len(t["bins"]) == 10
    y2 = np.zeros(n, dtype=int)
    y2[:500] = 1  # accuracy 0.5 vs confidence 0.8
    assert calibration_table(y2, P)["ece"] == pytest.approx(0.3, abs=1e-9)
