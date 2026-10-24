import numpy as np
import pytest

from llm_preference_audit.calibrate import (
    PROB_FLOOR,
    MatrixScaler,
    TemperatureScaler,
    assert_cv_rows,
    cross_fit,
    log_features,
)
from llm_preference_audit.scoring import log_loss


def _true_probs(n, seed):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet([2.0, 2.0, 1.5], size=n)
    return P, rng


def _draw(P, rng):
    u = rng.random(len(P))[:, None]
    return (u > P.cumsum(axis=1)).sum(axis=1).clip(0, 2)


def test_log_features_floors_zeros_and_rows_renormalise():
    P = np.array([[0.0, 0.5, 0.5], [1.0, 0.0, 0.0]])
    F = log_features(P)
    assert np.isfinite(F).all()
    assert np.allclose(np.exp(F).sum(axis=1), 1.0)
    assert F[1, 1] == pytest.approx(F[1, 2])
    assert F[1, 1] < F[1, 0]
    assert PROB_FLOOR == 1e-3


def test_temperature_scaler_cools_overconfident_judge():
    P, rng = _true_probs(5000, 0)
    y = _draw(P, rng)
    sharp = P**3
    sharp /= sharp.sum(axis=1, keepdims=True)
    ts = TemperatureScaler().fit(sharp, y)
    assert ts.T > 1
    assert log_loss(y, ts.predict(sharp)) < log_loss(y, sharp)


def test_temperature_scaler_near_one_when_already_calibrated():
    P, rng = _true_probs(5000, 1)
    y = _draw(P, rng)
    assert TemperatureScaler().fit(P, y).T == pytest.approx(1.0, abs=0.15)


def _biased_case(n=4000, seed=2):
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=[0.35, 0.35, 0.30])
    P = np.full((n, 3), 0.0)
    P[:, :2] = 0.05
    P[np.arange(n), np.where(y == 2, rng.integers(0, 2, n), y)] = 0.9
    P[:, 2] = 0.0  # judge never gives tie mass
    P = P / P.sum(axis=1, keepdims=True)
    return P, y


def test_matrix_scaler_beats_raw_on_constant_class_bias():
    P, y = _biased_case()
    m = MatrixScaler().fit(P, y)
    out = m.predict(P)
    assert out.shape == (len(P), 3)
    assert log_loss(y, out) < log_loss(y, P)
    assert out[:, 2].mean() == pytest.approx(0.30, abs=0.05)


def test_matrix_scaler_columns_follow_class_order():
    P, y = _biased_case()
    out = MatrixScaler().fit(P, y).predict(P)
    assert (out[y == 0, 0] > out[y == 0, 1]).mean() > 0.9
    assert (out[y == 1, 1] > out[y == 1, 0]).mean() > 0.9


def test_matrix_scaler_raises_when_a_class_is_missing():
    P, y = _biased_case(n=200)
    with pytest.raises(ValueError):
        MatrixScaler().fit(P[y != 2], y[y != 2])


def test_assert_cv_rows_rejects_test_rows_and_names_count():
    assert_cv_rows(np.array(["cv", "cv"]))
    with pytest.raises(ValueError, match="2"):
        assert_cv_rows(np.array(["cv", "test", "test"]))


class _Spy:
    made = 0

    def __init__(self):
        type(self).made += 1

    def fit(self, P, y):
        raise AssertionError("must not fit")


def test_cross_fit_raises_on_test_split_before_fitting_anything():
    P, y = _biased_case(n=30)
    splits = np.array(["cv"] * 29 + ["test"])
    _Spy.made = 0
    with pytest.raises(ValueError):
        cross_fit(P, y, np.arange(30) % 3, splits, _Spy)
    assert _Spy.made == 0


def test_cross_fit_fold_predictions_ignore_own_fold_labels():
    P, y = _biased_case(n=900)
    folds = np.arange(900) % 3
    splits = np.array(["cv"] * 900)
    base = cross_fit(P, y, folds, splits, MatrixScaler)
    y2 = y.copy()
    f0 = folds == 0
    y2[f0] = (y2[f0] + 1) % 3
    alt = cross_fit(P, y2, folds, splits, MatrixScaler)
    assert base.shape == (900, 3)
    assert np.allclose(base[f0], alt[f0])
    assert not np.allclose(base[~f0], alt[~f0])
    assert np.allclose(base.sum(axis=1), 1.0)


def test_cross_fit_keeps_row_order_with_interleaved_folds():
    P, y = _biased_case(n=600)
    folds = np.random.default_rng(3).integers(0, 4, 600)
    out = cross_fit(P, y, folds, np.array(["cv"] * 600), TemperatureScaler)
    for f in range(4):
        tr = folds != f
        expect = TemperatureScaler().fit(P[tr], y[tr]).predict(P[~tr])
        assert np.allclose(out[~tr], expect)
