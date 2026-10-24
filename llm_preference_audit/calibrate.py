"""Post-hoc calibration of zero-shot judge probabilities (cv rows only)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

PROB_FLOOR = 1e-3
_GOLD = (np.sqrt(5) - 1) / 2


def log_features(P: np.ndarray) -> np.ndarray:
    """Log of P clipped to [PROB_FLOOR, 1] and renormalised per row."""
    P = np.clip(np.asarray(P, dtype=float), PROB_FLOOR, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    return np.log(P)


def _softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


class TemperatureScaler:
    """One temperature T > 0 on the log-probabilities, fit by NLL."""

    T: float = 1.0

    def _nll(self, F: np.ndarray, y: np.ndarray, log_t: float) -> float:
        P = _softmax(F / np.exp(log_t))
        return float(-np.log(P[np.arange(len(y)), y] + 1e-15).mean())

    def fit(self, P: np.ndarray, y: np.ndarray) -> TemperatureScaler:
        F, y = log_features(P), np.asarray(y, dtype=int)
        lo, hi = -3.0, 3.0
        for _ in range(60):
            c, d = hi - _GOLD * (hi - lo), lo + _GOLD * (hi - lo)
            if self._nll(F, y, c) < self._nll(F, y, d):
                hi = d
            else:
                lo = c
        self.T = float(np.exp((lo + hi) / 2))
        return self

    def predict(self, P: np.ndarray) -> np.ndarray:
        return _softmax(log_features(P) / self.T)


class MatrixScaler:
    """Multinomial logistic regression on log-probabilities (3x3 map plus bias)."""

    def fit(self, P: np.ndarray, y: np.ndarray) -> MatrixScaler:
        y = np.asarray(y, dtype=int)
        if set(np.unique(y).tolist()) != {0, 1, 2}:
            raise ValueError(f"training labels must contain classes 0, 1, 2; got {np.unique(y)}")
        self._lr = LogisticRegression(C=1.0, max_iter=1000).fit(log_features(P), y)
        return self

    def predict(self, P: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(log_features(P))


def assert_cv_rows(splits) -> None:
    """Raise if any row is not a cv (training-fold) row."""
    bad = int((np.asarray(splits) != "cv").sum())
    if bad:
        raise ValueError(f"{bad} non-cv rows reached calibration")


def cross_fit(P, y, folds, splits, make_scaler) -> np.ndarray:
    """Out-of-fold calibrated probabilities: each fold is scored by a scaler fit on the rest."""
    assert_cv_rows(splits)
    P, y, folds = np.asarray(P, dtype=float), np.asarray(y), np.asarray(folds)
    out = np.zeros((len(P), 3))
    for f in sorted(np.unique(folds).tolist()):
        te = folds == f
        out[te] = make_scaler().fit(P[~te], y[~te]).predict(P[te])
    return out
