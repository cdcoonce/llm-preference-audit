"""Bias read-outs on out-of-fold predictions."""

from __future__ import annotations

import numpy as np


def longer_wins_readout(
    y: np.ndarray, P: np.ndarray, words_a: np.ndarray, words_b: np.ndarray
) -> dict:
    """Among non-ties with unequal word counts, how often the human label and the model's
    higher win probability both go to the longer response. Codes: 0=a, 1=b, 2=tie."""
    m = (y != 2) & (words_a != words_b)
    longer_is_a = (words_a > words_b)[m]
    human = ((y[m] == 0) == longer_is_a).mean()
    model = ((P[m, 0] > P[m, 1]) == longer_is_a).mean()
    return {
        "n": int(m.sum()),
        "human_longer_share": float(human),
        "model_longer_share": float(model),
    }


def identical_rows_readout(P: np.ndarray, identical: np.ndarray) -> dict:
    """On rows whose a and b features are identical: how often the model favours a."""
    p = P[identical]
    if len(p) == 0:
        return {
            "n": 0,
            "share_favours_a": None,
            "mean_p_a_minus_p_b": None,
            "max_abs_p_a_minus_p_b": None,
        }
    d = p[:, 0] - p[:, 1]
    return {
        "n": len(p),
        "share_favours_a": float((d > 0).mean()),
        "mean_p_a_minus_p_b": float(d.mean()),
        "max_abs_p_a_minus_p_b": float(np.abs(d).max()),
    }
