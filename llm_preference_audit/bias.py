"""Bias read-outs on out-of-fold predictions."""

from __future__ import annotations

import numpy as np

from llm_preference_audit.scoring import group_bootstrap_diff


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
        "direction_agreement": float(model),
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


def calibrated_longer_readout(
    y: np.ndarray,
    P: np.ndarray,
    words_a: np.ndarray,
    words_b: np.ndarray,
    groups: np.ndarray,
    n_buckets: int = 5,
) -> dict:
    """Like-for-like verbosity comparison.

    Among non-ties with unequal word counts: the human share where the longer response won,
    versus the model's mean p_longer / (p_longer + p_shorter). diff = model - human, with a
    paired group-bootstrap 95% CI. Buckets are quintiles of |log1p word ratio|.
    """
    m = (y != 2) & (words_a != words_b)
    wa, wb = words_a[m].astype(float), words_b[m].astype(float)
    longer_is_a = wa > wb
    p = P[m]
    p_long = np.where(longer_is_a, p[:, 0], p[:, 1])
    p_short = np.where(longer_is_a, p[:, 1], p[:, 0])
    model = p_long / (p_long + p_short)
    human = ((y[m] == 0) == longer_is_a).astype(float)
    boot = group_bootstrap_diff(model, human, groups[m])
    mag = np.abs(np.log1p(wa) - np.log1p(wb))  # log1p: empty responses are valid
    edges = np.quantile(mag, np.linspace(0, 1, n_buckets + 1))
    idx = np.clip(np.searchsorted(edges, mag, side="right") - 1, 0, n_buckets - 1)
    buckets = [
        {
            "lo": float(edges[b]),
            "hi": float(edges[b + 1]),
            "n": int((idx == b).sum()),
            "human_share": float(human[idx == b].mean()),
            "model_share": float(model[idx == b].mean()),
        }
        for b in range(n_buckets)
    ]
    return {
        "n": int(m.sum()),
        "human_share": float(human.mean()),
        "model_share": float(model.mean()),
        "diff": boot["diff"],
        "ci95": [boot["lo"], boot["hi"]],
        "buckets": buckets,
    }


def swap_flip_readout(P_ab: np.ndarray, P_ba: np.ndarray) -> dict:
    """Order sensitivity of a judge. Both arrays are (n, 3) in original label order (the
    swapped call already mapped back); rows with a NaN in either are excluded."""
    P_ab, P_ba = np.asarray(P_ab, dtype=float), np.asarray(P_ba, dtype=float)
    ok = ~(np.isnan(P_ab).any(axis=1) | np.isnan(P_ba).any(axis=1))
    a, b = P_ab[ok], P_ba[ok]
    return {
        "n": int(ok.sum()),
        "flip_rate": float((a.argmax(axis=1) != b.argmax(axis=1)).mean()),
        "mean_abs_diff": float(np.abs(a - b).max(axis=1).mean()),
    }


def first_position_readout(raw_ab: np.ndarray, raw_ba: np.ndarray, groups: np.ndarray) -> dict:
    """Position bias. Raw arrays are (n, 3) in presented order (col 0 = shown first, col 1 =
    shown second), NaN rows for failed calls. Each valid call is one observation; a row's
    group applies to both its calls. diff = mean p_first - mean p_second."""
    raw = np.vstack([raw_ab, raw_ba]).astype(float)
    g = np.concatenate([np.asarray(groups), np.asarray(groups)])
    ok = ~np.isnan(raw).any(axis=1)
    raw, g = raw[ok], g[ok]
    p_first, p_second = raw[:, 0], raw[:, 1]
    boot = group_bootstrap_diff(p_first, p_second, g)
    top = raw.argmax(axis=1)
    nontie = top != 2
    return {
        "n_calls": int(ok.sum()),
        "mean_p_first": float(p_first.mean()),
        "mean_p_second": float(p_second.mean()),
        "diff": boot["diff"],
        "ci95": [boot["lo"], boot["hi"]],
        "share_first_wins_argmax_nontie": float((top[nontie] == 0).mean()),
    }
