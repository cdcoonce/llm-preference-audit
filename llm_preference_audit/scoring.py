"""Log loss, paired group bootstrap, calibration."""

from __future__ import annotations

import numpy as np

EPS = 1e-15
BOOT_SEED = 20261023
N_BOOT = 1000


def row_losses(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Per-row natural-log loss. P rows are renormalised after clipping to [EPS, 1-EPS]."""
    P = np.clip(np.asarray(P, dtype=float), EPS, 1 - EPS)
    P = P / P.sum(axis=1, keepdims=True)
    return -np.log(P[np.arange(len(y)), np.asarray(y, dtype=int)])


def log_loss(y: np.ndarray, P: np.ndarray) -> float:
    return float(row_losses(y, P).mean())


def fold_scores(y: np.ndarray, P: np.ndarray, folds: np.ndarray) -> dict[int, float]:
    return {int(f): log_loss(y[folds == f], P[folds == f]) for f in sorted(set(folds.tolist()))}


def group_bootstrap_diff(
    loss_ref: np.ndarray,
    loss_new: np.ndarray,
    groups: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, float]:
    """Paired group bootstrap of mean(loss_ref) - mean(loss_new).

    Positive means `new` is better. Resamples whole prompt groups with replacement.
    """
    _, gidx = np.unique(np.asarray(groups), return_inverse=True)
    n_g = gidx.max() + 1
    diff = np.asarray(loss_ref, dtype=float) - np.asarray(loss_new, dtype=float)
    g_sum = np.bincount(gidx, weights=diff, minlength=n_g)
    g_cnt = np.bincount(gidx, minlength=n_g).astype(float)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, n_g, n_g), minlength=n_g)
        stats[i] = (w * g_sum).sum() / (w * g_cnt).sum()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"diff": float(diff.mean()), "lo": float(lo), "hi": float(hi)}


def calibration_table(y: np.ndarray, P: np.ndarray, n_bins: int = 10) -> dict:
    """Bin rows by max predicted probability (equal-width bins on [1/3, 1]).

    Returns per-bin count, mean confidence, accuracy, and the expected calibration error.
    """
    P = np.asarray(P, dtype=float)
    conf = P.max(axis=1)
    correct = (P.argmax(axis=1) == np.asarray(y)).astype(float)
    edges = np.linspace(1 / 3, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)
    bins = []
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            bins.append(
                {"lo": edges[b], "hi": edges[b + 1], "n": 0, "confidence": None, "accuracy": None}
            )
            continue
        c, a = float(conf[m].mean()), float(correct[m].mean())
        ece += n / len(conf) * abs(a - c)
        bins.append(
            {
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "n": n,
                "confidence": c,
                "accuracy": a,
            }
        )
    return {"bins": bins, "ece": float(ece)}
