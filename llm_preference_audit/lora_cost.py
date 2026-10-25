"""Torch-free cost arithmetic for rung 6 (Modal H100 billing and the probe projection)."""

from __future__ import annotations

# Modal H100 list price: $0.001097 per second = $3.95 per hour (https://modal.com/pricing,
# checked 2026-10-25). Modal may run an "H100" request on an H200 at the same price.
PRICE_PER_HOUR = 3.95
OVERHEAD = 0.15  # image start, downloads, idle: the plan's 15% line
G0_TOTAL_CAP_USD = 70.0  # G0: the projected total must be at most this


def est_cost_usd(seconds: float) -> float:
    return float(seconds) * PRICE_PER_HOUR / 3600


def project(
    *,
    n_cv: int,
    train_samples_per_s: float,
    infer_seqs_per_s: float,
    load_seconds: float,
    n_evals: int = 2,
    inner_share: float = 0.1,
    fold_share: float = 0.2,
) -> dict:
    """Project GPU seconds and dollars from probe rates.

    fold  = load + train(80% x 90% of cv) + n_evals x both-order inner holdout + both-order fold.
    full  = load + train(all cv rows), no eval, no inference.
    The cost lines carry the 15% overhead separately (`*_usd`) from the raw rate x time.
    """
    if train_samples_per_s <= 0 or infer_seqs_per_s <= 0:
        raise ValueError("rates must be positive")
    n_train = n_cv * (1 - fold_share) * (1 - inner_share)
    n_inner = n_cv * (1 - fold_share) * inner_share
    n_fold = n_cv * fold_share
    train_s = n_train / train_samples_per_s
    eval_s = n_evals * 2 * n_inner / infer_seqs_per_s
    pred_s = 2 * n_fold / infer_seqs_per_s
    fold_s = load_seconds + train_s + eval_s + pred_s
    full_s = load_seconds + n_cv / train_samples_per_s
    fold_usd, full_usd = est_cost_usd(fold_s), est_cost_usd(full_s)
    raw_total = 5 * fold_usd + full_usd
    total = raw_total * (1 + OVERHEAD)
    return {
        "price_per_hour": PRICE_PER_HOUR,
        "fold_seconds": fold_s,
        "fold_train_seconds": train_s,
        "fold_eval_seconds": eval_s,
        "fold_predict_seconds": pred_s,
        "full_seconds": full_s,
        "fold_usd": fold_usd,
        "five_folds_usd": 5 * fold_usd,
        "full_usd": full_usd,
        "folds_1_to_4_plus_full_usd": 4 * fold_usd + full_usd,
        "overhead_share": OVERHEAD,
        "total_usd_raw": raw_total,
        "total_usd_with_overhead": total,
        "g0_total_cap_usd": G0_TOTAL_CAP_USD,
        "meets_g0_total_cap": total <= G0_TOTAL_CAP_USD,
    }
