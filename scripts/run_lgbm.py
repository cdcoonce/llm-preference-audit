"""Rung 2: hand-built features + LightGBM, 5-fold grouped CV on the cv split only.

Commit the code before running: each ledger row records HEAD as git_sha.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from llm_preference_audit.bias import identical_rows_readout, longer_wins_readout
from llm_preference_audit.data import LABELS, load_train
from llm_preference_audit.features import RESPONSE_FEATURES, build_features, feature_columns
from llm_preference_audit.scoring import (
    calibration_table,
    fold_scores,
    group_bootstrap_diff,
    log_loss,
    row_losses,
)
from llm_preference_audit.split import MANIFEST_PATH, SHA_PATH, build_manifest, manifest_sha256
from llm_preference_audit.train_lgbm import (
    EARLY_STOP,
    INNER_BUCKETS,
    MAX_ROUNDS,
    PARAMS,
    assert_cv_only,
    run_cv,
)

RAW = Path("data/raw/train.csv")
LEDGER = Path("results/ledger.jsonl")
PRED_DIR = Path("data/predictions")
SUMMARY = Path("results/rung-2-summary.json")
PRIOR_LOG_LOSS = 1.0972  # class-prior log loss from results/profile.json (gate threshold)
CONFIGS = {
    "length_only": {"features": "length_only", "swap": True},
    "full": {"features": "full", "swap": True},
    "full_no_swap": {"features": "full", "swap": False},
}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def oof_prior(y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Out-of-fold class prior: each fold gets the label frequencies of the other folds."""
    P = np.zeros((len(y), 3))
    for k in sorted(set(folds.tolist())):
        freq = np.bincount(y[folds != k], minlength=3) / (folds != k).sum()
        P[folds == k] = freq
    return P


def main(argv: list[str]) -> None:
    names = argv or list(CONFIGS)
    unknown = set(names) - set(CONFIGS)
    if unknown:
        raise SystemExit(f"unknown configs {sorted(unknown)}; choose from {list(CONFIGS)}")
    dirty = git("status", "--porcelain", "--", "llm_preference_audit", "scripts", "pyproject.toml")
    if dirty:
        raise SystemExit(f"uncommitted code; commit before running:\n{dirty}")
    sha = git("rev-parse", "HEAD")
    df = load_train(RAW)
    manifest = build_manifest(df)
    if manifest_sha256(manifest) != SHA_PATH.read_text().split()[0]:
        raise SystemExit("split manifest does not match the committed digest")
    print(f"manifest ok ({MANIFEST_PATH} digest matches)")

    t0 = time.time()
    cache = Path("data/features.parquet")  # gitignored with the rest of data/
    if cache.exists() and len(pd.read_parquet(cache)) == len(df):
        X_all = pd.read_parquet(cache)
    else:
        X_all = build_features(df)
        X_all.to_parquet(cache)
    print(f"features: {X_all.shape} in {time.time() - t0:.0f}s")
    m = manifest.set_index("id").loc[X_all.index]
    cv = (m["split"] == "cv").to_numpy()
    X_cv = X_all[cv]
    assert_cv_only(X_cv.index.to_numpy(), manifest)
    lab = df.set_index("id")["label"].loc[X_cv.index].astype(str)
    y = lab.map({c: i for i, c in enumerate(LABELS)}).to_numpy(dtype=int)
    groups = m.loc[X_cv.index, "group"].to_numpy()
    folds = m.loc[X_cv.index, "fold"].to_numpy(dtype=int)
    print(f"cv rows {len(y)}, groups {len(set(groups))}")

    prior = oof_prior(y, folds)
    prior_loss = row_losses(y, prior)
    print(f"oof prior log loss {prior_loss.mean():.4f}")

    ts = datetime.now(UTC)
    stamp = ts.strftime("%Y%m%dT%H%M%S")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {
        "prior_oof_log_loss": float(prior_loss.mean()),
        "n_rows": len(y),
        "configs": {},
    }
    losses: dict[str, np.ndarray] = {}
    words_a = X_cv["n_words_a"].to_numpy()
    words_b = X_cv["n_words_b"].to_numpy()
    ident = np.all(
        [X_cv[f"{k}_a"].to_numpy() == X_cv[f"{k}_b"].to_numpy() for k in RESPONSE_FEATURES], axis=0
    )
    for name in names:
        cfg = CONFIGS[name]
        cols = feature_columns(cfg["features"])
        X = X_cv[cols]
        t1 = time.time()
        res = run_cv(X, y, groups, folds, manifest, cfg["swap"])
        secs = time.time() - t1
        P = res["oof"]
        loss = row_losses(y, P)
        fs = fold_scores(y, P, folds)
        ci_prior = group_bootstrap_diff(prior_loss, loss, groups)
        cal = calibration_table(y, P)
        config = {
            "name": name,
            "features": cols,
            "swap": cfg["swap"],
            "params": PARAMS,
            "max_rounds": MAX_ROUNDS,
            "early_stopping_rounds": EARLY_STOP,
            "inner_holdout_buckets": INNER_BUCKETS,
        }
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        run_id = f"{stamp}-{name}"
        pred_path = PRED_DIR / f"{run_id}.parquet"
        pd.DataFrame(
            {"id": X_cv.index.to_numpy(), "p_a": P[:, 0], "p_b": P[:, 1], "p_tie": P[:, 2]}
        ).to_parquet(pred_path, index=False)
        pred_sha = hashlib.sha256(pred_path.read_bytes()).hexdigest()
        row = {
            "run_id": run_id,
            "timestamp_utc": ts.isoformat(timespec="seconds"),
            "git_sha": sha,
            "config_hash": config_hash,
            "model": "lightgbm",
            "config": name,
            "split": "cv-oof",
            "fold_scores": {str(k): v for k, v in fs.items()},
            "log_loss": log_loss(y, P),
            "cost_usd": 0,
            "predictions_path": str(pred_path),
            "predictions_sha256": pred_sha,
            "n_rows": len(y),
            "ci95_vs_prior": [ci_prior["lo"], ci_prior["hi"]],
            "best_iterations": {str(k): v for k, v in res["best_iterations"].items()},
        }
        with LEDGER.open("a") as f:
            f.write(json.dumps(row) + "\n")
        top = res["gain"].sort_values(ascending=False)
        out["configs"][name] = {
            "run_id": run_id,
            "log_loss": row["log_loss"],
            "fold_scores": row["fold_scores"],
            "improvement_vs_prior": ci_prior,
            "calibration": cal,
            "best_iterations": row["best_iterations"],
            "runtime_seconds": secs,
            "n_features": len(cols),
            "top_features": [
                {"feature": k, "gain_share": float(v / top.sum())} for k, v in top.head(20).items()
            ],
            "longer_wins": longer_wins_readout(y, P, words_a, words_b),
            "identical_rows": identical_rows_readout(P, ident),
            "raw_asymmetry": {
                "mean": float(res["asym"].mean()),
                "max": float(res["asym"].max()),
            },
            "predictions_sha256": pred_sha,
        }
        print(
            f"{name}: oof {row['log_loss']:.4f} folds {min(fs.values()):.4f}-{max(fs.values()):.4f}"
            f" vs prior {ci_prior['diff']:+.4f} [{ci_prior['lo']:+.4f},{ci_prior['hi']:+.4f}]"
            f" ece {cal['ece']:.4f} iters {row['best_iterations']} {secs:.0f}s"
        )
    if set(out["configs"]) == set(CONFIGS):
        losses = {}
        for name, c in out["configs"].items():
            pr = (
                pd.read_parquet(PRED_DIR / f"{c['run_id']}.parquet").set_index("id").loc[X_cv.index]
            )
            losses[name] = row_losses(y, pr[["p_a", "p_b", "p_tie"]].to_numpy())
        out["paired"] = {
            f"{a}_vs_{b}": group_bootstrap_diff(losses[a], losses[b], groups)
            for a, b in (("length_only", "full"), ("full_no_swap", "full"))
        }
        full = out["configs"]["full"]
        ci = full["improvement_vs_prior"]
        out["gate"] = {
            "threshold": PRIOR_LOG_LOSS,
            "pass": bool(full["log_loss"] < PRIOR_LOG_LOSS and ci["lo"] > 0),
        }
        print(json.dumps({"gate": out["gate"], "paired": out["paired"]}, indent=2))
    SUMMARY.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
