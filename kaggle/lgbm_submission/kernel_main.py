"""Main of the rung-2 LightGBM submission kernel (CPU, internet off, no attached dataset).

`build_kernel.py` appends this file to a header that registers the inlined repo modules, so the
imports below resolve to the committed sources. Under pytest they resolve to the repo itself.

Order of work: write a prior-filled submission.csv at once, verify train.csv, fit ONE `full`
LightGBM model on the cv-split rows only (the locked test split is never fitted on), predict the
competition test.csv swap-averaged, overwrite submission.csv atomically and write run_log.json.
Nothing written here carries dataset text.

Environment (both optional, harmless): LPA_INPUT_ROOT (default /kaggle/input) and LPA_OUTPUT_DIR
(default /kaggle/working). The sha checks can only be skipped by a build-time flag baked into the
script (`build["skip_sha_checks"]`); the production render never sets it.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from llm_preference_audit.cv_common import assert_cv_only
from llm_preference_audit.data import LABELS, load_train
from llm_preference_audit.features import (
    build_features,
    feature_columns,
    swap_labels,
    swap_pair,
    symmetric_predict,
)
from llm_preference_audit.kernel_common import MANIFEST_SHA256, TRAIN_SHA256, sha256_file
from llm_preference_audit.split import build_manifest, manifest_sha256
from llm_preference_audit.submission import (
    class_prior,
    load_test,
    locate_input,
    prior_submission,
    submission_frame,
)
from llm_preference_audit.train_lgbm import PARAMS

DEFAULT_INPUT_ROOT = "/kaggle/input"
DEFAULT_OUTPUT_DIR = "/kaggle/working"
COMPETITION_HINT = "llm-classification-finetuning"


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def select_cv(df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """The rows allowed into training: the cv split, in train.csv order."""
    split = manifest.set_index("id")["split"].loc[df["id"].to_numpy()].to_numpy()
    out = df[split == "cv"]
    assert_cv_only(out["id"].to_numpy(), manifest)
    return out


def labels_of(df: pd.DataFrame) -> np.ndarray:
    return df["label"].astype(str).map({c: i for i, c in enumerate(LABELS)}).to_numpy(dtype=int)


def fit_final(X: pd.DataFrame, y: np.ndarray, rounds: int) -> lgb.Booster:
    """One model on all rows plus their swapped copies (rung 2's swap augmentation), fixed rounds."""
    X_aug = pd.concat([X, swap_pair(X)])
    y_aug = np.concatenate([y, swap_labels(y)])
    return lgb.train(PARAMS, lgb.Dataset(X_aug, label=y_aug), num_boost_round=rounds)


def predict_test(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Swap-averaged probabilities (a, b, tie)."""
    return symmetric_predict(lambda frame: booster.predict(frame), X)


def main(build: dict) -> None:
    t_start = time.perf_counter()
    roots = [os.environ.get("LPA_INPUT_ROOT", DEFAULT_INPUT_ROOT)]
    out_dir = Path(os.environ.get("LPA_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)
    skip_sha = bool(build.get("skip_sha_checks", False))
    rounds = int(build["rounds"])
    if rounds < 1:
        raise SystemExit(f"rounds must be positive, got {rounds}")
    secs: dict[str, float] = {}
    t = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal t
        now = time.perf_counter()
        secs[name] = round(now - t, 3)
        t = now
        print(f"phase {name}: {secs[name]}s", flush=True)

    train_path = locate_input(
        "train.csv", roots, expect_sha=None if skip_sha else TRAIN_SHA256, prefer=COMPETITION_HINT
    )
    test_path = locate_input("test.csv", roots, prefer=COMPETITION_HINT)
    if not skip_sha and sha256_file(train_path) != TRAIN_SHA256:
        raise SystemExit("train.csv sha256 does not match TRAIN_SHA256")
    test = load_test(test_path)
    test_ids = test["id"].to_numpy()
    sub_path = out_dir / "submission.csv"
    write_csv_atomic(prior_submission(test_ids, class_prior(train_path)), sub_path)
    print(f"wrote the prior-filled fallback {sub_path} ({len(test)} rows)", flush=True)
    lap("locate_and_prior")

    df = load_train(train_path)
    manifest = build_manifest(df)
    if not skip_sha:
        digest = manifest_sha256(manifest)
        if digest != MANIFEST_SHA256:
            raise SystemExit(f"split manifest digest {digest} != committed {MANIFEST_SHA256}")
    cv = select_cv(df, manifest)
    if len(cv) == 0 or cv["label"].isna().any():
        raise SystemExit("no usable cv rows (empty, or rows without exactly one winner)")
    print(f"train rows {len(df)}, cv rows used {len(cv)}; test split rows not touched", flush=True)
    lap("load_and_verify")

    cols = feature_columns("full")
    X_cv = build_features(cv)[cols]
    y = labels_of(cv)
    lap("features_train")
    X_test = build_features(test)[cols]
    lap("features_test")

    booster = fit_final(X_cv, y, rounds)
    lap("fit")

    P = predict_test(booster, X_test)
    frame = submission_frame(X_test.index.to_numpy(), P, test_ids)
    write_csv_atomic(frame, sub_path)
    lap("predict_and_write")

    log = {
        "n_cv_rows": len(cv),
        "n_train_rows": int(2 * len(cv)),
        "n_test_rows": len(test),
        "rounds": rounds,
        "seconds": {**secs, "total": round(time.perf_counter() - t_start, 3)},
        "versions": {
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "source_commit": build["source_commit"],
        "module_sha256": build["module_sha256"],
        "skip_sha_checks": skip_sha,
        "inlined": bool(build.get("inlined", False)),
        "features_module_file": sys.modules[build_features.__module__].__file__,
    }
    (out_dir / "run_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"done: {sub_path} rows={len(frame)} rounds={rounds}", flush=True)
