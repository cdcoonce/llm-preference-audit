"""Score any out-of-fold prediction file over the cv split (rungs 2-5 share this scorer).

Input parquet, one row per cv id: `id`; `p_a, p_b, p_tie` (final probabilities, original a/b
order); optional `raw_ab_{a,b,tie}` and `raw_ba_{a,b,tie}` (each order's probabilities in
PRESENTED order, column a = shown first); optional bool `both_untruncated`, `hard_cut`.
Haiku-style files (`winner_model_a/b`, `winner_tie`) are read as p_a/p_b/p_tie.

Writes aggregates only (never dataset text). With --ledger, appends one ledger row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from llm_preference_audit.bias import (
    calibrated_longer_readout,
    first_position_readout,
    swap_flip_readout,
)
from llm_preference_audit.calibrate import MatrixScaler, TemperatureScaler, cross_fit
from llm_preference_audit.data import LABELS, conversation_text, load_train
from llm_preference_audit.scoring import group_bootstrap_diff, log_loss, row_losses
from llm_preference_audit.split import SHA_PATH, build_manifest, manifest_sha256

RAW = Path("data/raw/train.csv")
LEDGER = Path("results/ledger.jsonl")
CODE_DIRS = ("llm_preference_audit", "scripts", "kaggle")
PROB_COLS = ["p_a", "p_b", "p_tie"]
HAIKU_COLS = {"winner_model_a": "p_a", "winner_model_b": "p_b", "winner_tie": "p_tie"}
SUM_TOL = 1e-6


def _probs(df: pd.DataFrame, cols: list[str], what: str) -> np.ndarray:
    P = df[cols].to_numpy(dtype=float)
    ok = ~np.isnan(P).any(axis=1)  # NaN rows in raw_* are allowed for failed calls
    if what == "p" and not ok.all():
        raise ValueError(f"{what}: {int((~ok).sum())} rows with non-finite probabilities")
    Q = P[ok]
    if not np.isfinite(Q).all() or (Q < 0).any():
        raise ValueError(f"{what}: probabilities must be finite and non-negative")
    bad = int((np.abs(Q.sum(axis=1) - 1) > SUM_TOL).sum())
    if bad:
        raise ValueError(f"{what}: {bad} rows do not sum to 1 within {SUM_TOL}")
    return P


def validate(oof: pd.DataFrame, rows: pd.DataFrame) -> None:
    dup = int(oof["id"].duplicated().sum())
    if dup:
        raise ValueError(f"oof has {dup} duplicate ids")
    o, r = set(oof["id"]), set(rows["id"])
    if o != r:
        raise ValueError(f"oof ids differ from cv rows: {len(r - o)} missing, {len(o - r)} extra")
    not_cv = int((rows["split"] != "cv").sum())
    if not_cv:
        raise ValueError(f"{not_cv} rows are not in the cv split")


def _readouts(y, P, wa, wb, groups, mask) -> dict:
    return calibrated_longer_readout(y[mask], P[mask], wa[mask], wb[mask], groups[mask])


def score_oof(
    oof: pd.DataFrame,
    rows: pd.DataFrame,
    prior: np.ndarray,
    comparators: dict[str, np.ndarray],
) -> tuple[dict, np.ndarray]:
    """Pure scoring. rows: id, y, fold, group, split, words_a, words_b. comparators and prior
    are (n,3) arrays aligned to `rows`. Returns (summary, primary calibrated P)."""
    validate(oof, rows)
    oof = oof.set_index("id").loc[rows["id"]].reset_index()
    P_raw = _probs(oof, PROB_COLS, "p")
    y = rows["y"].to_numpy(dtype=int)
    folds = rows["fold"].to_numpy(dtype=int)
    groups = rows["group"].to_numpy()
    splits = rows["split"].to_numpy()
    wa = rows["words_a"].to_numpy(dtype=float)
    wb = rows["words_b"].to_numpy(dtype=float)

    P_temp = cross_fit(P_raw, y, folds, splits, TemperatureScaler)
    P_cal = cross_fit(P_raw, y, folds, splits, MatrixScaler)
    loss_cal = row_losses(y, P_cal)

    def vs(P_other: np.ndarray) -> dict:
        return group_bootstrap_diff(row_losses(y, P_other), loss_cal, groups)

    summary: dict = {
        "n_rows": len(rows),
        "log_loss": {
            "prior_oof": log_loss(y, prior),
            "raw": log_loss(y, P_raw),
            "temperature_cv": log_loss(y, P_temp),
            "matrix_cv": float(loss_cal.mean()),
        },
        "gain_vs_prior": vs(prior),
        "comparators": {},
    }
    summary["gate_pass"] = bool(summary["gain_vs_prior"]["lo"] > 0)

    has_bu = "both_untruncated" in oof.columns
    bu = oof["both_untruncated"].to_numpy(dtype=bool) if has_bu else None
    masks = {"all": np.ones(len(rows), dtype=bool)}
    if has_bu:
        masks["both_untruncated"] = bu
        masks["truncated_any"] = ~bu

    counts: dict = {"n_rows": len(rows)}
    if "hard_cut" in oof.columns:
        counts["hard_cut_share"] = float(oof["hard_cut"].to_numpy(dtype=bool).mean())
    if has_bu:
        counts["both_untruncated_share"] = float(bu.mean())
    summary["counts"] = counts

    raw_cols = [f"raw_{o}_{c}" for o in ("ab", "ba") for c in ("a", "b", "tie")]
    if all(c in oof.columns for c in raw_cols):
        raw_ab = _probs(oof, ["raw_ab_a", "raw_ab_b", "raw_ab_tie"], "raw_ab")
        raw_ba = _probs(oof, ["raw_ba_a", "raw_ba_b", "raw_ba_tie"], "raw_ba")
        summary["position"] = {
            "swap_flip": swap_flip_readout(raw_ab, raw_ba[:, [1, 0, 2]]),
            "first_position": first_position_readout(raw_ab, raw_ba, groups),
        }

    models = {"primary": P_cal, **comparators}
    summary["verbosity"] = {
        subset: {
            "n_rows": int(m.sum()),
            **{name: _readouts(y, P, wa, wb, groups, m) for name, P in models.items()},
        }
        for subset, m in masks.items()
    }

    d_self = P_cal[:, 0] - P_cal[:, 1]
    for name, Pc in comparators.items():
        d_c = Pc[:, 0] - Pc[:, 1]
        corr = float(np.corrcoef(d_self, d_c)[0, 1]) if d_self.std() and d_c.std() else None
        summary["comparators"][name] = {
            "log_loss": log_loss(y, Pc),
            "gain_vs_comparator": vs(Pc),
            "pa_minus_pb_correlation": corr,
        }
    return summary, P_cal


# ---------------------------------------------------------------------------------- CLI


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def require_clean() -> str:
    dirty = git("status", "--porcelain", "--", *CODE_DIRS)
    if dirty:
        raise SystemExit(f"uncommitted code; commit before running:\n{dirty}")
    return git("rev-parse", "HEAD")


def ledger_rows() -> list[dict]:
    return [json.loads(x) for x in LEDGER.read_text().splitlines()] if LEDGER.exists() else []


def read_predictions(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.rename(columns=HAIKU_COLS) if "p_a" not in df.columns else df


def prior_by_fold(rows: pd.DataFrame) -> np.ndarray:
    """Per row, label frequencies of the cv rows in the other folds."""
    y, f = rows["y"].to_numpy(dtype=int), rows["fold"].to_numpy(dtype=int)
    P = np.zeros((len(rows), 3))
    for k in sorted(set(f.tolist())):
        P[f == k] = np.bincount(y[f != k], minlength=3) / (f != k).sum()
    return P


def load_cv_rows() -> pd.DataFrame:
    df = load_train(RAW)
    manifest = build_manifest(df)
    if manifest_sha256(manifest) != SHA_PATH.read_text().split()[0]:
        raise SystemExit("split manifest does not match the committed digest")
    m = manifest.set_index("id")
    df = df.assign(
        split=m.loc[df["id"], "split"].to_numpy(),
        fold=m.loc[df["id"], "fold"].to_numpy(),
        group=m.loc[df["id"], "group"].to_numpy(),
    )
    cv = df[(df["split"] == "cv") & df["label"].notna()].sort_values("id").reset_index(drop=True)
    cv["y"] = cv["label"].astype(str).map({c: i for i, c in enumerate(LABELS)}).astype(int)
    for col, short in (("response_a", "a"), ("response_b", "b")):
        cv[f"words_{short}"] = [len(conversation_text(t).split()) for t in cv[col]]
    return cv[["id", "y", "fold", "group", "split", "words_a", "words_b"]]


def load_comparators(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    specs = {
        "lgbm_full": lambda r: r.get("config") == "full" and r.get("model") == "lightgbm",
        "haiku_full": lambda r: r.get("config") == "haiku_full_cv",
    }
    out: dict[str, np.ndarray] = {}
    for name, match in specs.items():
        hits = [r for r in ledger_rows() if match(r)]
        if not hits or not Path(hits[-1]["predictions_path"]).exists():
            continue
        df = read_predictions(hits[-1]["predictions_path"]).set_index("id")
        if not set(rows["id"]) <= set(df.index):
            continue
        out[name] = df.loc[rows["id"], PROB_COLS].to_numpy(dtype=float)
    return out


def write_ledger_row(args, sha: str, summary: dict, pred_path: Path) -> bool:
    digest = sha256_file(pred_path)
    if any(r.get("predictions_sha256") == digest for r in ledger_rows()):
        print(f"ledger already has predictions {digest[:12]}; nothing appended")
        return False
    cfg = Path(args.config_json).read_bytes() if args.config_json else b"{}"
    cfg_obj = json.loads(cfg)
    ts = datetime.now(UTC)
    row = {
        "run_id": f"{ts.strftime('%Y%m%dT%H%M%S')}-{args.name}",
        "timestamp_utc": ts.isoformat(timespec="seconds"),
        "git_sha": sha,
        "config_hash": hashlib.sha256(json.dumps(cfg_obj, sort_keys=True).encode()).hexdigest(),
        "model": args.model,
        "config": args.name,
        "split": "cv-oof",
        "log_loss": summary["log_loss"]["matrix_cv"],
        "cost_usd": args.cost_usd,
        "gpu_hours": args.gpu_hours,
        "predictions_path": str(pred_path),
        "predictions_sha256": digest,
        "n_rows": summary["n_rows"],
        "ci95_vs_prior": [summary["gain_vs_prior"]["lo"], summary["gain_vs_prior"]["hi"]],
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"ledger row {row['run_id']}")
    return True


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--oof", required=True, type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--gpu-hours", type=float, default=None)
    p.add_argument("--cost-usd", type=float, default=0, help="real GPU cost for the ledger row")
    p.add_argument("--config-json", type=Path, default=None)
    p.add_argument("--summary", type=Path, default=None)
    p.add_argument("--ledger", action="store_true")
    args = p.parse_args(argv)

    sha = require_clean() if args.ledger else None
    rows = load_cv_rows()
    oof = read_predictions(args.oof)
    summary, _ = score_oof(oof, rows, prior_by_fold(rows), load_comparators(rows))
    summary.update(name=args.name, model=args.model, gpu_hours=args.gpu_hours)
    text = json.dumps(summary, indent=2) + "\n"
    if args.summary:
        args.summary.write_text(text)
    print(json.dumps({k: summary[k] for k in ("log_loss", "gain_vs_prior", "gate_pass")}, indent=2))
    if args.ledger:
        write_ledger_row(args, sha, summary, args.oof)


if __name__ == "__main__":
    main(sys.argv[1:])
