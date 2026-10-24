"""Compare the head+tail ablation to the main run on one fold (default 0).

  uv run python kaggle/rung5_deberta/compare_ablation.py \
      --main data/predictions/deberta_base_t1024.parquet --ablation data/kaggle_out/abl0/oof_fold0.parquet

Fold-k rows only (score_oof.py needs all cv rows, so it cannot score a one-fold ablation). Prints
log loss for both, their paired group-bootstrap difference (positive = main better), and the
calibrated verbosity readout of each. Aggregates only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from llm_preference_audit.bias import calibrated_longer_readout
from llm_preference_audit.scoring import group_bootstrap_diff, row_losses

REPO_ROOT = Path(__file__).resolve().parents[2]
PROB = ["p_a", "p_b", "p_tie"]


def compare_fold(
    main: pd.DataFrame, abl: pd.DataFrame, rows: pd.DataFrame, fold: int, n_boot: int = 2000
) -> dict:
    """`rows`: cv rows with id, y, fold, group, words_a, words_b. main/abl: OOF frames with PROB."""
    r = rows[rows["fold"] == fold].sort_values("id").reset_index(drop=True)
    ids = set(r["id"])
    if set(abl["id"]) != ids:
        raise ValueError("ablation ids are not exactly the fold's cv ids")
    if not ids <= set(main["id"]):
        raise ValueError("main run lacks some of the fold's ids")
    pm = main.set_index("id").loc[r["id"], PROB].to_numpy()
    pa = abl.set_index("id").loc[r["id"], PROB].to_numpy()
    y, g = r["y"].to_numpy(dtype=int), r["group"].to_numpy()
    lm, la = row_losses(y, pm), row_losses(y, pa)
    wa, wb = r["words_a"].to_numpy(), r["words_b"].to_numpy()

    def readout(P):
        d = calibrated_longer_readout(y, P, wa, wb, g)
        return {k: v for k, v in d.items() if k != "buckets"}

    return {
        "fold": fold,
        "n": len(r),
        "log_loss_main": float(lm.mean()),
        "log_loss_ablation": float(la.mean()),
        "main_better_by": group_bootstrap_diff(la, lm, g, n_boot=n_boot),
        "verbosity_main": readout(pm),
        "verbosity_ablation": readout(pa),
    }


def main(argv=None) -> int:
    spec = importlib.util.spec_from_file_location("score_oof", REPO_ROOT / "scripts/score_oof.py")
    score_oof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score_oof)  # reuse its manifest-checked cv rows (words, folds, groups)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--ablation", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args(argv)
    out = compare_fold(
        pd.read_parquet(a.main), pd.read_parquet(a.ablation), score_oof.load_cv_rows(), a.fold
    )
    print(
        json.dumps(
            out, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
