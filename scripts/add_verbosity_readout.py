"""Recompute the verbosity read-outs for the current runs from their out-of-fold parquets.

No retraining: reads results/rung-2-summary.json and the parquets it names.
"""

import json
from pathlib import Path

import pandas as pd

from llm_preference_audit.bias import calibrated_longer_readout, longer_wins_readout
from llm_preference_audit.data import LABELS, load_train
from llm_preference_audit.split import build_manifest
from llm_preference_audit.train_lgbm import assert_cv_only

SUMMARY = Path("results/rung-2-summary.json")
PRED_DIR = Path("data/predictions")


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    df = load_train("data/raw/train.csv")
    manifest = build_manifest(df)
    m = manifest.set_index("id")
    meta = df.set_index("id")
    for name, cfg in summary["configs"].items():
        pr = pd.read_parquet(PRED_DIR / f"{cfg['run_id']}.parquet")
        assert_cv_only(pr["id"].to_numpy(), manifest)
        ids = pr["id"].to_numpy()
        y = meta.loc[ids, "label"].astype(str).map({c: i for i, c in enumerate(LABELS)}).to_numpy()
        P = pr[["p_a", "p_b", "p_tie"]].to_numpy()
        wa = meta.loc[ids, "response_a"].map(lambda t: len("\n".join(t).split())).to_numpy()
        wb = meta.loc[ids, "response_b"].map(lambda t: len("\n".join(t).split())).to_numpy()
        groups = m.loc[ids, "group"].to_numpy()
        cfg.pop("longer_wins", None)
        cfg["direction_agreement"] = longer_wins_readout(y, P, wa, wb)
        cfg["verbosity_calibrated"] = calibrated_longer_readout(y, P, wa, wb, groups)
        v = cfg["verbosity_calibrated"]
        print(
            f"{name}: human {v['human_share']:.4f} model {v['model_share']:.4f} "
            f"diff {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] "
            f"agreement {cfg['direction_agreement']['direction_agreement']:.4f}"
        )
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
