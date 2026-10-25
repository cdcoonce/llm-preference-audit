"""Merge per-fold OOF files from one or more session dirs into one prediction parquet.

  uv run python kaggle/rung5_deberta/merge_oof.py \
      --in data/kaggle_out/f01 --in data/kaggle_out/f23 --in data/kaggle_out/f4 --tag deberta_base_t1024

Every oof_fold{k}.parquet needs a log_fold{k}.json with no error; fold-k ids must be exactly the
manifest's fold-k cv ids (rebuilt from data/raw/train.csv); all logs must share git_sha, t,
middle_share, model and epochs (so an ablation fold can never join the main run); folds 0-4 each
once. Writes data/predictions/<tag>.parquet and <tag>.run.json (no text columns).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from llm_preference_audit.data import load_train
from llm_preference_audit.deberta_inputs import validate_fold_parts
from llm_preference_audit.split import SHA_PATH, build_manifest, manifest_sha256

RAW = Path("data/raw/train.csv")
PRED_DIR = Path("data/predictions")


def collect_parts(dirs: list[Path]) -> list[tuple[int, pd.DataFrame, dict]]:
    parts = []
    for d in dirs:
        for f in sorted(d.glob("oof_fold*.parquet")):
            m = re.fullmatch(r"oof_fold(\d+)\.parquet", f.name)
            if not m:
                raise SystemExit(f"unexpected file name {f}")
            fold = int(m.group(1))
            log_path = d / f"log_fold{fold}.json"
            if not log_path.exists():
                raise SystemExit(f"{f} has no matching {log_path.name}")
            parts.append((fold, pd.read_parquet(f), json.loads(log_path.read_text())))
    if not parts:
        raise SystemExit(f"no oof_fold*.parquet in {', '.join(map(str, dirs))}")
    return parts


def gpu_seconds(log: dict) -> float:
    if "wall_seconds" in log:  # rung 6 logs the whole container-side wall time
        return float(log["wall_seconds"])
    return float(log.get("train_stats", {}).get("train_seconds", 0.0)) + float(
        log.get("predict_seconds", 0.0)
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="src", type=Path, action="append", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--raw", type=Path, default=RAW)
    a = ap.parse_args(argv)
    parts = collect_parts(a.src)
    manifest = build_manifest(load_train(a.raw))
    if manifest_sha256(manifest) != SHA_PATH.read_text().split()[0]:
        raise SystemExit("rebuilt split manifest does not match splits/split_manifest.sha256")
    cv = manifest[manifest["split"] == "cv"]
    try:
        merged = validate_fold_parts(parts, cv)
    except ValueError as e:
        raise SystemExit(f"refusing to merge: {e}") from e
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = PRED_DIR / f"{a.tag}.parquet"
    merged.to_parquet(out, index=False)
    logs = [log for _, _, log in parts]
    run = {
        "tag": a.tag,
        "folds": sorted(f for f, _, _ in parts),
        "session_tags": sorted({str(log.get("tag", "")) for log in logs}),
        "git_sha": logs[0].get("git_sha"),
        **{k: logs[0].get(k) for k in ("t", "middle_share", "model", "epochs")},
        "n_rows": len(merged),
        "gpu_seconds": sum(gpu_seconds(log) for log in logs),
        "gpu_hours_estimate": sum(gpu_seconds(log) for log in logs) / 3600,
        "est_cost_usd": sum(float(log.get("est_cost_usd", 0)) for log in logs),
    }
    (PRED_DIR / f"{a.tag}.run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"merged folds {run['folds']} -> {out} ({len(merged)} cv rows, each id exactly once)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
