"""Rung-6 spend guard (torch-free). Refuses a Modal launch that would break a pre-registered gate.

Sources: results/ledger.jsonl (every recorded run) and data/lora_out/spend.jsonl (one `launch`
line with the estimate BEFORE each launch, one `reconcile` line with the real cost after).

A launch is refused when
  (a) it is a probe whose estimate is over $5 (G0);
  (b) cumulative rung-6 spend after it would pass the user-approved amount (default $15 = G0+G1);
  (c) cumulative GPU spend across the ledger and the spend file would pass $85;
  (d) it is a fold 1-4 (or the all-cv `full` adapter) launch and data/lora_out/g1_decision.json
      does not say {"pass": true, "fold0_log_loss": x} with x <= 1.00.
When G1 passes, folds 1-4 and `full` are auto-approved up to the probe's projected cost plus a 20%
margin on top of the approved amount, still under the $85 ceiling.

  python lora/spend_guard.py check --mode fold --folds 0 --estimate-usd 8.3
  python lora/spend_guard.py g1 --oof data/lora_out/g1/oof_fold0.parquet --probe data/lora_out/probe/probe.json
  python lora/spend_guard.py reconcile --job-id JOB --from-logs data/lora_out/g1
  python lora/spend_guard.py status
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

LEDGER = Path("results/ledger.jsonl")
SPEND = Path("data/lora_out/spend.jsonl")
G1_FILE = Path("data/lora_out/g1_decision.json")
PROBE_CAP_USD = 5.0
DEFAULT_APPROVED_USD = 15.0
CEILING_USD = 85.0
G1_MAX_LOG_LOSS = 1.00
AUTO_MARGIN = 0.20
RECONCILE_OVERHEAD = 0.15  # container boot / image pull not in est_cost_usd; matches lora_cost
RUNG6_PREFIX = "lora"  # ledger `config` names for rung 6 start with this


class SpendRefused(Exception):
    pass


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def is_rung6(row: dict) -> bool:
    return str(row.get("config", "")).startswith(RUNG6_PREFIX)


def ledger_spend(rows: Iterable[dict]) -> tuple[float, float]:
    """(non-rung-6 GPU spend, rung-6 spend) from ledger rows. GPU rows carry `gpu_hours`."""
    other = rung6 = 0.0
    for r in rows:
        cost = float(r.get("cost_usd") or 0)
        if is_rung6(r):
            rung6 += cost
        elif r.get("gpu_hours") is not None:
            other += cost
    return other, rung6


def spend_file_total(lines: Iterable[dict]) -> float:
    """Sum over jobs of the reconciled actual, else the launch estimate."""
    est: dict[str, float] = {}
    act: dict[str, float] = {}
    for ln in lines:
        jid = ln["job_id"]
        if ln.get("event") == "launch":
            est[jid] = float(ln["estimate_usd"])
        elif ln.get("event") == "reconcile":
            act[jid] = float(ln["actual_usd"])
    return sum(act.get(j, e) for j, e in est.items()) + sum(
        a for j, a in act.items() if j not in est
    )


def totals(ledger: Path = LEDGER, spend: Path = SPEND) -> dict:
    other, led6 = ledger_spend(read_jsonl(ledger))
    file6 = spend_file_total(read_jsonl(spend))
    rung6 = max(led6, file6)  # a ledgered rung-6 run is also in the spend file: never add both
    return {"other_gpu_usd": other, "rung6_usd": rung6, "gpu_total_usd": other + rung6}


def g1_check(
    oof: pd.DataFrame,
    y_by_id: dict | pd.Series,
    projected_remaining_usd: float | None = None,
    expect_ids: Iterable[int] | None = None,
) -> dict:
    """Fold-0 decision: swap-averaged raw log loss of the OOF against the labels.

    `pass` is log loss <= 1.00 (inclusive). Refuses duplicate/foreign/missing ids and
    non-finite or non-normalised probabilities.
    """
    from llm_preference_audit.scoring import log_loss

    ids = oof["id"].astype(int)
    if ids.duplicated().any():
        raise ValueError("oof has duplicate ids")
    y_map = dict(y_by_id.items()) if hasattr(y_by_id, "items") else dict(y_by_id)
    missing = [i for i in ids if i not in y_map]
    if missing:
        raise ValueError(f"{len(missing)} oof ids have no label")
    if expect_ids is not None and set(map(int, expect_ids)) != set(ids):
        raise ValueError("oof ids differ from fold 0's ids")
    P = oof[["p_a", "p_b", "p_tie"]].to_numpy(dtype=float)
    if not np.isfinite(P).all() or (np.abs(P.sum(axis=1) - 1) > 1e-5).any():
        raise ValueError("oof probabilities are non-finite or do not sum to 1")
    y = np.array([int(y_map[i]) for i in ids])
    ll = float(log_loss(y, P))
    out = {"pass": bool(ll <= G1_MAX_LOG_LOSS), "fold0_log_loss": ll, "threshold": G1_MAX_LOG_LOSS}
    out["n_rows"] = len(oof)
    if projected_remaining_usd is not None:
        out["projected_remaining_usd"] = float(projected_remaining_usd)
    return out


def read_g1(path: Path = G1_FILE) -> dict | None:
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def g1_passed(decision: dict | None) -> bool:
    """The file's own number is re-checked: a hand-edited `pass: true` over a bad loss fails."""
    if not decision or decision.get("pass") is not True:
        return False
    x = decision.get("fold0_log_loss")
    return isinstance(x, int | float) and bool(np.isfinite(x)) and x <= G1_MAX_LOG_LOSS


def needs_g1(mode: str, folds: Iterable[int]) -> bool:
    return mode == "full" or (mode == "fold" and any(int(f) != 0 for f in folds))


def check_launch(
    mode: str,
    folds: Iterable[int],
    estimate_usd: float,
    *,
    approved_usd: float = DEFAULT_APPROVED_USD,
    ledger: Path = LEDGER,
    spend: Path = SPEND,
    g1_file: Path = G1_FILE,
) -> dict:
    """Return the accounting for a permitted launch; raise SpendRefused (all reasons) otherwise."""
    folds = list(folds)
    if mode not in ("probe", "fold", "full"):
        raise SpendRefused(f"unknown mode {mode!r}")
    tot = totals(ledger, spend)
    reasons = []
    if estimate_usd < 0 or not np.isfinite(estimate_usd):
        reasons.append(f"estimate {estimate_usd} is not a non-negative number")
    if mode == "probe" and estimate_usd > PROBE_CAP_USD:
        reasons.append(
            f"G0: probe estimate ${estimate_usd:.2f} is over the ${PROBE_CAP_USD:.0f} cap"
        )
    g1 = read_g1(g1_file)
    cap = approved_usd
    if needs_g1(mode, folds):
        if not g1_passed(g1):
            reasons.append(
                f"G1: {g1_file} must say pass=true with fold0_log_loss <= {G1_MAX_LOG_LOSS:.2f} "
                "before folds 1-4 or the full adapter"
            )
        elif g1.get("projected_remaining_usd") is not None:
            cap = approved_usd + (1 + AUTO_MARGIN) * float(g1["projected_remaining_usd"])
    after6 = tot["rung6_usd"] + estimate_usd
    if after6 > cap + 1e-9:
        reasons.append(
            f"approved amount: rung-6 spend would be ${after6:.2f} (> ${cap:.2f} approved)"
        )
    after = tot["gpu_total_usd"] + estimate_usd
    if after > CEILING_USD + 1e-9:
        reasons.append(
            f"ceiling: cumulative GPU spend would be ${after:.2f} (> ${CEILING_USD:.0f})"
        )
    if reasons:
        raise SpendRefused("refusing launch: " + "; ".join(reasons))
    return tot | {"estimate_usd": estimate_usd, "rung6_after_usd": after6, "approved_cap_usd": cap}


def _append(path: Path, line: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(line) + "\n")


def record_launch(
    job_id: str, mode: str, tag: str, folds: list[int], estimate_usd: float, spend: Path = SPEND
) -> None:
    """Append the estimate BEFORE the launch (so a crash mid-launch still counts)."""
    _append(
        spend,
        {
            "event": "launch",
            "job_id": job_id,
            "ts": time.time(),
            "mode": mode,
            "tag": tag,
            "folds": folds,
            "estimate_usd": float(estimate_usd),
        },
    )


def reconcile(job_id: str, actual_usd: float, spend: Path = SPEND, note: str = "") -> None:
    if job_id not in {ln["job_id"] for ln in read_jsonl(spend) if ln.get("event") == "launch"}:
        raise ValueError(f"no launch recorded for job {job_id!r}")
    _append(
        spend,
        {
            "event": "reconcile",
            "job_id": job_id,
            "ts": time.time(),
            "actual_usd": float(actual_usd),
            "note": note,
        },
    )


def cost_from_logs(out_dir: Path) -> float:
    """Sum est_cost_usd over probe/log files in an output dir, plus the unlogged boot overhead."""
    out = Path(out_dir)
    files = [*out.glob("log_fold*.json"), *out.glob("log_full.json"), *out.glob("probe.json")]
    if not files:
        raise ValueError(f"no probe/log files under {out}")
    raw = 0.0
    for f in files:
        d = json.loads(f.read_text())
        if "est_cost_usd" not in d:
            raise ValueError(f"{f.name} has no est_cost_usd")
        raw += float(d["est_cost_usd"])
    return raw * (1 + RECONCILE_OVERHEAD)


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--mode", required=True, choices=["probe", "fold", "full"])
    c.add_argument("--folds", default="0")
    c.add_argument("--estimate-usd", type=float, required=True)
    c.add_argument("--approved-usd", type=float, default=DEFAULT_APPROVED_USD)
    g = sub.add_parser("g1")
    g.add_argument("--oof", type=Path, required=True)
    g.add_argument("--probe", type=Path, default=None, help="probe.json for the projection")
    g.add_argument("--raw", type=Path, default=Path("data/raw/train.csv"))
    g.add_argument("--out", type=Path, default=G1_FILE)
    r = sub.add_parser("reconcile")
    r.add_argument("--job-id", required=True)
    r.add_argument("--from-logs", type=Path, default=None)
    r.add_argument("--actual-usd", type=float, default=None)
    sub.add_parser("status")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        folds = [int(f) for f in a.folds.split(",") if f != ""]
        try:
            print(
                json.dumps(check_launch(a.mode, folds, a.estimate_usd, approved_usd=a.approved_usd))
            )
        except SpendRefused as e:
            print(str(e), file=sys.stderr)
            return 2
    elif a.cmd == "g1":
        from llm_preference_audit.data import LABELS, load_train
        from llm_preference_audit.split import build_manifest

        df = load_train(a.raw)
        manifest = build_manifest(df)
        f0 = manifest[(manifest["split"] == "cv") & (manifest["fold"] == 0)]
        lab = df.set_index("id")["label"].astype(str)
        y = {int(i): LABELS.index(lab[i]) for i in f0["id"]}
        proj = None
        if a.probe:
            p = json.loads(a.probe.read_text()).get("projection", {})
            proj = p.get("folds_1_to_4_plus_full_usd")
        dec = g1_check(pd.read_parquet(a.oof), y, proj, expect_ids=f0["id"])
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dec, indent=2) + "\n")
        print(json.dumps(dec, indent=2))
    elif a.cmd == "reconcile":
        actual = a.actual_usd if a.actual_usd is not None else cost_from_logs(a.from_logs)
        reconcile(a.job_id, actual, note="from logs" if a.actual_usd is None else "manual")
        print(f"reconciled {a.job_id}: ${actual:.2f}")
    else:
        print(json.dumps(totals() | {"g1": read_g1()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
