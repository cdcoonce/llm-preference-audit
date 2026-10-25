"""Rung-6 spend guard (torch-free). Refuses a Modal launch that would break a pre-registered gate.

Sources: results/ledger.jsonl (every recorded run) and data/lora_out/spend.jsonl (an append-only
`launch` line with the estimate AND the worst case BEFORE each launch, `reconcile` lines after).

A fold or full launch is refused unless
  G0  a real probe exists for the SAME git sha (tiny=false, model google/gemma-2-9b-it, T=2048)
      with a finite loss, no non-finite-loss note, an attention decision and a projected total
      (5 folds + full + 15% overhead) <= $70;
  G1  (folds 1-4 and `full` only) data/lora_out/g1_decision.json says pass=true, was made on the
      same git sha as the launch, and is bound to its OOF file (sha256) and labels (train.csv
      sha256); the loss is RECOMPUTED from that OOF file, never trusted from the decision.
A launch is also refused when
  (a) it is a probe whose estimate is over $5;
  (b) cumulative rung-6 spend after it would pass the approved amount (default $15 = G0+G1; after
      G1 passes, plus 1.2 x the probe's folds-1-4-plus-full projection);
  (c) cumulative GPU spend across the ledger and the spend file would pass $85;
  (d) its budget-derived time limit is shorter than the job needs (see deadline_plan).

A launch is charged at max(estimate, worst case) until reconciled. The automatic reconcile never
goes below what could have been billed (unlogged jobs are charged their full worst case); only a
manual `reconcile --actual-usd X --source modal-dashboard` can lower it, and it records the
previous value. The spend file is append-only.

  python lora/spend_guard.py check --mode fold --folds 0 --estimate-usd 8.3 --git-sha SHA
  python lora/spend_guard.py g1 --oof data/lora_out/g1/oof_fold0.parquet
  python lora/spend_guard.py reconcile --job-id JOB --from-logs data/lora_out/g1
  python lora/spend_guard.py reconcile --job-id JOB --actual-usd 6.5 --source modal-dashboard
  python lora/spend_guard.py status
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from llm_preference_audit.lora_cost import G0_TOTAL_CAP_USD, OVERHEAD, PRICE_PER_HOUR

LEDGER = Path("results/ledger.jsonl")
OUT_ROOT = Path("data/lora_out")
SPEND = OUT_ROOT / "spend.jsonl"
G1_FILE = OUT_ROOT / "g1_decision.json"
LOCK_FILE = OUT_ROOT / "launch.lock"
PROBE_CAP_USD = 5.0
DEFAULT_APPROVED_USD = 15.0
CEILING_USD = 85.0
G1_MAX_LOG_LOSS = 1.00
AUTO_MARGIN = 0.20
RECONCILE_OVERHEAD = 0.15  # container boot / image pull not in est_cost_usd; matches lora_cost
RUNG6_PREFIX = "lora"  # ledger `config` names for rung 6 start with this
REAL_MODEL = "google/gemma-2-9b-it"
REAL_T = 2048
MAX_JOB_S = 6 * 3600  # longest any one job may run
SAFETY_MARGIN_S = 600  # boot / teardown slack added to the deadline to get the Modal timeout
MIN_DEADLINE_S = 1200  # used as the job's need when there is no probe projection
MANUAL_SOURCE = "modal-dashboard"
# CPU-only work (the weight prefetch). Modal lists $0.0000131/core-s and $0.00000222/GiB-s
# (modal.com/pricing, fetched 2026-10-25): 2 cores + 8 GiB = $0.16/h; $0.20/h is the rounded-up cap.
CPU_PRICE_PER_HOUR = 0.20
PREFETCH_TIMEOUT_S = 1800


class SpendRefused(Exception):
    pass


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def is_rung6(row: dict) -> bool:
    return str(row.get("config", "")).startswith(RUNG6_PREFIX)


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def job_states(lines: Iterable[dict]) -> dict[str, dict]:
    """Per job: its launch line, the current reconciled figure (last reconcile) and the charge.

    Until reconciled a launch is charged at max(estimate, worst case): what the job could have
    cost if it ran to its time limit.
    """
    jobs: dict[str, dict] = {}
    orphans: dict[str, float] = {}
    for ln in lines:
        jid = ln["job_id"]
        if ln.get("event") == "launch":
            jobs[jid] = {"launch": ln, "actual": None}
        elif ln.get("event") == "reconcile":
            if jid in jobs:
                jobs[jid]["actual"] = float(ln["actual_usd"])
            else:
                orphans[jid] = float(ln["actual_usd"])
    for jid, st in jobs.items():
        ln = st["launch"]
        unreconciled = max(float(ln["estimate_usd"]), float(ln.get("worst_case_usd") or 0))
        st["charge"] = st["actual"] if st["actual"] is not None else unreconciled
    for jid, a in orphans.items():
        jobs[jid] = {"launch": None, "actual": a, "charge": a}
    return jobs


def spend_file_total(lines: Iterable[dict]) -> float:
    """Sum over jobs of the reconciled figure, else max(estimate, worst case)."""
    return sum(st["charge"] for st in job_states(lines).values())


def totals(ledger: Path = LEDGER, spend: Path = SPEND) -> dict:
    other, led6 = ledger_spend(read_jsonl(ledger))
    file6 = spend_file_total(read_jsonl(spend))
    rung6 = max(led6, file6)  # a ledgered rung-6 run is also in the spend file: never add both
    return {"other_gpu_usd": other, "rung6_usd": rung6, "gpu_total_usd": other + rung6}


# ---------------------------------------------------------------- G0 (the probe)


def select_probe(root: Path, git_sha: str) -> dict | None:
    """The real probe for `git_sha`: not tiny, Gemma-2-9B-it, T=2048. Newest `finished_at` wins.

    Deliberately not by file mtime (a copied, touched or tiny dry-run probe must not shadow it)
    and not by tag.
    """
    best: tuple[float, str] | None = None
    chosen = None
    for f in sorted(Path(root).glob("*/probe.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if not (
            d.get("git_sha") == git_sha
            and d.get("tiny") is False
            and d.get("model") == REAL_MODEL
            and d.get("t") == REAL_T
        ):
            continue
        key = (float(d.get("finished_at") or 0), str(f))
        if best is None or key > best:
            best, chosen = key, d | {"_path": str(f)}
    return chosen


def g0_problems(probe: dict) -> list[str]:
    out = []
    if probe.get("error"):
        out.append(f"probe recorded error {probe['error']!r}")
    if probe.get("complete") is not True:
        out.append("probe is not complete (killed or partial)")
    if probe.get("loss_finite") is not True:
        out.append("probe loss is not finite")
    if probe.get("non_finite_loss") is not None:
        out.append(f"probe recorded a non-finite loss: {probe.get('non_finite_loss')}")
    if probe.get("attention_decision") not in ("eager", "sdpa"):
        out.append("probe has no attention decision")
    total = (probe.get("projection") or {}).get("total_usd_with_overhead")
    if not isinstance(total, int | float) or not np.isfinite(total) or total > G0_TOTAL_CAP_USD:
        out.append(
            f"probe projected total {total!r} is missing or over the ${G0_TOTAL_CAP_USD:.0f} cap"
        )
    return out


# ---------------------------------------------------------------- G1 (fold 0)


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


def build_g1_decision(
    oof_path: Path,
    y_by_id: dict | pd.Series,
    *,
    git_sha: str,
    tag: str,
    train_sha256: str,
    projected_remaining_usd: float | None = None,
) -> dict:
    """The decision, bound to the run that produced it: git sha, tag, OOF hash, labels hash."""
    dec = g1_check(pd.read_parquet(oof_path), y_by_id, projected_remaining_usd, expect_ids=y_by_id)
    return dec | {
        "git_sha": git_sha,
        "tag": tag,
        "oof_path": str(oof_path),
        "oof_sha256": sha256_path(oof_path),
        "train_sha256": train_sha256,
    }


def read_g1(path: Path = G1_FILE) -> dict | None:
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def g1_passed(decision: dict | None) -> bool:
    """The decision's own claim. verify_g1 re-derives it from the OOF file."""
    if not decision or decision.get("pass") is not True:
        return False
    x = decision.get("fold0_log_loss")
    return isinstance(x, int | float) and bool(np.isfinite(x)) and x <= G1_MAX_LOG_LOSS


def default_labels() -> dict:
    """Fold 0's labels from the raw train.csv, after checking that file's sha256."""
    from llm_preference_audit.data import LABELS, load_train
    from llm_preference_audit.kernel_common import TRAIN_SHA256
    from llm_preference_audit.split import build_manifest

    raw = Path("data/raw/train.csv")
    got = sha256_path(raw)
    if got != TRAIN_SHA256:
        raise SpendRefused(f"{raw} sha256 {got} != expected {TRAIN_SHA256}")
    df = load_train(raw)
    manifest = build_manifest(df)
    f0 = manifest[(manifest["split"] == "cv") & (manifest["fold"] == 0)]
    lab = df.set_index("id")["label"].astype(str)
    return {int(i): LABELS.index(lab[i]) for i in f0["id"]}


def verify_g1(decision: dict, git_sha: str, labels: Callable[[], dict] | None) -> list[str]:
    """Reasons the G1 decision does not hold for a launch at `git_sha` (empty = it holds)."""
    from llm_preference_audit.kernel_common import TRAIN_SHA256

    if decision.get("git_sha") != git_sha:
        return [
            f"G1: decision was made on git sha {decision.get('git_sha')!r}, launch is {git_sha!r}"
        ]
    if decision.get("train_sha256") != TRAIN_SHA256:
        return ["G1: decision is not bound to the verified train.csv (train_sha256 mismatch)"]
    oof = Path(str(decision.get("oof_path") or ""))
    if not oof.is_file():
        return [f"G1: the OOF file {oof} the decision was made from is missing"]
    if sha256_path(oof) != decision.get("oof_sha256"):
        return [f"G1: {oof} sha256 differs from the one the decision recorded"]
    try:
        y = (labels or default_labels)()
        redo = g1_check(pd.read_parquet(oof), y, expect_ids=y)
    except (ValueError, SpendRefused) as e:
        return [f"G1: the OOF could not be re-scored: {e}"]
    claimed = decision.get("fold0_log_loss")
    if not redo["pass"]:
        return [
            f"G1: recomputed fold-0 log loss {redo['fold0_log_loss']:.4f} > {G1_MAX_LOG_LOSS:.2f}"
        ]
    if not isinstance(claimed, int | float) or abs(claimed - redo["fold0_log_loss"]) > 1e-6:
        return [
            (
                f"G1: decision says loss {claimed!r} but the recomputed loss is "
                f"{redo['fold0_log_loss']:.6f}"
            )
        ]
    return []


def needs_g1(mode: str, folds: Iterable[int]) -> bool:
    return mode == "full" or (mode == "fold" and any(int(f) != 0 for f in folds))


# ---------------------------------------------------------------- the launch check


def check_launch(
    mode: str,
    folds: Iterable[int],
    estimate_usd: float,
    *,
    approved_usd: float = DEFAULT_APPROVED_USD,
    ledger: Path = LEDGER,
    spend: Path = SPEND,
    g1_file: Path = G1_FILE,
    out_root: Path = OUT_ROOT,
    git_sha: str = "",
    labels: Callable[[], dict] | None = None,
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
    cap = approved_usd
    probe = None
    if mode in ("fold", "full"):
        if not git_sha:
            reasons.append("G0: the launch git sha is required to find its probe")
        else:
            probe = select_probe(out_root, git_sha)
            if probe is None:
                reasons.append(
                    f"G0: no real probe.json (not tiny, {REAL_MODEL}, T={REAL_T}) for git sha "
                    f"{git_sha[:12]} under {out_root}; run the probe on this commit first"
                )
            else:
                reasons += [f"G0: {p}" for p in g0_problems(probe)]
        if needs_g1(mode, folds):
            g1 = read_g1(g1_file)
            if not g1_passed(g1):
                reasons.append(
                    f"G1: {g1_file} must say pass=true with fold0_log_loss <= "
                    f"{G1_MAX_LOG_LOSS:.2f} before folds 1-4 or the full adapter"
                )
            else:
                reasons += verify_g1(g1, git_sha, labels) if git_sha else []
                proj = (probe or {}).get("projection") or {}
                if proj.get("folds_1_to_4_plus_full_usd") is not None and not reasons:
                    cap = approved_usd + (1 + AUTO_MARGIN) * float(
                        proj["folds_1_to_4_plus_full_usd"]
                    )
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
    remaining = max(0.0, min(cap - tot["rung6_usd"], CEILING_USD - tot["gpu_total_usd"]))
    return tot | {
        "estimate_usd": estimate_usd,
        "rung6_after_usd": after6,
        "approved_cap_usd": cap,
        "remaining_usd": remaining,
        "probe": probe,
    }


def deadline_plan(mode: str, n_jobs: int, remaining_usd: float, probe: dict | None) -> dict:
    """Per-job time limit from the remaining approved budget.

    deadline_s = min(MAX_JOB_S, budget x 3600 / price / n_jobs) - SAFETY_MARGIN_S, where budget is
    the remaining approved amount (a probe: also at most its $5 cap). The Modal timeout is
    deadline + margin, so the worst case any job can bill is exactly its timeout x price. Refused
    when the deadline is shorter than the job's need: the probe's projection x 1.15 for the job
    (fold_seconds / full_seconds), or 20 minutes when there is no projection.
    """
    if n_jobs < 1:
        raise SpendRefused("a launch needs at least one job")
    budget = min(PROBE_CAP_USD, remaining_usd) if mode == "probe" else remaining_usd
    deadline = min(MAX_JOB_S, budget * 3600 / PRICE_PER_HOUR / n_jobs) - SAFETY_MARGIN_S
    proj = (probe or {}).get("projection") or {}
    key = {"fold": "fold_seconds", "full": "full_seconds"}.get(mode)
    need = proj[key] * (1 + OVERHEAD) if key and proj.get(key) else float(MIN_DEADLINE_S)
    if deadline < need:
        raise SpendRefused(
            f"deadline: the budget allows {deadline:.0f}s per job but the job needs {need:.0f}s "
            f"(${remaining_usd:.2f} remaining, {n_jobs} parallel job(s))"
        )
    timeout = deadline + SAFETY_MARGIN_S
    per_job = timeout * PRICE_PER_HOUR / 3600
    return {
        "deadline_s": deadline,
        "timeout_s": timeout,
        "per_job_worst_usd": per_job,
        "worst_case_usd": per_job * n_jobs,
        "n_jobs": n_jobs,
        "need_s": need,
    }


# ---------------------------------------------------------------- the spend file


def _append(path: Path, line: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(line) + "\n")


def record_launch(
    job_id: str,
    mode: str,
    tag: str,
    folds: list[int],
    estimate_usd: float,
    spend: Path = SPEND,
    *,
    worst_case_usd: float | None = None,
    per_job_worst_usd: float | None = None,
    deadline_s: float | None = None,
    n_jobs: int | None = None,
    git_sha: str = "",
    kind: str = "gpu",
) -> None:
    """Append the estimate and worst case BEFORE the launch (so a crash mid-launch still counts)."""
    _append(
        spend,
        {
            "event": "launch",
            "job_id": job_id,
            "ts": time.time(),
            "kind": kind,
            "mode": mode,
            "tag": tag,
            "folds": folds,
            "git_sha": git_sha,
            "estimate_usd": float(estimate_usd),
            "worst_case_usd": None if worst_case_usd is None else float(worst_case_usd),
            "per_job_worst_usd": None if per_job_worst_usd is None else float(per_job_worst_usd),
            "deadline_s": deadline_s,
            "n_jobs": n_jobs,
        },
    )


def _launch_line(job_id: str, spend: Path) -> dict:
    for ln in read_jsonl(spend):
        if ln.get("event") == "launch" and ln["job_id"] == job_id:
            return ln
    raise ValueError(f"no launch recorded for job {job_id!r}")


def launch_floor(launch: dict) -> float:
    """The least an automatic reconcile may record: min(launch estimate, worst case)."""
    est = float(launch["estimate_usd"])
    worst = launch.get("worst_case_usd")
    return est if worst is None else min(est, float(worst))


def reconcile(
    job_id: str,
    actual_usd: float,
    spend: Path = SPEND,
    note: str = "",
    *,
    source: str = "auto-logs",
    manual: bool = False,
) -> dict:
    """Append a reconcile line. Automatic ones are clamped up to the launch floor; only a manual
    one with source `modal-dashboard` may go lower. Records the previous charge either way."""
    launch = _launch_line(job_id, spend)
    if manual and source != MANUAL_SOURCE:
        raise ValueError(f"a manual reconcile needs --source {MANUAL_SOURCE}")
    previous = job_states(read_jsonl(spend))[job_id]["charge"]
    value, clamped = float(actual_usd), False
    if not manual and value < launch_floor(launch):
        value, clamped = launch_floor(launch), True
    line = {
        "event": "reconcile",
        "job_id": job_id,
        "ts": time.time(),
        "actual_usd": value,
        "source": source,
        "manual": manual,
        "previous_usd": previous,
        "clamped": clamped,
        "note": note,
    }
    _append(spend, line)
    return line


def job_logs(launch: dict, out_dir: Path) -> list[tuple[str, Path]]:
    out = Path(out_dir)
    if launch["mode"] == "probe":
        return [("probe", out / "probe.json")]
    if launch["mode"] == "full":
        return [("full", out / "log_full.json")]
    return [(f"fold{k}", out / f"log_fold{k}.json") for k in launch["folds"]]


def auto_reconcile(job_id: str, out_dir: Path, spend: Path = SPEND) -> dict:
    """Charge each launched job its log's est_cost_usd x 1.15 (at most its worst case), or its
    full worst case when it left no usable log; never below the launch floor."""
    launch = _launch_line(job_id, spend)
    n = int(launch.get("n_jobs") or len(launch["folds"]) or 1)
    worst = launch.get("per_job_worst_usd")
    if worst is None:
        worst = launch.get("worst_case_usd")
        worst = float(launch["estimate_usd"]) if worst is None else float(worst) / n
    charges, notes = [], []
    for name, path in job_logs(launch, out_dir):
        est = None
        try:
            est = float(json.loads(Path(path).read_text())["est_cost_usd"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if est is None:
            charges.append(float(worst))
            notes.append(f"{name}: no usable log, worst case ${worst:.2f}")
        else:
            charges.append(min(est * (1 + RECONCILE_OVERHEAD), float(worst)))
            notes.append(f"{name}: log ${est:.2f} x 1.15")
    return reconcile(
        job_id, sum(charges), spend, note="; ".join(notes), source="auto-logs", manual=False
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


@contextlib.contextmanager
def launch_lock(path: Path = LOCK_FILE):
    """Exclusive lock from the guard check through record_launch (two launches must not both
    pass the guard against the same spend file). A leftover lock refuses; remove it by hand."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SpendRefused(
            f"{path} exists: another launch is between its guard check and its spend record, or "
            f"a previous launch crashed (stale lock; read it, then remove it by hand)"
        ) from None
    try:
        os.write(fd, f"pid {os.getpid()} at {time.time():.0f}\n".encode())
        os.close(fd)
        yield path
    finally:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--mode", required=True, choices=["probe", "fold", "full"])
    c.add_argument("--folds", default="0")
    c.add_argument("--estimate-usd", type=float, required=True)
    c.add_argument("--approved-usd", type=float, default=DEFAULT_APPROVED_USD)
    c.add_argument("--git-sha", default="")
    g = sub.add_parser("g1")
    g.add_argument("--oof", type=Path, required=True)
    g.add_argument("--log", type=Path, default=None, help="default: log_fold0.json beside the OOF")
    g.add_argument("--raw", type=Path, default=Path("data/raw/train.csv"))
    g.add_argument("--out", type=Path, default=G1_FILE)
    r = sub.add_parser("reconcile")
    r.add_argument("--job-id", required=True)
    r.add_argument("--from-logs", type=Path, default=None)
    r.add_argument("--actual-usd", type=float, default=None)
    r.add_argument("--source", default=None, help=f"required with --actual-usd: {MANUAL_SOURCE}")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        folds = [int(f) for f in a.folds.split(",") if f != ""]
        try:
            acc = check_launch(
                a.mode, folds, a.estimate_usd, approved_usd=a.approved_usd, git_sha=a.git_sha
            )
            print(json.dumps(acc | {"probe": None}))
        except SpendRefused as e:
            print(str(e), file=sys.stderr)
            return 2
    elif a.cmd == "g1":
        from llm_preference_audit.data import LABELS, load_train
        from llm_preference_audit.kernel_common import TRAIN_SHA256
        from llm_preference_audit.split import build_manifest

        got = sha256_path(a.raw)
        if got != TRAIN_SHA256:
            print(
                f"{a.raw} sha256 {got} != {TRAIN_SHA256}; refusing to label from it",
                file=sys.stderr,
            )
            return 2
        log = json.loads((a.log or a.oof.parent / "log_fold0.json").read_text())
        if log.get("error") or log.get("mode") != "fold" or log.get("fold") != 0:
            print("the fold-0 log is missing, errored or not fold 0; refusing", file=sys.stderr)
            return 2
        if not log.get("git_sha") or log.get("oof_rows") != len(pd.read_parquet(a.oof)):
            print(
                "the fold-0 log has no git_sha or does not match the OOF row count", file=sys.stderr
            )
            return 2
        df = load_train(a.raw)
        manifest = build_manifest(df)
        f0 = manifest[(manifest["split"] == "cv") & (manifest["fold"] == 0)]
        lab = df.set_index("id")["label"].astype(str)
        y = {int(i): LABELS.index(lab[i]) for i in f0["id"]}
        dec = build_g1_decision(
            a.oof,
            y,
            git_sha=log["git_sha"],
            tag=log.get("tag", ""),
            train_sha256=got,
            projected_remaining_usd=None,
        )
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dec, indent=2) + "\n")
        print(json.dumps(dec, indent=2))
    elif a.cmd == "reconcile":
        if a.actual_usd is not None:
            line = reconcile(
                a.job_id,
                a.actual_usd,
                note="manual",
                source=a.source or "",
                manual=True,
            )
        elif a.from_logs is not None:
            line = auto_reconcile(a.job_id, a.from_logs)
        else:
            print(
                "give --from-logs, or --actual-usd with --source modal-dashboard", file=sys.stderr
            )
            return 2
        print(f"reconciled {a.job_id}: ${line['actual_usd']:.2f} (was ${line['previous_usd']:.2f})")
    else:
        print(json.dumps(totals() | {"g1": read_g1()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
