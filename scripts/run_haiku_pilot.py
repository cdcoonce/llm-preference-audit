"""Rung 3 pilot: the Haiku judge on 500 cv pairs, both A/B orders, through the Batch API.

Stages, run in order (each is resumable; artifacts live in gitignored data/judge/pilot-500/):
  estimate  offline: sample the pairs, truncate, print local token and cost estimates
  count     count input tokens with messages.count_tokens (free) and print the cost estimate
  submit    submit one batch of 2 x 500 requests; refuses without --yes or past the allotment
  collect   wait for the batch, save raw results, score, append a ledger row, write the summary

Only cv rows are ever sent; the locked test split is never read past the manifest. Commit the
code before `submit` and `collect`: the ledger row records HEAD as git_sha.
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
from llm_preference_audit.judge.haiku import (
    BATCH_INPUT_USD_PER_MTOK,
    BATCH_OUTPUT_USD_PER_MTOK,
    EFFORT,
    MAX_TOKENS,
    MODEL,
    OUTPUT_SCHEMA,
    SYSTEM,
    HaikuBatchJudge,
    build_params,
    build_request,
    cost_usd,
    parse_result,
)
from llm_preference_audit.judge.harness import combine, make_items, refusal_rate, status_counts
from llm_preference_audit.profile import TOKENS_PER_WORD
from llm_preference_audit.scoring import group_bootstrap_diff, log_loss, row_losses
from llm_preference_audit.split import SHA_PATH, build_manifest, manifest_sha256
from llm_preference_audit.train_lgbm import assert_cv_only
from llm_preference_audit.truncate import (
    PROMPT_BUDGET_TOKENS,
    RESPONSE_BUDGET_TOKENS,
    truncate_turns,
)

RAW = Path("data/raw/train.csv")  # the Hugging Face copy: the only one sent to the API
WORK = Path("data/judge/pilot-500")
LEDGER = Path("results/ledger.jsonl")
PRED_DIR = Path("data/predictions")
SUMMARY = Path("results/rung-3-pilot.json")
N_PILOT = 500
SAMPLE_SALT = "haiku-pilot-2026-10-23"
JUDGE_ALLOTMENT_USD = 40.0  # decision #20: judges get $40 of the $150 cap
PILOT_CEILING_USD = 1.0  # refuse to submit if the estimate is above this
EST_OUTPUT_TOKENS = 400  # per call, for the pre-submit estimate only (thinking + JSON)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def require_clean() -> str:
    dirty = git("status", "--porcelain", "--", "llm_preference_audit", "scripts", "pyproject.toml")
    if dirty:
        raise SystemExit(f"uncommitted code; commit before running:\n{dirty}")
    return git("rev-parse", "HEAD")


def config() -> dict:
    return {
        "model": MODEL,
        "effort": EFFORT,
        "max_tokens": MAX_TOKENS,
        "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "schema": OUTPUT_SCHEMA,
        "prompt_budget_tokens": PROMPT_BUDGET_TOKENS,
        "response_budget_tokens": RESPONSE_BUDGET_TOKENS,
        "n_pilot": N_PILOT,
        "sample_salt": SAMPLE_SALT,
        "orders": ["ab", "ba"],
        "calibration": "matrix scaling (multinomial LR on log p), cross-fitted by fold",
    }


def config_hash() -> str:
    return hashlib.sha256(json.dumps(config(), sort_keys=True).encode()).hexdigest()


def load_pilot() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pilot rows with label, group, fold and raw word counts, the full manifest)."""
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
    cv = df[(df["split"] == "cv") & df["label"].notna()].copy()
    cv["rank"] = [hashlib.sha256(f"{SAMPLE_SALT}:{i}".encode()).hexdigest() for i in cv["id"]]
    pilot = cv.sort_values("rank").head(N_PILOT).sort_values("id").reset_index(drop=True)
    assert_cv_only(pilot["id"].to_numpy(), manifest)
    pilot["y"] = pilot["label"].astype(str).map({c: i for i, c in enumerate(LABELS)}).astype(int)
    for col, short in (("prompt", "prompt"), ("response_a", "a"), ("response_b", "b")):
        pilot[f"words_{short}"] = [len(conversation_text(t).split()) for t in pilot[col]]
    return pilot, manifest


def build_items(pilot: pd.DataFrame) -> tuple[list, dict]:
    tp = [truncate_turns(t, PROMPT_BUDGET_TOKENS) for t in pilot["prompt"]]
    ta = [truncate_turns(t, RESPONSE_BUDGET_TOKENS) for t in pilot["response_a"]]
    tb = [truncate_turns(t, RESPONSE_BUDGET_TOKENS) for t in pilot["response_b"]]
    items = make_items(
        [str(i) for i in pilot["id"]],
        [t.text for t in tp],
        [t.text for t in ta],
        [t.text for t in tb],
    )
    stats = {
        "prompt_truncated_share": float(np.mean([t.truncated for t in tp])),
        "response_truncated_share": float(np.mean([t.truncated for t in ta + tb])),
        "rows_any_truncated_share": float(
            np.mean([p.truncated or a.truncated or b.truncated for p, a, b in zip(tp, ta, tb)])
        ),
        "words_omitted_share": float(
            sum(t.words_omitted for t in tp + ta + tb)
            / max(1, sum(t.words_in for t in tp + ta + tb))
        ),
    }
    return items, stats


def estimate_usd(input_tokens: int, n_calls: int) -> float:
    out = n_calls * EST_OUTPUT_TOKENS
    return (input_tokens * BATCH_INPUT_USD_PER_MTOK + out * BATCH_OUTPUT_USD_PER_MTOK) / 1e6


def prior_by_fold(manifest: pd.DataFrame, labels: pd.Series, folds: np.ndarray) -> np.ndarray:
    """Per pilot row, the label frequencies of all cv rows in the other folds."""
    cv = manifest[manifest["split"] == "cv"].set_index("id")
    y_all = labels.loc[cv.index].astype(str).map({c: i for i, c in enumerate(LABELS)})
    keep = y_all.notna()
    y_all, f_all = y_all[keep].astype(int).to_numpy(), cv["fold"][keep].to_numpy(dtype=int)
    P = np.zeros((len(folds), 3))
    for k in sorted(set(folds.tolist())):
        P[folds == k] = np.bincount(y_all[f_all != k], minlength=3) / (f_all != k).sum()
    return P


def stage_estimate(_: argparse.Namespace) -> None:
    pilot, _m = load_pilot()
    items, stats = build_items(pilot)
    words = sum(len(f"{SYSTEM} {i.prompt} {i.response_a} {i.response_b}".split()) for i in items)
    est_in = int(words * TOKENS_PER_WORD)
    print(f"pilot rows {len(pilot)}, calls {len(items)}, folds {sorted(set(pilot.fold))}")
    print(json.dumps(stats, indent=2))
    print(f"local input-token estimate {est_in:,} (words x {TOKENS_PER_WORD})")
    print(f"estimated cost ${estimate_usd(est_in, len(items)):.4f} at batch prices")


def requests_sha256(items: list) -> str:
    """Digest of the exact request bodies, so `submit` sends what `count` counted."""
    body = json.dumps([build_request(i) for i in items], sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


def ledger_rows() -> list[dict]:
    return [json.loads(x) for x in LEDGER.read_text().splitlines()] if LEDGER.exists() else []


def stage_count(_: argparse.Namespace) -> None:
    import anthropic

    pilot, _m = load_pilot()
    items, _stats = build_items(pilot)
    client = anthropic.Anthropic()
    counts = {}
    for k, item in enumerate(items):
        p = build_params(item)
        r = client.messages.count_tokens(
            model=p["model"],
            system=p["system"],
            messages=p["messages"],
            output_config=p["output_config"],
        )
        counts[item.custom_id] = r.input_tokens
        if (k + 1) % 100 == 0:
            print(f"counted {k + 1}/{len(items)}")
    WORK.mkdir(parents=True, exist_ok=True)
    record = {"requests_sha256": requests_sha256(items), "counts": counts}
    (WORK / "token_counts.json").write_text(json.dumps(record) + "\n")
    total = sum(counts.values())
    v = np.array(list(counts.values()))
    print(f"input tokens {total:,} (per call median {np.median(v):.0f}, max {v.max()})")
    print(
        f"estimated cost ${estimate_usd(total, len(items)):.4f} (output {EST_OUTPUT_TOKENS}/call)"
    )


def stage_submit(args: argparse.Namespace) -> None:
    require_clean()
    batch_file = WORK / "batch_id.txt"
    lock = WORK / "submit.lock"
    if batch_file.exists():
        raise SystemExit(f"already submitted: {batch_file.read_text().strip()}")
    if lock.exists():
        raise SystemExit(f"{lock} exists without a batch id: check the Console before resubmitting")
    counts_file = WORK / "token_counts.json"
    if not counts_file.exists():
        raise SystemExit("run `count` first")
    pilot, _m = load_pilot()
    items, _stats = build_items(pilot)
    record = json.loads(counts_file.read_text())
    if record["requests_sha256"] != requests_sha256(items):
        raise SystemExit("token counts do not match the current requests; re-run `count`")
    est = estimate_usd(sum(record["counts"].values()), len(items))
    spent = sum(r.get("cost_usd", 0) for r in ledger_rows())
    print(f"estimate ${est:.4f}; ledger spend so far ${spent:.4f} of ${JUDGE_ALLOTMENT_USD}")
    if est > PILOT_CEILING_USD or spent + est > JUDGE_ALLOTMENT_USD:
        raise SystemExit("estimate exceeds the pilot ceiling or the judge allotment; stopping")
    if not args.yes:
        raise SystemExit("dry run: pass --yes to submit")
    lock.write_text(datetime.now(UTC).isoformat() + "\n")  # left behind if create dies mid-way
    batch_id = HaikuBatchJudge().submit(items)
    batch_file.write_text(batch_id + "\n")
    lock.unlink()
    print(f"submitted {batch_id}")


def lgbm_reference(pilot: pd.DataFrame, loss_judge: np.ndarray) -> dict | None:
    """Rung 2's out-of-fold LightGBM (`full`) on the same pilot rows, paired with the judge."""
    full = [r for r in ledger_rows() if r.get("config") == "full" and r.get("model") == "lightgbm"]
    if not full or not Path(full[-1]["predictions_path"]).exists():
        return None
    oof = pd.read_parquet(full[-1]["predictions_path"]).set_index("id").loc[pilot["id"]]
    loss_lgbm = row_losses(pilot["y"].to_numpy(), oof[["p_a", "p_b", "p_tie"]].to_numpy())
    return {
        "run_id": full[-1]["run_id"],
        "log_loss": float(loss_lgbm.mean()),
        "judge_gain_vs_lgbm": group_bootstrap_diff(loss_lgbm, loss_judge, pilot["group"]),
    }


def stage_collect(_: argparse.Namespace) -> None:
    sha = require_clean()
    batch_id = (WORK / "batch_id.txt").read_text().strip()
    raw_file = WORK / "results.jsonl"
    if not raw_file.exists():
        judge = HaikuBatchJudge()
        judge.wait(batch_id)
        raw = judge.collect(batch_id)
        raw_file.write_text("".join(json.dumps(r) + "\n" for r in raw))
    raw = [json.loads(line) for line in raw_file.read_text().splitlines()]
    outputs = [parse_result(r) for r in raw]

    pilot, manifest = load_pilot()
    labels = load_train(RAW).set_index("id")["label"]
    prior = prior_by_fold(manifest, labels, pilot["fold"].to_numpy(dtype=int))
    summary, P_cal = score(pilot, prior, outputs)
    _items, summary["truncation"] = build_items(pilot)
    summary["batch_id"] = batch_id
    summary["lgbm_reference"] = lgbm_reference(pilot, row_losses(pilot["y"].to_numpy(), P_cal))
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    if any(r.get("batch_id") == batch_id for r in ledger_rows()):
        print(f"ledger already has batch {batch_id}; summary rewritten, no new row")
        return
    cost, ci, loss = summary["cost_usd"], summary["gain_vs_prior"], summary["log_loss"]
    _write_ledger(sha, pilot, P_cal, batch_id, cost, ci, loss["matrix_cv"], summary)


def score(pilot: pd.DataFrame, prior: np.ndarray, outputs: list) -> tuple[dict, np.ndarray]:
    """Pure scoring of judge outputs on the pilot rows. prior: (n,3) out-of-fold class prior."""
    ids = [str(i) for i in pilot["id"]]
    y = pilot["y"].to_numpy()
    folds = pilot["fold"].to_numpy(dtype=int)
    groups = pilot["group"].to_numpy()
    prior_vec = prior.mean(axis=0)  # fallback for refused calls; folds differ in the 4th digit
    comb = combine(outputs, ids, prior_vec)

    splits = pilot["split"].to_numpy()  # cross_fit refuses anything but cv rows
    P_raw = comb["P"]
    P_cal = cross_fit(P_raw, y, folds, splits, MatrixScaler)
    P_temp = cross_fit(P_raw, y, folds, splits, TemperatureScaler)
    loss_prior = row_losses(y, prior)
    loss_cal = row_losses(y, P_cal)
    ci = group_bootstrap_diff(loss_prior, loss_cal, groups)
    ok_rows = ~comb["fallback"]
    wa, wb = pilot["words_a"].to_numpy(), pilot["words_b"].to_numpy()
    cost = cost_usd(outputs)
    ok = [o for o in outputs if o.status == "ok"]
    agree = [o.winner == ("A", "B", "tie")[int(np.argmax([o.p_a, o.p_b, o.p_tie]))] for o in ok]

    summary = {
        "run": "pilot-500",
        "n_rows": len(pilot),
        "n_calls": len(outputs),
        "status_counts": status_counts(outputs),
        "refusal_rate": refusal_rate(outputs),
        "fallback_row_share": float(comb["fallback"].mean()),  # rows with any non-ok call
        "winner_matches_argmax_share": float(np.mean(agree)) if agree else None,
        "log_loss": {
            "prior_oof": float(loss_prior.mean()),
            "raw_swap_avg": log_loss(y, P_raw),
            "temperature_cv": log_loss(y, P_temp),
            "matrix_cv": float(loss_cal.mean()),
            "matrix_cv_excluding_fallback_rows": (
                log_loss(y[ok_rows], P_cal[ok_rows]) if ok_rows.any() else None
            ),
            "ab_only_raw": log_loss(y, comb["P_ab"]),
        },
        "gain_vs_prior": ci,
        "gate_pass": bool(ci["lo"] > 0),
        "position": {
            # raw arrays keep NaN for failed calls, so prior-filled rows can't count as flips
            "swap_flip": swap_flip_readout(comb["raw_ab"], comb["raw_ba"][:, [1, 0, 2]]),
            "first_position": first_position_readout(comb["raw_ab"], comb["raw_ba"], groups),
        },
        "verbosity_calibrated": calibrated_longer_readout(y, P_cal, wa, wb, groups),
        "verbosity_raw_swap_avg": calibrated_longer_readout(y, P_raw, wa, wb, groups),
        "tokens": {
            "input": sum(o.input_tokens for o in outputs),
            "output": sum(o.output_tokens for o in outputs),
        },
        "cost_usd": cost,
        "config": config(),
    }
    return summary, P_cal


def _write_ledger(sha, pilot, P_cal, batch_id, cost, ci, loss_cal, summary) -> None:
    ts = datetime.now(UTC)
    run_id = f"{ts.strftime('%Y%m%dT%H%M%S')}-haiku-pilot"
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = PRED_DIR / f"{run_id}.parquet"
    pd.DataFrame(
        {
            "id": pilot["id"],
            "winner_model_a": P_cal[:, 0],
            "winner_model_b": P_cal[:, 1],
            "winner_tie": P_cal[:, 2],
        }
    ).to_parquet(pred_path, index=False)
    row = {
        "run_id": run_id,
        "timestamp_utc": ts.isoformat(timespec="seconds"),
        "git_sha": sha,
        "config_hash": config_hash(),
        "model": MODEL,
        "config": "haiku_pilot_500",
        "split": "cv-pilot",
        "log_loss": loss_cal,
        "cost_usd": cost,
        "predictions_path": str(pred_path),
        "predictions_sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest(),
        "n_rows": len(pilot),
        "ci95_vs_prior": [ci["lo"], ci["hi"]],
        "refusal_rate": summary["refusal_rate"],
        "batch_id": batch_id,
    }
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps({k: summary[k] for k in ("log_loss", "gain_vs_prior", "gate_pass")}, indent=2))
    print(f"cost ${cost:.4f}; summary {SUMMARY}; ledger row {run_id}")


STAGES = {
    "estimate": stage_estimate,
    "count": stage_count,
    "submit": stage_submit,
    "collect": stage_collect,
}


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=list(STAGES))
    parser.add_argument("--yes", action="store_true", help="actually submit the paid batch")
    args = parser.parse_args(argv)
    STAGES[args.stage](args)


if __name__ == "__main__":
    main(sys.argv[1:])
