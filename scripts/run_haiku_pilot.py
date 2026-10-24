"""Rung 3: the Haiku judge on cv pairs, both A/B orders, through the Batch API.

Runs (--run): pilot-500 (default, 500 sampled cv rows) or full-cv (every cv row with a label).
Stages, run in order (each is resumable; artifacts live in gitignored data/judge/<run>/):
  estimate  offline: sample the pairs, truncate, print local token and cost estimates
  count     count input tokens with messages.count_tokens (free); `--sample N` counts a
            deterministic row sample and projects the total with a ratio estimator
  submit    submit the batches (chunks of CHUNK_REQUESTS; resumable); needs --yes
  collect   wait for every chunk, score, append one ledger row, write the run's summary

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
LEDGER = Path("results/ledger.jsonl")
PRED_DIR = Path("data/predictions")
PILOT_SUMMARY = Path("results/rung-3-pilot.json")  # measured pilot rates feed the estimates
N_PILOT = 500
SAMPLE_SALT = "haiku-pilot-2026-10-23"
COUNT_SALT = "haiku-count-2026-10-24"
JUDGE_ALLOTMENT_USD = 40.0  # decision #20: judges get $40 of the $150 cap
EST_OUTPUT_TOKENS = 400  # per call fallback when no pilot summary exists (thinking + JSON)
OUTPUT_SAFETY = 2.0  # multiplier on the measured pilot output rate, for the gate only
CHUNK_REQUESTS = 10_000  # Batch API caps: 100,000 requests / 256 MB per batch
MAX_CHUNK_BYTES = 200_000_000
BOOT_RESAMPLES = 1000
BOOT_SEED = 20261024

RUNS = {
    "pilot-500": {
        "n_rows": N_PILOT,
        "work": Path("data/judge/pilot-500"),
        "summary": Path("results/rung-3-pilot.json"),
        "ledger_config": "haiku_pilot_500",
        "split_tag": "cv-pilot",
        "ceiling_usd": 1.0,  # refuse to submit if the estimate is above this
        "default_sample": None,  # count every request
        "id_tag": "haiku-pilot",
    },
    "full-cv": {
        "n_rows": None,  # every cv row with a label
        "work": Path("data/judge/full-cv"),
        "summary": Path("results/rung-3-full.json"),
        "ledger_config": "haiku_full_cv",
        "split_tag": "cv-full",
        "ceiling_usd": None,  # only the allotment binds
        "default_sample": 2000,
        "id_tag": "haiku-full",
    },
}
RUN = "pilot-500"
WORK = RUNS[RUN]["work"]  # the active run's paths; select_run() rebinds them
SUMMARY = RUNS[RUN]["summary"]


def select_run(name: str) -> None:
    global RUN, WORK, SUMMARY
    RUN, WORK, SUMMARY = name, RUNS[name]["work"], RUNS[name]["summary"]


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
        "run": RUN,
        "n_rows": RUNS[RUN]["n_rows"],
        "sample_salt": SAMPLE_SALT,
        "orders": ["ab", "ba"],
        "calibration": "matrix scaling (multinomial LR on log p), cross-fitted by fold",
    }


def config_hash() -> str:
    return hashlib.sha256(json.dumps(config(), sort_keys=True).encode()).hexdigest()


def sample_rows(cv: pd.DataFrame, n_rows: int | None) -> pd.DataFrame:
    """The run's rows, sorted by id: every row for None, else the n smallest salted hashes."""
    if n_rows is None:
        return cv.sort_values("id").reset_index(drop=True)
    cv = cv.assign(
        rank=[hashlib.sha256(f"{SAMPLE_SALT}:{i}".encode()).hexdigest() for i in cv["id"]]
    )
    return cv.sort_values("rank").head(n_rows).sort_values("id").reset_index(drop=True)


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
    pilot = sample_rows(cv, RUNS[RUN]["n_rows"])
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


def output_tokens_per_call() -> float:
    """Gate estimate of output tokens per call: the measured pilot rate x OUTPUT_SAFETY."""
    if PILOT_SUMMARY.exists():
        pilot = json.loads(PILOT_SUMMARY.read_text())
        if pilot.get("n_calls"):
            return pilot["tokens"]["output"] / pilot["n_calls"] * OUTPUT_SAFETY
    return float(EST_OUTPUT_TOKENS)


def estimate_usd(input_tokens: float, n_calls: int) -> float:
    out = n_calls * output_tokens_per_call()
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


def local_tokens(item) -> float:
    """Local input-token estimate for one request: words x TOKENS_PER_WORD."""
    words = len(f"{SYSTEM} {item.prompt} {item.response_a} {item.response_b}".split())
    return words * TOKENS_PER_WORD


def row_key(custom_id: str) -> str:
    return custom_id.rsplit("-", 1)[0]


def chunk_items(items: list) -> list[list]:
    """Split into chunks of at most CHUNK_REQUESTS requests; a row's -ab/-ba stay together."""
    rows: list[list] = []
    for item in items:
        if rows and row_key(rows[-1][0].custom_id) == row_key(item.custom_id):
            rows[-1].append(item)
        else:
            rows.append([item])
    chunks: list[list] = []
    size = 0
    for row in rows:
        if not chunks or size + len(row) > CHUNK_REQUESTS:
            chunks.append([])
            size = 0
        chunks[-1].extend(row)
        size += len(row)
    return chunks


def chunk_bytes(chunk: list) -> int:
    return len(json.dumps([build_request(i) for i in chunk]).encode())


def stage_estimate(_: argparse.Namespace) -> None:
    pilot, _m = load_pilot()
    items, stats = build_items(pilot)
    est_in = int(sum(local_tokens(i) for i in items))
    chunks = chunk_items(items)
    print(f"run {RUN}: rows {len(pilot)}, calls {len(items)}, folds {sorted(set(pilot.fold))}")
    print(json.dumps(stats, indent=2))
    print(f"local input-token estimate {est_in:,} (words x {TOKENS_PER_WORD})")
    print(f"estimated cost ${estimate_usd(est_in, len(items)):.4f} at batch prices")
    print(f"output tokens per call for the gate {output_tokens_per_call():.1f}")
    biggest = max(chunk_bytes(c) for c in chunks)
    print(f"chunks {len(chunks)} (<= {CHUNK_REQUESTS} requests), biggest {biggest:,} bytes")


def requests_sha256(items: list) -> str:
    """Digest of the exact request bodies, so `submit` sends what `count` counted."""
    h = hashlib.sha256()
    h.update(b"[")
    for n, item in enumerate(items):
        h.update((b", " if n else b"") + json.dumps(build_request(item), sort_keys=True).encode())
    h.update(b"]")
    return h.hexdigest()


def ledger_rows() -> list[dict]:
    return [json.loads(x) for x in LEDGER.read_text().splitlines()] if LEDGER.exists() else []


def count_sample_ids(row_ids, n: int | None) -> set[str]:
    """Deterministic row sample for token counting: the n smallest salted hashes."""
    row_ids = [str(r) for r in row_ids]
    if n is None or n >= len(row_ids):
        return set(row_ids)
    key = lambda r: hashlib.sha256(f"{COUNT_SALT}:{r}".encode()).hexdigest()
    return set(sorted(row_ids, key=key)[:n])


def ratio_projection(
    real: np.ndarray, local: np.ndarray, local_total: float
) -> tuple[float, float]:
    """Ratio estimator real/local over sampled rows -> (projected total, 99% upper bound)."""
    ratio = real.sum() / local.sum()
    rng = np.random.default_rng(BOOT_SEED)
    n = len(real)
    idx = rng.integers(0, n, size=(BOOT_RESAMPLES, n))
    boot = real[idx].sum(axis=1) / local[idx].sum(axis=1)
    projected = float(ratio * local_total)
    return projected, max(projected, float(np.percentile(boot, 99) * local_total))


def projected_tokens(record: dict) -> tuple[float, float]:
    """(projected, p99) input tokens; the old format (counts only) is an exact sum."""
    if "projected_input_tokens" not in record:
        exact = float(sum(record["counts"].values()))
        return exact, exact
    projected = float(record["projected_input_tokens"])
    return projected, float(record.get("projected_input_tokens_p99", projected))


def stage_count(args: argparse.Namespace) -> None:
    import anthropic

    pilot, _m = load_pilot()
    items, _stats = build_items(pilot)
    sample = getattr(args, "sample", None)
    if sample is None:
        sample = RUNS[RUN]["default_sample"]
    keep = count_sample_ids([str(i) for i in pilot["id"]], sample)
    todo = [i for i in items if row_key(i.custom_id) in keep]
    client = anthropic.Anthropic()
    counts = {}
    for k, item in enumerate(todo):
        p = build_params(item)
        r = client.messages.count_tokens(
            model=p["model"],
            system=p["system"],
            messages=p["messages"],
            output_config=p["output_config"],
        )
        counts[item.custom_id] = r.input_tokens
        if (k + 1) % 100 == 0:
            print(f"counted {k + 1}/{len(todo)}")
    local = {i.custom_id: local_tokens(i) for i in items}
    local_total = float(sum(local.values()))
    sampled = len(todo) < len(items)
    if sampled:
        rows = sorted({row_key(c) for c in counts})
        real = np.array([counts[f"{r}-ab"] + counts[f"{r}-ba"] for r in rows], dtype=float)
        loc = np.array([local[f"{r}-ab"] + local[f"{r}-ba"] for r in rows], dtype=float)
        projected, p99 = ratio_projection(real, loc, local_total)
    else:
        projected = p99 = float(sum(counts.values()))
    WORK.mkdir(parents=True, exist_ok=True)
    record = {
        "requests_sha256": requests_sha256(items),
        "sampled": sampled,
        "n_counted": len(counts),
        "counts": counts,
        "projected_input_tokens": projected,
        "projected_input_tokens_p99": p99,
    }
    (WORK / "token_counts.json").write_text(json.dumps(record) + "\n")
    v = np.array(list(counts.values()))
    kind = f"ratio-projected from {len(counts)} counted" if sampled else "exact"
    print(f"input tokens {projected:,.0f} ({kind}); 99% upper bound {p99:,.0f}")
    print(
        f"per counted call median {np.median(v):.0f}, max {v.max()}; local total {local_total:,.0f}"
    )
    print(f"estimated cost ${estimate_usd(p99, len(items)):.4f} at the 99% bound")


def load_batches(requests_digest: str | None) -> dict:
    """State of the run's batches: {"chunks": {k: id}, "n_chunks", "requests_sha256"}."""
    state_file = WORK / "batches.json"
    if state_file.exists():
        return json.loads(state_file.read_text())
    legacy = WORK / "batch_id.txt"  # the pilot's single batch, before chunking existed
    if legacy.exists():
        return {
            "chunks": {"0": legacy.read_text().strip()},
            "n_chunks": 1,
            "requests_sha256": requests_digest,
        }
    return {"chunks": {}, "n_chunks": 0, "requests_sha256": requests_digest}


def save_batches(state: dict) -> None:
    tmp = WORK / "batches.json.tmp"
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(WORK / "batches.json")


def check_gate(estimate: float, spent: float, ceiling: float | None) -> None:
    if ceiling is not None and estimate > ceiling:
        raise SystemExit(
            f"estimate ${estimate:.4f} exceeds this run's ceiling ${ceiling}; stopping"
        )
    if spent + estimate > JUDGE_ALLOTMENT_USD:
        raise SystemExit(
            f"spent ${spent:.4f} + estimate ${estimate:.4f} exceeds the judge allotment "
            f"${JUDGE_ALLOTMENT_USD}; stopping"
        )


def stage_submit(args: argparse.Namespace) -> None:
    require_clean()
    leftover = sorted(WORK.glob("submit*.lock"))
    if leftover:
        raise SystemExit(
            f"{leftover[0]} exists without a batch id (chunk {leftover[0].stem.split('-')[-1]}): "
            "check the Console before resubmitting"
        )
    counts_file = WORK / "token_counts.json"
    if not counts_file.exists():
        raise SystemExit("run `count` first")
    pilot, _m = load_pilot()
    items, _stats = build_items(pilot)
    record = json.loads(counts_file.read_text())
    digest = requests_sha256(items)
    if record["requests_sha256"] != digest:
        raise SystemExit("token counts do not match the current requests; re-run `count`")
    chunks = chunk_items(items)
    state = load_batches(digest)
    if state["chunks"] and state["requests_sha256"] not in (digest, None):
        raise SystemExit("batches.json was written for different requests; investigate")
    if (
        state["chunks"]
        and state["n_chunks"] != len(chunks)
        and not (WORK / "batch_id.txt").exists()
    ):
        raise SystemExit("batches.json has a different chunk count; investigate")
    state.update(n_chunks=len(chunks), requests_sha256=digest)
    pending = [k for k in range(len(chunks)) if str(k) not in state["chunks"]]
    if not pending:
        print(f"already submitted: {len(state['chunks'])} batches")
        return

    _projected, p99 = projected_tokens(record)
    est = estimate_usd(p99, len(items))
    spent = sum(r.get("cost_usd", 0) for r in ledger_rows())
    print(
        f"{len(pending)} of {len(chunks)} chunks to submit; estimate ${est:.4f} "
        f"(99% input bound {p99:,.0f} tokens); ledger spend ${spent:.4f}; "
        f"remaining allotment ${JUDGE_ALLOTMENT_USD - spent:.4f}"
    )
    check_gate(est, spent, RUNS[RUN]["ceiling_usd"])
    for k in pending:
        size = chunk_bytes(chunks[k])
        if size > MAX_CHUNK_BYTES:
            raise SystemExit(f"chunk {k} is {size:,} bytes, over the {MAX_CHUNK_BYTES:,} limit")
    if not args.yes:
        raise SystemExit("dry run: pass --yes to submit")
    judge = HaikuBatchJudge()
    WORK.mkdir(parents=True, exist_ok=True)
    for k in pending:
        lock = WORK / f"submit-{k}.lock"
        lock.write_text(datetime.now(UTC).isoformat() + "\n")  # left behind if create dies
        batch_id = judge.submit(chunks[k])
        state["chunks"][str(k)] = batch_id
        save_batches(state)
        lock.unlink()
        print(f"submitted chunk {k}/{len(chunks)}: {batch_id}")


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


def raw_path(k: int) -> Path:
    """Chunk k's saved results; the pilot's single chunk keeps its original file name."""
    return WORK / ("results.jsonl" if RUN == "pilot-500" and k == 0 else f"results-{k}.jsonl")


def batch_ids_of(row: dict) -> list[str]:
    return sorted(row.get("batch_ids") or ([row["batch_id"]] if row.get("batch_id") else []))


def stage_collect(_: argparse.Namespace) -> None:
    sha = require_clean()
    state = load_batches(None)
    n_chunks = state["n_chunks"]
    if not n_chunks or len(state["chunks"]) != n_chunks:
        raise SystemExit(
            f"only {len(state['chunks'])} of {n_chunks} chunks are submitted; run `submit` first"
        )
    batch_ids = sorted(state["chunks"].values())
    judge = None
    raw: list[dict] = []
    for k in range(n_chunks):
        path = raw_path(k)
        if not path.exists():
            judge = judge or HaikuBatchJudge()
            judge.wait(state["chunks"][str(k)])
            rows = judge.collect(state["chunks"][str(k)])
            tmp = path.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
            tmp.replace(path)
        raw.extend(json.loads(line) for line in path.read_text().splitlines())
    outputs = [parse_result(r) for r in raw]

    pilot, manifest = load_pilot()
    labels = load_train(RAW).set_index("id")["label"]
    prior = prior_by_fold(manifest, labels, pilot["fold"].to_numpy(dtype=int))
    summary, P_cal = score(pilot, prior, outputs)
    _items, summary["truncation"] = build_items(pilot)
    summary["batch_ids"] = batch_ids
    summary["n_chunks"] = n_chunks
    if n_chunks == 1:
        summary["batch_id"] = batch_ids[0]
    summary["lgbm_reference"] = lgbm_reference(pilot, row_losses(pilot["y"].to_numpy(), P_cal))
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    if any(batch_ids_of(r) == batch_ids for r in ledger_rows()):
        print(f"ledger already has batches {batch_ids}; summary rewritten, no new row")
        return
    cost, ci, loss = summary["cost_usd"], summary["gain_vs_prior"], summary["log_loss"]
    _write_ledger(sha, pilot, P_cal, batch_ids, cost, ci, loss["matrix_cv"], summary)


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
        "run": RUN,
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


def _write_ledger(sha, pilot, P_cal, batch_ids, cost, ci, loss_cal, summary) -> None:
    ts = datetime.now(UTC)
    run_id = f"{ts.strftime('%Y%m%dT%H%M%S')}-{RUNS[RUN]['id_tag']}"
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
        "config": RUNS[RUN]["ledger_config"],
        "split": RUNS[RUN]["split_tag"],
        "log_loss": loss_cal,
        "cost_usd": cost,
        "predictions_path": str(pred_path),
        "predictions_sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest(),
        "n_rows": len(pilot),
        "ci95_vs_prior": [ci["lo"], ci["hi"]],
        "refusal_rate": summary["refusal_rate"],
        "batch_ids": batch_ids,
    }
    if len(batch_ids) == 1:
        row["batch_id"] = batch_ids[0]
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
    parser.add_argument("--run", choices=list(RUNS), default="pilot-500")
    parser.add_argument("--sample", type=int, default=None, help="count: rows to count_tokens")
    parser.add_argument("--yes", action="store_true", help="actually submit the paid batch")
    args = parser.parse_args(argv)
    select_run(args.run)
    STAGES[args.stage](args)


if __name__ == "__main__":
    main(sys.argv[1:])
