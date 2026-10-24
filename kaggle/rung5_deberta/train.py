"""Rung 5: fine-tune microsoft/deberta-v3-base for 3-way a/b/tie preference classification.

One script for Kaggle (T4 x2) and a local CPU dry run (`--tiny`). Input per row:
`[CLS] prompt [SEP] response_a [SEP] response_b [SEP]`, fitted to T tokens by the shared
adaptive head/middle/tail truncation (`fit_pair`), hard-cut at the tokenizer when it still
overflows. Modes:
  probe  one fold's training data for a few optimizer steps, then inference in both orders;
         writes probe.json with throughput, memory and a projected GPU-hour budget.
  folds  per outer fold: train (best epoch by inner-holdout log loss), predict the held-out
         fold in both orders, write oof_fold{k}.parquet + log_fold{k}.json. One subprocess
         per fold, pinned to its own GPU when two exist.

Safety: the train.csv sha256 and the split manifest digest are verified on the FULL file before
anything is restricted. The competition test.csv is never referenced. Only ids, probabilities,
flags and aggregates are written to --out-dir; dataset text lives in memory only; model weights
stay in memory (never in --out-dir).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import pickle
import queue
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from llm_preference_audit.data import load_train
from llm_preference_audit.deberta_inputs import (
    N_SPECIAL,
    assert_no_text,
    build_input_ids,
    fold_plan,
    length_grouped_batches,
    oof_frame,
    prepare_rows,
    swap_example,
)
from llm_preference_audit.scoring import log_loss
from llm_preference_audit.split import build_manifest, manifest_sha256
from llm_preference_audit.train_lgbm import assert_cv_only
from llm_preference_audit.truncate import MIDDLE_SHARE

# The Hugging Face train.csv (byte-identical to Kaggle's copy) and the committed split digest.
TRAIN_SHA256 = "0692154aaf20fc6649090c3f49b6b5dd1e693a765ecb5f0856ae0da5946f5be2"
MANIFEST_SHA256 = "0a77276dd887690580c1004a7d44224117095f2912947ba9137dd602a4995523"
DEFAULT_DATA_DIR = "/kaggle/input/llm-classification-finetuning"
EFFECTIVE_BATCH = 16
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP = 0.1
GRAD_CLIP = 1.0
SAMPLE_SEED = 20261024  # --max-rows sampling
GIB = 1024**3


class NonFiniteLoss(RuntimeError):
    pass


class Encoded(NamedTuple):
    meta: pd.DataFrame  # id, y, fold, group, hard_cut, both_untruncated, n_* (no text)
    p: list
    a: list
    b: list
    n_cv_full: int  # cv rows in the full manifest (for projections)
    shares: dict


# ---------------------------------------------------------------- inputs


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def locate_train_csv(data_dir: Path, expect_sha: str) -> Path:
    """data_dir/train.csv, else the train.csv under data_dir's parent whose sha256 matches.

    Kaggle's competition mount point has moved between images, so the hash decides, not the path.
    """
    direct = data_dir / "train.csv"
    if direct.is_file():
        return direct
    candidates = sorted(data_dir.parent.rglob("train.csv"))
    for path in candidates:
        if sha256_file(path) == expect_sha:
            print(f"found train.csv at {path} (searched under {data_dir.parent})", flush=True)
            return path
    raise SystemExit(
        f"no train.csv with sha256 {expect_sha} under {data_dir.parent}; candidates: "
        f"{[str(c) for c in candidates]}"
    )


def load_verified(data_dir: Path, expect_sha: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hash train.csv, build the manifest on the full file and check its digest. Never test.csv."""
    path = locate_train_csv(data_dir, expect_sha)
    got = sha256_file(path)
    if got != expect_sha:
        raise SystemExit(f"train.csv sha256 {got} != expected {expect_sha}")
    df = load_train(path)
    manifest = build_manifest(df)
    digest = manifest_sha256(manifest)
    if digest != MANIFEST_SHA256:
        raise SystemExit(f"split manifest digest {digest} != committed {MANIFEST_SHA256}")
    print(f"verified train.csv sha256 and split manifest digest ({len(df)} rows)", flush=True)
    return df, manifest


def make_tokenizer(args):
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.model_max_length = 10**9  # we fit to the window ourselves
    return tok


def encode(args, tok, df: pd.DataFrame, manifest: pd.DataFrame) -> Encoded:
    n_cv_full = int((manifest["split"] == "cv").sum())
    if args.max_rows:
        cv_ids = manifest.loc[manifest["split"] == "cv", "id"].to_numpy()
        keep = np.random.default_rng(SAMPLE_SEED).choice(cv_ids, size=args.max_rows, replace=False)
        df = df[df["id"].isin(keep)]

    def count(s: str) -> int:
        return len(tok(s, add_special_tokens=False).input_ids)

    t0 = time.time()
    rows = prepare_rows(df, manifest, count, args.t - N_SPECIAL, args.middle_share)
    print(f"fitted {len(rows)} cv rows to T={args.t} in {time.time() - t0:.0f}s", flush=True)

    def ids(col: str) -> list:
        enc = tok(rows[col].tolist(), add_special_tokens=False)["input_ids"]
        return [np.asarray(x, dtype=np.int32) for x in enc]

    p, a, b = ids("prompt_text"), ids("a_text"), ids("b_text")
    meta = rows.drop(columns=["prompt_text", "a_text", "b_text"])
    shares = {
        "n_rows": len(meta),
        "hard_cut_share": float(meta["hard_cut"].mean()),
        "both_untruncated_share": float(meta["both_untruncated"].mean()),
    }
    return Encoded(meta, p, a, b, n_cv_full, shares)


def cache_path(args) -> Path:
    key = f"{args.tag or 'run'}_{args.t}_{args.middle_share}_{args.max_rows}"
    return Path("/tmp") / f"rung5_cache_{key}.pkl"  # ids only, no text


def load_or_encode(args) -> tuple[Encoded, object, pd.DataFrame | None]:
    tok = make_tokenizer(args)
    if args.cache and Path(args.cache).exists():
        with open(args.cache, "rb") as f:
            enc, manifest = pickle.load(f)
        print(f"loaded encoded rows from {args.cache}", flush=True)
        return enc, tok, manifest
    df, manifest = load_verified(Path(args.data_dir), args.expect_train_sha256)
    return encode(args, tok, df, manifest), tok, manifest


# ---------------------------------------------------------------- model


def build_model(args, tok):
    if args.tiny:
        from transformers import DebertaV2Config

        cfg = DebertaV2Config(
            vocab_size=len(tok),
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=args.t + 8,
            relative_attention=True,
            position_buckets=256,
            norm_rel_ebd="layer_norm",
            share_att_key=True,
            pos_att_type=["p2c", "c2p"],
            position_biased_input=False,
            type_vocab_size=0,
            num_labels=3,
            pad_token_id=tok.pad_token_id,
        )
        model = AutoModelForSequenceClassification.from_config(cfg)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    return model


def make_seq(enc: Encoded, i: int, swap: bool, tok, limit: int) -> tuple[list[int], int]:
    y = int(enc.meta["y"].iat[i])
    p, a, b = enc.p[i], enc.a[i], enc.b[i]
    if swap:
        p, a, b, y = swap_example(p, a, b, y)
    return build_input_ids(p, a, b, tok.cls_token_id, tok.sep_token_id, limit), y


def collate(seqs: list[list[int]], pad_id: int, device):
    n = max(map(len, seqs))
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[r, : len(s)] = 1
    return ids.to(device), mask.to(device)


def autocast(device):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda")


@torch.no_grad()
def predict(model, enc: Encoded, idx, swap: bool, tok, args, device) -> np.ndarray:
    """Probabilities in PRESENTED order ([p_first, p_second, p_tie]) for rows `idx`."""
    model.eval()
    seqs = [make_seq(enc, i, swap, tok, args.t)[0] for i in idx]
    order = np.argsort([len(s) for s in seqs], kind="stable")
    out = np.zeros((len(seqs), 3), dtype=np.float64)
    for s in range(0, len(order), args.infer_batch):
        sel = order[s : s + args.infer_batch]
        ids, mask = collate([seqs[j] for j in sel], tok.pad_token_id, device)
        with autocast(device):
            logits = model(input_ids=ids, attention_mask=mask).logits
        out[sel] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return out


def predict_both(model, enc, idx, tok, args, device) -> tuple[np.ndarray, np.ndarray]:
    return (
        predict(model, enc, idx, False, tok, args, device),
        predict(model, enc, idx, True, tok, args, device),
    )


def swap_avg(raw_ab: np.ndarray, raw_ba: np.ndarray) -> np.ndarray:
    return (raw_ab + raw_ba[:, [1, 0, 2]]) / 2


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------- training


def train_epochs(model, enc, train_idx, eval_idx, tok, args, seed, device, epochs, max_steps=None):
    """Train; after each epoch score the inner holdout. Returns stats and best-epoch state."""
    from transformers import get_linear_schedule_with_warmup

    micro = args.micro_batch
    accum = max(1, EFFECTIVE_BATCH // micro)
    lengths = np.minimum(
        enc.meta[["n_prompt", "n_a", "n_b"]].to_numpy()[train_idx].sum(axis=1) + N_SPECIAL, args.t
    )
    steps_per_epoch = math.ceil(math.ceil(len(train_idx) / micro) / accum)
    total_steps = max_steps or steps_per_epoch * epochs
    no_decay = ("bias", "LayerNorm.weight", "layernorm", "norm")
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    opt = torch.optim.AdamW(params, lr=LR)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * total_steps), total_steps)
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    history, best, step = [], {"loss": float("inf"), "epoch": None, "state": None}, 0
    skips = real_tokens = padded_tokens = samples = 0
    t_start = time.time()
    t_after_warm = samples_at_warm = None
    for epoch in range(epochs):
        model.train()
        swap_rng = np.random.default_rng(seed * 1000 + epoch)
        swaps = swap_rng.random(len(train_idx)) < 0.5
        batches = length_grouped_batches(lengths, micro, seed * 1000 + epoch)
        groups = [batches[i : i + accum] for i in range(0, len(batches), accum)]
        loss_sum, loss_n, t_epoch = 0.0, 0, time.time()
        for group in groups:
            for bidx in group:
                pairs = [
                    make_seq(enc, int(train_idx[j]), bool(swaps[j]), tok, args.t) for j in bidx
                ]
                ids, mask = collate([s for s, _ in pairs], tok.pad_token_id, device)
                y = torch.tensor([y for _, y in pairs], device=device)
                with autocast(device):
                    logits = model(input_ids=ids, attention_mask=mask).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), y)
                if not torch.isfinite(loss):
                    raise NonFiniteLoss(f"non-finite loss at step {step}, epoch {epoch}")
                scaler.scale(loss / len(group)).backward()
                loss_sum += loss.item() * len(bidx)
                loss_n += len(bidx)
                samples += len(bidx)
                real_tokens += int(mask.sum())
                padded_tokens += int(mask.numel())
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            skips += int(use_cuda and scaler.get_scale() < before)
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step == 2:
                sync(device)
                t_after_warm, samples_at_warm = time.time(), samples
            if step % 100 == 0:
                print(f"step {step}/{total_steps} loss {loss_sum / loss_n:.4f}", flush=True)
            if max_steps and step >= max_steps:
                break
        rec = {"epoch": epoch + 1, "train_loss": loss_sum / max(1, loss_n)}
        sync(device)
        rec["train_seconds"] = time.time() - t_epoch
        if eval_idx is not None and len(eval_idx) and not max_steps:
            t0 = time.time()
            ab, ba = predict_both(model, enc, eval_idx, tok, args, device)
            P = swap_avg(ab, ba)
            if not np.isfinite(P).all():
                raise NonFiniteLoss("non-finite inner-holdout probabilities")
            rec["inner_log_loss"] = log_loss(enc.meta["y"].to_numpy()[eval_idx], P)
            rec["eval_seconds"] = time.time() - t0
            if rec["inner_log_loss"] < best["loss"]:
                best.update(loss=rec["inner_log_loss"], epoch=epoch + 1)
                best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(rec)
        print(f"epoch {epoch + 1}: {rec}", flush=True)
        if max_steps and step >= max_steps:
            break
    sync(device)
    now = time.time()
    if t_after_warm is not None and samples > samples_at_warm:
        meas_s, meas_n = now - t_after_warm, samples - samples_at_warm
    else:
        meas_s, meas_n = now - t_start, samples
    stats = {
        "micro_batch": micro,
        "accumulation": accum,
        "optimizer_steps": step,
        "scaler_skipped_steps": skips,
        "train_samples_per_s": meas_n / max(meas_s, 1e-9),
        "real_tokens_per_s": real_tokens / max(now - t_start, 1e-9),
        "padded_tokens_per_s": padded_tokens / max(now - t_start, 1e-9),
        "pad_waste_share": 1 - real_tokens / max(padded_tokens, 1),
        "train_seconds": now - t_start,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / GIB if use_cuda else None,
    }
    return history, best, stats


def versions(device) -> dict:
    names = (
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else ["cpu"]
    )
    return {
        "device": names,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
    }


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- probe


def run_probe(args) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    enc, tok, manifest = load_or_encode(args)
    device = pick_device()
    seed = 1000
    set_seeds(seed)
    meta = enc.meta
    assert_cv_only(meta["id"].to_numpy(), manifest)
    plan = fold_plan(meta, 0, manifest)
    pos = {int(i): k for k, i in enumerate(meta["id"])}
    train_idx = np.array([pos[int(i)] for i in plan["train_ids"]])
    pred_idx = np.array([pos[int(i)] for i in plan["predict_ids"]])
    eff = args.micro_batch * max(1, EFFECTIVE_BATCH // args.micro_batch)
    rng = np.random.default_rng(SAMPLE_SEED)
    n_sample = min(len(train_idx), args.probe_steps * eff)
    sample = rng.choice(train_idx, size=n_sample, replace=False)
    steps = min(args.probe_steps, -(-n_sample // eff))

    model = build_model(args, tok).to(device)
    probe = {"mode": "probe", "t": args.t, "middle_share": args.middle_share, "tiny": args.tiny}
    probe |= versions(device) | enc.shares | {"git_sha": args.git_sha}
    probe["model"] = "tiny-from-scratch" if args.tiny else args.model
    try:
        _, _, tstats = train_epochs(
            model, enc, sample, None, tok, args, seed, device, epochs=1, max_steps=steps
        )
        probe["non_finite_loss"] = None
    except NonFiniteLoss as e:
        probe["non_finite_loss"] = str(e)
        tstats = {}
    probe |= {"train": tstats}

    n_inf = min(args.probe_infer_rows, len(pred_idx))
    inf_idx = rng.choice(pred_idx, size=n_inf, replace=False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    predict(model, enc, inf_idx[: min(32, n_inf)], False, tok, args, device)  # warm-up
    sync(device)
    t0 = time.time()
    ab, ba = predict_both(model, enc, inf_idx, tok, args, device)
    sync(device)
    dt = time.time() - t0
    seq_per_s = 2 * n_inf / max(dt, 1e-9)
    probe["infer"] = {
        "rows": n_inf,
        "sequences_per_s": seq_per_s,
        "rows_both_orders_per_s": n_inf / max(dt, 1e-9),
        "batch": args.infer_batch,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / GIB if device.type == "cuda" else None,
        "finite": bool(np.isfinite(ab).all() and np.isfinite(ba).all()),
    }

    if tstats:
        sps, n = tstats["train_samples_per_s"], enc.n_cv_full
        n_train, n_inner, n_fold = n * 0.8 * 0.9, n * 0.8 * 0.1, n * 0.2
        train_h = args.epochs * n_train / sps / 3600
        eval_h = args.epochs * 2 * n_inner / seq_per_s / 3600
        pred_h = 2 * n_fold / seq_per_s / 3600
        per_fold = train_h + eval_h + pred_h
        probe["projection"] = {
            "n_cv_rows_full": n,
            "epochs": args.epochs,
            "gpu_hours_per_fold": per_fold,
            "gpu_hours_5_folds": 5 * per_fold,
            "gpu_hours_5_folds_plus_ablation_fold": 6 * per_fold,
            # two folds run at a time, so 5 folds take ceil(5 / 2) = 3 rounds
            "wall_hours_on_2_gpus_5_folds": 3 * per_fold,
            "note": "single-GPU measured rates; excludes model load, fitting and cache time",
        }
    (out / "probe.json").write_text(json.dumps(probe, indent=2) + "\n")
    print(json.dumps(probe, indent=2))


# ---------------------------------------------------------------- folds


def run_fold(args, fold: int) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = 1000 + fold
    log = {"fold": fold, "seed": seed, "tag": args.tag, "git_sha": args.git_sha}
    log |= {"t": args.t, "middle_share": args.middle_share, "epochs": args.epochs}
    log["model"] = "tiny-from-scratch" if args.tiny else args.model
    try:
        enc, tok, manifest = load_or_encode(args)
        device = pick_device()
        log |= versions(device)
        set_seeds(seed)
        meta = enc.meta
        plan = fold_plan(meta, fold, manifest)
        log["group_disjoint_checked"] = True
        pos = {int(i): k for k, i in enumerate(meta["id"])}
        tr, inn, pr = (
            np.array([pos[int(i)] for i in plan[k]])
            for k in ("train_ids", "inner_ids", "predict_ids")
        )
        log |= {"n_train": len(tr), "n_inner": len(inn), "n_predict": len(pr)}
        log |= enc.shares
        model = build_model(args, tok).to(device)
        t0 = time.time()
        history, best, stats = train_epochs(
            model, enc, tr, inn, tok, args, seed, device, args.epochs
        )
        log |= {"epochs_log": history, "best_epoch": best["epoch"], "train_stats": stats}
        if best["state"] is not None:
            model.load_state_dict(best["state"])
        t1 = time.time()
        ab, ba = predict_both(model, enc, pr, tok, args, device)
        log["predict_seconds"] = time.time() - t1
        log["total_seconds"] = time.time() - t0
        sub = meta.iloc[pr]
        frame = oof_frame(sub["id"], ab, ba, sub["both_untruncated"], sub["hard_cut"])
        assert_no_text(frame)
        frame.to_parquet(out / f"oof_fold{fold}.parquet", index=False)
        log["oof_rows"] = len(frame)
        log["fold_log_loss_swap_avg"] = log_loss(
            sub["y"].to_numpy(), frame[["p_a", "p_b", "p_tie"]].to_numpy()
        )
        code = 0
    except Exception as e:  # noqa: BLE001  recorded, never skipped; the fold produces no OOF file
        log["error"] = f"{type(e).__name__}: {e}"
        print(f"[fold {fold}] FAILED: {log['error']}", flush=True)
        code = 3
    (out / f"log_fold{fold}.json").write_text(json.dumps(log, indent=2) + "\n")
    return code


def worker_argv(args, fold: int, cache: Path) -> list[str]:
    a = [
        "--worker",
        "--mode",
        "folds",
        "--folds",
        str(fold),
        "--cache",
        str(cache),
        "--t",
        str(args.t),
        "--middle-share",
        str(args.middle_share),
        "--epochs",
        str(args.epochs),
        "--model",
        args.model,
        "--micro-batch",
        str(args.micro_batch),
        "--infer-batch",
        str(args.infer_batch),
        "--max-rows",
        str(args.max_rows),
        "--data-dir",
        args.data_dir,
        "--out-dir",
        args.out_dir,
        "--expect-train-sha256",
        args.expect_train_sha256,
        "--tag",
        args.tag,
        "--git-sha",
        args.git_sha,
    ]
    return a + (["--tiny"] if args.tiny else []) + (["--grad-ckpt"] if args.grad_ckpt else [])


def run_folds(args) -> int:
    folds = [int(f) for f in args.folds.split(",")]
    tok = make_tokenizer(args)
    df, manifest = load_verified(Path(args.data_dir), args.expect_train_sha256)
    enc = encode(args, tok, df, manifest)
    del df
    cache = cache_path(args)
    with open(cache, "wb") as f:
        pickle.dump((enc, manifest), f)
    n_gpu = torch.cuda.device_count()
    slots: queue.Queue = queue.Queue()
    for s in range(max(1, n_gpu)):
        slots.put(s)

    def job(fold: int) -> int:
        slot = slots.get()
        try:
            env = dict(os.environ)
            if n_gpu >= 2:
                env["CUDA_VISIBLE_DEVICES"] = str(slot)
            print(f"fold {fold} -> slot {slot}", flush=True)
            cmd = [sys.executable, str(Path(__file__).resolve()), *worker_argv(args, fold, cache)]
            return subprocess.run(cmd, check=False, env=env).returncode
        finally:
            slots.put(slot)

    with ThreadPoolExecutor(max_workers=max(1, n_gpu)) as pool:
        codes = list(pool.map(job, folds))
    with contextlib.suppress(OSError):
        cache.unlink()
    print(f"fold exit codes: {dict(zip(folds, codes, strict=True))}", flush=True)
    return int(any(codes))


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["probe", "folds"], required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--t", type=int, default=1024)
    ap.add_argument("--middle-share", type=float, default=MIDDLE_SHARE)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--tiny", action="store_true", help="tiny from-scratch model (dry run)")
    ap.add_argument("--max-rows", type=int, default=0, help="restrict cv rows (dry runs)")
    ap.add_argument("--probe-steps", type=int, default=200)
    ap.add_argument("--probe-infer-rows", type=int, default=2000)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--infer-batch", type=int, default=16)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", default="/kaggle/working")
    ap.add_argument("--expect-train-sha256", default=TRAIN_SHA256)
    ap.add_argument("--tag", default="")
    ap.add_argument("--git-sha", default="")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cache", default="", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    if EFFECTIVE_BATCH % args.micro_batch:
        raise SystemExit("--micro-batch must divide 16")
    if args.mode == "probe":
        run_probe(args)
        return 0
    if args.worker:
        return run_fold(args, int(args.folds))
    return run_folds(args)


if __name__ == "__main__":
    sys.exit(main())
