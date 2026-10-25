"""Rung 6: LoRA fine-tune of google/gemma-2-9b-it for 3-way a/b/tie preference classification.

Runs inside a Modal H100 container (lora/modal_app.py) or on CPU with `--tiny` (a tiny Gemma2
config built from scratch, so no gated weights are touched). Modes:
  probe     attention check (eager vs sdpa on 200 rows), `--probe-steps` optimizer steps on a random
            sample of fold 0's training rows, then both-order inference on `--probe-infer-rows`
            rows; writes probe.json with throughput, memory, the attention decision and a cost
            projection.
  fold      train on fold `--fold`'s training rows (best adapter by inner-holdout log loss at 50%
            and 100% of the single epoch), predict the held-out fold in both orders, write
            oof_fold{k}.parquet + log_fold{k}.json.
  full      train the submission adapter on ALL cv rows for one epoch (no selection, no OOF) and
            save the adapter; writes log_full.json.
  selftest  CPU checks on the tiny model: right-padding invariance, peft wiring, dtypes.

Safety: the train.csv sha256 and the split-manifest digest are verified on the FULL file before
anything is restricted; only cv rows are ever encoded (the locked test split is never loaded into
training or inference); the competition test.csv is never referenced; only ids, probabilities,
flags and aggregates are written to --out-dir (dataset text lives in memory only).

Dry-run tokenizer note: Gemma's tokenizer is gated, so `--tiny` uses the Qwen2.5 tokenizer (open).
It has a pad token but no BOS, so the tiny template omits BOS; the real Gemma path requires one.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from llm_preference_audit.deberta_inputs import (
    assert_no_text,
    fold_plan,
    length_grouped_batches,
    oof_frame,
    prepare_rows,
    swap_example,
)
from llm_preference_audit.kernel_common import GIB, TRAIN_SHA256, load_verified
from llm_preference_audit.lora_cost import PRICE_PER_HOUR, est_cost_usd, project
from llm_preference_audit.lora_inputs import Template, build_lora_ids, field_window, make_template
from llm_preference_audit.scoring import log_loss
from llm_preference_audit.train_lgbm import assert_cv_only
from llm_preference_audit.truncate import MIDDLE_SHARE

EFFECTIVE_BATCH = 16
LR = 1e-4
WEIGHT_DECAY = 0.0
WARMUP = 0.05
GRAD_CLIP = 1.0
SEED_BASE = 2000  # fold k trains with seed 2000 + k; the all-cv adapter uses 2000 + 5
FULL_SEED = SEED_BASE + 5
SAMPLE_SEED = 20261024
DEFAULT_MODEL = "google/gemma-2-9b-it"
TINY_TOKENIZER = "Qwen/Qwen2.5-0.5B"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ATTN_TOL = 1e-3  # max abs logit difference for preferring sdpa over eager
ATTN_MIN_LOGIT_SCALE = 0.05  # below this the check cannot see a difference; stay on eager
CONFIG_LOG_KEYS = ("t", "middle_share", "epochs", "micro_batch", "lr", "git_sha", "model")


class NonFiniteLoss(RuntimeError):
    pass


class Encoded(NamedTuple):
    meta: pd.DataFrame  # id, y, fold, group, hard_cut, both_untruncated, n_* (no text)
    p: list
    a: list
    b: list
    n_cv_full: int
    shares: dict


# ---------------------------------------------------------------- inputs


def make_tokenizer(args):
    tok = AutoTokenizer.from_pretrained(TINY_TOKENIZER if args.tiny else args.model)
    tok.model_max_length = 10**9  # we fit to the window ourselves
    if tok.pad_token_id is None:
        raise SystemExit("tokenizer has no pad token")
    if not args.tiny and tok.bos_token_id is None:
        raise SystemExit("the real tokenizer must have a BOS token")
    return tok


def tokenizer_template(tok) -> Template:
    return make_template(lambda s: tok(s, add_special_tokens=False).input_ids, tok.bos_token_id)


def encode(args, tok, template: Template, df: pd.DataFrame, manifest: pd.DataFrame) -> Encoded:
    n_cv_full = int((manifest["split"] == "cv").sum())
    if args.max_rows:
        cv_ids = manifest.loc[manifest["split"] == "cv", "id"].to_numpy()
        keep = np.random.default_rng(SAMPLE_SEED).choice(cv_ids, size=args.max_rows, replace=False)
        df = df[df["id"].isin(keep)]

    def count(s: str) -> int:
        return len(tok(s, add_special_tokens=False).input_ids)

    t0 = time.time()
    rows = prepare_rows(df, manifest, count, field_window(args.t, template), args.middle_share)
    print(f"fitted {len(rows)} cv rows to T={args.t} in {time.time() - t0:.0f}s", flush=True)

    def ids(col: str) -> list:
        enc = tok(rows[col].tolist(), add_special_tokens=False)["input_ids"]
        return [np.asarray(x, dtype=np.int32) for x in enc]

    p, a, b = ids("prompt_text"), ids("a_text"), ids("b_text")
    meta = rows.drop(columns=["prompt_text", "a_text", "b_text"])
    assert_cv_only(meta["id"].to_numpy(), manifest)  # the locked test split never gets this far
    shares = {
        "n_rows": len(meta),
        "hard_cut_share": float(meta["hard_cut"].mean()),
        "both_untruncated_share": float(meta["both_untruncated"].mean()),
    }
    return Encoded(meta, p, a, b, n_cv_full, shares)


def make_seq(enc: Encoded, i: int, swap: bool, template: Template, limit: int):
    y = int(enc.meta["y"].iat[i])
    p, a, b = enc.p[i], enc.a[i], enc.b[i]
    if swap:
        p, a, b, y = swap_example(p, a, b, y)
    return build_lora_ids(template, p, a, b, limit), y


def collate(seqs: list[list[int]], pad_id: int, device):
    """Right-pad to the longest row. Gemma2ForSequenceClassification pools the rightmost
    non-pad token, so right padding is correct as long as pad_token_id is set on the config."""
    n = max(map(len, seqs))
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[r, : len(s)] = 1
    return ids.to(device), mask.to(device)


# ---------------------------------------------------------------- model


def tiny_config(vocab: int, pad_id: int, t: int, softcap: float = 50.0):
    from transformers import Gemma2Config

    return Gemma2Config(
        vocab_size=vocab,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        query_pre_attn_scalar=16,
        max_position_embeddings=t + 16,
        sliding_window=64,
        attn_logit_softcapping=softcap,
        final_logit_softcapping=30.0,
        num_labels=3,
        pad_token_id=pad_id,
    )


def build_tiny_model(vocab, pad_id, t, attn, dtype, softcap=50.0, seed=0):
    from transformers import Gemma2ForSequenceClassification

    torch.manual_seed(seed)
    cfg = tiny_config(vocab, pad_id, t, softcap)
    model = Gemma2ForSequenceClassification._from_config(cfg, attn_implementation=attn, dtype=dtype)
    with torch.no_grad():  # a larger head so logits (and an attention-impl difference) are visible
        model.score.weight.normal_(0, 1.0)
    return model


def load_base(args, tok, attn: str):
    """Base classifier in bf16 with the tokenizer's pad id; asserts nothing is fp32/fp16."""
    if args.tiny:
        model = build_tiny_model(
            len(tok), tok.pad_token_id, args.t, attn, torch.bfloat16, args.tiny_softcap
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=3,
            dtype=torch.bfloat16,
            attn_implementation=attn,
            pad_token_id=tok.pad_token_id,
        )
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    bad = {str(p.dtype) for p in model.parameters() if p.dtype != torch.bfloat16}
    if bad:
        raise SystemExit(f"base parameters must all be bf16, found {sorted(bad)}")
    return model


def wrap_lora(model, args):
    """Gradient checkpointing, LoRA on every decoder linear, the `score` head fully trainable.

    LoRA and head params are cast to fp32 (fp32 masters; bf16 autocast does the compute), the
    base stays bf16. Returns the peft model.
    """
    import peft

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = peft.LoraConfig(
        task_type=peft.TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGETS,
        modules_to_save=["score"],
        bias="none",
    )
    pm = peft.get_peft_model(model, cfg)
    for p in pm.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    trainable = {n: p for n, p in pm.named_parameters() if p.requires_grad}
    bad = {n for n, p in trainable.items() if p.dtype != torch.float32}
    if bad:
        raise SystemExit(f"trainable params must be fp32, found non-fp32: {sorted(bad)[:3]}")
    if not any(".score." in n or n.endswith("score.weight") for n in trainable):
        raise SystemExit("the classification head `score` is not trainable")
    if not any("lora_A" in n for n in trainable):
        raise SystemExit("no LoRA parameters are trainable")
    return pm


def trainable_state(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model, state: dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in state.items():
            params[n].copy_(v.to(params[n].device))


def autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


@torch.no_grad()
def predict(model, enc, idx, swap, tok, template, args, device) -> tuple[np.ndarray, int]:
    """Probabilities in PRESENTED order ([p_first, p_second, p_tie]) and real tokens processed."""
    model.eval()
    seqs = [make_seq(enc, int(i), swap, template, args.t)[0] for i in idx]
    order = np.argsort([len(s) for s in seqs], kind="stable")
    out = np.zeros((len(seqs), 3), dtype=np.float64)
    tokens = 0
    for s in range(0, len(order), args.infer_batch):
        sel = order[s : s + args.infer_batch]
        ids, mask = collate([seqs[j] for j in sel], tok.pad_token_id, device)
        with autocast(device):
            logits = model(input_ids=ids, attention_mask=mask).logits
        out[sel] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        tokens += int(mask.sum())
    return out, tokens


def predict_both(model, enc, idx, tok, template, args, device):
    ab, n1 = predict(model, enc, idx, False, tok, template, args, device)
    ba, n2 = predict(model, enc, idx, True, tok, template, args, device)
    return ab, ba, n1 + n2


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


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_name(device) -> str:
    return torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"


def peak_gib(device) -> float | None:
    return torch.cuda.max_memory_allocated() / GIB if device.type == "cuda" else None


# ---------------------------------------------------------------- training


def train(model, enc, template, train_idx, eval_idx, tok, args, seed, device, max_steps=None):
    """One epoch. With `eval_idx`, scores the inner holdout at 50% and 100% of the epoch and keeps
    the better trainable state (LoRA + head only). Returns (history, best, stats)."""
    from transformers import get_cosine_schedule_with_warmup

    micro = args.micro_batch
    accum = max(1, EFFECTIVE_BATCH // micro)
    lengths = np.minimum(
        enc.meta[["n_prompt", "n_a", "n_b"]].to_numpy()[train_idx].sum(axis=1) + template.overhead,
        args.t,
    )
    steps_per_epoch = math.ceil(math.ceil(len(train_idx) / micro) / accum)
    total_steps = min(max_steps, steps_per_epoch) if max_steps else steps_per_epoch
    eval_at = {}
    if eval_idx is not None and len(eval_idx) and not max_steps:
        eval_at = {max(1, total_steps // 2): 0.5, total_steps: 1.0}
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    sched = get_cosine_schedule_with_warmup(opt, int(WARMUP * total_steps), total_steps)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    swaps = np.random.default_rng(seed).random(len(train_idx)) < 0.5
    batches = length_grouped_batches(lengths, micro, seed)
    groups = [batches[i : i + accum] for i in range(0, len(batches), accum)]
    history, best = [], {"loss": float("inf"), "epoch": None, "state": None}
    step = samples = real_tokens = padded_tokens = 0
    loss_sum, loss_n = 0.0, 0
    train_time = eval_time = 0.0
    warm_time = warm_samples = None
    model.train()
    for group in groups:
        t_step = time.perf_counter()
        for bidx in group:
            pairs = [
                make_seq(enc, int(train_idx[j]), bool(swaps[j]), template, args.t) for j in bidx
            ]
            ids, mask = collate([s for s, _ in pairs], tok.pad_token_id, device)
            y = torch.tensor([y for _, y in pairs], device=device)
            with autocast(device):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), y)
            if not torch.isfinite(loss):
                raise NonFiniteLoss(f"non-finite loss at step {step}")
            (loss / len(group)).backward()
            loss_sum += loss.item() * len(bidx)
            loss_n += len(bidx)
            samples += len(bidx)
            real_tokens += int(mask.sum())
            padded_tokens += int(mask.numel())
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        sync(device)
        train_time += time.perf_counter() - t_step
        if step == 2:
            warm_time, warm_samples = train_time, samples
        if step % 50 == 0 or step == total_steps:
            print(f"step {step}/{total_steps} loss {loss_sum / loss_n:.4f}", flush=True)
        if step in eval_at:
            t0 = time.perf_counter()
            ab, ba, _ = predict_both(model, enc, eval_idx, tok, template, args, device)
            P = swap_avg(ab, ba)
            if not np.isfinite(P).all():
                raise NonFiniteLoss("non-finite inner-holdout probabilities")
            rec = {
                "epoch": eval_at[step],
                "step": step,
                "train_loss": loss_sum / max(1, loss_n),
                "inner_log_loss": log_loss(enc.meta["y"].to_numpy()[eval_idx], P),
            }
            rec["eval_seconds"] = time.perf_counter() - t0
            eval_time += rec["eval_seconds"]
            history.append(rec)
            print(f"eval: {rec}", flush=True)
            if rec["inner_log_loss"] < best["loss"]:
                best.update(loss=rec["inner_log_loss"], epoch=eval_at[step])
                best["state"] = trainable_state(model)
            model.train()
        if step >= total_steps:
            break
    if warm_time is not None and samples > warm_samples and train_time > warm_time:
        sps = (samples - warm_samples) / (train_time - warm_time)
    else:
        sps = samples / max(train_time, 1e-9)
    stats = {
        "micro_batch": micro,
        "accumulation": accum,
        "optimizer_steps": step,
        "train_samples": samples,
        "train_samples_per_s": sps,
        "train_real_tokens_per_s": real_tokens / max(train_time, 1e-9),
        "train_padded_tokens_per_s": padded_tokens / max(train_time, 1e-9),
        "pad_waste_share": 1 - real_tokens / max(padded_tokens, 1),
        "train_seconds": train_time,
        "eval_seconds": eval_time,
        "mean_train_loss": loss_sum / max(1, loss_n),
        "peak_gpu_gib": peak_gib(device),
    }
    return history, best, stats


# ---------------------------------------------------------------- shared run setup


def expected_config(args, epochs: int = 1) -> dict:
    return {
        "t": args.t,
        "middle_share": args.middle_share,
        "epochs": epochs,
        "micro_batch": args.micro_batch,
        "lr": args.lr,
        "git_sha": args.git_sha,
        "model": "tiny-from-scratch" if args.tiny else args.model,
    }


def base_log(args, device, attn: str, template: Template) -> dict:
    log = {"tag": args.tag} | expected_config(args)
    log |= {
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "attn_implementation": attn,
        "dtype": "base bf16; LoRA+head fp32 masters; bf16 autocast",
        "template_overhead": template.overhead,
        "price_per_hour": PRICE_PER_HOUR,
        "gpu_name": gpu_name(device),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
    }
    try:
        import peft

        log["peft"] = peft.__version__
    except ImportError:  # pragma: no cover
        pass
    return log


def resolve_attn(args) -> str:
    return "eager" if args.attn == "auto" else args.attn


def setup(args):
    """Verified data, tokenizer, template, encoded cv rows."""
    tok = make_tokenizer(args)
    template = tokenizer_template(tok)
    df, manifest = load_verified(Path(args.data_dir), args.expect_train_sha256)
    enc = encode(args, tok, template, df, manifest)
    del df
    return tok, template, manifest, enc


def positions(meta: pd.DataFrame, ids) -> np.ndarray:
    pos = {int(i): k for k, i in enumerate(meta["id"])}
    return np.array([pos[int(i)] for i in ids])


# ---------------------------------------------------------------- attention check


@torch.no_grad()
def attention_check(model, enc, tok, template, args, device, n_rows: int) -> dict:
    """Same weights, eager vs sdpa, on `n_rows` rows: max abs logit difference.

    transformers' sdpa path swallows Gemma2's `softcap` kwarg (checked in the installed source),
    so a difference is expected on real attention scores; this measures it. sdpa is chosen only
    when the difference is within ATTN_TOL AND the logits are large enough to show one.
    """
    rng = np.random.default_rng(SAMPLE_SEED)
    idx = rng.choice(len(enc.meta), size=min(n_rows, len(enc.meta)), replace=False)
    seqs = [make_seq(enc, int(i), False, template, args.t)[0] for i in idx]
    order = np.argsort([len(s) for s in seqs], kind="stable")
    outs = {}
    model.eval()
    for impl in ("eager", "sdpa"):
        model.set_attn_implementation(impl)
        logits = np.zeros((len(seqs), 3), dtype=np.float64)
        for s in range(0, len(order), args.infer_batch):
            sel = order[s : s + args.infer_batch]
            ids, mask = collate([seqs[j] for j in sel], tok.pad_token_id, device)
            with autocast(device):
                logits[sel] = model(input_ids=ids, attention_mask=mask).logits.float().cpu().numpy()
        outs[impl] = logits
    diff = float(np.abs(outs["eager"] - outs["sdpa"]).max())
    scale = float(np.abs(outs["eager"]).max())
    sensitive = scale >= ATTN_MIN_LOGIT_SCALE
    choice = "sdpa" if (diff <= ATTN_TOL and sensitive) else "eager"
    return {
        "rows": len(seqs),
        "max_abs_logit_diff": diff,
        "max_abs_logit_eager": scale,
        "check_sensitive": sensitive,
        "tolerance": ATTN_TOL,
        "choice": choice,
    }


# ---------------------------------------------------------------- probe


def run_probe(args) -> int:
    t_run = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    seed = SEED_BASE
    set_seeds(seed)
    t0 = time.time()
    tok, template, manifest, enc = setup(args)
    data_s = time.time() - t0
    meta = enc.meta
    plan = fold_plan(meta, 0, manifest)
    train_idx, pred_idx = positions(meta, plan["train_ids"]), positions(meta, plan["predict_ids"])
    eff = args.micro_batch * max(1, EFFECTIVE_BATCH // args.micro_batch)
    rng = np.random.default_rng(SAMPLE_SEED)
    n_sample = min(len(train_idx), args.probe_steps * eff)
    sample = rng.choice(train_idx, size=n_sample, replace=False)  # random = length-representative
    steps = min(args.probe_steps, -(-n_sample // eff))

    t0 = time.time()
    model = load_base(args, tok, "eager").to(device)
    load_s = time.time() - t0
    probe = {"mode": "probe", "tiny": args.tiny, "seed": seed} | enc.shares
    probe |= base_log(args, device, "pending", template)
    probe["model_load_seconds"] = load_s
    probe["data_seconds"] = data_s

    check = attention_check(model, enc, tok, template, args, device, args.attn_rows)
    probe["attention_check"] = check
    attn = check["choice"] if args.attn == "auto" else args.attn
    probe["attention_decision"] = attn
    probe["attn_implementation"] = attn
    model.set_attn_implementation(attn)
    model = wrap_lora(model, args)

    try:
        _, _, tstats = train(model, enc, template, sample, None, tok, args, seed, device, steps)
        probe["non_finite_loss"] = None
    except NonFiniteLoss as e:
        probe["non_finite_loss"] = str(e)
        tstats = {}
    probe["train"] = tstats

    n_inf = min(args.probe_infer_rows, len(pred_idx))
    inf_idx = rng.choice(pred_idx, size=n_inf, replace=False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    predict(model, enc, inf_idx[: min(16, n_inf)], False, tok, template, args, device)  # warm-up
    sync(device)
    t0 = time.perf_counter()
    ab, ba, tokens = predict_both(model, enc, inf_idx, tok, template, args, device)
    sync(device)
    dt = time.perf_counter() - t0
    seq_per_s = 2 * n_inf / max(dt, 1e-9)
    probe["infer"] = {
        "rows": n_inf,
        "sequences_per_s": seq_per_s,
        "real_tokens_per_s": tokens / max(dt, 1e-9),
        "batch": args.infer_batch,
        "peak_gpu_gib": peak_gib(device),
        "finite": bool(np.isfinite(ab).all() and np.isfinite(ba).all()),
    }
    probe["peak_gpu_gib"] = peak_gib(device)
    probe["loss_finite"] = probe["non_finite_loss"] is None and bool(tstats)
    if tstats:
        probe["projection"] = project(
            n_cv=enc.n_cv_full,
            train_samples_per_s=tstats["train_samples_per_s"],
            infer_seqs_per_s=seq_per_s,
            load_seconds=load_s,
        )
    probe["wall_seconds"] = time.time() - t_run
    probe["est_cost_usd"] = est_cost_usd(probe["wall_seconds"])
    (out / "probe.json").write_text(json.dumps(probe, indent=2) + "\n")
    print(json.dumps(probe, indent=2))
    return 0


# ---------------------------------------------------------------- fold / full


def finish_log(log: dict, t_run: float, device) -> None:
    log["wall_seconds"] = time.time() - t_run
    log["est_cost_usd"] = est_cost_usd(log["wall_seconds"])
    log["peak_gpu_gib"] = peak_gib(device)


def run_fold(args) -> int:
    t_run = time.time()
    fold = args.fold
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = SEED_BASE + fold
    attn = resolve_attn(args)
    log = {"mode": "fold", "fold": fold, "seed": seed, "attn_implementation": attn}
    code = 3
    device = pick_device()
    try:
        set_seeds(seed)
        t0 = time.time()
        tok, template, manifest, enc = setup(args)
        log |= base_log(args, device, attn, template) | enc.shares
        log["data_seconds"] = time.time() - t0
        meta = enc.meta
        plan = fold_plan(meta, fold, manifest)
        log["group_disjoint_checked"] = True
        tr, inn, pr = (positions(meta, plan[k]) for k in ("train_ids", "inner_ids", "predict_ids"))
        log |= {"n_train": len(tr), "n_inner": len(inn), "n_predict": len(pr)}
        t0 = time.time()
        model = wrap_lora(load_base(args, tok, attn).to(device), args)
        log["model_load_seconds"] = time.time() - t0
        history, best, stats = train(model, enc, template, tr, inn, tok, args, seed, device)
        log |= {"epochs_log": history, "best_epoch": best["epoch"], "train_stats": stats}
        if best["state"] is not None:
            load_trainable_state(model, best["state"])
        elif args.max_rows:
            log["inner_holdout_empty_dry_run"] = True
            print(f"[fold {fold}] empty inner holdout; using the final state (--max-rows dry run)")
        else:
            raise RuntimeError("inner holdout is empty: no best state to select")
        t1 = time.time()
        ab, ba, tokens = predict_both(model, enc, pr, tok, template, args, device)
        log["predict_seconds"] = time.time() - t1
        log["infer_real_tokens_per_s"] = tokens / max(log["predict_seconds"], 1e-9)
        sub = meta.iloc[pr]
        frame = oof_frame(sub["id"], ab, ba, sub["both_untruncated"], sub["hard_cut"])
        assert_no_text(frame)
        log["fold_log_loss_swap_avg"] = log_loss(
            sub["y"].to_numpy(), frame[["p_a", "p_b", "p_tie"]].to_numpy()
        )
        log["oof_rows"] = len(frame)
        finish_log(log, t_run, device)
        (out / f"log_fold{fold}.json").write_text(json.dumps(log, indent=2) + "\n")
        frame.to_parquet(out / f"oof_fold{fold}.parquet", index=False)  # only after a clean score
        return 0
    except Exception as e:  # noqa: BLE001  recorded, never skipped; the fold produces no OOF file
        log["error"] = f"{type(e).__name__}: {e}"
        print(f"[fold {fold}] FAILED: {log['error']}", flush=True)
        log |= {k: v for k, v in expected_config(args).items() if k not in log}
        finish_log(log, t_run, device)
        (out / f"log_fold{fold}.json").write_text(json.dumps(log, indent=2) + "\n")
        return code


def run_full(args) -> int:
    """The submission adapter: all cv rows, one epoch, no selection, no OOF."""
    t_run = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    attn = resolve_attn(args)
    log = {"mode": "full", "seed": FULL_SEED, "attn_implementation": attn}
    device = pick_device()
    try:
        set_seeds(FULL_SEED)
        t0 = time.time()
        tok, template, manifest, enc = setup(args)
        log |= base_log(args, device, attn, template) | enc.shares
        log["data_seconds"] = time.time() - t0
        meta = enc.meta
        assert_cv_only(meta["id"].to_numpy(), manifest)
        if "split" in meta.columns and (meta["split"] != "cv").any():
            raise AssertionError("non-cv rows reached the full-adapter training set")
        all_idx = np.arange(len(meta))
        log["n_train"] = len(all_idx)
        t0 = time.time()
        model = wrap_lora(load_base(args, tok, attn).to(device), args)
        log["model_load_seconds"] = time.time() - t0
        _, _, stats = train(model, enc, template, all_idx, None, tok, args, FULL_SEED, device)
        log["train_stats"] = stats
        adapter = Path(args.adapter_dir)
        adapter.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter))
        (adapter / "adapter_meta.json").write_text(json.dumps(expected_config(args), indent=2))
        log["adapter_dir"] = str(adapter)
        code = 0
    except Exception as e:  # noqa: BLE001
        log["error"] = f"{type(e).__name__}: {e}"
        print(f"[full] FAILED: {log['error']}", flush=True)
        log |= {k: v for k, v in expected_config(args).items() if k not in log}
        code = 3
    finish_log(log, t_run, device)
    (out / "log_full.json").write_text(json.dumps(log, indent=2) + "\n")
    return code


def check_worker_config(out_dir: Path, fold: int, want: dict) -> None:
    """The fold's log must record the config the launcher asked for."""
    log = json.loads((Path(out_dir) / f"log_fold{fold}.json").read_text())
    got = {k: log.get(k) for k in want}
    if got != want:
        raise SystemExit(f"fold {fold} ran with {got}, launcher asked for {want}")


# ---------------------------------------------------------------- selftest


def run_selftest(args) -> int:
    """CPU checks on the tiny model: padding invariance, peft wiring, dtypes."""
    torch.manual_seed(0)
    vocab, pad = 300, 0
    for dtype, tol in ((torch.float32, 1e-4), (torch.bfloat16, 5e-2)):
        for attn in ("eager", "sdpa"):
            model = build_tiny_model(vocab, pad, 64, attn, dtype).eval()
            ids = torch.randint(1, vocab, (3, 20))
            lens = [20, 13, 7]
            logits_alone, logits_padded = [], []
            for r, n in enumerate(lens):
                with torch.no_grad():
                    logits_alone.append(model(input_ids=ids[r : r + 1, :n]).logits)
            padded = torch.full((3, 20), pad)
            mask = torch.zeros((3, 20), dtype=torch.long)
            for r, n in enumerate(lens):
                padded[r, :n] = ids[r, :n]
                mask[r, :n] = 1
            with torch.no_grad():
                logits_padded = model(input_ids=padded, attention_mask=mask).logits
            diff = max(
                float((logits_alone[r][0].float() - logits_padded[r].float()).abs().max())
                for r in range(3)
            )
            print(f"padding invariance {dtype} {attn}: max abs diff {diff:.2e} (tol {tol})")
            if diff > tol:
                raise SystemExit(f"padded and unpadded logits differ by {diff} > {tol}")
    base = build_tiny_model(vocab, pad, 64, "eager", torch.bfloat16)
    args.lora_r, args.lora_alpha, args.lora_dropout = 16, 32, 0.05
    pm = wrap_lora(base, args)
    names = [n for n, p in pm.named_parameters() if p.requires_grad]
    print(f"{len(names)} trainable tensors; head: {[n for n in names if 'score' in n]}")
    return 0


# ---------------------------------------------------------------- cli


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["probe", "fold", "full", "selftest"], required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--t", type=int, default=2048)
    ap.add_argument("--middle-share", type=float, default=MIDDLE_SHARE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tiny", action="store_true", help="tiny from-scratch Gemma2 (dry run)")
    ap.add_argument("--tiny-softcap", type=float, default=0.5, help=argparse.SUPPRESS)
    ap.add_argument("--max-rows", type=int, default=0, help="restrict cv rows (dry runs)")
    ap.add_argument("--probe-steps", type=int, default=200)
    ap.add_argument("--probe-infer-rows", type=int, default=2000)
    ap.add_argument("--attn-rows", type=int, default=200)
    ap.add_argument("--attn", choices=["auto", "eager", "sdpa"], default="auto")
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--infer-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--data-dir", default="/vol/data")
    ap.add_argument("--out-dir", required=False, default="/vol/out/run")
    ap.add_argument("--adapter-dir", default="/vol/adapters/run")
    ap.add_argument("--expect-train-sha256", default=TRAIN_SHA256)
    ap.add_argument("--tag", default="")
    ap.add_argument("--git-sha", default="")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    if EFFECTIVE_BATCH % args.micro_batch:
        raise SystemExit("--micro-batch must divide 16")
    if not 0 <= args.fold <= 4:
        raise SystemExit("--fold must be 0-4")
    return {"probe": run_probe, "fold": run_fold, "full": run_full, "selftest": run_selftest}[
        args.mode
    ](args)


if __name__ == "__main__":
    sys.exit(main())
