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

Rung 6b generalisation (Qwen3-1.7B on Kaggle T4s; the Gemma/Modal defaults are unchanged):
`--model`, `--precision {bf16,fp16-autocast,fp32}`, `--fp32-head-norm`, Qwen3 as a second
architecture (`--tiny-arch qwen3` for the CPU dry run), and the Kaggle-side probe in
`lora_kaggle.py`, which reuses everything here. fp16-autocast: base weights fp16, LoRA params and
the score head fp32, torch.autocast(fp16) + GradScaler, unscale before the 1.0 clip, and any
non-finite loss stops the run with a first-overflow report (forward hooks, one re-run).
Qwen3 is trained with the same plain template as Gemma, NOT its chat template (no <|im_start|>,
no <think> tags): the task is classification from the last token, not generation.

Dry-run tokenizer note: Gemma's tokenizer is gated, so `--tiny` uses the Qwen2.5 tokenizer (open).
It has a pad token but no BOS, so the tiny template omits BOS; the real Gemma path requires one.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from llm_preference_audit.cv_common import assert_cv_only
from llm_preference_audit.deberta_inputs import (
    assert_no_text,
    fold_plan,
    length_grouped_batches,
    oof_frame,
    prepare_rows,
    swap_example,
)
from llm_preference_audit.kernel_common import GIB, TRAIN_SHA256, load_verified
from llm_preference_audit.lora_cost import (
    PRICE_PER_HOUR,
    Deadline,
    DeadlineExceeded,
    clear_stale_outputs,
    est_cost_usd,
    project,
)
from llm_preference_audit.lora_g0 import attention_tolerance, choose_attention
from llm_preference_audit.lora_inputs import Template, build_lora_ids, field_window, make_template
from llm_preference_audit.scoring import log_loss
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
TINY_QWEN3_TOKENIZER = "Qwen/Qwen3-1.7B"  # public; Qwen3's tokenizer has a pad token, no BOS
PRECISIONS = ("bf16", "fp16-autocast", "fp32")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ATTN_TOL = 1e-3  # max abs logit difference for preferring sdpa over eager
ATTN_MIN_LOGIT_SCALE = 0.05  # below this the check cannot see a difference; stay on eager
CONFIG_LOG_KEYS = ("t", "middle_share", "epochs", "micro_batch", "lr", "git_sha", "model")


class NonFiniteLoss(RuntimeError):
    """A non-finite training loss. `first_overflow` is the hook report naming the first module
    whose output held inf/nan when the failing micro-batch was re-run (None if not produced)."""

    def __init__(self, msg: str, first_overflow: dict | None = None):
        super().__init__(msg)
        self.first_overflow = first_overflow


class WorkerConfigMismatch(RuntimeError):
    """A worker's log records a different config than the launcher asked for."""


DEADLINE_EXIT = 4


def tick(args, stage: str) -> None:
    """Stop cleanly (DeadlineExceeded) once the in-process wall-clock budget is spent."""
    deadline = getattr(args, "deadline", None)
    if deadline is not None:
        deadline.check(stage)


class Encoded(NamedTuple):
    meta: pd.DataFrame  # id, y, fold, group, hard_cut, both_untruncated, n_* (no text)
    p: list
    a: list
    b: list
    n_cv_full: int
    shares: dict


# ---------------------------------------------------------------- inputs


def model_family(args) -> str:
    """ "qwen3" or "gemma2". Tiny runs follow --tiny-arch; a local model dir follows its
    config.json model_type; otherwise the model name decides (Gemma is the default)."""
    if args.tiny:
        return getattr(args, "tiny_arch", "gemma2")
    model_dir = getattr(args, "model_dir", "") or ""
    cfg = Path(model_dir) / "config.json" if model_dir else None
    if cfg is not None and cfg.is_file():
        return "qwen3" if json.loads(cfg.read_text()).get("model_type") == "qwen3" else "gemma2"
    return "qwen3" if "qwen3" in args.model.lower().replace("-", "") else "gemma2"


def model_ref(args) -> str:
    """What from_pretrained loads: the local directory when one is given, else the hub name."""
    return getattr(args, "model_dir", "") or args.model


def make_tokenizer(args):
    if args.tiny:
        name = TINY_QWEN3_TOKENIZER if model_family(args) == "qwen3" else TINY_TOKENIZER
    else:
        name = model_ref(args)
    tok = AutoTokenizer.from_pretrained(
        name,
        local_files_only=bool(getattr(args, "local_files_only", False)) and not args.tiny,
    )
    tok.model_max_length = 10**9  # we fit to the window ourselves
    if tok.pad_token_id is None:
        raise SystemExit("tokenizer has no pad token")
    # Gemma's template needs a BOS; Qwen3 has none (the template then starts at the prompt).
    if not args.tiny and model_family(args) != "qwen3" and tok.bos_token_id is None:
        raise SystemExit("the real tokenizer must have a BOS token")
    return tok


def tokenizer_template(tok) -> Template:
    return make_template(lambda s: tok(s, add_special_tokens=False).input_ids, tok.bos_token_id)


def text_ids(tok, texts: list[str]) -> list[list[int]]:
    """Token ids for dataset text. split_special_tokens keeps a literal "<|endoftext|>" in a
    response from becoming Qwen's pad id (and a special token generally) mid-sequence."""
    return tok(texts, add_special_tokens=False, split_special_tokens=True)["input_ids"]


def encode(args, tok, template: Template, df: pd.DataFrame, manifest: pd.DataFrame) -> Encoded:
    n_cv_full = int((manifest["split"] == "cv").sum())
    if args.max_rows:
        cv_ids = manifest.loc[manifest["split"] == "cv", "id"].to_numpy()
        keep = np.random.default_rng(SAMPLE_SEED).choice(cv_ids, size=args.max_rows, replace=False)
        df = df[df["id"].isin(keep)]

    def count(s: str) -> int:
        return len(text_ids(tok, [s])[0])

    t0 = time.time()
    rows = prepare_rows(df, manifest, count, field_window(args.t, template), args.middle_share)
    print(f"fitted {len(rows)} cv rows to T={args.t} in {time.time() - t0:.0f}s", flush=True)

    def ids(col: str) -> list:
        enc = text_ids(tok, rows[col].tolist())
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


def tiny_config_qwen3(vocab: int, pad_id: int, t: int):
    from transformers import Qwen3Config

    return Qwen3Config(
        vocab_size=vocab,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=t + 16,
        num_labels=3,
        pad_token_id=pad_id,
        use_cache=False,
    )


def build_tiny_model(vocab, pad_id, t, attn, dtype, softcap=50.0, seed=0, arch="gemma2"):
    torch.manual_seed(seed)
    if arch == "qwen3":
        from transformers import Qwen3ForSequenceClassification as Cls

        cfg = tiny_config_qwen3(vocab, pad_id, t)
    else:
        from transformers import Gemma2ForSequenceClassification as Cls

        cfg = tiny_config(vocab, pad_id, t, softcap)
    model = Cls._from_config(cfg, attn_implementation=attn, dtype=dtype)
    with torch.no_grad():  # a larger head so logits (and an attention-impl difference) are visible
        model.score.weight.normal_(0, 1.0)
    return model


def precision_dtypes(args, device=None):
    """(base weight dtype, autocast dtype or None) for `args.precision` on this device.

    fp16 autocast does not exist on CPU, so a CPU dry run of fp16-autocast substitutes bf16 for
    both (the GradScaler is then off; `--force-scaler` turns it on to exercise that code path).
    """
    precision = getattr(args, "precision", "bf16")
    if precision == "bf16":
        return torch.bfloat16, torch.bfloat16
    if precision == "fp32":
        return torch.float32, None
    if precision != "fp16-autocast":
        raise SystemExit(f"unknown --precision {precision!r}")
    device = device or pick_device()
    if device.type == "cuda":
        return torch.float16, torch.float16
    return torch.bfloat16, torch.bfloat16


def fp16_emulated(args, device=None) -> bool:
    return getattr(args, "precision", "bf16") == "fp16-autocast" and (
        (device or pick_device()).type != "cuda"
    )


def dtype_label(dtype) -> str:
    return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[dtype]


class Fp32Head(torch.nn.Linear):
    """A Linear that always computes in fp32, even inside an autocast region (autocast would
    otherwise cast its input and weight to fp16). Same parameter names as the Linear it replaces."""

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            bias = None if self.bias is None else self.bias.float()
            return torch.nn.functional.linear(x.float(), self.weight.float(), bias)


def make_head_norm_fp32(model) -> None:
    """Keep the final norm and the `score` head in fp32 (the T4 fp16 overflow guard)."""
    old = model.score
    head = Fp32Head(
        old.in_features, old.out_features, bias=old.bias is not None, dtype=torch.float32
    ).to(old.weight.device)
    with torch.no_grad():
        head.weight.copy_(old.weight.float())
        if old.bias is not None:
            head.bias.copy_(old.bias.float())
    model.score = head
    model.base_model.norm.float()  # RMSNorm then returns fp32 (fp32 weight * fp16 input)


def load_base(args, tok, attn: str, device=None):
    """Base classifier in the precision's weight dtype with the tokenizer's pad id; asserts every
    parameter has exactly that dtype (bf16 by default: the Gemma/Modal behaviour)."""
    base_dtype, _ = precision_dtypes(args, device)
    if args.tiny:
        model = build_tiny_model(
            len(tok),
            tok.pad_token_id,
            args.t,
            attn,
            base_dtype,
            args.tiny_softcap,
            arch=model_family(args),
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_ref(args),
            num_labels=3,
            dtype=base_dtype,
            attn_implementation=attn,
            pad_token_id=tok.pad_token_id,
            local_files_only=bool(getattr(args, "local_files_only", False)),
        )
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    bad = {str(p.dtype) for p in model.parameters() if p.dtype != base_dtype}
    if bad:
        raise SystemExit(
            f"base parameters must all be {dtype_label(base_dtype)}, found {sorted(bad)}"
        )
    if getattr(args, "fp32_head_norm", False):
        make_head_norm_fp32(model)
    return model


_FP32_BASE_OK = re.compile(r"(^|\.)model\.norm\.weight$|(^|\.)score\.")


def assert_precision_dtypes(pm, args, device=None) -> dict:
    """Before training: every trainable parameter (LoRA + head) is fp32, and every frozen base
    parameter has the precision's weight dtype, except the final norm and the head when
    --fp32-head-norm keeps them fp32. Raises SystemExit; returns the dtype census."""
    base_dtype, _ = precision_dtypes(args, device)
    keep32 = getattr(args, "fp32_head_norm", False)
    census: dict[str, set] = {"trainable": set(), "frozen": set()}
    for n, p in pm.named_parameters():
        if p.requires_grad:
            census["trainable"].add(str(p.dtype))
            if p.dtype != torch.float32:
                raise SystemExit(f"trainable parameter {n} is {p.dtype}, must be fp32")
        else:
            census["frozen"].add(str(p.dtype))
            allowed = {base_dtype} | (
                {torch.float32} if keep32 and _FP32_BASE_OK.search(n) else set()
            )
            if p.dtype not in allowed:
                raise SystemExit(f"frozen parameter {n} is {p.dtype}, expected {base_dtype}")
    if keep32 and not isinstance(
        getattr(pm.base_model.model, "score", None).modules_to_save["default"], Fp32Head
    ):
        raise SystemExit("--fp32-head-norm requested but the trainable head is not an Fp32Head")
    return {k: sorted(v) for k, v in census.items()}


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
    assert_precision_dtypes(pm, args)
    return pm


def trainable_state(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model, state: dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in state.items():
            params[n].copy_(v.to(params[n].device))


def autocast(device, args=None):
    """bf16 autocast unless `args.precision` says otherwise (fp16-autocast, or fp32: no autocast)."""
    if args is None:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    _, ac = precision_dtypes(args, device)
    return torch.autocast(
        device_type=device.type,
        dtype=ac if ac is not None else torch.bfloat16,
        enabled=ac is not None,
    )


def make_scaler(args, device):
    """GradScaler for fp16-autocast on CUDA. Disabled otherwise (bf16/fp32 need none), where every
    scaler call is a no-op and the loop is the original one. `--force-scaler` is a CPU test hook."""
    enabled = (
        getattr(args, "precision", "bf16") == "fp16-autocast" and device.type == "cuda"
    ) or bool(getattr(args, "force_scaler", False))
    return torch.amp.GradScaler(device.type, enabled=enabled)


@torch.no_grad()
def predict(model, enc, idx, swap, tok, template, args, device) -> tuple[np.ndarray, int]:
    """Probabilities in PRESENTED order ([p_first, p_second, p_tie]) and real tokens processed."""
    model.eval()
    seqs = [make_seq(enc, int(i), swap, template, args.t)[0] for i in idx]
    order = np.argsort([len(s) for s in seqs], kind="stable")
    out = np.zeros((len(seqs), 3), dtype=np.float64)
    tokens = 0
    for s in range(0, len(order), args.infer_batch):
        tick(args, "inference")
        sel = order[s : s + args.infer_batch]
        ids, mask = collate([seqs[j] for j in sel], tok.pad_token_id, device)
        with autocast(device, args):
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


# ---------------------------------------------------------------- first-overflow report

_HOOKED = re.compile(
    r"(^|\.)(embed_tokens|norm|score)$"
    r"|(^|\.)layers\.\d+$"
    r"|(^|\.)layers\.\d+\.(self_attn|mlp|input_layernorm|post_attention_layernorm"
    r"|pre_feedforward_layernorm|post_feedforward_layernorm)$"
    r"|(^|\.)(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


def _nonfinite(out) -> dict | None:
    """Counts of inf/nan in a module output (tensor, or any nesting of tuple/list/dict), else None."""
    n_nan = n_inf = 0
    absmax = 0.0
    stack = [out]
    while stack:
        x = stack.pop()
        if isinstance(x, torch.Tensor):
            if x.is_floating_point() and x.numel():
                n_nan += int(torch.isnan(x).sum())
                n_inf += int(torch.isinf(x).sum())
                fin = x[torch.isfinite(x)]
                if fin.numel():
                    absmax = max(absmax, float(fin.abs().max()))
        elif isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, (tuple, list)):
            stack.extend(x)
    if n_nan or n_inf:
        return {"n_nan": n_nan, "n_inf": n_inf, "absmax_finite": absmax}
    return None


def first_overflow_report(model, ids, mask, args, device) -> dict:
    """Re-run one micro-batch forward-only with hooks on the embeddings, decoder layers, their
    attention/MLP/norm sub-blocks, the linears, the final norm and the head. A forward hook fires
    when a module FINISHES, so the first hit is the innermost module that produced an inf/nan.
    Returns {"first_overflow_module": name or None, "reproduced": bool, ...}."""
    hits: list[dict] = []
    handles = []

    def make(name):
        def hook(_module, _inputs, output):
            if not hits:
                bad = _nonfinite(output)
                if bad:
                    hits.append({"module": name, **bad})

        return hook

    for name, mod in model.named_modules():
        if name and _HOOKED.search(name):
            handles.append(mod.register_forward_hook(make(name)))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), autocast(device, args):
            out = model(input_ids=ids, attention_mask=mask).logits
        logits_finite = bool(torch.isfinite(out).all())
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    first = hits[0] if hits else None
    return {
        "first_overflow_module": first["module"] if first else None,
        "reproduced": first is not None,
        "detail": first,
        "rerun_logits_finite": logits_finite,
        "hooked_modules": len(handles),
        "micro_batch_rows": int(ids.shape[0]),
        "micro_batch_max_len": int(ids.shape[1]),
    }


def _all_finite(finite: bool, device, world: int) -> bool:
    """Every rank must agree on a non-finite loss, or the others hang in the next all-reduce."""
    if world <= 1:
        return finite
    import torch.distributed as dist

    flag = torch.tensor([1 if finite else 0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


# ---------------------------------------------------------------- training


def train(
    model,
    enc,
    template,
    train_idx,
    eval_idx,
    tok,
    args,
    seed,
    device,
    max_steps=None,
    *,
    world: int = 1,
    rank: int = 0,
    fwd=None,
):
    """One epoch. With `eval_idx`, scores the inner holdout at 50% and 100% of the epoch and keeps
    the better trainable state (LoRA + head only). Returns (history, best, stats).

    `world`/`rank`/`fwd` are the DDP hooks (rung 6b): `fwd` is the DDP-wrapped module used for the
    forward pass, each optimizer step takes EFFECTIVE_BATCH rows in total, and rank r processes
    micro-batches r, r+world, ... of every group. With world=1 this is the original loop."""
    from transformers import get_cosine_schedule_with_warmup

    fwd = fwd if fwd is not None else model
    micro = args.micro_batch
    if EFFECTIVE_BATCH % (micro * world):
        raise SystemExit(f"--micro-batch {micro} x {world} ranks must divide {EFFECTIVE_BATCH}")
    accum = max(1, EFFECTIVE_BATCH // (micro * world))
    lengths = np.minimum(
        enc.meta[["n_prompt", "n_a", "n_b"]].to_numpy()[train_idx].sum(axis=1) + template.overhead,
        args.t,
    )
    steps_per_epoch = math.ceil(math.ceil(len(train_idx) / micro) / (accum * world))
    total_steps = min(max_steps, steps_per_epoch) if max_steps else steps_per_epoch
    eval_at = {}
    if eval_idx is not None and len(eval_idx) and not max_steps:
        eval_at = {max(1, total_steps // 2): 0.5, total_steps: 1.0}
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    sched = get_cosine_schedule_with_warmup(opt, int(WARMUP * total_steps), total_steps)
    scaler = make_scaler(args, device)
    skipped_steps = 0
    grad_norm_max = grad_norm_last = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    swaps = np.random.default_rng(seed).random(len(train_idx)) < 0.5
    batches = length_grouped_batches(lengths, micro, seed)
    groups = [batches[i : i + accum * world] for i in range(0, len(batches), accum * world)]
    if world > 1:  # every rank needs the same number of micro-batches in a step
        groups = [g[: len(g) - len(g) % world] for g in groups]
        groups = [g for g in groups if g]
    history, best = [], {"loss": float("inf"), "epoch": None, "state": None}
    step = samples = real_tokens = padded_tokens = 0
    loss_sum, loss_n = 0.0, 0
    train_time = eval_time = 0.0
    warm_time = warm_samples = None
    model.train()
    for group in groups:
        t_step = time.perf_counter()
        mine = group[rank::world] if world > 1 else group
        for k, bidx in enumerate(mine):
            tick(args, "train")
            pairs = [
                make_seq(enc, int(train_idx[j]), bool(swaps[j]), template, args.t) for j in bidx
            ]
            ids, mask = collate([s for s, _ in pairs], tok.pad_token_id, device)
            y = torch.tensor([y for _, y in pairs], device=device)
            sync_ctx = (
                fwd.no_sync() if (world > 1 and k < len(mine) - 1) else contextlib.nullcontext()
            )
            with sync_ctx:
                with autocast(device, args):
                    logits = fwd(input_ids=ids, attention_mask=mask).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), y)
                if not _all_finite(bool(torch.isfinite(loss)), device, world):
                    report = first_overflow_report(model, ids, mask, args, device)
                    raise NonFiniteLoss(
                        f"non-finite loss at step {step}; first overflow: "
                        f"{report['first_overflow_module']}",
                        first_overflow=report | {"step": step},
                    )
                scaler.scale(loss / len(mine)).backward()
            loss_sum += loss.item() * len(bidx)
            loss_n += len(bidx)
            samples += len(bidx)
            real_tokens += int(mask.sum())
            padded_tokens += int(mask.numel())
        scaler.unscale_(opt)  # BEFORE the clip: the 1.0 bound is on true, not scaled, gradients
        gn = float(torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP))
        if math.isfinite(gn):
            grad_norm_max, grad_norm_last = max(grad_norm_max, gn), gn
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if scaler.is_enabled() and scaler.get_scale() < scale_before:
            skipped_steps += 1  # the scaler found inf/nan gradients and skipped this step
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
    warm_steps_per_s = (
        (step - 2) / (train_time - warm_time)
        if warm_time is not None and step > 2 and train_time > warm_time
        else step / max(train_time, 1e-9)
    )
    stats = {
        "micro_batch": micro,
        "accumulation": accum,
        "world_size": world,
        "requested_steps": total_steps,
        "optimizer_steps": step,
        "optimizer_steps_per_s_warm": warm_steps_per_s,
        "global_samples_per_s_warm": warm_steps_per_s * EFFECTIVE_BATCH,
        "scaler_enabled": scaler.is_enabled(),
        "scaler_skipped_steps": skipped_steps,
        "scaler_final_scale": scaler.get_scale() if scaler.is_enabled() else None,
        "grad_norm_max": grad_norm_max,
        "grad_norm_last": grad_norm_last,
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


def dtype_description(args, device) -> str:
    base, ac = precision_dtypes(args, device)
    head = "; final norm+head fp32" if getattr(args, "fp32_head_norm", False) else ""
    auto = f"{dtype_label(ac)} autocast" if ac is not None else "no autocast"
    return f"base {dtype_label(base)}; LoRA+head fp32 masters; {auto}{head}"


def base_log(args, device, attn: str, template: Template) -> dict:
    log = {"tag": args.tag} | expected_config(args)
    log |= {
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "attn_implementation": attn,
        "dtype": dtype_description(args, device),
        "precision": getattr(args, "precision", "bf16"),
        "fp32_head_norm": bool(getattr(args, "fp32_head_norm", False)),
        "fp16_emulated_on_cpu": fp16_emulated(args, device),
        "family": model_family(args),
        "template_overhead": template.overhead,
        "price_per_hour": PRICE_PER_HOUR,
        "deadline_s": getattr(args, "deadline_s", 0.0) or None,
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
            tick(args, "attention")
            sel = order[s : s + args.infer_batch]
            ids, mask = collate([seqs[j] for j in sel], tok.pad_token_id, device)
            with autocast(device, args):
                logits[sel] = model(input_ids=ids, attention_mask=mask).logits.float().cpu().numpy()
        outs[impl] = logits
    diff = float(np.abs(outs["eager"] - outs["sdpa"]).max())
    scale = float(np.abs(outs["eager"]).max())
    sensitive = scale >= ATTN_MIN_LOGIT_SCALE
    # Gemma (bf16): tight, because sdpa drops its softcap. fp16-autocast: 2e-2, see lora_g0.
    tol = attention_tolerance(getattr(args, "precision", "bf16"))
    choice = choose_attention(diff, scale, tol, ATTN_MIN_LOGIT_SCALE)
    return {
        "rows": len(seqs),
        "max_abs_logit_diff": diff,
        "max_abs_logit_eager": scale,
        "check_sensitive": sensitive,
        "tolerance": tol,
        "choice": choice,
    }


# ---------------------------------------------------------------- probe


def write_probe(out: Path, probe: dict, t_run: float, complete: bool = False) -> None:
    """Rewrite probe.json from what is known so far, so a killed job still leaves evidence."""
    probe["complete"] = complete
    probe["finished_at"] = time.time()
    probe["wall_seconds"] = time.time() - t_run
    probe["est_cost_usd"] = est_cost_usd(probe["wall_seconds"])
    (out / "probe.json").write_text(json.dumps(probe, indent=2) + "\n")


def run_probe(args) -> int:
    t_run = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clear_stale_outputs("probe", out, args.adapter_dir)
    device = pick_device()
    seed = SEED_BASE
    probe = {"mode": "probe", "tiny": args.tiny, "seed": seed} | expected_config(args)
    probe["deadline_s"] = getattr(args, "deadline_s", 0.0) or None
    write_probe(out, probe, t_run)  # evidence even if the job dies in data prep
    try:
        _probe_body(args, out, device, seed, probe, t_run)
    except DeadlineExceeded as e:
        probe |= {"error": "deadline", "deadline_stage": e.stage}
        print(f"[probe] deadline reached during {e.stage}", flush=True)
        write_probe(out, probe, t_run)
        return DEADLINE_EXIT
    except Exception as e:  # noqa: BLE001  recorded, then the failure is visible in the exit code
        probe["error"] = f"{type(e).__name__}: {e}"
        print(f"[probe] FAILED: {probe['error']}", flush=True)
        write_probe(out, probe, t_run)
        return 3
    print(json.dumps(probe, indent=2))
    return 0


def _probe_body(args, out: Path, device, seed: int, probe: dict, t_run: float) -> None:
    set_seeds(seed)
    t0 = time.time()
    tok, template, manifest, enc = setup(args)
    data_s = time.time() - t0
    tick(args, "setup")
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
    probe |= enc.shares
    probe |= base_log(args, device, "pending", template)
    probe["model_load_seconds"] = load_s
    probe["data_seconds"] = data_s
    write_probe(out, probe, t_run)
    tick(args, "model load")

    check = attention_check(model, enc, tok, template, args, device, args.attn_rows)
    probe["attention_check"] = check
    attn = check["choice"] if args.attn == "auto" else args.attn
    probe["attention_decision"] = attn
    probe["attn_implementation"] = attn
    write_probe(out, probe, t_run)  # after the attention check
    model.set_attn_implementation(attn)
    model = wrap_lora(model, args)

    try:
        _, _, tstats = train(model, enc, template, sample, None, tok, args, seed, device, steps)
        probe["non_finite_loss"] = None
    except NonFiniteLoss as e:
        probe["non_finite_loss"] = str(e)
        tstats = {}
    probe["train"] = tstats
    write_probe(out, probe, t_run)  # after training

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
    write_probe(out, probe, t_run, complete=True)  # after inference


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
    # G1 binds to this run: nothing an earlier run left under this tag may survive into it.
    clear_stale_outputs("fold", out, args.adapter_dir, fold)
    log = {"mode": "fold", "fold": fold, "seed": seed, "attn_implementation": attn}
    code = 3
    device = pick_device()
    try:
        set_seeds(seed)
        t0 = time.time()
        tok, template, manifest, enc = setup(args)
        log |= base_log(args, device, attn, template) | enc.shares
        log["data_seconds"] = time.time() - t0
        tick(args, "setup")
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
    except DeadlineExceeded as e:
        log |= {"error": "deadline", "deadline_stage": e.stage}
        print(f"[fold {fold}] deadline reached during {e.stage}", flush=True)
        log |= {k: v for k, v in expected_config(args).items() if k not in log}
        finish_log(log, t_run, device)
        (out / f"log_fold{fold}.json").write_text(json.dumps(log, indent=2) + "\n")
        return DEADLINE_EXIT
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
    clear_stale_outputs("full", out, args.adapter_dir)
    log = {"mode": "full", "seed": FULL_SEED, "attn_implementation": attn}
    device = pick_device()
    try:
        set_seeds(FULL_SEED)
        t0 = time.time()
        tok, template, manifest, enc = setup(args)
        log |= base_log(args, device, attn, template) | enc.shares
        log["data_seconds"] = time.time() - t0
        tick(args, "setup")
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
    except DeadlineExceeded as e:
        log |= {"error": "deadline", "deadline_stage": e.stage}
        print(f"[full] deadline reached during {e.stage}", flush=True)
        log |= {k: v for k, v in expected_config(args).items() if k not in log}
        code = DEADLINE_EXIT
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
        raise WorkerConfigMismatch(f"fold {fold} ran with {got}, launcher asked for {want}")


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
    ap.add_argument(
        "--mode",
        choices=["probe", "fold", "full", "selftest", "qprobe", "ddp-child"],
        required=True,
    )
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--t", type=int, default=2048)
    ap.add_argument("--middle-share", type=float, default=MIDDLE_SHARE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tiny", action="store_true", help="tiny from-scratch model (dry run)")
    ap.add_argument("--tiny-arch", choices=["gemma2", "qwen3"], default="gemma2")
    ap.add_argument("--precision", choices=PRECISIONS, default="bf16")
    ap.add_argument(
        "--fp32-head-norm", action="store_true", help="keep the final norm and score head in fp32"
    )
    ap.add_argument("--force-scaler", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--model-dir", default="", help="local model directory (Kaggle mount)")
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
    ap.add_argument("--deadline-s", type=float, default=0.0, help="wall-clock budget; 0 = none")
    ap.add_argument(
        "--local-files-only",
        action="store_true",
        help="load weights/tokenizer from the HF cache only (fail fast if not prefetched)",
    )
    ap.add_argument("--tag", default="")
    ap.add_argument("--git-sha", default="")
    # rung 6b (Kaggle probe, lora_kaggle.py)
    ap.add_argument("--kaggle", action="store_true", help="discover model and data under /kaggle")
    ap.add_argument("--probe-encode-rows", type=int, default=12000)
    ap.add_argument("--qattn-rows", type=int, default=100)
    ap.add_argument("--ddp-steps", type=int, default=50)
    ap.add_argument("--ddp-timeout-s", type=float, default=900.0)
    ap.add_argument("--no-ddp", action="store_true")
    ap.add_argument("--ddp-cpu-procs", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--ddp-cache", default="", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    return run(parse(argv))


def run(args) -> int:
    args.deadline = Deadline(args.deadline_s)  # starts now: the budget covers data prep and load
    if EFFECTIVE_BATCH % args.micro_batch:
        raise SystemExit("--micro-batch must divide 16")
    if not 0 <= args.fold <= 4:
        raise SystemExit("--fold must be 0-4")
    if args.mode in ("qprobe", "ddp-child"):  # rung 6b: Kaggle probe (lora_kaggle owns it)
        raise SystemExit(f"--mode {args.mode} runs through llm_preference_audit.lora_kaggle")
    return {"probe": run_probe, "fold": run_fold, "full": run_full, "selftest": run_selftest}[
        args.mode
    ](args)


if __name__ == "__main__":
    sys.exit(main())
