"""Torch-free decisions for rung 6b (free Qwen3 LoRA on Kaggle T4s).

Everything here is arithmetic or file discovery, so CI exercises it without torch:
  * `choose_attention`      sdpa vs eager from an attention-check result
  * `discover_model_dir`    find the Kaggle-mounted Qwen3 directory by config.json model_type
  * `run_with_fallback`     micro-batch ladder 4 -> 2 -> 1 on out-of-memory, recording attempts
  * `project_kaggle`        wall-hour projection of fold 0 and the all-cv model, single vs DDP
  * `g0_verdict`            the G0 pass/fail object, with a reason for every failed check
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

# The exact Kaggle model instance (owner/model/framework/variation/version), read-only verified
# 2026-10-25 with `kaggle models instances get qwen-lm/qwen-3/transformers/1.7b`: versionNumber 1,
# Apache 2.0, 4,079,437,039 bytes, post-trained Qwen3-1.7B.
QWEN3_MODEL_HANDLE = "qwen-lm/qwen-3/transformers/1.7b/1"

# ---- thresholds (pre-registered in docs/plans/rung-6b-qwen3-kaggle.md)
MAX_SKIP_SHARE = 0.05  # G0: GradScaler may skip fewer than 5% of optimizer steps
MAX_FOLD0_HOURS = 10.0  # G0: projected fold-0 train + inference, best of single/DDP
MEM_FRACTION = 0.95  # peak allocated memory must stay under this share of the card
MICRO_BATCH_LADDER = (4, 2, 1)
INFER_BATCH_LADDER = (8, 4, 2, 1)

# Attention-check tolerance on logits. Gemma2 (rung 6, bf16): sdpa drops the attention softcap, so
# the check is tight. Qwen3 has no softcap: sdpa and eager are the same maths, and what remains is
# fp16 rounding. fp16 carries ~1e-3 relative precision (eps 9.8e-4) and the logits come out of 28
# layers of differently-ordered accumulation, so sub-1e-2 absolute differences on logits of order 1
# are rounding. 2e-2 absolute is about a 0.5% shift in a probability: well under a logit-calibration
# error, and a real kernel disagreement (a wrong mask, a missed scale) is orders larger.
ATTN_TOL_STRICT = 1e-3
ATTN_TOL_FP16 = 2e-2
ATTN_MIN_LOGIT_SCALE = 0.05  # below this the check cannot see a difference: stay on eager


def attention_tolerance(precision: str) -> float:
    return ATTN_TOL_FP16 if precision == "fp16-autocast" else ATTN_TOL_STRICT


def choose_attention(
    max_abs_diff: float, logit_scale: float, tol: float, min_scale: float = ATTN_MIN_LOGIT_SCALE
) -> str:
    """sdpa only when the two implementations agree within `tol` AND the logits are large enough
    for the check to have seen a disagreement; a NaN difference never passes."""
    sensitive = logit_scale >= min_scale
    return "sdpa" if (max_abs_diff <= tol and sensitive) else "eager"


# ---------------------------------------------------------------- model discovery


def discover_model_dir(root: Path | str, model_type: str = "qwen3") -> Path:
    """The directory under `root` holding a config.json whose `model_type` is `model_type`.

    Kaggle's model mount path varies (`/kaggle/input/qwen-3/transformers/1.7b/1`), so the config
    decides, not the path. Directories that also hold weights win over a bare config; more than
    one distinct match is an error (attach exactly one Qwen3 model).
    """
    hits: list[Path] = []
    for cfg in sorted(Path(root).rglob("config.json")):
        try:
            data = json.loads(cfg.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("model_type") == model_type:
            hits.append(cfg.parent)
    if not hits:
        raise SystemExit(f"no config.json with model_type {model_type!r} under {root}")
    with_weights = [d for d in hits if any(d.glob("*.safetensors")) or any(d.glob("*.bin"))]
    pool = with_weights or hits
    if len(pool) > 1:
        raise SystemExit(f"more than one {model_type} model under {root}: {[str(p) for p in pool]}")
    return pool[0]


# ---------------------------------------------------------------- micro-batch ladder


def run_with_fallback(
    fn: Callable[[int], object],
    ladder: Sequence[int],
    is_oom: Callable[[BaseException], bool],
) -> tuple[object | None, int | None, list[dict]]:
    """Try `fn(size)` down the ladder; only an out-of-memory error steps down.

    Returns (result, chosen size or None, attempts). Any other exception propagates.
    """
    attempts: list[dict] = []
    for size in ladder:
        try:
            result = fn(size)
        except BaseException as e:
            if not is_oom(e):
                raise
            attempts.append({"size": size, "oom": True})
            continue
        attempts.append({"size": size, "oom": False})
        return result, size, attempts
    return None, None, attempts


# ---------------------------------------------------------------- projection


def _fold0_seconds(
    *,
    n_cv: int,
    train_sps: float,
    infer_sps: float,
    load_s: float,
    prep_s: float,
    n_evals: int,
    inner_share: float,
    fold_share: float,
) -> dict:
    n_train = n_cv * (1 - fold_share) * (1 - inner_share)
    n_inner = n_cv * (1 - fold_share) * inner_share
    n_pred = n_cv * fold_share
    train_s = n_train / train_sps
    eval_s = n_evals * 2 * n_inner / infer_sps  # inner-holdout selection, both orders
    pred_s = 2 * n_pred / infer_sps  # fold 0 itself, both orders
    total = prep_s + load_s + train_s + eval_s + pred_s
    return {
        "prep_seconds": prep_s,
        "load_seconds": load_s,
        "train_seconds": train_s,
        "eval_seconds": eval_s,
        "predict_seconds": pred_s,
        "seconds": total,
        "hours": total / 3600,
    }


def project_kaggle(
    *,
    n_cv: int,
    train_sps_single: float,
    infer_sps_single: float,
    load_seconds: float,
    prep_seconds: float = 0.0,
    train_sps_ddp: float | None = None,
    infer_parallel: int = 2,
    n_evals: int = 2,
    inner_share: float = 0.1,
    fold_share: float = 0.2,
) -> dict:
    """Wall hours for fold 0 (train + inner-holdout selection + both-order prediction) and for the
    all-cv submission-model train (no eval, no inference), single GPU and DDP.

    `train_sps_*` is training samples/s; `infer_sps_single` is sequences/s (a row is two
    sequences). The DDP inference rate is ASSUMED to be `infer_parallel` x the single rate (rows
    are independent, so two processes each infer half); it is not measured. If `train_sps_ddp`
    is None the DDP column is absent and `best` is the single-GPU figure.
    """
    if min(train_sps_single, infer_sps_single) <= 0:
        raise ValueError("rates must be positive")
    common = {
        "n_cv": n_cv,
        "load_s": load_seconds,
        "prep_s": prep_seconds,
        "n_evals": n_evals,
        "inner_share": inner_share,
        "fold_share": fold_share,
    }
    out: dict = {
        "single": {
            "fold0": _fold0_seconds(
                train_sps=train_sps_single, infer_sps=infer_sps_single, **common
            ),
            "all_cv_train_seconds": prep_seconds + load_seconds + n_cv / train_sps_single,
        },
        "ddp": None,
        "assumptions": {
            "ddp_inference_speedup": infer_parallel,
            "ddp_inference_measured": False,
            "fold0_includes_inner_holdout_evals": n_evals,
        },
    }
    out["single"]["all_cv_train_hours"] = out["single"]["all_cv_train_seconds"] / 3600
    if train_sps_ddp is not None:
        if train_sps_ddp <= 0:
            raise ValueError("rates must be positive")
        out["ddp"] = {
            "fold0": _fold0_seconds(
                train_sps=train_sps_ddp, infer_sps=infer_sps_single * infer_parallel, **common
            ),
            "all_cv_train_seconds": prep_seconds + load_seconds + n_cv / train_sps_ddp,
        }
        out["ddp"]["all_cv_train_hours"] = out["ddp"]["all_cv_train_seconds"] / 3600
    candidates = {"single": out["single"]["fold0"]["hours"]}
    if out["ddp"] is not None:
        candidates["ddp"] = out["ddp"]["fold0"]["hours"]
    out["best_fold0_mode"] = min(candidates, key=candidates.get)
    out["best_fold0_hours"] = candidates[out["best_fold0_mode"]]
    return out


# ---------------------------------------------------------------- G0


def g0_verdict(probe: dict) -> dict:
    """The G0 gate over a (possibly partial) probe record. Every check must hold to pass.

    Reads: non_finite_loss, train{optimizer_steps, requested_steps, scaler_skipped_steps},
    infer{finite}, peak_gpu_gib, gpu_total_gib, micro_batch_chosen, projection.best_fold0_hours.
    Anything absent counts as not established, so a probe that stopped early cannot pass.
    """
    train = probe.get("train") or {}
    steps = train.get("optimizer_steps")
    requested = train.get("requested_steps")
    skipped = train.get("scaler_skipped_steps")
    infer = probe.get("infer") or {}
    peak, total = probe.get("peak_gpu_gib"), probe.get("gpu_total_gib")
    best = (probe.get("projection") or {}).get("best_fold0_hours")

    trained = bool(steps) and requested is not None and steps >= requested
    skip_share = (skipped / steps) if (steps and skipped is not None) else None
    checks = {
        "trained_all_steps": trained,
        "loss_finite": trained and probe.get("non_finite_loss") is None,
        "inference_finite": infer.get("finite") is True,
        "scaler_skips_ok": skip_share is not None and skip_share < MAX_SKIP_SHARE,
        "fits_memory": (
            probe.get("micro_batch_chosen") is not None
            and peak is not None
            and total is not None
            and peak <= MEM_FRACTION * total
        ),
        "projected_fold0_within_cap": best is not None and best <= MAX_FOLD0_HOURS,
    }
    reasons = []
    if not checks["trained_all_steps"]:
        reasons.append(f"training incomplete: {steps} of {requested} optimizer steps")
    if probe.get("non_finite_loss") is not None:
        reasons.append(f"non-finite loss: {probe['non_finite_loss']}")
    if not checks["inference_finite"]:
        reasons.append("inference not shown finite")
    if not checks["scaler_skips_ok"]:
        share = "unknown" if skip_share is None else f"{skip_share:.3f}"
        reasons.append(f"scaler skip share {share} is not < {MAX_SKIP_SHARE}")
    if not checks["fits_memory"]:
        reasons.append(
            f"memory not shown to fit: peak {peak} GiB of {total} GiB "
            f"(micro-batch {probe.get('micro_batch_chosen')})"
        )
    if not checks["projected_fold0_within_cap"]:
        reasons.append(f"projected fold-0 hours {best} is not <= {MAX_FOLD0_HOURS}")
    return {
        "g0_pass": all(checks.values()),
        "checks": checks,
        "reasons": reasons,
        "thresholds": {
            "max_skip_share": MAX_SKIP_SHARE,
            "max_fold0_hours": MAX_FOLD0_HOURS,
            "mem_fraction": MEM_FRACTION,
        },
        "scaler_skip_share": skip_share,
        "best_fold0_hours": best,
    }
