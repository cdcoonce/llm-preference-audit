"""Rung 6b: the Kaggle side of the free Qwen3 LoRA run (T4 x2, fp16).

  python -m llm_preference_audit.lora_kaggle --mode qprobe --kaggle --precision fp16-autocast \\
      --fp32-head-norm --micro-batch 4 --t 2048 --deadline-s 3300 ...

Modes
  qprobe     the single-session probe (stages below), writes probe.json after EVERY stage
  ddp-child  one rank of the DDP stage, started by qprobe through torch.distributed.run
  fold/full/probe/selftest  delegate to lora_train (single GPU, same generalised code; not
             validated on Kaggle yet)

qprobe stages (each rewrites probe.json; `complete` is true only when every stage ran):
  setup      verified data, tokenizer, template, a length-representative encode of cv rows
  attention  sdpa vs eager logits on --qattn-rows rows in the run's precision -> the attention used
  train      --probe-steps optimizer steps (effective batch 16) on fold 0's training rows, ONE GPU,
             micro-batch 4 -> 2 -> 1 on out-of-memory; throughput, memory, scaler skips, finiteness
  infer      --probe-infer-rows rows, both orders, one GPU
  ddp        --ddp-steps steps across both GPUs through torch.distributed.run (NCCL); a failure is
             recorded and the probe goes on
  verdict    projection (fold 0 and all-cv, single vs DDP, wall hours) and the G0 object

Only ids, aggregates and module names are written; dataset text lives in memory. The DDP children
get the encoded token ids through a pickle in a temp directory that is deleted afterwards (never
in /kaggle/working).
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from llm_preference_audit import lora_train as lt
from llm_preference_audit.deberta_inputs import fold_plan
from llm_preference_audit.lora_cost import Deadline, DeadlineExceeded
from llm_preference_audit.lora_g0 import (
    INFER_BATCH_LADDER,
    MICRO_BATCH_LADDER,
    QWEN3_MODEL_HANDLE,
    discover_model_dir,
    g0_verdict,
    project_kaggle,
    run_with_fallback,
)

KAGGLE_INPUT = "/kaggle/input"
KAGGLE_DATA_DIR = "/kaggle/input/llm-classification-finetuning"
KAGGLE_OUT_DIR = "/kaggle/working"
MODEL_HANDLE = QWEN3_MODEL_HANDLE
DEFAULT_PROBE_DEADLINE_S = 55 * 60
PARSE_DEFAULTS = {
    "data_dir": "/vol/data",
    "out_dir": "/vol/out/run",
    "adapter_dir": "/vol/adapters/run",
}
GIB = 1024**3


# ---------------------------------------------------------------- setup


def apply_kaggle(args) -> None:
    """Point the shared trainer at Kaggle's mounts: the model directory is found by config.json
    model_type under /kaggle/input; train.csv is found by sha256 inside `kernel_common`."""
    if not args.kaggle:
        return
    if not args.tiny and not args.model_dir:
        args.model_dir = str(discover_model_dir(KAGGLE_INPUT, "qwen3"))
        print(f"model directory: {args.model_dir}", flush=True)
    if args.model == lt.DEFAULT_MODEL:
        args.model = MODEL_HANDLE
    if args.data_dir == PARSE_DEFAULTS["data_dir"]:
        args.data_dir = KAGGLE_DATA_DIR
    if args.out_dir == PARSE_DEFAULTS["out_dir"]:
        args.out_dir = KAGGLE_OUT_DIR
    if args.adapter_dir == PARSE_DEFAULTS["adapter_dir"]:
        args.adapter_dir = f"{KAGGLE_OUT_DIR}/adapter"


def is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gpu_facts(device) -> dict:
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    facts = {"n_gpu": n, "gpu_names": [torch.cuda.get_device_name(i) for i in range(n)]}
    if device.type == "cuda":
        facts["gpu_total_gib"] = torch.cuda.get_device_properties(0).total_memory / GIB
    return facts


class Probe:
    """probe.json, rewritten atomically after every stage so a killed session leaves evidence."""

    def __init__(self, out: Path, t_run: float):
        self.path, self.t_run, self.data = out / "probe.json", t_run, {}

    def write(self, complete: bool = False) -> None:
        self.data |= {
            "complete": complete,
            "finished_at": time.time(),
            "wall_seconds": time.time() - self.t_run,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str) + "\n")
        tmp.replace(self.path)

    def stage(self, name: str) -> None:
        self.data["stage"] = name
        print(f"[qprobe] stage {name} at {time.time() - self.t_run:.0f}s", flush=True)

    def done(self, name: str) -> None:
        self.data.setdefault("stages_done", []).append(name)
        self.write()


# ---------------------------------------------------------------- the probe


def run_qprobe(args) -> int:
    t_run = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("probe.json", "ddp_result.json"):
        (out / stale).unlink(missing_ok=True)
    device = lt.pick_device()
    probe = Probe(out, t_run)
    probe.data |= {"mode": "qprobe", "tiny": args.tiny, "seed": lt.SEED_BASE}
    probe.data |= lt.expected_config(args) | gpu_facts(device)
    probe.data |= {
        "precision": args.precision,
        "fp32_head_norm": args.fp32_head_norm,
        "deadline_s": args.deadline_s or None,
        "stages_done": [],
        "stage": "start",
    }
    probe.write()
    code = 0
    try:
        _qprobe_body(args, out, device, probe)
    except DeadlineExceeded as e:
        probe.data |= {"error": "deadline", "deadline_stage": e.stage}
        print(f"[qprobe] deadline reached during {e.stage}", flush=True)
        code = lt.DEADLINE_EXIT
    except Exception as e:  # noqa: BLE001  recorded; the exit code shows the failure
        probe.data["error"] = f"{type(e).__name__}: {e}"
        print(f"[qprobe] FAILED: {probe.data['error']}", flush=True)
        code = 3
    probe.data["g0"] = g0_verdict(probe.data)
    probe.write(complete=code == 0)
    print(json.dumps(probe.data["g0"], indent=2), flush=True)
    return code


def _qprobe_body(args, out: Path, device, probe: Probe) -> None:
    d = probe.data
    seed = lt.SEED_BASE
    lt.set_seeds(seed)
    if (
        not args.max_rows
    ):  # a length-representative subset is plenty for rates; sizes use the full cv
        args.max_rows = args.probe_encode_rows

    probe.stage("setup")
    tok = lt.make_tokenizer(args)
    template = lt.tokenizer_template(tok)
    t0 = time.time()
    df, manifest = lt.load_verified(Path(args.data_dir), args.expect_train_sha256)
    load_s = time.time() - t0  # hashing + manifest: a fixed cost, not per row
    t0 = time.time()
    enc = lt.encode(args, tok, template, df, manifest)
    del df
    encode_s = time.time() - t0
    lt.tick(args, "setup")
    meta = enc.meta
    plan = fold_plan(meta, 0, manifest)
    train_idx = lt.positions(meta, plan["train_ids"])
    pred_idx = lt.positions(meta, plan["predict_ids"])
    rng = np.random.default_rng(lt.SAMPLE_SEED)
    n_sample = min(len(train_idx), args.probe_steps * lt.EFFECTIVE_BATCH)
    sample = rng.choice(train_idx, size=n_sample, replace=False)  # random = length-representative
    d |= enc.shares | {
        "n_cv_full": enc.n_cv_full,
        "n_encoded": len(meta),
        "data_load_seconds": load_s,
        "encode_seconds": encode_s,
        "prep_seconds_full_cv_est": load_s + encode_s * enc.n_cv_full / max(len(meta), 1),
        "template_overhead": template.overhead,
        "family": lt.model_family(args),
        "torch": torch.__version__,
        "transformers": lt.transformers.__version__,
        "python": sys.version.split()[0],
        "probe_steps_requested": args.probe_steps,
        "train_sample_rows": int(n_sample),
        "pad_token_id": int(tok.pad_token_id),
        "bos_token_id": tok.bos_token_id,
    }
    with contextlib.suppress(ImportError):
        import peft

        d["peft"] = peft.__version__
    probe.done("setup")

    # ---- attention check (model loaded once here, then rebuilt per micro-batch attempt)
    probe.stage("attention")
    t0 = time.time()
    model = lt.load_base(args, tok, "eager", device).to(device)
    d["model_load_seconds"] = time.time() - t0
    d["base_dtype_census"] = sorted({str(p.dtype) for p in model.parameters()})
    lt.tick(args, "model load")

    def attn_attempt(batch: int):
        cleanup()
        args.infer_batch = batch
        return lt.attention_check(model, enc, tok, template, args, device, args.qattn_rows)

    check, batch, attempts = run_with_fallback(attn_attempt, INFER_BATCH_LADDER, is_oom)
    d["attention_batch_attempts"] = attempts
    if check is None:
        raise RuntimeError("attention check ran out of memory even at batch 1")
    attn = check["choice"] if args.attn == "auto" else args.attn
    d |= {"attention_check": check, "attention_decision": attn, "attn_implementation": attn}
    model = None
    cleanup()
    probe.done("attention")

    # ---- single-GPU training, micro-batch ladder
    probe.stage("train")
    holder: dict = {}

    def train_attempt(mb: int):
        cleanup()
        holder.clear()
        args.micro_batch = mb
        pm = lt.wrap_lora(lt.load_base(args, tok, attn, device).to(device), args)
        holder["model"] = pm
        d["dtype_census"] = lt.assert_precision_dtypes(pm, args, device)
        try:
            _, _, stats = lt.train(
                pm, enc, template, sample, None, tok, args, seed, device, args.probe_steps
            )
            return {"stats": stats}
        except lt.NonFiniteLoss as e:
            return {"non_finite": str(e), "first_overflow": e.first_overflow}

    res, mb, attempts = run_with_fallback(train_attempt, MICRO_BATCH_LADDER, is_oom)
    d["micro_batch_attempts"] = attempts
    d["micro_batch_chosen"] = mb
    d["non_finite_loss"] = None
    d["train"] = {}
    if res is None:
        d["train_error"] = "out of memory at every micro-batch (4, 2, 1)"
    elif "non_finite" in res:
        d["non_finite_loss"] = res["non_finite"]
        d["first_overflow"] = res["first_overflow"]
        print(f"[qprobe] first overflow: {res['first_overflow']}", flush=True)
    else:
        d["train"] = res["stats"] | {"requested_steps": args.probe_steps}
        d["peak_gpu_gib"] = res["stats"]["peak_gpu_gib"]
    probe.done("train")
    model = holder.get("model")
    finite = res is not None and "stats" in res

    # ---- inference, both orders, one GPU (on the model the training just left)
    probe.stage("infer")
    if model is not None:
        n_inf = min(args.probe_infer_rows, len(pred_idx))
        inf_idx = rng.choice(pred_idx, size=n_inf, replace=False)

        def infer_attempt(batch: int):
            cleanup()
            args.infer_batch = batch
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            lt.predict(model, enc, inf_idx[: min(16, n_inf)], False, tok, template, args, device)
            lt.sync(device)
            t = time.perf_counter()
            ab, ba, tokens = lt.predict_both(model, enc, inf_idx, tok, template, args, device)
            lt.sync(device)
            return ab, ba, tokens, time.perf_counter() - t

        got, batch, attempts = run_with_fallback(infer_attempt, INFER_BATCH_LADDER, is_oom)
        d["infer_batch_attempts"] = attempts
        if got is not None:
            ab, ba, tokens, dt = got
            d["infer"] = {
                "rows": n_inf,
                "batch": batch,
                "rows_per_s": n_inf / max(dt, 1e-9),
                "sequences_per_s": 2 * n_inf / max(dt, 1e-9),
                "real_tokens_per_s": tokens / max(dt, 1e-9),
                "peak_gpu_gib": lt.peak_gib(device),
                "finite": bool(np.isfinite(ab).all() and np.isfinite(ba).all()),
            }
            if d["infer"]["peak_gpu_gib"] is not None:
                d["peak_gpu_gib"] = max(d.get("peak_gpu_gib") or 0.0, d["infer"]["peak_gpu_gib"])
    model = None
    holder.clear()
    cleanup()
    probe.done("infer")

    # ---- DDP across both GPUs
    probe.stage("ddp")
    d["ddp"] = ddp_stage(args, enc, template, sample, out, attn, d, finite, probe)
    probe.done("ddp")

    # ---- projection and verdict
    probe.stage("verdict")
    infer = d.get("infer") or {}
    if finite and infer.get("sequences_per_s"):
        ddp_sps = (d["ddp"].get("result") or {}).get("train", {}).get("global_samples_per_s_warm")
        d["projection"] = project_kaggle(
            n_cv=enc.n_cv_full,
            train_sps_single=d["train"]["global_samples_per_s_warm"],
            infer_sps_single=infer["sequences_per_s"],
            load_seconds=d["model_load_seconds"],
            prep_seconds=d["prep_seconds_full_cv_est"],
            train_sps_ddp=ddp_sps if (d["ddp"].get("ok") and ddp_sps) else None,
        )
        if d["projection"]["ddp"] is not None:
            d["ddp"]["speedup_vs_single"] = ddp_sps / d["train"]["global_samples_per_s_warm"]
    probe.done("verdict")


# ---------------------------------------------------------------- DDP


CHILD_FLAGS = [
    "t",
    "middle_share",
    "model",
    "model_dir",
    "precision",
    "tiny_arch",
    "micro_batch",
    "attn",
    "lr",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "tag",
    "git_sha",
    "data_dir",
    "out_dir",
    "ddp_steps",
]


def child_argv(args, cache: Path) -> list[str]:
    argv = ["--mode", "ddp-child", "--ddp-cache", str(cache)]
    for k in CHILD_FLAGS:
        argv += [f"--{k.replace('_', '-')}", str(getattr(args, k))]
    for flag in ("tiny", "fp32_head_norm", "force_scaler", "local_files_only"):
        if getattr(args, flag, False):
            argv.append(f"--{flag.replace('_', '-')}")
    return argv


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def launch_ddp(argv: list[str], world: int, env_extra: dict, timeout: float) -> dict:
    """GPU: torch.distributed.run (NCCL, env:// rendezvous). CPU dry run: the same module started
    `world` times by hand with a file:// rendezvous (a TCP store can hang on a locked-down host)."""
    base_env = dict(os.environ) | {"TOKENIZERS_PARALLELISM": "false"} | env_extra
    t0 = time.time()
    if torch.cuda.is_available():
        cmd = [
            sys.executable, "-m", "torch.distributed.run", "--rdzv-backend", "c10d",
            "--rdzv-endpoint", f"127.0.0.1:{_free_port()}", "--nproc_per_node", str(world),
            "--module", "llm_preference_audit.lora_kaggle", *argv,
        ]  # fmt: skip
        procs = [subprocess.Popen(cmd, env=base_env, start_new_session=True)]
    else:
        init = Path(tempfile.mkdtemp(prefix="rung6b_init_")) / "store"
        procs = []
        for r in range(world):
            env = base_env | {
                "RANK": str(r), "WORLD_SIZE": str(world), "LOCAL_RANK": str(r),
                "RUNG6B_INIT_FILE": str(init),
            }  # fmt: skip
            cmd = [sys.executable, "-m", "llm_preference_audit.lora_kaggle", *argv]
            procs.append(subprocess.Popen(cmd, env=env, start_new_session=True))
    deadline = time.time() + max(timeout, 1.0)
    timed_out, codes = False, []
    for proc in procs:
        try:
            codes.append(proc.wait(timeout=max(deadline - time.time(), 0.1)))
        except subprocess.TimeoutExpired:
            timed_out = True
            break
    if timed_out:
        for proc in procs:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
        codes = [proc.wait() for proc in procs]
    return {
        "returncode": max(codes, key=abs) if codes else -1,
        "timed_out": timed_out,
        "seconds": time.time() - t0,
        "env_extra": sorted(env_extra),
    }


def ddp_stage(
    args, enc, template, sample, out: Path, attn: str, d: dict, finite: bool, probe
) -> dict:
    """50 steps across both GPUs. Any failure is recorded; the probe carries on."""
    world = min(2, torch.cuda.device_count()) if torch.cuda.is_available() else args.ddp_cpu_procs
    if args.no_ddp:
        return {"skipped": "--no-ddp"}
    if world < 2:
        return {"skipped": f"needs 2 processes, found {world}"}
    if not finite or d.get("micro_batch_chosen") is None:
        return {"skipped": "single-GPU training did not finish finite"}
    remaining = args.deadline.remaining() if getattr(args, "deadline", None) else float("inf")
    if remaining <= 60:
        raise DeadlineExceeded("ddp")
    args.micro_batch, args.attn = d["micro_batch_chosen"], attn
    tmp = Path(tempfile.mkdtemp(prefix="rung6b_ddp_"))
    cache = tmp / "cache.pkl"
    attempts, result, ok = [], None, False
    try:
        with cache.open("wb") as f:
            pickle.dump({"enc": enc, "template": template, "sample": sample}, f)
        for env_extra in ({}, {"NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1"}):
            left = args.deadline.remaining() if getattr(args, "deadline", None) else float("inf")
            if left <= 60 and attempts:
                break
            (out / "ddp_result.json").unlink(missing_ok=True)
            run = launch_ddp(
                child_argv(args, cache), world, env_extra, min(args.ddp_timeout_s, left - 30)
            )
            res_path = out / "ddp_result.json"
            result = json.loads(res_path.read_text()) if res_path.exists() else None
            run["result_written"] = result is not None
            attempts.append(run)
            probe.data["ddp"] = {"attempts": attempts, "in_progress": True}
            probe.write()
            ok = run["returncode"] == 0 and bool(result and result.get("ok"))
            if ok or (result and result.get("non_finite_loss")):
                break  # success, or a genuine numerical failure that a retry cannot fix
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out_d = {"ok": ok, "world_size": world, "attempts": attempts, "result": result}
    if result and not ok:
        out_d["error"] = result.get("error") or result.get("non_finite_loss")
    elif not result:
        out_d["error"] = "no result written (hang, crash or timeout); see attempts"
    return out_d


def run_ddp_child(args) -> int:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(local)
    device = torch.device("cuda", local) if cuda else torch.device("cpu")
    backend = "nccl" if cuda else "gloo"
    init_file = os.environ.get("RUNG6B_INIT_FILE")
    if init_file:  # CPU dry run (see launch_ddp)
        dist.init_process_group(
            backend, init_method=f"file://{init_file}", rank=rank, world_size=world
        )
    else:
        dist.init_process_group(backend)
    result: dict = {"ok": False, "world_size": world, "backend": "nccl" if cuda else "gloo"}
    args.deadline = None  # the parent enforces the wall-clock limit by killing the process group
    try:
        with open(args.ddp_cache, "rb") as f:
            cache = pickle.load(f)
        tok = lt.make_tokenizer(args)
        seed = lt.SEED_BASE
        lt.set_seeds(seed)
        t0 = time.time()
        pm = lt.wrap_lora(lt.load_base(args, tok, args.attn, device).to(device), args)
        result["setup_seconds"] = time.time() - t0
        ddp = DDP(pm, device_ids=[local] if cuda else None, find_unused_parameters=False)
        _, _, stats = lt.train(
            pm, cache["enc"], cache["template"], cache["sample"], None, tok, args, seed, device,
            args.ddp_steps, world=world, rank=rank, fwd=ddp,
        )  # fmt: skip
        peak = torch.tensor([stats["peak_gpu_gib"] or 0.0], device=device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        skipped = torch.tensor([float(stats["scaler_skipped_steps"])], device=device)
        dist.all_reduce(skipped, op=dist.ReduceOp.MAX)
        result |= {
            "ok": True,
            "train": stats | {"requested_steps": args.ddp_steps},
            "peak_gpu_gib_max": float(peak.item()),
            "scaler_skipped_steps_max": int(skipped.item()),
        }
    except lt.NonFiniteLoss as e:
        result |= {"non_finite_loss": str(e), "first_overflow": e.first_overflow}
    except BaseException as e:  # noqa: BLE001  recorded for the parent, then re-raised as exit 3
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        if rank == 0:
            (Path(args.out_dir) / "ddp_result.json").write_text(json.dumps(result, indent=2))
        with contextlib.suppress(Exception):
            if result["ok"]:
                dist.barrier()
            dist.destroy_process_group()
    return 0 if result["ok"] else 3


# ---------------------------------------------------------------- cli


def main(argv=None) -> int:
    args = lt.parse(argv)
    if lt.EFFECTIVE_BATCH % args.micro_batch:
        raise SystemExit("--micro-batch must divide 16")
    if args.mode == "ddp-child":
        return run_ddp_child(args)
    apply_kaggle(args)
    if args.mode == "qprobe":
        args.deadline = Deadline(args.deadline_s or DEFAULT_PROBE_DEADLINE_S)
        args.deadline_s = args.deadline_s or DEFAULT_PROBE_DEADLINE_S
        return run_qprobe(args)
    return lt.run(args)


if __name__ == "__main__":
    sys.exit(main())
