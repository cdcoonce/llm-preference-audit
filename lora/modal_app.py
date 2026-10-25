"""Rung 6 on Modal: Gemma-2-9B LoRA probe, one-fold-per-container OOF runs, and the all-cv adapter.

  LORA_SHA=$(git rev-parse HEAD) modal run lora/modal_app.py --mode probe --tag probe --sha $LORA_SHA

`LORA_SHA` is read at import time and baked into the image build (a pinned `git+https` install of
this repo), so each commit is a distinct image. The GPU functions only exist when `LORA_SHA` is
set, so `--mode fetch` runs without it. The local entrypoint refuses unless the SHA is HEAD, the
code dirs are clean and the SHA is on the remote, then runs the SPEND GUARD
(lora/spend_guard.py) before anything is launched; the guard derives each job's time limit from
the remaining approved budget, and the Modal timeout, the in-process deadline and the recorded
worst case all come from that one number. See lora/README.md for the full sequence.

Preemption (checked 2026-10-25): modal 1.6.2 has `retries=` (set to 0 on every GPU function here)
but Modal's preemption guide says a preempted input is restarted on the same input, does not say
whether that counts against retries, and `nonpreemptible` is not supported for GPU functions. So a
preemption can re-bill an input outside the guard's one-attempt worst case. The container deletes
its own stale outputs at start, so a rerun cannot mix runs; check the dashboard after each launch.

Layout on the Modal volume `llm-pref-audit` (mounted at /vol):
  /vol/data/train.csv        the Hugging Face copy (uploaded by this entrypoint; never test.csv)
  /vol/hf-cache              HF_HOME, model weights
  /vol/out/<tag>/            probe.json, oof_fold{k}.parquet, log_fold{k}.json, log_full.json
  /vol/adapters/<tag>/       the all-cv submission adapter (full mode)

The container functions only import the pinned package (`llm_preference_audit`); the local-only
helpers (spend_guard, launch_plan) are imported inside `main`, never at module level.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import modal

APP_NAME = "llm-pref-audit-lora"
REPO = "cdcoonce/llm-preference-audit"
SHA = os.environ.get("LORA_SHA", "")
VOLUME_NAME = "llm-pref-audit"
VOL = "/vol"
MODEL = "google/gemma-2-9b-it"
PREFETCH_MARKER = f"{VOL}/hf-cache/.prefetched/gemma-2-9b-it"
# "H100!" opts out of Modal's automatic H100 -> H200 upgrade, so measured throughput and the
# $3.95/h price in llm_preference_audit/lora_cost.py describe the same hardware
# (https://modal.com/docs/guide/gpu, checked 2026-10-25).
GPU = "H100!"
# Fallback timeouts only: every launch overrides them with `Function.with_options(timeout=...)`
# (modal 1.6.2 has it) computed by the spend guard from the remaining approved budget.
PROBE_TIMEOUT_S = 4600  # $5.05 at $3.95/h
FOLD_TIMEOUT_S = 2 * 3600
MAX_PARALLEL_FOLDS = 5

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")  # env HF_TOKEN; only the CPU prefetch gets it

# A tiny CPU image for the data check and the weight prefetch: no pinned repo needed.
light = modal.Image.debian_slim(python_version="3.12")
prefetch_image = light.pip_install("huggingface_hub").env({"HF_HOME": f"{VOL}/hf-cache"})


def _dirs(tag: str) -> tuple[str, str, str]:
    return f"{VOL}/data", f"{VOL}/out/{tag}", f"{VOL}/adapters/{tag}"


def _argv(
    mode: str,
    tag: str,
    sha: str,
    attn: str,
    micro_batch: int,
    fold: int | None,
    deadline_s: float,
) -> list[str]:
    data, out, adapter = _dirs(tag)
    a = ["--mode", mode, "--data-dir", data, "--out-dir", out, "--adapter-dir", adapter]
    a += ["--tag", tag, "--git-sha", sha, "--attn", attn, "--micro-batch", str(micro_batch)]
    a += ["--deadline-s", str(deadline_s), "--local-files-only"]
    return a + (["--fold", str(fold)] if fold is not None else [])


def _run(argv: list[str], sha: str) -> int:
    if os.environ.get("LORA_SHA") != sha:
        raise RuntimeError(f"container image is for {os.environ.get('LORA_SHA')}, asked for {sha}")
    volume.reload()  # see the model cache and data other containers committed
    if not Path(PREFETCH_MARKER).exists():  # fail fast, before any GPU time is spent on a download
        raise RuntimeError(f"weights not prefetched ({PREFETCH_MARKER} missing); run the prefetch")
    from llm_preference_audit import lora_train

    t0 = time.time()
    code = lora_train.main(argv)
    print(f"lora_train exit {code} after {time.time() - t0:.0f}s", flush=True)
    volume.commit()  # persist outputs
    return code


@app.function(image=light, volumes={VOL: volume}, timeout=1800)
def data_status() -> str | None:
    """sha256 of /vol/data/train.csv, or None when absent."""
    volume.reload()
    path = Path(f"{VOL}/data/train.csv")
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@app.function(image=light, volumes={VOL: volume}, timeout=600)
def prefetch_status() -> bool:
    volume.reload()
    return Path(PREFETCH_MARKER).exists()


@app.function(
    image=prefetch_image,
    volumes={VOL: volume},
    secrets=[hf_secret],
    timeout=1800,  # spend_guard.PREFETCH_TIMEOUT_S
    cpu=2,
    memory=8192,
    retries=0,
)
def prefetch(model: str = MODEL) -> str:
    """CPU only: download the weights into the volume's HF cache and commit it."""
    from huggingface_hub import snapshot_download

    volume.reload()
    path = snapshot_download(
        model,
        token=os.environ["HF_TOKEN"],
        allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
    )
    snapshot_download(model, local_files_only=True, allow_patterns=["config.json"])  # in the cache
    marker = Path(PREFETCH_MARKER)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{model}\n{path}\n")
    volume.commit()
    return path


if SHA:
    # Defined only with a pinned commit: `modal run` builds every function's image, and the GPU
    # image is a pip install of this repo at SHA. `--mode fetch` needs none of it.
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "libgomp1")
        .pip_install(
            "torch==2.14.1",
            "transformers==5.20.2",
            "peft==0.21.2",
            "accelerate",
            "sentencepiece",
            "protobuf",
            "pandas",
            "numpy",
            "pyarrow",
            "scikit-learn",
        )
        .pip_install(f"git+https://github.com/{REPO}@{SHA}", extra_options="--no-deps")
        .env(
            {
                "HF_HOME": f"{VOL}/hf-cache",
                "HF_HUB_OFFLINE": "1",  # weights come from the prefetched cache only
                "LORA_SHA": SHA,
                "PYTHONUNBUFFERED": "1",
            }
        )
    )
    GPU_KW = {"image": image, "gpu": GPU, "volumes": {VOL: volume}, "retries": 0}

    @app.function(timeout=PROBE_TIMEOUT_S, **GPU_KW)
    def probe(tag: str, sha: str, attn: str, micro_batch: int, deadline_s: float) -> int:
        return _run(_argv("probe", tag, sha, attn, micro_batch, None, deadline_s), sha)

    @app.function(timeout=FOLD_TIMEOUT_S, max_containers=MAX_PARALLEL_FOLDS, **GPU_KW)
    def run_fold(
        fold: int, tag: str, sha: str, attn: str, micro_batch: int, deadline_s: float
    ) -> int:
        argv = _argv("fold", tag, sha, attn, micro_batch, fold, deadline_s)
        code = _run(argv, sha)
        from llm_preference_audit import lora_train

        # The log must record the config the launcher asked for (a dropped flag would otherwise
        # train a silently different run). A mismatch raises WorkerConfigMismatch, which Modal
        # reports as the call's exception.
        want = lora_train.expected_config(lora_train.parse(argv)) | {"attn_implementation": attn}
        lora_train.check_worker_config(Path(_dirs(tag)[1]), fold, want)
        return code

    @app.function(timeout=FOLD_TIMEOUT_S, **GPU_KW)
    def train_full(tag: str, sha: str, attn: str, micro_batch: int, deadline_s: float) -> int:
        return _run(_argv("full", tag, sha, attn, micro_batch, None, deadline_s), sha)


# ---------------------------------------------------------------- local entrypoint


def _upload_data(raw: Path) -> None:
    """Put the Hugging Face train.csv on the volume if it is absent or its hash differs."""
    from llm_preference_audit.kernel_common import TRAIN_SHA256, sha256_file

    local = sha256_file(raw)
    if local != TRAIN_SHA256:
        raise SystemExit(f"{raw} sha256 {local} != expected {TRAIN_SHA256}; refusing to upload")
    remote = data_status.remote()
    if remote == TRAIN_SHA256:
        print("volume already has the verified train.csv")
        return
    print(f"uploading {raw.name} ({raw.stat().st_size / 1e6:.0f} MB) to {VOLUME_NAME}:/data")
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(raw), "/data/train.csv")  # never test.csv


def _download(tag: str, dest: Path) -> list[Path]:
    """Copy /out/<tag>/* from the volume to data/lora_out/<tag>/ (gitignored under data/)."""
    dest.mkdir(parents=True, exist_ok=True)
    got = []
    for entry in volume.listdir(f"out/{tag}", recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        rel = Path(entry.path.lstrip("/")).relative_to(f"out/{tag}")
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as fh:
            for chunk in volume.read_file(entry.path):
                fh.write(chunk)
        got.append(target)
    return got


@app.local_entrypoint()
def main(
    mode: str,
    folds: str = "0",
    tag: str = "",
    sha: str = "",
    attn: str = "auto",
    micro_batch: int = 4,
    yes: bool = False,
    approved_usd: float = 15.0,
    dry: bool = False,
):
    """mode: probe | fold | full | fetch.  `yes` confirms the estimate; `dry` stops before launch."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import launch_plan as lp
    import spend_guard as sg

    if mode not in ("probe", "fold", "full", "fetch"):
        raise SystemExit("--mode must be probe, fold, full or fetch")
    lp.validate_tag(tag)
    out_dir = lp.REPO_ROOT / lp.OUT_ROOT / tag
    if mode == "fetch":
        files = _download(tag, out_dir)
        print(f"fetched {len(files)} files to {out_dir}")
        return
    if not SHA:
        raise SystemExit(
            "LORA_SHA is not set: export LORA_SHA=$(git rev-parse HEAD) before modal run"
        )
    if sha != SHA:
        raise SystemExit(f"--sha {sha!r} must equal the LORA_SHA the image was built for ({SHA!r})")
    lp.preflight(sha)
    fold_list = lp.parse_folds(folds) if mode == "fold" else []
    n_jobs = len(fold_list) if mode == "fold" else 1
    os.chdir(lp.REPO_ROOT)  # the guard's default paths are repo-relative
    lp.refuse_stale_outputs(out_dir, mode, fold_list)

    def guard():
        """The guard's verdict, the attention choice, the estimate and the time limits."""
        probe_json = lp.latest_probe(lp.REPO_ROOT / lp.OUT_ROOT, sha)
        estimate = lp.estimate_usd(mode, n_jobs, probe_json)
        try:
            acc = sg.check_launch(mode, fold_list, estimate, approved_usd=approved_usd, git_sha=sha)
            plan = sg.deadline_plan(mode, n_jobs, acc["remaining_usd"], acc["probe"] or probe_json)
        except sg.SpendRefused as e:
            raise SystemExit(str(e)) from e
        return acc, plan, estimate, lp.resolve_attn(attn, probe_json)

    acc, plan, estimate, attn_used = guard()
    print(
        f"spend guard OK: estimate ${estimate:.2f}, worst case ${plan['worst_case_usd']:.2f} "
        f"(deadline {plan['deadline_s']:.0f}s + {sg.SAFETY_MARGIN_S}s margin per job, {n_jobs} "
        f"job(s)); rung-6 after ${acc['rung6_after_usd']:.2f} of ${acc['approved_cap_usd']:.2f} "
        f"approved, GPU total ${acc['gpu_total_usd'] + estimate:.2f} of ${sg.CEILING_USD:.0f}; "
        f"attn={attn_used}"
    )
    if dry:
        print("--dry: stopping before upload and launch")
        return
    if not yes:
        raise SystemExit("re-run with --yes to launch (this spends money)")

    _upload_data(lp.REPO_ROOT / "data/raw/train.csv")
    if not prefetch_status.remote():
        pf_usd = sg.PREFETCH_TIMEOUT_S / 3600 * sg.CPU_PRICE_PER_HOUR
        with sg.launch_lock():  # CPU spend at a conservative $0.20/h (Modal lists ~$0.16/h here)
            sg.record_launch(
                f"prefetch-{int(time.time())}", "prefetch", "-", [], pf_usd, worst_case_usd=pf_usd,
                git_sha=sha, kind="cpu",
            )  # fmt: skip
        print(f"prefetching {MODEL} on CPU (recorded ${pf_usd:.2f} worst case)")
        prefetch.remote(MODEL)
    job_id = f"{mode}-{tag}-{int(time.time())}"
    with sg.launch_lock():  # guard check through record_launch: no second launch can slip between
        acc, plan, estimate, attn_used = guard()  # the prefetch spend is now on the books
        sg.record_launch(
            job_id, mode, tag, fold_list, estimate, worst_case_usd=plan["worst_case_usd"],
            per_job_worst_usd=plan["per_job_worst_usd"], deadline_s=plan["deadline_s"],
            n_jobs=n_jobs, git_sha=sha,
        )  # fmt: skip
    timeout = math.ceil(plan["timeout_s"])
    dl = plan["deadline_s"]
    codes: dict = {}
    try:
        if mode == "probe":
            fn = probe.with_options(timeout=timeout)
            codes = {"probe": fn.remote(tag, sha, attn_used, micro_batch, dl)}
        elif mode == "full":
            fn = train_full.with_options(timeout=timeout)
            codes = {"full": fn.remote(tag, sha, attn_used, micro_batch, dl)}
        else:
            kw = {
                "tag": tag, "sha": sha, "attn": attn_used, "micro_batch": micro_batch,
                "deadline_s": dl,
            }  # fmt: skip
            fn = run_fold.with_options(timeout=timeout)
            results = fn.map(fold_list, kwargs=kw, return_exceptions=True)
            codes = {f: r for f, r in zip(fold_list, results, strict=True)}
    finally:
        try:
            files = _download(tag, out_dir)
            print(f"fetched {len(files)} files to {out_dir}")
        finally:
            line = sg.auto_reconcile(job_id, out_dir)  # never below what could have been billed
            print(
                f"reconciled {job_id}: ${line['actual_usd']:.2f} (was ${line['previous_usd']:.2f}); "
                "check the Modal dashboard, then lower it with `reconcile --actual-usd X "
                "--source modal-dashboard` if the bill is lower"
            )
    print(json.dumps({str(k): repr(v) for k, v in codes.items()}))
    if any(v != 0 for v in codes.values()):
        raise SystemExit("at least one job failed; read its log before spending more")
