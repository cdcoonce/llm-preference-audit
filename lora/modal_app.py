"""Rung 6 on Modal: Gemma-2-9B LoRA probe, one-fold-per-container OOF runs, and the all-cv adapter.

  LORA_SHA=$(git rev-parse HEAD) modal run lora/modal_app.py --mode probe --tag probe --sha $LORA_SHA

`LORA_SHA` is read at import time and baked into the image build (a pinned `git+https` install of
this repo), so each commit is a distinct image. The local entrypoint refuses unless the SHA is
HEAD, the code dirs are clean and the SHA is on the remote, then runs the SPEND GUARD
(lora/spend_guard.py) before anything is launched. See lora/README.md for the full sequence.

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
# "H100!" opts out of Modal's automatic H100 -> H200 upgrade, so measured throughput and the
# $3.95/h price in llm_preference_audit/lora_cost.py describe the same hardware
# (https://modal.com/docs/guide/gpu, checked 2026-10-25).
GPU = "H100!"
FOLD_TIMEOUT_S = int(os.environ.get("LORA_TIMEOUT_S", str(6 * 3600)))  # per fold/full call
PROBE_TIMEOUT_S = 3600  # one hour = $3.95, under the $5 G0 cap
MAX_PARALLEL_FOLDS = 5

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")  # env HF_TOKEN

# Versions are the ones the CPU dry runs used (torch 2.14.1, transformers 5.20.2, peft 0.21.2).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
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
        "lightgbm",  # imported by llm_preference_audit.train_lgbm
    )
    .pip_install(f"git+https://github.com/{REPO}@{SHA or 'UNSET'}", extra_options="--no-deps")
    .env({"HF_HOME": f"{VOL}/hf-cache", "LORA_SHA": SHA, "PYTHONUNBUFFERED": "1"})
)
# A tiny CPU image for the data check: no pinned repo needed.
light = modal.Image.debian_slim(python_version="3.12")

COMMON = {"volumes": {VOL: volume}, "secrets": [hf_secret]}


def _dirs(tag: str) -> tuple[str, str, str]:
    return f"{VOL}/data", f"{VOL}/out/{tag}", f"{VOL}/adapters/{tag}"


def _argv(
    mode: str, tag: str, sha: str, attn: str, micro_batch: int, fold: int | None
) -> list[str]:
    data, out, adapter = _dirs(tag)
    a = ["--mode", mode, "--data-dir", data, "--out-dir", out, "--adapter-dir", adapter]
    a += ["--tag", tag, "--git-sha", sha, "--attn", attn, "--micro-batch", str(micro_batch)]
    return a + (["--fold", str(fold)] if fold is not None else [])


def _run(argv: list[str], sha: str) -> int:
    if os.environ.get("LORA_SHA") != sha:
        raise RuntimeError(f"container image is for {os.environ.get('LORA_SHA')}, asked for {sha}")
    volume.reload()  # see the model cache and data other containers committed
    from llm_preference_audit import lora_train

    t0 = time.time()
    code = lora_train.main(argv)
    print(f"lora_train exit {code} after {time.time() - t0:.0f}s", flush=True)
    volume.commit()  # persist outputs and any newly downloaded weights
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


@app.function(image=image, gpu=GPU, timeout=PROBE_TIMEOUT_S, **COMMON)
def probe(tag: str, sha: str, attn: str, micro_batch: int) -> int:
    return _run(_argv("probe", tag, sha, attn, micro_batch, None), sha)


@app.function(
    image=image, gpu=GPU, timeout=FOLD_TIMEOUT_S, max_containers=MAX_PARALLEL_FOLDS, **COMMON
)
def run_fold(fold: int, tag: str, sha: str, attn: str, micro_batch: int) -> int:
    code = _run(_argv("fold", tag, sha, attn, micro_batch, fold), sha)
    from llm_preference_audit import lora_train

    # The log must record the config the launcher asked for (a dropped flag would otherwise
    # train a silently different run).
    argv = _argv("fold", tag, sha, attn, micro_batch, fold)
    want = lora_train.expected_config(lora_train.parse(argv)) | {"attn_implementation": attn}
    lora_train.check_worker_config(Path(_dirs(tag)[1]), fold, want)
    return code


@app.function(image=image, gpu=GPU, timeout=FOLD_TIMEOUT_S, **COMMON)
def train_full(tag: str, sha: str, attn: str, micro_batch: int) -> int:
    return _run(_argv("full", tag, sha, attn, micro_batch, None), sha)


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
    if sha != SHA:
        raise SystemExit(f"--sha {sha!r} must equal the LORA_SHA the image was built for ({SHA!r})")
    lp.preflight(sha)
    fold_list = lp.parse_folds(folds) if mode == "fold" else []
    probe_json = lp.latest_probe(lp.REPO_ROOT / lp.OUT_ROOT)
    attn = lp.resolve_attn(attn, probe_json)
    n_jobs = len(fold_list) if mode == "fold" else 1
    estimate = lp.estimate_usd(mode, n_jobs, probe_json)

    os.chdir(lp.REPO_ROOT)  # the guard's default paths are repo-relative
    try:
        accounting = sg.check_launch(mode, fold_list, estimate, approved_usd=approved_usd)
    except sg.SpendRefused as e:
        raise SystemExit(str(e)) from e
    print(
        f"spend guard OK: estimate ${estimate:.2f}, rung-6 after ${accounting['rung6_after_usd']:.2f}"
        f" of ${accounting['approved_cap_usd']:.2f} approved, GPU total "
        f"${accounting['gpu_total_usd'] + estimate:.2f} of ${sg.CEILING_USD:.0f}; attn={attn}"
    )
    if dry:
        print("--dry: stopping before upload and launch")
        return
    if not yes:
        raise SystemExit("re-run with --yes to launch (this spends money)")

    _upload_data(lp.REPO_ROOT / "data/raw/train.csv")
    job_id = f"{mode}-{tag}-{int(time.time())}"
    sg.record_launch(job_id, mode, tag, fold_list, estimate)  # BEFORE the launch
    codes: dict = {}
    try:
        if mode == "probe":
            codes = {"probe": probe.remote(tag, sha, attn, micro_batch)}
        elif mode == "full":
            codes = {"full": train_full.remote(tag, sha, attn, micro_batch)}
        else:
            kw = {"tag": tag, "sha": sha, "attn": attn, "micro_batch": micro_batch}
            results = run_fold.map(fold_list, kwargs=kw, return_exceptions=True)
            codes = {f: r for f, r in zip(fold_list, results, strict=True)}
    finally:
        files = _download(tag, out_dir)
        print(f"fetched {len(files)} files to {out_dir}")
        try:
            actual = sg.cost_from_logs(out_dir)
            sg.reconcile(job_id, actual, note="est_cost_usd x 1.15 from logs")
            print(f"reconciled {job_id}: ${actual:.2f} (check the Modal dashboard for the bill)")
        except ValueError as e:
            print(f"NOT reconciled ({e}); the launch estimate stays on the books")
    print(json.dumps({str(k): repr(v) for k, v in codes.items()}))
    if any(v != 0 for v in codes.values()):
        raise SystemExit("at least one job failed; read its log before spending more")
