"""Render and push the rung-6 timing / submission kernels (internet OFF). Not run by tests.

  uv run python kaggle/rung6_submission/push.py --mode timing --dry
  uv run python kaggle/rung6_submission/push.py --mode submission --dry
  uv run python kaggle/rung6_submission/push.py --mode timing --fetch
  uv run python kaggle/rung6_submission/push.py --mode submission --print-submit-cmd --version 3

Each kernel is `infer.py` plus a generated header that fixes its arguments (the repo code itself
comes from the wheel in the attached dataset, built by build_dataset.py). Refuses unless HEAD is
clean in `llm_preference_audit/` and `kaggle/`, is on the remote, and the built dataset's
MANIFEST.json was made from that same HEAD. `--dry` prints the commands and runs nothing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BUILD_ROOT = Path("data/kaggle_build")
OUT_ROOT = Path("data/kaggle_out")
DATASET_DIR = BUILD_ROOT / "rung6_dataset"
COMPETITION = "llm-classification-finetuning"
DATASET = "charlescoonce/llm-pref-rung6-adapter"
# owner/model/framework/variation/version. Verified read-only on 2026-10-25 with
# `kaggle models instances get google/gemma-2/transformers/gemma-2-9b-it` (versionNumber 2).
GEMMA_MODEL_SOURCE = "google/gemma-2/transformers/gemma-2-9b-it/2"
# "NvidiaTeslaT4" gave two Tesla T4 on the rung-5 probe (probe.json device list)
ACCELERATOR = "NvidiaTeslaT4"
SLUGS = {"timing": "llm-pref-rung6-timing", "submission": "llm-pref-rung6-submission"}
FUTURE = "from __future__ import annotations\n"
TIMING_LIMIT_S = 9000
TIMING_TIMEOUT_S = 10800
SUBMISSION_TIMEOUT_S = 32400  # Kaggle's 9 h session cap


def _rung5():
    spec = importlib.util.spec_from_file_location(
        "rung5_push", REPO_ROOT / "kaggle/rung5_deberta/push.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def preflight(sha: str | None, allow_mismatch: bool = False) -> str:
    sha = _rung5().preflight(sha)
    manifest = REPO_ROOT / DATASET_DIR / "MANIFEST.json"
    if not manifest.is_file():
        raise SystemExit(f"refusing: {manifest} does not exist; run build_dataset.py first")
    built = json.loads(manifest.read_text()).get("git_sha")
    if built != sha and not allow_mismatch:
        raise SystemExit(
            f"refusing: the dataset was built from {built}, HEAD is {sha}; rebuild the dataset"
        )
    print(f"check: dataset manifest git sha {str(built)[:12]} matches")
    return sha


def kernel_args(a: argparse.Namespace) -> list[str]:
    out = ["--limit-s", str(a.limit_s if a.mode == "submission" else TIMING_LIMIT_S)]
    if a.mode == "timing":
        out += ["--timing-rows", str(a.timing_rows)]
    if a.t:
        out += ["--t", str(a.t)]
    return out + ["--batch-tokens", str(a.batch_tokens), "--max-batch", str(a.max_batch)]


def header(args: list[str]) -> str:
    return f"""
# --- generated kernel header (kaggle/rung6_submission/push.py); the rest is infer.py ---
import sys as _sys

if "--worker" not in _sys.argv[1:]:  # worker subprocesses re-exec this file with their own argv
    _sys.argv = ["infer.py", *{args!r}]
# --- end header ---
"""


def metadata(a: argparse.Namespace) -> dict:
    return {
        "id": f"{a.username}/{SLUGS[a.mode]}",
        "title": SLUGS[a.mode],
        "code_file": f"rung6_{a.mode}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,  # the competition's rule for code submissions; no pip, no HF
        "machine_shape": ACCELERATOR,
        "competition_sources": [COMPETITION],
        "dataset_sources": [DATASET],
        "kernel_sources": [],
        "model_sources": [GEMMA_MODEL_SOURCE],
    }


def render(a: argparse.Namespace) -> Path:
    src = (HERE / "infer.py").read_text()
    if src.count(FUTURE) != 1:
        raise SystemExit("infer.py must contain exactly one `from __future__ import annotations`")
    code = src.replace(FUTURE, FUTURE + header(kernel_args(a)))
    build = REPO_ROOT / BUILD_ROOT / f"rung6_{a.mode}"
    build.mkdir(parents=True, exist_ok=True)
    meta = metadata(a)
    (build / meta["code_file"]).write_text(code)
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return build


def run(cmd: list[str], dry: bool) -> None:
    print("$ " + shlex.join(cmd))
    if not dry:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def submit_command(a: argparse.Namespace) -> list[str]:
    """Code competitions are submitted from a finished notebook version, not by uploading a CSV."""
    if not a.version:
        raise SystemExit("--version N (the kernel version that finished) is required")
    return [
        "kaggle", "competitions", "submit", COMPETITION,
        "-k", f"{a.username}/{SLUGS['submission']}", "-v", str(a.version),
        "-f", "submission.csv", "-m", a.message,
    ]  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--username", default="charlescoonce")
    ap.add_argument("--mode", choices=sorted(SLUGS), required=True)
    ap.add_argument("--timing-rows", type=int, default=1000)
    ap.add_argument("--t", type=int, default=0, help="0 = the adapter's training T")
    ap.add_argument("--batch-tokens", type=int, default=4096)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--limit-s", type=int, default=8 * 3600 + 15 * 60)
    ap.add_argument("--sha", help="commit to pin (must be HEAD); default HEAD")
    ap.add_argument("--allow-sha-mismatch", action="store_true")
    ap.add_argument("--dry", action="store_true", help="print commands instead of running them")
    ap.add_argument("--fetch", action="store_true", help="status + download output, no push")
    ap.add_argument("--print-submit-cmd", action="store_true", help="print the submit command only")
    ap.add_argument("--version", type=int, default=0)
    ap.add_argument("--message", default="rung 6 Gemma-2-9B LoRA, both orders")
    a = ap.parse_args(argv)
    ref = f"{a.username}/{SLUGS[a.mode]}"
    if a.print_submit_cmd:
        print(shlex.join(submit_command(a)))
        return 0
    if a.fetch:
        run(["kaggle", "kernels", "status", ref], a.dry)
        run(["kaggle", "kernels", "output", ref, "-p", str(OUT_ROOT / f"rung6_{a.mode}")], a.dry)
        return 0
    preflight(a.sha, a.allow_sha_mismatch)
    build = render(a)
    print(f"rendered {build}")
    timeout = SUBMISSION_TIMEOUT_S if a.mode == "submission" else TIMING_TIMEOUT_S
    run(["kaggle", "kernels", "push", "-p", str(build), "--accelerator", ACCELERATOR,
         "-t", str(timeout)], a.dry)  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
