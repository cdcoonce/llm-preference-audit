"""Render and push the rung-6b Qwen3 LoRA kernel to Kaggle (or fetch its output).

The kernel is one generated script: a header that pip-installs this repo at a pinned commit plus
peft/accelerate/transformers, then `train.py` with fixed arguments. Refuses unless the pinned SHA
is HEAD, `llm_preference_audit/` and `kaggle/` are clean, and the SHA exists on the remote.
`--dry` prints instead of running (it still renders the build directory).

  uv run python kaggle/rung6b_qwen3/push.py --username charlescoonce --mode probe --tag probe1 --dry
  uv run python kaggle/rung6b_qwen3/push.py --username charlescoonce --mode probe --tag probe1
  uv run python kaggle/rung6b_qwen3/push.py --username charlescoonce --mode probe --tag probe1 --fetch

Each tag gets its own kernel slug (`llm-pref-rung6b-qwen3-<tag>`) and output directory
(data/kaggle_out/<tag>/). The kernel is private, uses the GPU, and has internet ON (training
kernels need it for the pip installs; the data and the model come from Kaggle mounts).
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from llm_preference_audit.lora_g0 import QWEN3_MODEL_HANDLE

BUILD_ROOT = Path("data/kaggle_build")  # gitignored with data/
OUT_ROOT = Path("data/kaggle_out")
CHECKED_DIRS = ("llm_preference_audit", "kaggle")
COMPETITION = "llm-classification-finetuning"
ACCELERATOR = "NvidiaTeslaT4"
BASE_SLUG = "llm-pref-rung6b-qwen3"
DEFAULT_REPO = "cdcoonce/llm-preference-audit"
FUTURE = "from __future__ import annotations\n"
HANDLE_RE = re.compile(
    r"[\w-]+/[\w-]+/[\w-]+/[\w.-]+/\d+"
)  # owner/model/framework/variation/version
PROBE_TIMEOUT = 4200  # seconds; the in-process deadline (3300) stops the probe cleanly before it
LONG_TIMEOUT = 43200
# Tested with transformers 5.20.2 (`dtype=` kwarg, set_attn_implementation); Kaggle's image is
# unverified, so the header installs it. torch stays Kaggle's.
EXTRA_PIP = ("transformers==5.20.2", "peft", "accelerate", "sentencepiece", "protobuf")


def kernel_slug(a: argparse.Namespace) -> str:
    if a.slug:
        return a.slug
    tag = a.tag.lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", tag):
        raise SystemExit(f"--tag {a.tag!r} must be lowercase letters, digits and hyphens")
    return f"{BASE_SLUG}-{tag}"


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], check=True, capture_output=True, text=True, cwd=REPO_ROOT)
    return out.stdout.strip()


def preflight(sha: str | None) -> str:
    head = git("rev-parse", "HEAD")
    sha = sha or head
    print(f"check: HEAD = {head}")
    if sha != head:
        raise SystemExit(f"refusing: --sha {sha} is not HEAD ({head})")
    dirty = git("status", "--porcelain", "--", *CHECKED_DIRS)
    if dirty:
        raise SystemExit(f"refusing: uncommitted changes in {CHECKED_DIRS}:\n{dirty}")
    print(f"check: {' and '.join(CHECKED_DIRS)} are clean")
    git("fetch", "--quiet")
    on_remote = git("branch", "-r", "--contains", sha)
    if not on_remote:
        raise SystemExit(f"refusing: {sha} is not on any remote branch; push first")
    print(f"check: {sha[:12]} is on remote: {on_remote.splitlines()[0].strip()}")
    return sha


def train_args(a: argparse.Namespace, sha: str) -> list[str]:
    common = [
        "--kaggle", "--model", a.model_source, "--precision", "fp16-autocast", "--fp32-head-norm",
        "--t", str(a.t), "--micro-batch", str(a.micro_batch), "--tag", a.tag, "--git-sha", sha,
    ]  # fmt: skip
    if a.mode == "probe":
        return [
            "--mode", "qprobe", *common, "--probe-steps", str(a.probe_steps),
            "--probe-infer-rows", "1000", "--qattn-rows", "100", "--ddp-steps", "50",
            "--deadline-s", str(a.timeout - 900),
        ]  # fmt: skip
    mode = ["--mode", a.mode] + (["--fold", str(a.fold)] if a.mode == "fold" else [])
    return [*mode, *common, "--deadline-s", str(a.timeout - 900)]


def header(sha: str, repo: str, args: list[str]) -> str:
    return f"""
# --- generated kernel header (kaggle/rung6b_qwen3/push.py); the rest is train.py ---
import subprocess as _sp
import sys as _sys

_sp.check_call([_sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                "--ignore-requires-python",  # Kaggle may run Python 3.11
                "git+https://github.com/{repo}@{sha}"])
_sp.check_call([_sys.executable, "-m", "pip", "install", "-q", *{list(EXTRA_PIP)!r}])
_sys.argv = ["train.py", *{args!r}]
# --- end header ---
"""


def kernel_metadata(a: argparse.Namespace, code_file: str) -> dict:
    meta = {
        "id": f"{a.username}/{kernel_slug(a)}",
        "title": kernel_slug(a),
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "competition_sources": [COMPETITION],
        "dataset_sources": [],
        "kernel_sources": [],
        "model_sources": [a.model_source],
    }
    check_metadata(meta)
    return meta


def check_metadata(meta: dict) -> None:
    """A training kernel must be private, on the GPU, with internet (pip) and the model attached."""
    problems = []
    if meta.get("is_private") is not True:
        problems.append("is_private must be true")
    if meta.get("enable_gpu") is not True:
        problems.append("enable_gpu must be true")
    if meta.get("enable_internet") is not True:
        problems.append("enable_internet must be true (the training kernel pip-installs)")
    if meta.get("competition_sources") != [COMPETITION]:
        problems.append(f"competition_sources must be [{COMPETITION!r}]")
    sources = meta.get("model_sources") or []
    if len(sources) != 1 or not HANDLE_RE.fullmatch(sources[0]):
        problems.append("model_sources must be exactly one owner/model/framework/variation/version")
    if problems:
        raise SystemExit("kernel metadata refused: " + "; ".join(problems))


def render(a: argparse.Namespace, sha: str) -> Path:
    src = (HERE / "train.py").read_text()
    if src.count(FUTURE) != 1:
        raise SystemExit("train.py must contain exactly one `from __future__ import annotations`")
    code = src.replace(FUTURE, FUTURE + header(sha, a.repo, train_args(a, sha)))
    build = REPO_ROOT / BUILD_ROOT / a.tag
    build.mkdir(parents=True, exist_ok=True)
    code_file = f"rung6b_{a.mode}.py"
    (build / code_file).write_text(code)
    meta = kernel_metadata(a, code_file)
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return build


def run(cmd: list[str], dry: bool) -> None:
    print("$ " + shlex.join(cmd))
    if not dry:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--username", required=True)
    ap.add_argument("--slug", default=None, help="override the per-tag kernel slug")
    ap.add_argument("--mode", choices=["probe", "fold", "full"], default="probe")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--t", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--probe-steps", type=int, default=200)
    ap.add_argument("--model-source", default=QWEN3_MODEL_HANDLE)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sha", help="commit to pin (must be HEAD); default HEAD")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--timeout", type=int, default=None, help="kernel run limit, seconds")
    ap.add_argument("--dry", action="store_true", help="print commands instead of running them")
    ap.add_argument("--fetch", action="store_true", help="status + download output, no push")
    a = ap.parse_args(argv)
    if a.timeout is None:
        a.timeout = PROBE_TIMEOUT if a.mode == "probe" else LONG_TIMEOUT
    return a


def main(argv=None) -> int:
    a = parse(argv)
    ref = f"{a.username}/{kernel_slug(a)}"
    if a.fetch:
        dest = OUT_ROOT / a.tag
        run(["kaggle", "kernels", "status", ref], a.dry)
        run(["kaggle", "kernels", "output", ref, "-p", str(dest)], a.dry)
        return 0
    sha = preflight(a.sha)
    build = render(a, sha)
    print(f"rendered {build}")
    cmd = ["kaggle", "kernels", "push", "-p", str(build), "--accelerator", ACCELERATOR]
    run([*cmd, "-t", str(a.timeout)], a.dry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
