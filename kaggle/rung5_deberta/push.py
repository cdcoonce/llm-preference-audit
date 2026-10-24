"""Render and push the rung-5 DeBERTa kernel to Kaggle (or fetch its output).

The kernel is one generated script: a header that pip-installs this repo at a pinned commit,
then `train.py` with fixed arguments. Refuses unless the pinned SHA is HEAD, `llm_preference_audit/`
and `kaggle/` are clean, and the SHA exists on the remote. `--dry` prints instead of running.

  uv run python kaggle/rung5_deberta/push.py --username U --mode probe --tag probe1 --dry
  uv run python kaggle/rung5_deberta/push.py --username U --mode folds --folds 0,1,2,3,4 --tag full
  uv run python kaggle/rung5_deberta/push.py --username U --tag full --fetch
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BUILD_ROOT = Path("data/kaggle_build")  # gitignored with data/
OUT_ROOT = Path("data/kaggle_out")
CHECKED_DIRS = ("llm_preference_audit", "kaggle")
COMPETITION = "llm-classification-finetuning"
ACCELERATOR = "NvidiaTeslaT4"
DEFAULT_REPO = "cdcoonce/llm-preference-audit"
FUTURE = "from __future__ import annotations\n"


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
    out = ["--mode", a.mode, "--folds", a.folds, "--middle-share", str(a.middle_share)]
    out += ["--t", str(a.t), "--epochs", str(a.epochs), "--micro-batch", str(a.micro_batch)]
    out += ["--tag", a.tag, "--git-sha", sha]
    return out + (["--grad-ckpt"] if a.grad_ckpt else [])


def header(sha: str, repo: str, args: list[str]) -> str:
    return f"""
# --- generated kernel header (kaggle/rung5_deberta/push.py); the rest is train.py ---
import subprocess as _sp
import sys as _sys

if "--worker" not in _sys.argv[1:]:  # worker subprocesses re-exec this file; skip the install
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "--ignore-requires-python",  # Kaggle may run Python 3.11
                    "git+https://github.com/{repo}@{sha}"])
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q", "sentencepiece", "protobuf"])
    _sys.argv = ["train.py", *{args!r}]
# --- end header ---
"""


def render(a: argparse.Namespace, sha: str) -> Path:
    src = (HERE / "train.py").read_text()
    if src.count(FUTURE) != 1:
        raise SystemExit("train.py must contain exactly one `from __future__ import annotations`")
    code = src.replace(FUTURE, FUTURE + header(sha, a.repo, train_args(a, sha)))
    build = REPO_ROOT / BUILD_ROOT / a.tag
    build.mkdir(parents=True, exist_ok=True)
    code_file = f"rung5_{a.mode}.py"
    (build / code_file).write_text(code)
    meta = {
        "id": f"{a.username}/{a.slug}",
        "title": a.slug,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "competition_sources": [COMPETITION],
        "dataset_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return build


def run(cmd: list[str], dry: bool) -> None:
    print("$ " + shlex.join(cmd))
    if not dry:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--username", required=True)
    ap.add_argument("--slug", default="llm-pref-rung5-deberta")
    ap.add_argument("--mode", choices=["probe", "folds"], default="probe")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--middle-share", type=float, default=0.2)
    ap.add_argument("--t", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sha", help="commit to pin (must be HEAD); default HEAD")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--timeout", type=int, default=43200, help="kernel run limit, seconds")
    ap.add_argument("--dry", action="store_true", help="print commands instead of running them")
    ap.add_argument("--fetch", action="store_true", help="status + download output, no push")
    a = ap.parse_args(argv)
    ref = f"{a.username}/{a.slug}"
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
