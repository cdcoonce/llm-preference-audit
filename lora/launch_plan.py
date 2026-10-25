"""Rung-6 launch helpers (torch-free and modal-free): preflight, estimates, attention choice.

`modal_app.py` imports this lazily inside its local entrypoint (the module is not importable
inside the container, which only ships the pinned package).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKED_DIRS = ("llm_preference_audit", "lora", "kaggle")
OUT_ROOT = Path("data/lora_out")
TAG_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,40}")
# Plan estimates (docs/plans/rung-6-lora.md) used before a probe exists: hours x $3.95/h.
PRICE = 3.95
PLAN_HOURS = {"probe": 0.5, "fold": 2.1, "full": 2.3}
AUTO_MARGIN = 0.20


def git(*args: str, root: Path | None = None) -> str:
    out = subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, cwd=root or REPO_ROOT
    )
    return out.stdout.strip()


def preflight(sha: str, root: Path | None = None) -> str:
    """HEAD is the pinned sha, the code dirs are clean, and the sha is on a remote branch."""
    head = git("rev-parse", "HEAD", root=root)
    print(f"check: HEAD = {head}")
    if not sha or sha != head:
        raise SystemExit(f"refusing: --sha {sha!r} is not HEAD ({head})")
    dirty = git("status", "--porcelain", "--", *CHECKED_DIRS, root=root)
    if dirty:
        raise SystemExit(f"refusing: uncommitted changes in {CHECKED_DIRS}:\n{dirty}")
    print(f"check: {', '.join(CHECKED_DIRS)} are clean")
    git("fetch", "--quiet", root=root)
    on_remote = git("branch", "-r", "--contains", sha, root=root)
    if not on_remote:
        raise SystemExit(f"refusing: {sha} is not on any remote branch; push first")
    print(f"check: {sha[:12]} is on remote: {on_remote.splitlines()[0].strip()}")
    return sha


def validate_tag(tag: str) -> str:
    if not TAG_RE.fullmatch(tag or ""):
        raise SystemExit(f"--tag {tag!r} must be 1-41 chars of lowercase letters, digits, hyphens")
    return tag


def parse_folds(folds: str) -> list[int]:
    try:
        out = [int(f) for f in folds.split(",") if f.strip() != ""]
    except ValueError as e:
        raise SystemExit(f"--folds must be comma-separated integers: {e}") from e
    if not out or any(not 0 <= f <= 4 for f in out) or len(set(out)) != len(out):
        raise SystemExit("--folds must be distinct integers in 0-4")
    return out


def latest_probe(root: Path = OUT_ROOT) -> dict | None:
    files = sorted(Path(root).glob("*/probe.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(files[-1].read_text()) if files else None


def resolve_attn(attn: str, probe: dict | None) -> str:
    """`auto` takes the probe's decision, else eager (the safe choice for Gemma2 softcapping)."""
    if attn in ("eager", "sdpa"):
        return attn
    if attn != "auto":
        raise SystemExit(f"--attn must be auto, eager or sdpa, not {attn!r}")
    choice = (probe or {}).get("attention_decision")
    return choice if choice in ("eager", "sdpa") else "eager"


def estimate_usd(mode: str, n_folds: int, probe: dict | None) -> float:
    """Dollar estimate for the guard: the probe projection x 1.2 when known, else the plan."""
    proj = (probe or {}).get("projection")
    if mode == "probe":
        return PLAN_HOURS["probe"] * PRICE
    if mode == "fold":
        per = proj["fold_usd"] * (1 + AUTO_MARGIN) if proj else PLAN_HOURS["fold"] * PRICE
        return per * n_folds
    if mode == "full":
        return proj["full_usd"] * (1 + AUTO_MARGIN) if proj else PLAN_HOURS["full"] * PRICE
    raise SystemExit(f"unknown mode {mode!r}")
