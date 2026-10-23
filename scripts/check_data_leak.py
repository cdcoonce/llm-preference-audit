"""Fail if tracked files could leak the (CC BY-NC) Chatbot Arena data.

Rules (violation ids): data-dir, oversize, text-columns-csv, text-columns-jsonl.
Plain stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

MAX_BYTES = 1_000_000  # 1 MB
DATA_EXTS = {
    ".csv", ".tsv", ".parquet", ".jsonl", ".json",
    ".feather", ".arrow", ".pkl", ".pickle", ".npz",
}  # fmt: skip
TEXT_COLUMNS = {"prompt", "response_a", "response_b"}
PREDICTION_HEADER = {"id", "winner_model_a", "winner_model_b", "winner_tie"}


class Violation(NamedTuple):
    rule: str
    path: str
    detail: str


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    ).stdout
    return [p for p in out.decode().split("\0") if p]


def _csv_header(path: Path, delimiter: str) -> list[str]:
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        return next(csv.reader(fh, delimiter=delimiter), [])


def _first_jsonl_record(path: Path) -> dict | None:
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            return rec if isinstance(rec, dict) else None
    return None


def check(root: Path, files: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    for rel in files:
        path = root / rel
        parts = Path(rel).parts
        if parts and parts[0] == "data":
            violations.append(Violation("data-dir", rel, "tracked path under data/"))
        ext = path.suffix.lower()
        if ext not in DATA_EXTS or not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_BYTES:
            violations.append(Violation("oversize", rel, f"{size} bytes > {MAX_BYTES}"))
        if ext in {".csv", ".tsv"}:
            header = _csv_header(path, "\t" if ext == ".tsv" else ",")
            if set(header) == PREDICTION_HEADER and len(header) == len(PREDICTION_HEADER):
                continue  # prediction file: id + three probabilities only
            hit = TEXT_COLUMNS & set(header)
            if hit:
                violations.append(Violation("text-columns-csv", rel, f"columns {sorted(hit)}"))
        elif ext == ".jsonl":
            rec = _first_jsonl_record(path)
            hit = TEXT_COLUMNS & set(rec) if rec else set()
            if hit:
                violations.append(Violation("text-columns-jsonl", rel, f"keys {sorted(hit)}"))
    return violations


def main(argv: list[str] | None = None, files: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if files is None:
        files = tracked_files(root)
    violations = check(root, files)
    for v in violations:
        print(f"{v.rule}: {v.path}: {v.detail}")
    if violations:
        print(f"FAIL: {len(violations)} violation(s)")
        return 1
    print(f"ok: {len(files)} tracked files checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
