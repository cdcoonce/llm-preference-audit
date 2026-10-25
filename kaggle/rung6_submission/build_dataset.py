"""Build the private Kaggle dataset the offline submission kernel installs from. Never uploads.

  uv run python kaggle/rung6_submission/build_dataset.py --adapter data/lora_out/full/adapter \
      --tag full [--calibrator path.json] [--include-wheels peft,bitsandbytes,accelerate]

Output (data/kaggle_build/rung6_dataset/, gitignored):
  llm_preference_audit-*.whl   this package, built from a clean pinned HEAD that is on the remote
  adapter/                     the LoRA adapter + trained score head (adapter_model.safetensors,
                               adapter_config.json, adapter_meta.json) copied from --adapter
  calibrator.json              optional matrix-scaling calibrator (calibrate.export_calibrator)
  wheels/                      optional third-party wheels for Kaggle's Python 3.13, offline installs
  MANIFEST.json                sha256 of every file, git sha, adapter tag (infer.py verifies it)
  dataset-metadata.json        charlescoonce/llm-pref-rung6-adapter, private

It prints, but does not run, the `kaggle datasets create` / `version` command: uploading is
Charles's call. `--dev` skips the clean-HEAD preflight and the wheel build (CPU dry runs only) and
stamps the manifest git_sha "dev", which infer.py refuses without --allow-dev-manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BUILD_ROOT = Path("data/kaggle_build")
DATASET_NAME = "rung6_dataset"
DATASET_ID = "charlescoonce/llm-pref-rung6-adapter"
KAGGLE_PYTHON = "3.13"  # per the rung-5 probe log (python 3.13.15)
PLATFORMS = (  # Kaggle's image is glibc 2.35; bitsandbytes ships manylinux_2_24 wheels
    "manylinux2014_x86_64",
    "manylinux_2_17_x86_64",
    "manylinux_2_24_x86_64",
    "manylinux_2_28_x86_64",
)
ADAPTER_REQUIRED = ("adapter_config.json", "adapter_model.safetensors", "adapter_meta.json")
SKIP_IN_MANIFEST = {"MANIFEST.json", "dataset-metadata.json"}


def _load_rung5_push():
    spec = importlib.util.spec_from_file_location(
        "rung5_push", REPO_ROOT / "kaggle/rung5_deberta/push.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(ds: Path, git_sha: str, adapter_tag: str, extra: dict | None = None) -> dict:
    """MANIFEST.json over every file in `ds` except itself and dataset-metadata.json."""
    files = {}
    for p in sorted(ds.rglob("*")):
        rel = str(p.relative_to(ds))
        if p.is_file() and rel not in SKIP_IN_MANIFEST:
            files[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    manifest = {"git_sha": git_sha, "adapter_tag": adapter_tag, "files": files} | (extra or {})
    (ds / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def check_adapter(adapter: Path) -> dict:
    missing = [f for f in ADAPTER_REQUIRED if not (adapter / f).is_file()]
    if missing:
        raise SystemExit(f"{adapter} is missing {missing}")
    return json.loads((adapter / "adapter_meta.json").read_text())


def check_calibrator(path: Path) -> None:
    coef = json.loads(path.read_text())
    if coef.get("kind") != "matrix_scaling" or list(coef.get("classes", [])) != [0, 1, 2]:
        raise SystemExit(f"{path} is not a matrix_scaling calibrator export")


def metadata() -> dict:
    return {
        "title": "llm-pref-rung6-adapter",
        "id": DATASET_ID,
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }


def download_wheels(names: list[str], dest: Path, dry: bool) -> None:
    """Binary wheels for Kaggle's Python and platform, no dependency resolution (torch and the
    rest are already in the image). pip is run through uvx so it does not need to be installed."""
    cmd = ["uvx", "pip", "download", "--only-binary=:all:", "--no-deps", "-d", str(dest)]
    cmd += ["--python-version", KAGGLE_PYTHON, "--implementation", "cp"]
    for plat in PLATFORMS:
        cmd += ["--platform", plat]
    cmd += names
    print("$ " + shlex.join(cmd))
    if not dry:
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def build_wheel(dest: Path) -> Path:
    for old in dest.glob("*.whl"):
        old.unlink()
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(dest)], check=True, cwd=REPO_ROOT)
    wheels = list(dest.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one wheel in {dest}, found {wheels}")
    return wheels[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", required=True, help="directory with the trained adapter")
    ap.add_argument("--tag", required=True, help="adapter tag recorded in the manifest")
    ap.add_argument("--calibrator", default=None)
    ap.add_argument("--include-wheels", default="", help="comma-separated third-party packages")
    ap.add_argument("--out", default=None, help=f"default {BUILD_ROOT / DATASET_NAME}")
    ap.add_argument("--sha", default=None, help="commit to pin (must be HEAD)")
    ap.add_argument("--dev", action="store_true", help="dry runs: no preflight, no wheel build")
    ap.add_argument("--dry", action="store_true", help="print the wheel download, build nothing")
    a = ap.parse_args(argv)

    adapter = Path(a.adapter)
    meta = check_adapter(adapter)
    if a.calibrator:
        check_calibrator(Path(a.calibrator))
    ds = Path(a.out) if a.out else REPO_ROOT / BUILD_ROOT / DATASET_NAME
    if a.dry:
        names = [n for n in a.include_wheels.split(",") if n]
        if names:
            download_wheels(names, ds / "wheels", dry=True)
        print(f"(dry) would build {ds} from adapter {adapter} tag {a.tag}")
        return 0

    sha = "dev"
    if not a.dev:
        sha = _load_rung5_push().preflight(a.sha)
        tracked = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            check=True, capture_output=True, text=True, cwd=REPO_ROOT,
        ).stdout.strip()  # fmt: skip
        if tracked:
            raise SystemExit(
                f"refusing: tracked files changed (the wheel is built from them):\n{tracked}"
            )
    if ds.exists():
        shutil.rmtree(ds)
    ds.mkdir(parents=True)
    if a.dev:
        print("dev build: no wheel (use --skip-install in infer.py)")
    else:
        print(f"built {build_wheel(ds).name}")
    shutil.copytree(adapter, ds / "adapter", ignore=shutil.ignore_patterns("README.md", "*.tmp"))
    if a.calibrator:
        shutil.copy(a.calibrator, ds / "calibrator.json")
    names = [n for n in a.include_wheels.split(",") if n]
    if names:
        download_wheels(names, ds / "wheels", dry=False)
    (ds / "dataset-metadata.json").write_text(json.dumps(metadata(), indent=2) + "\n")
    manifest = write_manifest(
        ds, sha, a.tag, {"adapter_meta": meta, "calibrated": bool(a.calibrator)}
    )
    total_mib = sum(f["bytes"] for f in manifest["files"].values()) / 2**20
    print(f"built {ds}: {len(manifest['files'])} files, {total_mib:.0f} MiB, git {sha[:12]}")
    print("\nNot uploaded. When approved, from the repo root:")
    print(f"  kaggle datasets create -p {shlex.quote(str(ds))} -r zip")
    print(
        f"  kaggle datasets version -p {shlex.quote(str(ds))} -r zip -m 'adapter {a.tag}, git {sha[:12]}'"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
