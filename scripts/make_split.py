"""Regenerate splits/split_manifest.csv from data/raw/train.csv."""

from llm_preference_audit.data import load_train
from llm_preference_audit.split import MANIFEST_PATH, build_manifest, write_manifest


def main() -> None:
    manifest = build_manifest(load_train("data/raw/train.csv"))
    write_manifest(manifest)
    size = MANIFEST_PATH.stat().st_size
    print(f"wrote {MANIFEST_PATH}: {len(manifest)} rows, {size} bytes")


if __name__ == "__main__":
    main()
