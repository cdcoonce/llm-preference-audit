"""Download the Kaggle "LLM Classification Finetuning" data into data/kaggle/.

Needs the kaggle CLI, credentials, and accepted competition rules. The data is
CC BY-NC 4.0 and must never be committed (data/ is gitignored).
"""

import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "llm-classification-finetuning"
DEST = Path("data/kaggle")


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(DEST)]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"kaggle download failed (exit {result.returncode})", file=sys.stderr)
        return result.returncode
    for archive in DEST.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(DEST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
