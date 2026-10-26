"""Rung 6b kernel entry: the free Qwen3-1.7B LoRA run on Kaggle T4s.

All logic lives in `llm_preference_audit.lora_kaggle` (and `lora_train`, `lora_g0`), pip-installed
by the header that `push.py` prepends at a pinned commit. This file only forwards the arguments
the header put in sys.argv, so the kernel is exactly the tested package code.
"""

from __future__ import annotations

import sys

from llm_preference_audit.lora_kaggle import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
