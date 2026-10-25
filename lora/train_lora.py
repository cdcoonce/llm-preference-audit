"""CLI wrapper: the rung-6 training code lives in llm_preference_audit/lora_train.py so that a
pinned `pip install` of the repo ships it to the Modal container.

  python lora/train_lora.py --mode probe|fold|full|selftest --tiny --max-rows 300 --t 256 \
      --data-dir data/raw --out-dir <scratch>
"""

import sys

from llm_preference_audit.lora_train import main

if __name__ == "__main__":
    sys.exit(main())
