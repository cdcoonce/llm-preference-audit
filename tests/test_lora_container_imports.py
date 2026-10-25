"""The Modal container ships no libgomp, so nothing the LoRA trainer imports may import lightgbm."""

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BLOCK = textwrap.dedent(
    """
    import sys
    sys.modules["lightgbm"] = None  # any `import lightgbm` now raises ImportError
    import importlib
    for name in sys.argv[1:]:
        importlib.import_module(name)
    print("ok")
    """
)


def _run(*modules):
    return subprocess.run(
        [sys.executable, "-c", BLOCK, *modules],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


def test_lora_torch_free_modules_do_not_import_lightgbm():
    r = _run(
        "llm_preference_audit.cv_common",
        "llm_preference_audit.deberta_inputs",
        "llm_preference_audit.lora_inputs",
        "llm_preference_audit.lora_cost",
        "llm_preference_audit.kernel_common",
    )
    assert r.returncode == 0, r.stderr


def test_lora_train_does_not_import_lightgbm():
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    r = _run("llm_preference_audit.lora_train")
    assert r.returncode == 0, r.stderr


def test_train_lgbm_still_reexports_the_guards():
    from llm_preference_audit import train_lgbm

    assert train_lgbm.assert_cv_only and train_lgbm.TestSplitLeak and train_lgbm.inner_holdout_mask
