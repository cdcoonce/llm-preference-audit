"""Static checks on lora/modal_app.py (no Modal contact): image, retries, per-launch timeout."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "lora" / "modal_app.py"
SRC = APP.read_text()


def test_image_installs_libgomp_and_not_lightgbm():
    assert '"libgomp1"' in SRC
    assert '"lightgbm"' not in SRC


def test_gpu_functions_set_retries_to_zero_explicitly():
    assert '"retries": 0' in SRC
    tree = ast.parse(SRC)
    gpu_kw = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "GPU_KW"
    )
    keys = {k.value: v for k, v in zip(gpu_kw.value.keys, gpu_kw.value.values, strict=True)}
    assert keys["retries"].value == 0


def test_every_launch_overrides_the_timeout_and_passes_the_deadline():
    assert SRC.count(".with_options(timeout=timeout)") == 3
    assert '"--local-files-only"' in SRC and '"--deadline-s"' in SRC
    assert "HF_HUB_OFFLINE" in SRC


def test_the_prefetch_function_is_cpu_only():
    tree = ast.parse(SRC)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "prefetch")
    deco = ast.unparse(fn.decorator_list[0])
    assert "gpu" not in deco and "cpu=2" in deco and "hf_secret" in deco


def test_gpu_functions_do_not_get_the_hf_token():
    assert SRC.count("secrets=[hf_secret]") == 1  # the prefetch only
    tree = ast.parse(SRC)
    gpu_kw = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "GPU_KW"
    )
    assert "secret" not in ast.unparse(gpu_kw)


def test_fetch_needs_no_lora_sha():
    pytest.importorskip("modal")
    code = (
        "import importlib.util as u\n"
        f"s = u.spec_from_file_location('ma', {str(APP)!r}); m = u.module_from_spec(s)\n"
        "s.loader.exec_module(m)\n"
        "assert not hasattr(m, 'probe') and hasattr(m, 'prefetch')\n"
    )
    env = {k: v for k, v in __import__("os").environ.items() if k != "LORA_SHA"}
    r = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr
