import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rung5_push", ROOT / "kaggle/rung5_deberta/push.py")
push = importlib.util.module_from_spec(spec)
spec.loader.exec_module(push)


def _args(tmp_path, **kw):
    argv = ["--username", "u", "--mode", "folds", "--folds", "0,1", "--tag", "t1", "--dry"]
    return push.argparse.Namespace(
        username="u",
        slug=None,
        mode="folds",
        folds="0,1",
        middle_share=0.2,
        t=1024,
        epochs=2,
        micro_batch=4,
        lr=2e-5,
        precision="fp16",
        grad_ckpt=False,
        tag="t1",
        repo="o/r",
        **kw,
    ), argv


def test_render_script_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(push, "REPO_ROOT", tmp_path)
    a, _ = _args(tmp_path)
    build = push.render(a, "abc123")
    code = (build / "rung5_folds.py").read_text()
    tree = ast.parse(code)  # generated script is valid Python
    assert isinstance(tree.body[1], ast.ImportFrom) and tree.body[1].module == "__future__"
    assert "git+https://github.com/o/r@abc123" in code and "--no-deps" in code
    assert "--ignore-requires-python" in code  # Kaggle may run an older Python
    assert "'--git-sha', 'abc123'" in code and "'--tag', 't1'" in code
    assert code.index("generated kernel header") < code.index("import torch")
    meta = json.loads((build / "kernel-metadata.json").read_text())
    assert meta["id"] == "u/llm-pref-rung5-deberta-t1" and meta["is_private"] and meta["enable_gpu"]
    assert meta["competition_sources"] == ["llm-classification-finetuning"]
    assert meta["code_file"] == "rung5_folds.py" and meta["kernel_type"] == "script"


def test_preflight_refuses_non_head_and_dirty(monkeypatch):
    calls = {"status": ""}

    def fake_git(*args):
        if args[0] == "rev-parse":
            return "headsha"
        if args[0] == "status":
            return calls["status"]
        if args[:2] == ("branch", "-r"):
            return ""
        return ""

    monkeypatch.setattr(push, "git", fake_git)
    with pytest.raises(SystemExit, match="not HEAD"):
        push.preflight("other")
    calls["status"] = " M kaggle/x.py"
    with pytest.raises(SystemExit, match="uncommitted"):
        push.preflight(None)
    calls["status"] = ""
    with pytest.raises(SystemExit, match="not on any remote"):
        push.preflight(None)


def test_slug_per_tag_and_probe_keeps_base():
    a, _ = _args(None)
    assert push.kernel_slug(a) == "llm-pref-rung5-deberta-t1"
    a.tag = "f01"
    assert push.kernel_slug(a) == "llm-pref-rung5-deberta-f01"
    a.tag = "Bad_Tag"
    with pytest.raises(SystemExit, match="lowercase"):
        push.kernel_slug(a)
    a.mode = "probe"
    assert push.kernel_slug(a) == "llm-pref-rung5-deberta"


def test_fetch_uses_tag_slug_and_dir(capsys):
    assert (
        push.main(["--username", "u", "--mode", "folds", "--tag", "f01", "--fetch", "--dry"]) == 0
    )
    out = capsys.readouterr().out
    assert "u/llm-pref-rung5-deberta-f01" in out and "data/kaggle_out/f01" in out


def test_ablation_args_render(tmp_path, monkeypatch):
    monkeypatch.setattr(push, "REPO_ROOT", tmp_path)
    a, _ = _args(tmp_path)
    a.folds, a.middle_share, a.tag = "0", 0.0, "abl0"
    code = (push.render(a, "abc123") / "rung5_folds.py").read_text()
    assert "'--middle-share', '0.0'" in code and "'--folds', '0'" in code


def test_debug_lr_and_precision_reach_the_kernel(tmp_path, monkeypatch):
    monkeypatch.setattr(push, "REPO_ROOT", tmp_path)
    a, _ = _args(tmp_path)
    a.folds, a.lr, a.precision, a.tag = "0", 3e-5, "fp32", "dbg-lr3e5-fp32"
    code = (push.render(a, "abc123") / "rung5_folds.py").read_text()
    assert "'--lr', '3e-05'" in code and "'--precision', 'fp32'" in code
