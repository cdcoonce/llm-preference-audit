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
        slug="s",
        mode="folds",
        folds="0,1",
        middle_share=0.2,
        t=1024,
        epochs=2,
        micro_batch=4,
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
    assert "'--git-sha', 'abc123'" in code and "'--tag', 't1'" in code
    assert code.index("generated kernel header") < code.index("import torch")
    meta = json.loads((build / "kernel-metadata.json").read_text())
    assert meta["id"] == "u/s" and meta["is_private"] and meta["enable_gpu"]
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
