"""The rung-6b kernel tooling (torch-free): metadata, rendered script, preflight, fetch."""

import ast
import importlib.util
import json
from pathlib import Path

import pytest

from llm_preference_audit.lora_g0 import QWEN3_MODEL_HANDLE

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rung6b_push", ROOT / "kaggle/rung6b_qwen3/push.py")
push = importlib.util.module_from_spec(spec)
spec.loader.exec_module(push)


def _args(**over):
    argv = ["--username", "u", "--mode", "probe", "--tag", "p1", "--repo", "o/r"]
    a = push.parse(argv)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_probe_metadata_is_private_gpu_internet_with_the_model_attached(tmp_path, monkeypatch):
    monkeypatch.setattr(push, "REPO_ROOT", tmp_path)
    (tmp_path / "kaggle").mkdir()
    build = push.render(_args(), "abc123")
    meta = json.loads((build / "kernel-metadata.json").read_text())
    assert meta["id"] == "u/llm-pref-rung6b-qwen3-p1" and meta["title"] == meta["id"].split("/")[1]
    assert meta["is_private"] is True and meta["enable_gpu"] is True
    assert meta["enable_internet"] is True  # training kernels pip-install
    assert meta["competition_sources"] == ["llm-classification-finetuning"]
    assert meta["model_sources"] == ["qwen-lm/qwen-3/transformers/1.7b/1"] == [QWEN3_MODEL_HANDLE]
    assert push.HANDLE_RE.fullmatch(meta["model_sources"][0])
    assert meta["code_file"] == "rung6b_probe.py" and meta["kernel_type"] == "script"


def test_rendered_script_pins_the_commit_installs_peft_and_forwards_the_probe_args(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(push, "REPO_ROOT", tmp_path)
    build = push.render(_args(), "abc123")
    code = (build / "rung6b_probe.py").read_text()
    tree = ast.parse(code)
    assert isinstance(tree.body[1], ast.ImportFrom) and tree.body[1].module == "__future__"
    assert "git+https://github.com/o/r@abc123" in code and "--no-deps" in code
    assert "--ignore-requires-python" in code and "'peft'" in code and "'accelerate'" in code
    assert code.index("generated kernel header") < code.index("lora_kaggle import main")
    for needle in ("'--mode', 'qprobe'", "'--kaggle'", "'--precision', 'fp16-autocast'",
                   "'--fp32-head-norm'", "'--git-sha', 'abc123'", "'--probe-steps', '200'",
                   "'--t', '2048'", "'--deadline-s', '3300'"):  # fmt: skip
        assert needle in code


def test_probe_timeout_and_deadline_leave_a_clean_stop_margin():
    a = _args()
    assert a.timeout == 4200
    args = push.train_args(a, "sha")
    assert int(args[args.index("--deadline-s") + 1]) == 3300 < a.timeout


@pytest.mark.parametrize(
    ("field", "value", "msg"),
    [
        ("enable_internet", False, "enable_internet"),
        ("is_private", False, "is_private"),
        ("enable_gpu", False, "enable_gpu"),
        ("model_sources", [], "model_sources"),
        ("model_sources", ["qwen-lm/qwen-3/transformers/1.7b"], "model_sources"),  # no version
        ("model_sources", ["a/b/c/d/1", "a/b/c/e/1"], "model_sources"),
    ],
)
def test_metadata_check_refuses_a_bad_training_kernel(field, value, msg):
    meta = push.kernel_metadata(_args(), "x.py") | {field: value}
    with pytest.raises(SystemExit, match=msg):
        push.check_metadata(meta)


def test_fold_mode_args_carry_the_fold_and_the_same_precision_flags():
    a = _args(mode="fold", fold=0, timeout=push.LONG_TIMEOUT)
    args = push.train_args(a, "sha")
    assert args[:4] == ["--mode", "fold", "--fold", "0"]
    assert "--fp32-head-norm" in args and "fp16-autocast" in args


def test_preflight_refuses_non_head_dirty_and_unpushed(monkeypatch):
    state = {"status": ""}

    def fake_git(*args):
        if args[0] == "rev-parse":
            return "headsha"
        if args[0] == "status":
            return state["status"]
        return ""

    monkeypatch.setattr(push, "git", fake_git)
    with pytest.raises(SystemExit, match="not HEAD"):
        push.preflight("other")
    state["status"] = " M kaggle/x.py"
    with pytest.raises(SystemExit, match="uncommitted"):
        push.preflight(None)
    state["status"] = ""
    with pytest.raises(SystemExit, match="not on any remote"):
        push.preflight(None)


def test_slug_per_tag_and_fetch_uses_the_tag_directory(monkeypatch, capsys):
    assert push.kernel_slug(_args(tag="ab-1")) == "llm-pref-rung6b-qwen3-ab-1"
    with pytest.raises(SystemExit, match="lowercase"):
        push.kernel_slug(_args(tag="Bad_Tag"))
    assert push.main(["--username", "u", "--tag", "p1", "--fetch", "--dry"]) == 0
    out = capsys.readouterr().out
    assert "kaggle kernels status u/llm-pref-rung6b-qwen3-p1" in out
    assert "-p data/kaggle_out/p1" in out
