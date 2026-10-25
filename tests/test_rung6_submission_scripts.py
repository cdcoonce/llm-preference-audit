import argparse
import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


infer = _load("rung6_infer", "kaggle/rung6_submission/infer.py")
build = _load("rung6_build", "kaggle/rung6_submission/build_dataset.py")
push = _load("rung6_push", "kaggle/rung6_submission/push.py")


def _dataset(tmp_path, sha="a" * 40, wheel=True, wheels_dir=False):
    ds = tmp_path / "ds"
    (ds / "adapter").mkdir(parents=True)
    for f in build.ADAPTER_REQUIRED:
        (ds / "adapter" / f).write_text(f"x-{f}")
    if wheel:
        (ds / "llm_preference_audit-0.1.0-py3-none-any.whl").write_bytes(b"whl")
    if wheels_dir:
        (ds / "wheels").mkdir()
        (ds / "wheels" / "peft-0.17.0-py3-none-any.whl").write_bytes(b"peft")
    (ds / "dataset-metadata.json").write_text(json.dumps(build.metadata()))
    build.write_manifest(ds, sha, "full")
    return ds


# ---------------------------------------------------------------- manifest and bootstrap


def test_manifest_roundtrip_and_tamper_detection(tmp_path):
    ds = _dataset(tmp_path)
    m = infer.verify_manifest(ds)
    assert m["adapter_tag"] == "full" and "adapter/adapter_config.json" in m["files"]
    (ds / "adapter" / "adapter_model.safetensors").write_text("tampered")
    with pytest.raises(SystemExit, match="sha256"):
        infer.verify_manifest(ds)


def test_manifest_rejects_missing_extra_and_dev(tmp_path):
    ds = _dataset(tmp_path)
    (ds / "stray.txt").write_text("x")
    with pytest.raises(SystemExit, match="does not list"):
        infer.verify_manifest(ds)
    (ds / "stray.txt").unlink()
    (ds / "adapter" / "adapter_meta.json").unlink()
    with pytest.raises(SystemExit, match="missing"):
        infer.verify_manifest(ds)
    dev = _dataset(tmp_path / "d", sha="dev")
    with pytest.raises(SystemExit, match="dev build"):
        infer.verify_manifest(dev)
    assert infer.verify_manifest(dev, allow_dev=True)["git_sha"] == "dev"


def test_find_dataset_dir_requires_exactly_one(tmp_path):
    ds = _dataset(tmp_path)
    assert infer.find_dataset_dir([str(tmp_path)], None) == ds
    other = tmp_path / "other"
    other.mkdir()
    (other / "MANIFEST.json").write_text(json.dumps({"files": {}}))  # not ours: no adapter_tag
    assert infer.find_dataset_dir([str(tmp_path)], None) == ds
    _dataset(tmp_path / "second")
    with pytest.raises(SystemExit, match="expected one"):
        infer.find_dataset_dir([str(tmp_path)], None)


def test_bootstrap_installs_wheel_offline_and_fills_gaps_from_wheels(tmp_path, monkeypatch):
    ds = _dataset(tmp_path, wheels_dir=True)
    manifest = infer.verify_manifest(ds)
    calls = []
    monkeypatch.setattr(infer, "pip_install", lambda args: calls.append(args))
    present = {"sentencepiece", "accelerate", "bitsandbytes"}  # peft is missing until its wheel

    def find_spec(name):
        peft_installed = any("peft-0.17.0" in str(c) for c in calls)
        return object() if name in present or (name == "peft" and peft_installed) else None

    monkeypatch.setattr(infer.importlib.util, "find_spec", find_spec)
    info = infer.bootstrap(ds, manifest, need_bnb=True)
    assert calls[0][:2] == ["--no-deps", "--force-reinstall"] and calls[0][2].endswith(".whl")
    assert any("peft-0.17.0" in str(c) for c in calls[1:])
    assert info["installed"][0].startswith("llm_preference_audit") and not info["missing_after"]


def test_bootstrap_fails_loud_when_a_package_has_no_wheel(tmp_path, monkeypatch):
    ds = _dataset(tmp_path)  # no wheels/ folder
    manifest = infer.verify_manifest(ds)
    monkeypatch.setattr(infer, "pip_install", lambda args: None)
    monkeypatch.setattr(infer.importlib.util, "find_spec", lambda n: None)
    with pytest.raises(SystemExit, match="missing packages"):
        infer.bootstrap(ds, manifest, need_bnb=True)
    monkeypatch.setattr(infer.importlib.util, "find_spec", lambda n: object())
    with pytest.raises(SystemExit, match="exactly one repo wheel"):
        infer.bootstrap(_dataset(tmp_path / "nw", wheel=False), {"files": {}}, need_bnb=False)


def test_bnb_is_not_required_without_quantisation(tmp_path, monkeypatch):
    ds = _dataset(tmp_path)
    manifest = infer.verify_manifest(ds)
    monkeypatch.setattr(infer, "pip_install", lambda args: None)
    monkeypatch.setattr(
        infer.importlib.util, "find_spec", lambda n: None if n == "bitsandbytes" else object()
    )
    assert not infer.bootstrap(ds, manifest, need_bnb=False)["missing_after"]


# ---------------------------------------------------------------- build_dataset


def test_build_dataset_dev_build_is_verifiable(tmp_path):
    src = tmp_path / "adapter"
    src.mkdir()
    for f in build.ADAPTER_REQUIRED:
        (src / f).write_text("{}" if f.endswith(".json") else "w")
    (src / "adapter_meta.json").write_text(json.dumps({"t": 2048, "middle_share": 0.2}))
    (src / "README.md").write_text("skipped")
    out = tmp_path / "out"
    assert build.main(["--dev", "--adapter", str(src), "--tag", "t", "--out", str(out)]) == 0
    m = infer.verify_manifest(out, allow_dev=True)
    assert m["adapter_meta"]["t"] == 2048 and "adapter/README.md" not in m["files"]
    meta = json.loads((out / "dataset-metadata.json").read_text())
    assert meta["id"] == "charlescoonce/llm-pref-rung6-adapter" and meta["isPrivate"] is True
    assert meta["licenses"] == [{"name": "other"}]


def test_build_dataset_refuses_incomplete_adapter_and_bad_calibrator(tmp_path):
    (tmp_path / "a").mkdir()
    with pytest.raises(SystemExit, match="missing"):
        build.check_adapter(tmp_path / "a")
    bad = tmp_path / "c.json"
    bad.write_text(json.dumps({"kind": "other"}))
    with pytest.raises(SystemExit, match="matrix_scaling"):
        build.check_calibrator(bad)


def test_wheel_download_command_targets_kaggle_python(capsys):
    build.download_wheels(["peft", "bitsandbytes"], Path("w"), dry=True)
    cmd = capsys.readouterr().out
    assert "--only-binary=:all:" in cmd and "--python-version 3.13" in cmd
    assert "--platform manylinux2014_x86_64" in cmd and "--no-deps" in cmd


# ---------------------------------------------------------------- push


def _push_args(mode="timing", **kw):
    base = {
        "username": "u",
        "mode": mode,
        "timing_rows": 1000,
        "t": 0,
        "batch_tokens": 4096,
        "max_batch": 8,
        "limit_s": 29700,
        "version": 3,
        "message": "m",
    }
    return argparse.Namespace(**(base | kw))


def test_render_both_kernels_internet_off_with_all_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(push, "REPO_ROOT", tmp_path)
    for mode in ("timing", "submission"):
        build_dir = push.render(_push_args(mode))
        meta = json.loads((build_dir / "kernel-metadata.json").read_text())
        assert meta["enable_internet"] is False and meta["enable_gpu"] and meta["is_private"]
        assert meta["id"] == f"u/llm-pref-rung6-{mode}"
        assert meta["competition_sources"] == ["llm-classification-finetuning"]
        assert meta["dataset_sources"] == ["charlescoonce/llm-pref-rung6-adapter"]
        assert meta["model_sources"] == ["google/gemma-2/transformers/gemma-2-9b-it/2"]
        assert meta["machine_shape"] == "NvidiaTeslaT4"
        code = (build_dir / meta["code_file"]).read_text()
        tree = ast.parse(code)
        assert isinstance(tree.body[1], ast.ImportFrom) and tree.body[1].module == "__future__"
        assert code.index("generated kernel header") < code.index("def worker_main")
        assert ("'--timing-rows', '1000'" in code) == (mode == "timing")


def test_submit_command_and_preflight_manifest_guard(tmp_path, monkeypatch):
    cmd = push.submit_command(_push_args("submission"))
    assert cmd[:4] == ["kaggle", "competitions", "submit", "llm-classification-finetuning"]
    assert cmd[cmd.index("-k") + 1] == "u/llm-pref-rung6-submission" and "-v" in cmd
    with pytest.raises(SystemExit):
        push.submit_command(_push_args("submission", version=0))
    monkeypatch.setattr(push, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(push, "_rung5", lambda: argparse.Namespace(preflight=lambda sha: "headsha"))
    with pytest.raises(SystemExit, match="build_dataset"):
        push.preflight(None)
    (tmp_path / push.DATASET_DIR).mkdir(parents=True)
    (tmp_path / push.DATASET_DIR / "MANIFEST.json").write_text(json.dumps({"git_sha": "old"}))
    with pytest.raises(SystemExit, match="rebuild the dataset"):
        push.preflight(None)
    assert push.preflight(None, allow_mismatch=True) == "headsha"
