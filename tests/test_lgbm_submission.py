"""Rung-2 LightGBM Kaggle kernel: the inlining mechanism, the kernel's guarantees, the metadata.

Torch-free. The end-to-end test executes a RENDERED script in a subprocess against a tiny
synthetic competition tree (sha checks disabled by a build-time flag that production never sets).
"""

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llm_preference_audit.data import load_train
from llm_preference_audit.features import build_features, feature_columns, swap_pair
from llm_preference_audit.split import build_manifest
from llm_preference_audit.submission import SUBMISSION_COLUMNS, load_test

ROOT = Path(__file__).resolve().parents[1]
KDIR = ROOT / "kaggle/lgbm_submission"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bk = _load("lgbm_build_kernel", "kaggle/lgbm_submission/build_kernel.py")
km = _load("lgbm_kernel_main", "kaggle/lgbm_submission/kernel_main.py")
dry = _load("lgbm_dry_run_input", "kaggle/lgbm_submission/dry_run_input.py")

N_TRAIN, N_TEST = 600, 40


def _text(rng, n_words):
    return " ".join(rng.choice(["alpha", "beta", "gamma", "delta", "eps"], n_words))


def _synthetic(tmp_path):
    """root/input/comp/{train,test}.csv. Test ids are shuffled (file order != id order)."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(N_TRAIN):
        la, lb = int(rng.integers(3, 40)), int(rng.integers(3, 40))
        pick = rng.random()
        win = "tie" if pick < 0.25 else ("a" if (la > lb) == (pick < 0.7) else "b")
        rows.append(
            {
                "id": 1000 + i,
                "model_a": "m1",
                "model_b": "m2",
                "prompt": json.dumps([f"question {i} " + _text(rng, 5)]),
                "response_a": json.dumps([_text(rng, la)]),
                "response_b": json.dumps([_text(rng, lb)]),
                "winner_model_a": int(win == "a"),
                "winner_model_b": int(win == "b"),
                "winner_tie": int(win == "tie"),
            }
        )
    comp = tmp_path / "input" / "llm-classification-finetuning"
    comp.mkdir(parents=True)
    train = pd.DataFrame(rows)
    train.to_csv(comp / "train.csv", index=False)
    test = train.sample(N_TEST, random_state=1)[["id", "prompt", "response_a", "response_b"]]
    test["id"] = test["id"] + 50_000  # unseen ids, in a shuffled order
    test.to_csv(comp / "test.csv", index=False)
    return comp, test


def _run(tmp_path, script_text):
    script = tmp_path / "lgbm_submission.py"
    script.write_text(script_text)
    env = {"LPA_INPUT_ROOT": str(tmp_path / "input"), "LPA_OUTPUT_DIR": str(tmp_path / "work")}
    return subprocess.run(
        [sys.executable, "-I", str(script)],
        check=False,
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------- end to end on a rendered script


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    _synthetic(tmp)
    proc = _run(tmp, bk.render_script(None, rounds=5, skip_sha_checks=True))
    return tmp, proc


def test_rendered_script_produces_a_valid_submission(e2e):
    tmp, proc = e2e
    assert proc.returncode == 0, proc.stderr[-2000:]
    sub = pd.read_csv(tmp / "work" / "submission.csv")
    assert list(sub.columns) == SUBMISSION_COLUMNS
    assert len(sub) == N_TEST
    P = sub[SUBMISSION_COLUMNS[1:]].to_numpy()
    assert np.isfinite(P).all() and (P > 0).all()
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-9)


def test_submission_keeps_the_test_file_row_order(e2e):
    tmp, _ = e2e
    test = pd.read_csv(tmp / "input/llm-classification-finetuning/test.csv")
    sub = pd.read_csv(tmp / "work" / "submission.csv")
    assert sub["id"].tolist() == test["id"].tolist()
    assert sub["id"].tolist() != sorted(sub["id"].tolist())  # the fixture really is shuffled


def test_run_log_records_cv_only_training_and_inlined_code(e2e):
    tmp, _ = e2e
    log = json.loads((tmp / "work" / "run_log.json").read_text())
    manifest = build_manifest(load_train(tmp / "input/llm-classification-finetuning/train.csv"))
    n_cv = int((manifest["split"] == "cv").sum())
    assert 0 < n_cv < N_TRAIN  # the synthetic data has a locked test split too
    assert log["n_cv_rows"] == n_cv
    assert log["n_train_rows"] == 2 * n_cv
    assert log["n_test_rows"] == N_TEST and log["rounds"] == 5
    assert log["inlined"] is True
    assert log["features_module_file"].startswith("<inlined llm_preference_audit/features.py")
    assert set(log["seconds"]) >= {"fit", "features_train", "total"}
    text = json.dumps(log)
    assert "question" not in text and "alpha" not in text  # no dataset text in the log


def test_outputs_carry_no_dataset_text(e2e):
    tmp, _ = e2e
    for name in ("submission.csv", "run_log.json"):
        assert "alpha" not in (tmp / "work" / name).read_text()


def test_default_render_enforces_the_sha_checks(tmp_path):
    _synthetic(tmp_path)
    script = bk.render_script(None, rounds=5)
    assert "'skip_sha_checks': False" in script
    proc = _run(tmp_path, script)
    assert proc.returncode != 0  # synthetic train.csv has the wrong sha: refuses, loudly
    assert "train.csv" in proc.stderr
    assert not (tmp_path / "work" / "run_log.json").exists()


# ---------------------------------------------------------------- unit pieces of the kernel


def _frames(tmp_path):
    comp, _ = _synthetic(tmp_path)
    df = load_train(comp / "train.csv")
    return df, build_manifest(df)


def test_select_cv_never_returns_locked_test_rows(tmp_path):
    df, manifest = _frames(tmp_path)
    cv = km.select_cv(df, manifest)
    test_ids = set(manifest.loc[manifest["split"] == "test", "id"])
    assert len(test_ids) > 0 and len(cv) > 0
    assert not test_ids & set(cv["id"])
    assert len(cv) + len(test_ids) == len(df)


def test_predict_test_is_swap_symmetric(tmp_path):
    df, manifest = _frames(tmp_path)
    cv = km.select_cv(df, manifest)
    cols = feature_columns("full")
    X = build_features(cv)[cols]
    booster = km.fit_final(X, km.labels_of(cv), rounds=5)
    Xt = X.iloc[:25]
    P, Q = km.predict_test(booster, Xt), km.predict_test(booster, swap_pair(Xt))
    np.testing.assert_allclose(Q, P[:, [1, 0, 2]], atol=1e-12)
    raw = booster.predict(Xt)  # an unaveraged prediction is NOT symmetric on this data
    assert np.abs(raw - booster.predict(swap_pair(Xt))[:, [1, 0, 2]]).max() > 1e-6


def test_build_features_needs_no_label_columns(tmp_path):
    comp, _ = _synthetic(tmp_path)
    labelled = load_train(comp / "train.csv").iloc[:30]
    unlabeled = load_test(comp / "test.csv")
    assert set(unlabeled.columns) == {"id", "prompt", "response_a", "response_b"}
    X = build_features(unlabeled)
    assert set(feature_columns("full")) <= set(X.columns)
    assert X.index.tolist() == unlabeled["id"].tolist()
    # the same rows give the same features whether or not the label columns are around
    only_text = labelled[["id", "prompt", "response_a", "response_b"]]
    pd.testing.assert_frame_equal(build_features(labelled), build_features(only_text))


# ---------------------------------------------------------------- build step


def test_rounds_from_synthetic_ledger(tmp_path):
    rows = [
        {"model": "lightgbm", "config": "full", "best_iterations": {"0": 10, "1": 20}},
        {"model": "lightgbm", "config": "full_no_swap", "best_iterations": {"0": 999}},
        {"model": "claude", "config": "full", "best_iterations": {"0": 999}},
        {"model": "lightgbm", "config": "full", "best_iterations": {"0": 100, "1": 120, "2": 140}},
    ]
    p = tmp_path / "ledger.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert bk.rounds_from_ledger(p) == 132  # latest `full` row: mean 120 x 1.1
    p.write_text(json.dumps({"model": "lightgbm", "config": "length_only"}) + "\n")
    with pytest.raises(SystemExit):
        bk.rounds_from_ledger(p)


def test_real_ledger_rounds_follow_the_formula():
    rows = [json.loads(x) for x in (ROOT / "results/ledger.jsonl").read_text().splitlines() if x]
    best = [r for r in rows if r.get("model") == "lightgbm" and r.get("config") == "full"][-1][
        "best_iterations"
    ]
    assert bk.rounds_from_ledger(ROOT / "results/ledger.jsonl") == round(
        sum(best.values()) / len(best) * 1.1
    )


def test_metadata_is_private_cpu_and_offline():
    m = bk.metadata()
    assert m["id"] == "charlescoonce/llm-pref-lgbm-sub"
    assert m["kernel_type"] == "script" and m["code_file"] == "lgbm_submission.py"
    assert m["is_private"] is True
    assert m["enable_gpu"] is False
    assert m["enable_internet"] is False
    assert m["competition_sources"] == ["llm-classification-finetuning"]
    assert m["dataset_sources"] == [] and m["model_sources"] == [] and m["kernel_sources"] == []


def test_render_writes_the_metadata_beside_the_script(tmp_path):
    out = bk.render(tmp_path / "b", None, rounds=7)
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False and (out / meta["code_file"]).is_file()


def _imports(module):
    tree = ast.parse((ROOT / "llm_preference_audit" / f"{module}.py").read_text())
    internal, external = set(), set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [node.module]
        for n in names:
            root = n.split(".")[0]
            if root == "llm_preference_audit":
                internal.add(n.split(".")[1])
            else:
                external.add(root)
    return internal, external


def test_inlined_modules_cover_their_imports_in_dependency_order():
    seen = []
    for mod in bk.MODULES:
        internal, external = _imports(mod)
        assert internal <= set(seen), f"{mod} imports {internal - set(seen)} before it is inlined"
        assert external <= set(sys.stdlib_module_names) | bk.THIRD_PARTY | {"__future__"}, (
            f"{mod} needs {external - bk.THIRD_PARTY - set(sys.stdlib_module_names)}"
        )
        seen.append(mod)
    main_imports = ast.parse((KDIR / "kernel_main.py").read_text())
    for node in ast.walk(main_imports):
        if isinstance(node, ast.ImportFrom) and node.module.startswith("llm_preference_audit."):
            assert node.module.split(".")[1] in bk.MODULES


def test_rendered_script_embeds_the_committed_sources_verbatim():
    script = bk.render_script(None, rounds=121)
    ns = {}
    for line in script.splitlines():
        for name in ("_BUILD", "_MODULE_SOURCES"):
            if line.startswith(name + " = "):
                ns[name] = ast.literal_eval(line.split(" = ", 1)[1])
    for mod in bk.MODULES:
        src = (ROOT / "llm_preference_audit" / f"{mod}.py").read_text()
        assert ns["_MODULE_SOURCES"][mod] == src
        assert ns["_BUILD"]["module_sha256"][mod] == bk.sha256_text(src)
        assert f"{mod}.py  {bk.sha256_text(src)}" in script.split("_BUILD = ")[0]
    assert ns["_BUILD"]["rounds"] == 121


def test_commands_are_printed_not_run_and_use_the_slug():
    cmds = "\n".join(bk.commands(Path("data/kaggle_build/lgbm-sub")))
    assert "kaggle kernels push -p data/kaggle_build/lgbm-sub" in cmds
    assert "kaggle competitions submit -k charlescoonce/llm-pref-lgbm-sub -v" in cmds
    assert "-f submission.csv" in cmds


def test_dry_run_input_has_no_labels_and_shuffles(tmp_path):
    _, manifest = _frames(tmp_path)
    ids = dry.pick_ids(manifest, "test", 5, 0)
    assert set(ids) <= set(manifest.loc[manifest["split"] == "test", "id"])
    ids = dry.pick_ids(manifest, "cv-fold0", 5, 0)
    sel = manifest[manifest["id"].isin(ids)]
    assert set(sel["split"]) == {"cv"} and set(sel["fold"]) == {0}
