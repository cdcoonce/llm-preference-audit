"""Rung-6 spend guard and launch helpers: every refusal path, the pass path, G1, reconciliation."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "lora" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sg = _load("spend_guard")
lp = _load("launch_plan")


@pytest.fixture
def paths(tmp_path):
    return {
        "ledger": tmp_path / "ledger.jsonl",
        "spend": tmp_path / "spend.jsonl",
        "g1_file": tmp_path / "g1.json",
    }


def write_ledger(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def g1_file(paths, loss=0.95, ok=True, projected=None):
    d = {"pass": ok, "fold0_log_loss": loss}
    if projected is not None:
        d["projected_remaining_usd"] = projected
    paths["g1_file"].write_text(json.dumps(d))


def check(paths, mode, folds, usd, **kw):
    return sg.check_launch(mode, folds, usd, **paths, **kw)


# ---- (a) probe cap
def test_probe_under_cap_passes(paths):
    assert check(paths, "probe", [], 4.99)["rung6_after_usd"] == pytest.approx(4.99)


def test_probe_over_five_dollars_refused(paths):
    with pytest.raises(sg.SpendRefused, match="G0"):
        check(paths, "probe", [], 5.01)


# ---- (b) approved amount
def test_cumulative_rung6_over_approved_refused(paths):
    sg.record_launch("p1", "probe", "p", [], 4.0, paths["spend"])
    sg.record_launch("f0", "fold", "g1", [0], 9.0, paths["spend"])
    with pytest.raises(sg.SpendRefused, match="approved"):
        check(paths, "fold", [0], 2.5)  # 4 + 9 + 2.5 = 15.5 > 15
    assert check(paths, "fold", [0], 2.0)  # exactly 15 is allowed


def test_approved_amount_is_a_parameter(paths):
    with pytest.raises(sg.SpendRefused, match="approved"):
        check(paths, "fold", [0], 12.0, approved_usd=10)
    assert check(paths, "fold", [0], 12.0, approved_usd=20)


# ---- (c) ceiling
def test_ceiling_counts_ledger_gpu_spend(paths):
    write_ledger(
        paths["ledger"],
        [
            {"config": "deberta", "gpu_hours": 11.0, "cost_usd": 78.0},
            {"config": "haiku_full_cv", "gpu_hours": None, "cost_usd": 9.6},  # judge, not GPU
        ],
    )
    assert check(paths, "fold", [0], 6.9, approved_usd=100)["gpu_total_usd"] == 78.0
    with pytest.raises(sg.SpendRefused, match="ceiling"):
        check(paths, "fold", [0], 7.5, approved_usd=100)  # 78 + 7.5 > 85


def test_ledgered_rung6_run_is_not_double_counted_with_spend_file(paths):
    sg.record_launch("f0", "fold", "g1", [0], 8.0, paths["spend"])
    write_ledger(paths["ledger"], [{"config": "lora_g1", "gpu_hours": 2.0, "cost_usd": 8.0}])
    assert sg.totals(paths["ledger"], paths["spend"])["rung6_usd"] == 8.0
    assert sg.totals(paths["ledger"], paths["spend"])["gpu_total_usd"] == 8.0


# ---- (d) G1
@pytest.mark.parametrize("mode,folds", [("fold", [1]), ("fold", [0, 3]), ("full", [])])
def test_folds_1_to_4_and_full_need_g1(paths, mode, folds):
    with pytest.raises(sg.SpendRefused, match="G1"):
        check(paths, mode, folds, 1.0)


def test_fold_zero_and_probe_do_not_need_g1(paths):
    assert check(paths, "fold", [0], 1.0)
    assert check(paths, "probe", [], 1.0)


def test_g1_pass_unlocks_folds_and_full(paths):
    g1_file(paths, 0.95)
    assert check(paths, "fold", [1, 2, 3, 4], 2.0)
    assert check(paths, "full", [], 2.0)


@pytest.mark.parametrize("loss,ok", [(1.00, True), (1.0001, False)])
def test_g1_threshold(loss, ok):
    n = 40
    y = {i: i % 3 for i in range(n)}
    # a uniform-ish probability row whose log loss we set exactly via the true-class prob
    q = float(np.exp(-loss))
    rest = (1 - q) / 2
    P = np.array([[rest] * 3 for _ in range(n)])
    for i in range(n):
        P[i, y[i]] = q
    oof = pd.DataFrame({"id": list(y), "p_a": P[:, 0], "p_b": P[:, 1], "p_tie": P[:, 2]})
    dec = sg.g1_check(oof, y)
    assert dec["pass"] is ok
    assert dec["fold0_log_loss"] == pytest.approx(loss, abs=1e-9)


def test_g1_file_pass_true_with_bad_loss_is_refused(paths):
    g1_file(paths, loss=1.0001, ok=True)  # hand-edited flag over a failing number
    with pytest.raises(sg.SpendRefused, match="G1"):
        check(paths, "fold", [1], 1.0)
    g1_file(paths, loss=0.9, ok=False)
    with pytest.raises(sg.SpendRefused, match="G1"):
        check(paths, "fold", [1], 1.0)


def test_g1_exactly_one_is_accepted_from_file(paths):
    g1_file(paths, loss=1.00)
    assert check(paths, "fold", [1], 1.0)


def test_g1_check_rejects_bad_oof():
    y = {1: 0, 2: 1}
    ok = pd.DataFrame({"id": [1, 2], "p_a": [0.5, 0.2], "p_b": [0.3, 0.5], "p_tie": [0.2, 0.3]})
    assert sg.g1_check(ok, y)["n_rows"] == 2
    with pytest.raises(ValueError, match="duplicate"):
        sg.g1_check(pd.concat([ok, ok]), y)
    with pytest.raises(ValueError, match="no label"):
        sg.g1_check(ok.assign(id=[1, 9]), y)
    with pytest.raises(ValueError, match="sum to 1"):
        sg.g1_check(ok.assign(p_a=[0.9, 0.9]), y)
    with pytest.raises(ValueError, match="fold 0"):
        sg.g1_check(ok, y, expect_ids=[1, 2, 3])


def test_auto_approval_after_g1_scales_with_the_projection(paths):
    sg.record_launch("p1", "probe", "p", [], 2.0, paths["spend"])
    sg.record_launch("f0", "fold", "g1", [0], 8.0, paths["spend"])  # 10 spent of 15
    g1_file(paths, 0.95, projected=30.0)  # cap = 15 + 1.2 * 30 = 51
    assert check(paths, "fold", [1, 2, 3, 4], 30.0)["approved_cap_usd"] == pytest.approx(51.0)
    with pytest.raises(sg.SpendRefused, match="approved"):
        check(paths, "fold", [1, 2, 3, 4], 41.5)  # 10 + 41.5 > 51
    # the ceiling still binds even when the approved cap is generous
    g1_file(paths, 0.95, projected=200.0)
    with pytest.raises(sg.SpendRefused, match="ceiling"):
        check(paths, "full", [], 80.0)


def test_g1_pass_without_projection_keeps_the_approved_cap(paths):
    sg.record_launch("f0", "fold", "g1", [0], 12.0, paths["spend"])
    g1_file(paths, 0.95)
    with pytest.raises(sg.SpendRefused, match="approved"):
        check(paths, "fold", [1], 5.0)


def test_all_reasons_are_reported(paths):
    with pytest.raises(sg.SpendRefused) as e:
        check(paths, "probe", [], 99.0)
    assert "G0" in str(e.value) and "approved" in str(e.value) and "ceiling" in str(e.value)


def test_bad_inputs_refused(paths):
    with pytest.raises(sg.SpendRefused, match="unknown mode"):
        check(paths, "serve", [], 1.0)
    with pytest.raises(sg.SpendRefused, match="non-negative"):
        check(paths, "fold", [0], float("nan"))


# ---- spend file / reconciliation
def test_reconcile_replaces_the_estimate(paths):
    sg.record_launch("f0", "fold", "g1", [0], 10.0, paths["spend"])
    assert sg.totals(paths["ledger"], paths["spend"])["rung6_usd"] == 10.0
    sg.reconcile("f0", 6.5, paths["spend"])
    assert sg.totals(paths["ledger"], paths["spend"])["rung6_usd"] == 6.5
    sg.record_launch("f1", "fold", "g2", [1], 3.0, paths["spend"])
    assert sg.totals(paths["ledger"], paths["spend"])["rung6_usd"] == 9.5


def test_unreconciled_launch_still_counts_and_reconcile_needs_a_launch(paths):
    sg.record_launch("f0", "fold", "g1", [0], 10.0, paths["spend"])
    with pytest.raises(ValueError, match="no launch"):
        sg.reconcile("nope", 1.0, paths["spend"])
    assert sg.totals(paths["ledger"], paths["spend"])["gpu_total_usd"] == 10.0


def test_cost_from_logs_adds_boot_overhead(tmp_path):
    (tmp_path / "log_fold0.json").write_text(json.dumps({"est_cost_usd": 4.0}))
    (tmp_path / "log_fold1.json").write_text(json.dumps({"est_cost_usd": 6.0}))
    assert sg.cost_from_logs(tmp_path) == pytest.approx(10.0 * 1.15)
    (tmp_path / "log_full.json").write_text("{}")
    with pytest.raises(ValueError, match="est_cost_usd"):
        sg.cost_from_logs(tmp_path)
    with pytest.raises(ValueError, match="no probe"):
        sg.cost_from_logs(tmp_path / "empty")


# ---- launch helpers
def test_estimates_and_attention_choice():
    assert lp.estimate_usd("probe", 1, None) == pytest.approx(0.5 * 3.95)
    assert lp.estimate_usd("fold", 2, None) == pytest.approx(2 * 2.1 * 3.95)
    probe = {
        "projection": {"fold_usd": 5.0, "full_usd": 6.0},
        "attention_decision": "sdpa",
    }
    assert lp.estimate_usd("fold", 4, probe) == pytest.approx(4 * 5.0 * 1.2)
    assert lp.estimate_usd("full", 1, probe) == pytest.approx(6.0 * 1.2)
    assert lp.resolve_attn("auto", probe) == "sdpa"
    assert lp.resolve_attn("auto", None) == "eager"
    assert lp.resolve_attn("auto", {"attention_decision": "weird"}) == "eager"
    assert lp.resolve_attn("eager", probe) == "eager"
    with pytest.raises(SystemExit):
        lp.resolve_attn("flash", probe)


def test_tag_and_fold_validation():
    assert lp.validate_tag("g1-fold0") == "g1-fold0"
    for bad in ("", "Upper", "a/b", "-x", "a b"):
        with pytest.raises(SystemExit):
            lp.validate_tag(bad)
    assert lp.parse_folds("1,2,3,4") == [1, 2, 3, 4]
    for bad in ("", "5", "1,1", "a"):
        with pytest.raises(SystemExit):
            lp.parse_folds(bad)


def test_latest_probe_picks_the_newest(tmp_path):
    assert lp.latest_probe(tmp_path) is None
    for i, tag in enumerate(("old", "new")):
        d = tmp_path / tag
        d.mkdir()
        (d / "probe.json").write_text(json.dumps({"attention_decision": tag}))
        import os

        os.utime(d / "probe.json", (1000 + i, 1000 + i))
    assert lp.latest_probe(tmp_path)["attention_decision"] == "new"


def _git(root, *a):
    subprocess.run(["git", *a], cwd=root, check=True, capture_output=True)


def test_preflight_refuses_dirty_unpushed_and_wrong_sha(tmp_path):
    remote, work = tmp_path / "remote.git", tmp_path / "work"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    _git(tmp_path, "init", "-b", "main", str(work))
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        _git(work, "config", k, v)
    (work / "lora").mkdir()
    (work / "lora" / "x.py").write_text("a = 1\n")
    _git(work, "add", ".")
    _git(work, "commit", "-m", "c1")
    _git(work, "remote", "add", "origin", str(remote))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=work, check=True, capture_output=True, text=True
    ).stdout.strip()
    with pytest.raises(SystemExit, match="not HEAD"):
        lp.preflight("deadbeef", root=work)
    with pytest.raises(SystemExit, match="remote"):
        lp.preflight(head, root=work)  # committed but never pushed
    _git(work, "push", "origin", "main")
    _git(work, "fetch")
    assert lp.preflight(head, root=work) == head
    (work / "lora" / "x.py").write_text("a = 2\n")
    with pytest.raises(SystemExit, match="uncommitted"):
        lp.preflight(head, root=work)
