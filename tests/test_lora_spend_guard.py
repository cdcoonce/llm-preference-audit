"""Rung-6 spend guard and launch helpers: every refusal path, the pass path, G1, reconciliation."""

import importlib.util
import json
import os
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


SHA = "a" * 40
OTHER_SHA = "b" * 40
Y = {i: i % 3 for i in range(40)}


@pytest.fixture
def paths(tmp_path):
    return {
        "ledger": tmp_path / "ledger.jsonl",
        "spend": tmp_path / "spend.jsonl",
        "g1_file": tmp_path / "g1.json",
        "out_root": tmp_path / "lora_out",
    }


def write_ledger(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def good_probe(sha=SHA, **over):
    d = {
        "mode": "probe",
        "tiny": False,
        "model": sg.REAL_MODEL,
        "t": 2048,
        "git_sha": sha,
        "loss_finite": True,
        "non_finite_loss": None,
        "attention_decision": "eager",
        "complete": True,
        "finished_at": 1000.0,
        "projection": {
            "fold_seconds": 7560.0,
            "full_seconds": 8280.0,
            "fold_usd": 8.3,
            "full_usd": 9.1,
            "folds_1_to_4_plus_full_usd": 30.0,
            "total_usd_with_overhead": 55.0,
        },
    }
    return d | over


def write_probe(paths, tag="probe", **over):
    d = paths["out_root"] / tag
    d.mkdir(parents=True, exist_ok=True)
    probe = good_probe(**over)
    probe = {k: v for k, v in probe.items() if v != "__drop__"}
    (d / "probe.json").write_text(json.dumps(probe))
    return d / "probe.json"


def make_oof(loss):
    q = float(np.exp(-loss))
    rest = (1 - q) / 2
    P = np.array([[rest] * 3 for _ in Y])
    for i, y in Y.items():
        P[i, y] = q
    return pd.DataFrame({"id": list(Y), "p_a": P[:, 0], "p_b": P[:, 1], "p_tie": P[:, 2]})


def g1_file(paths, loss=0.95, sha=SHA, tag="g1"):
    """A genuine OOF file plus the decision the g1 CLI would write for it."""
    from llm_preference_audit.kernel_common import TRAIN_SHA256

    oof = paths["out_root"] / tag / "oof_fold0.parquet"
    oof.parent.mkdir(parents=True, exist_ok=True)
    make_oof(loss).to_parquet(oof, index=False)
    dec = sg.build_g1_decision(
        oof, Y, git_sha=sha, tag=tag, train_sha256=TRAIN_SHA256, projected_remaining_usd=None
    )
    paths["g1_file"].write_text(json.dumps(dec))
    return oof, dec


def check(paths, mode, folds, usd, **kw):
    kw = {"git_sha": SHA, "labels": lambda: Y} | kw
    if mode != "probe" and not (paths["out_root"] / "probe" / "probe.json").exists():
        write_probe(paths)
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
    """A hand-edited decision over a genuinely failing OOF: the loss is recomputed, not trusted."""
    _, dec = g1_file(paths, loss=1.0001)
    dec["pass"] = True
    paths["g1_file"].write_text(json.dumps(dec))
    with pytest.raises(sg.SpendRefused, match="G1"):
        check(paths, "fold", [1], 1.0)
    _, dec = g1_file(paths, loss=0.9)
    dec["pass"] = False
    paths["g1_file"].write_text(json.dumps(dec))
    with pytest.raises(sg.SpendRefused, match="G1"):
        check(paths, "fold", [1], 1.0)


def test_g1_tampered_loss_is_refused(paths):
    _, dec = g1_file(paths, loss=1.2)  # genuinely fails
    dec["pass"], dec["fold0_log_loss"] = True, 0.5
    paths["g1_file"].write_text(json.dumps(dec))
    with pytest.raises(sg.SpendRefused, match="recomputed"):
        check(paths, "fold", [1], 1.0)
    # a passing OOF with a lying (lower) loss number is also refused
    _, dec = g1_file(paths, loss=0.95)
    dec["fold0_log_loss"] = 0.5
    paths["g1_file"].write_text(json.dumps(dec))
    with pytest.raises(sg.SpendRefused, match="recomputed"):
        check(paths, "fold", [1], 1.0)


def test_g1_from_another_git_sha_is_refused(paths):
    g1_file(paths, loss=0.95, sha=OTHER_SHA)  # stale OOF: produced by other code
    with pytest.raises(sg.SpendRefused, match="git sha"):
        check(paths, "fold", [1], 1.0)
    g1_file(paths, loss=0.95, sha=SHA)
    assert check(paths, "fold", [1], 1.0)


def test_g1_oof_hash_mismatch_is_refused(paths):
    oof, _ = g1_file(paths, loss=0.95)
    make_oof(0.90).to_parquet(oof, index=False)  # a different (even better) OOF swapped in
    with pytest.raises(sg.SpendRefused, match="sha256"):
        check(paths, "fold", [1], 1.0)


def test_g1_missing_oof_or_foreign_train_hash_is_refused(paths):
    oof, dec = g1_file(paths, loss=0.95)
    dec["train_sha256"] = "0" * 64
    paths["g1_file"].write_text(json.dumps(dec))
    with pytest.raises(sg.SpendRefused, match="train.csv"):
        check(paths, "fold", [1], 1.0)
    oof, dec = g1_file(paths, loss=0.95)
    oof.unlink()
    with pytest.raises(sg.SpendRefused, match="OOF"):
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


def test_auto_approval_after_g1_scales_with_the_probe_projection(paths):
    sg.record_launch("p1", "probe", "p", [], 2.0, paths["spend"])
    sg.record_launch("f0", "fold", "g1", [0], 8.0, paths["spend"])  # 10 spent of 15
    g1_file(paths, 0.95)  # the probe projects 30 for folds 1-4 + full: cap = 15 + 1.2 * 30 = 51
    acc = check(paths, "fold", [1, 2, 3, 4], 30.0)
    assert acc["approved_cap_usd"] == pytest.approx(51.0)
    assert acc["remaining_usd"] == pytest.approx(41.0)
    with pytest.raises(sg.SpendRefused, match="approved"):
        check(paths, "fold", [1, 2, 3, 4], 41.5)  # 10 + 41.5 > 51


def test_ceiling_still_binds_when_the_projection_is_huge(paths):
    sg.record_launch("f0", "fold", "g1", [0], 8.0, paths["spend"])
    write_probe(
        paths,
        projection=good_probe()["projection"]
        | {"folds_1_to_4_plus_full_usd": 200.0, "total_usd_with_overhead": 60.0},
    )
    g1_file(paths, 0.95)
    with pytest.raises(sg.SpendRefused, match="ceiling"):
        check(paths, "full", [], 80.0)


def test_g1_decision_projection_field_is_not_trusted(paths):
    """The cap comes from the sha-matched probe, never from a number written in the decision."""
    sg.record_launch("f0", "fold", "g1", [0], 12.0, paths["spend"])
    _, dec = g1_file(paths, 0.95)
    dec["projected_remaining_usd"] = 1e6
    paths["g1_file"].write_text(json.dumps(dec))
    assert check(paths, "fold", [1], 1.0)["approved_cap_usd"] == pytest.approx(15 + 1.2 * 30.0)


def test_all_reasons_are_reported(paths):
    with pytest.raises(sg.SpendRefused) as e:
        check(paths, "probe", [], 99.0)
    assert "G0" in str(e.value) and "approved" in str(e.value) and "ceiling" in str(e.value)


def test_bad_inputs_refused(paths):
    with pytest.raises(sg.SpendRefused, match="unknown mode"):
        check(paths, "serve", [], 1.0)
    with pytest.raises(sg.SpendRefused, match="non-negative"):
        check(paths, "fold", [0], float("nan"))


# ---- G0 (machine-enforced)
def test_fold_and_full_need_a_real_probe_for_the_same_sha(paths):
    with pytest.raises(sg.SpendRefused, match="G0"):
        sg.check_launch("fold", [0], 1.0, **paths, git_sha=SHA)  # no probe at all
    write_probe(paths, git_sha=OTHER_SHA)
    with pytest.raises(sg.SpendRefused, match="G0.*git sha"):
        sg.check_launch("fold", [0], 1.0, **paths, git_sha=SHA)
    write_probe(paths, git_sha=SHA)
    assert sg.check_launch("fold", [0], 1.0, **paths, git_sha=SHA)


def test_launch_needs_a_git_sha(paths):
    write_probe(paths)
    with pytest.raises(sg.SpendRefused, match="git sha"):
        sg.check_launch("fold", [0], 1.0, **paths, git_sha="")


@pytest.mark.parametrize(
    "over,why",
    [
        ({"tiny": True}, "G0"),
        ({"model": "tiny-from-scratch"}, "G0"),
        ({"model": "google/gemma-2-2b-it"}, "G0"),
        ({"t": 256}, "G0"),
        ({"loss_finite": False}, "finite"),
        ({"non_finite_loss": "non-finite loss at step 3"}, "non-finite"),
        ({"attention_decision": None}, "attention"),
        ({"attention_decision": "__drop__"}, "attention"),
        ({"complete": False}, "complete"),
        ({"error": "deadline"}, "error"),
        ({"projection": "__drop__"}, "projected total"),
    ],
)
def test_each_g0_failure_is_refused(paths, over, why):
    write_probe(paths, **over)
    with pytest.raises(sg.SpendRefused, match="G0"):
        sg.check_launch("fold", [0], 1.0, **paths, git_sha=SHA)


def test_g0_projected_total_over_seventy_is_refused(paths):
    proj = good_probe()["projection"]
    write_probe(paths, projection=proj | {"total_usd_with_overhead": 70.01})
    with pytest.raises(sg.SpendRefused, match="70"):
        sg.check_launch("fold", [0], 1.0, **paths, git_sha=SHA)
    write_probe(paths, projection=proj | {"total_usd_with_overhead": 70.0})
    assert sg.check_launch("fold", [0], 1.0, **paths, git_sha=SHA)
    write_probe(paths, projection=proj | {"total_usd_with_overhead": float("nan")})
    with pytest.raises(sg.SpendRefused, match="70"):
        sg.check_launch("fold", [0], 1.0, **paths, git_sha=SHA)


def test_select_probe_is_by_sha_and_realness_not_mtime(paths):
    root = paths["out_root"]
    good = write_probe(paths, "real", finished_at=10.0)
    os.utime(good, (1000, 1000))
    newer_tiny = write_probe(paths, "tiny", tiny=True, finished_at=20.0)
    newer_other = write_probe(paths, "other", git_sha=OTHER_SHA, finished_at=30.0)
    for p in (newer_tiny, newer_other):
        os.utime(p, (9000, 9000))
    assert sg.select_probe(root, SHA)["finished_at"] == 10.0
    assert sg.select_probe(root, OTHER_SHA)["finished_at"] == 30.0
    assert sg.select_probe(root, "c" * 40) is None
    write_probe(paths, "real2", finished_at=11.0)  # newest finished_at wins among real ones
    assert sg.select_probe(root, SHA)["finished_at"] == 11.0


# ---- spend file / reconciliation
def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def totals6(paths):
    return sg.totals(paths["ledger"], paths["spend"])["rung6_usd"]


def test_manual_reconcile_replaces_the_estimate_and_records_source_and_previous(paths):
    sg.record_launch("f0", "fold", "g1", [0], 10.0, paths["spend"])
    assert totals6(paths) == 10.0
    sg.reconcile("f0", 6.5, paths["spend"], manual=True, source=sg.MANUAL_SOURCE)
    assert totals6(paths) == 6.5
    last = lines(paths["spend"])[-1]
    assert last["source"] == "modal-dashboard" and last["previous_usd"] == 10.0
    sg.record_launch("f1", "fold", "g2", [1], 3.0, paths["spend"])
    assert totals6(paths) == 9.5


def test_manual_reconcile_needs_the_dashboard_source(paths):
    sg.record_launch("f0", "fold", "g1", [0], 10.0, paths["spend"])
    with pytest.raises(ValueError, match="modal-dashboard"):
        sg.reconcile("f0", 1.0, paths["spend"], manual=True, source="guess")
    with pytest.raises(ValueError, match="modal-dashboard"):
        sg.reconcile("f0", 1.0, paths["spend"], manual=True)


def test_automatic_reconcile_never_goes_below_the_floor(paths):
    sg.record_launch("f0", "fold", "g1", [0], 10.0, paths["spend"], worst_case_usd=12.0)
    sg.reconcile("f0", 0.5, paths["spend"], source="auto-logs")  # not manual: clamped
    assert totals6(paths) == 10.0  # min(estimate 10, worst 12)
    last = lines(paths["spend"])[-1]
    assert last["clamped"] is True and last["actual_usd"] == 10.0
    sg.record_launch("f1", "fold", "g2", [1], 10.0, paths["spend"], worst_case_usd=7.0)
    sg.reconcile("f1", 0.5, paths["spend"], source="auto-logs")
    assert totals6(paths) == 10.0 + 7.0  # min(estimate 10, worst 7) = 7
    sg.reconcile("f1", 9.0, paths["spend"], source="auto-logs")  # raising is always allowed
    assert totals6(paths) == 10.0 + 9.0


def test_unreconciled_launch_is_charged_its_worst_case(paths):
    sg.record_launch("f0", "fold", "g1", [0], 8.0, paths["spend"], worst_case_usd=12.0)
    assert totals6(paths) == 12.0
    sg.record_launch("old", "fold", "g1", [0], 5.0, paths["spend"])  # no worst case recorded
    assert totals6(paths) == 17.0


def _launch(paths, mode, folds, est, worst_per_job, n=None):
    n = n or max(1, len(folds))
    sg.record_launch(
        "job",
        mode,
        "t",
        folds,
        est,
        paths["spend"],
        worst_case_usd=worst_per_job * n,
        per_job_worst_usd=worst_per_job,
        n_jobs=n,
    )


def test_auto_reconcile_charges_logs_x_1_15_and_missing_logs_their_worst_case(paths, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    _launch(paths, "fold", [1, 2, 3], est=12.0, worst_per_job=5.0)
    (out / "log_fold1.json").write_text(json.dumps({"est_cost_usd": 4.0}))
    (out / "log_fold2.json").write_text("{}")  # a log without a cost: not evidence of a cheap run
    # fold 3 never wrote a log (killed): full worst case
    res = sg.auto_reconcile("job", out, paths["spend"])
    assert res["actual_usd"] == pytest.approx(4.0 * 1.15 + 5.0 + 5.0)
    assert totals6(paths) == pytest.approx(14.6)
    assert lines(paths["spend"])[-1]["source"] == "auto-logs"


def test_auto_reconcile_cannot_go_below_the_launch_floor(paths, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    _launch(paths, "fold", [0], est=8.0, worst_per_job=12.0)
    (out / "log_fold0.json").write_text(json.dumps({"est_cost_usd": 0.01}))  # crashed instantly
    res = sg.auto_reconcile("job", out, paths["spend"])
    assert res["actual_usd"] == pytest.approx(8.0)  # min(estimate 8, worst 12)


def test_auto_reconcile_caps_a_log_at_its_worst_case(paths, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    _launch(paths, "probe", [], est=2.0, worst_per_job=5.0)
    (out / "probe.json").write_text(json.dumps({"est_cost_usd": 4.9}))
    assert sg.auto_reconcile("job", out, paths["spend"])["actual_usd"] == pytest.approx(5.0)


def test_spend_file_is_append_only(paths, tmp_path):
    _launch(paths, "fold", [0], est=8.0, worst_per_job=12.0)
    before = paths["spend"].read_text()
    sg.reconcile("job", 9.0, paths["spend"], source="auto-logs")
    sg.reconcile("job", 3.0, paths["spend"], manual=True, source=sg.MANUAL_SOURCE)
    after = paths["spend"].read_text()
    assert after.startswith(before) and len(lines(paths["spend"])) == 3


def test_reconcile_needs_a_launch(paths):
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


# ---- budget-derived time limits
def test_deadline_is_budget_over_price_over_jobs_minus_margin():
    plan = sg.deadline_plan("fold", 1, remaining_usd=12.0, probe=good_probe())
    assert plan["deadline_s"] == pytest.approx(12.0 * 3600 / 3.95 - 600)
    assert plan["timeout_s"] == pytest.approx(12.0 * 3600 / 3.95)
    assert plan["worst_case_usd"] == pytest.approx(12.0)
    plan4 = sg.deadline_plan("fold", 4, remaining_usd=48.0, probe=good_probe())
    assert plan4["deadline_s"] == pytest.approx(12.0 * 3600 / 3.95 - 600)
    assert plan4["per_job_worst_usd"] == pytest.approx(12.0)
    assert plan4["worst_case_usd"] == pytest.approx(48.0)


def test_deadline_is_capped_by_the_longest_job():
    plan = sg.deadline_plan("full", 1, remaining_usd=70.0, probe=good_probe())
    assert plan["deadline_s"] == sg.MAX_JOB_S - 600
    assert plan["worst_case_usd"] == pytest.approx(sg.MAX_JOB_S * 3.95 / 3600)


def test_probe_deadline_uses_the_probe_cap_or_the_remaining_budget():
    p = sg.deadline_plan("probe", 1, remaining_usd=70.0, probe=None)
    assert p["worst_case_usd"] == pytest.approx(5.0)
    assert p["deadline_s"] == pytest.approx(5.0 * 3600 / 3.95 - 600)
    p = sg.deadline_plan("probe", 1, remaining_usd=3.0, probe=None)
    assert p["worst_case_usd"] == pytest.approx(3.0)


def test_launch_refused_when_the_deadline_is_shorter_than_the_projected_need():
    need = 7560 * 1.15  # fold_seconds x the 15% overhead line
    ok_usd = (need + 600) * 3.95 / 3600
    assert sg.deadline_plan("fold", 1, ok_usd + 0.01, good_probe())
    with pytest.raises(sg.SpendRefused, match="deadline"):
        sg.deadline_plan("fold", 1, ok_usd - 0.05, good_probe())
    with pytest.raises(sg.SpendRefused, match="deadline"):  # 4 parallel jobs split the budget
        sg.deadline_plan("fold", 4, ok_usd * 2, good_probe())


def test_without_a_projection_the_job_needs_twenty_minutes():
    min_usd = (1200 + 600) * 3.95 / 3600
    assert sg.deadline_plan("probe", 1, min_usd + 0.01, None)
    with pytest.raises(sg.SpendRefused, match="deadline"):
        sg.deadline_plan("probe", 1, min_usd - 0.01, None)


def test_check_launch_reports_the_remaining_budget(paths):
    sg.record_launch("p1", "probe", "p", [], 2.0, paths["spend"], worst_case_usd=3.0)
    assert check(paths, "fold", [0], 1.0)["remaining_usd"] == pytest.approx(12.0)  # 15 - 3
    write_ledger(paths["ledger"], [{"config": "deberta", "gpu_hours": 1.0, "cost_usd": 78.0}])
    acc = check(paths, "fold", [0], 1.0, approved_usd=100)
    assert acc["remaining_usd"] == pytest.approx(85 - 78 - 3.0)  # the ceiling binds


# ---- launch lock
def test_launch_lock_is_exclusive_and_removed_after(tmp_path):
    lock = tmp_path / "launch.lock"
    with sg.launch_lock(lock):
        assert lock.exists()
        with pytest.raises(sg.SpendRefused, match="launch.lock"), sg.launch_lock(lock):
            pass
        assert lock.exists()  # the failed second attempt must not delete the holder's lock
    assert not lock.exists()


def test_stale_lock_refuses_with_a_message_and_lock_is_removed_after_an_error(tmp_path):
    lock = tmp_path / "launch.lock"
    lock.write_text("pid 1 at 0")
    with pytest.raises(sg.SpendRefused, match="stale|remove"), sg.launch_lock(lock):
        pass
    lock.unlink()
    with pytest.raises(RuntimeError), sg.launch_lock(lock):
        raise RuntimeError("boom")
    assert not lock.exists()


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


def test_latest_probe_selects_by_sha_and_realness_not_mtime(tmp_path):
    assert lp.latest_probe(tmp_path, SHA) is None
    for i, (tag, over) in enumerate(
        (("old", {"finished_at": 5.0}), ("tinynew", {"tiny": True}), ("othernew", {"git_sha": "z"}))
    ):
        d = tmp_path / tag
        d.mkdir()
        (d / "probe.json").write_text(json.dumps(good_probe() | {"attention_decision": tag} | over))
        os.utime(d / "probe.json", (1000 + i, 1000 + i))
    assert lp.latest_probe(tmp_path, SHA)["attention_decision"] == "old"


def test_stale_outputs_for_a_reused_tag_are_refused(tmp_path):
    lp.refuse_stale_outputs(tmp_path, "fold", [0, 1])
    (tmp_path / "log_fold1.json").write_text("{}")
    with pytest.raises(SystemExit, match="new --tag"):
        lp.refuse_stale_outputs(tmp_path, "fold", [0, 1])
    lp.refuse_stale_outputs(tmp_path, "fold", [0])  # fold 1's log is not this launch's output
    (tmp_path / "probe.json").write_text("{}")
    with pytest.raises(SystemExit, match="new --tag"):
        lp.refuse_stale_outputs(tmp_path, "probe", [])


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


def test_g1_cli_refuses_a_raw_file_that_is_not_the_verified_train_csv(tmp_path, capsys):
    raw = tmp_path / "train.csv"
    raw.write_text("id,prompt\n1,x\n")  # not the Hugging Face copy
    rc = sg._main(["g1", "--oof", str(tmp_path / "oof.parquet"), "--raw", str(raw)])
    assert rc == 2 and "sha256" in capsys.readouterr().err
    assert not (tmp_path / "g1_decision.json").exists()


def test_g1_cli_refuses_an_errored_or_unrelated_fold_log(tmp_path, capsys):
    from llm_preference_audit.kernel_common import sha256_file

    raw = ROOT / "data" / "raw" / "train.csv"
    if not raw.exists():
        pytest.skip("data/raw/train.csv not present")
    from llm_preference_audit.kernel_common import TRAIN_SHA256

    if sha256_file(raw) != TRAIN_SHA256:
        pytest.skip("raw train.csv is not the verified copy")
    (tmp_path / "log_fold0.json").write_text(
        json.dumps({"mode": "fold", "fold": 0, "error": "deadline", "git_sha": "x", "oof_rows": 1})
    )
    pd.DataFrame({"id": [1], "p_a": [0.4], "p_b": [0.4], "p_tie": [0.2]}).to_parquet(
        tmp_path / "oof_fold0.parquet", index=False
    )
    rc = sg._main(["g1", "--oof", str(tmp_path / "oof_fold0.parquet"), "--raw", str(raw)])
    assert rc == 2
