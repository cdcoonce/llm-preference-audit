"""The in-process wall-clock deadline (torch-free part)."""

import pytest

from llm_preference_audit.lora_cost import Deadline, DeadlineExceeded


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_deadline_raises_once_the_budget_is_spent():
    c = Clock()
    d = Deadline(10, clock=c)
    d.check("a")
    c.t = 9.9
    d.check("b")
    c.t = 10.0
    with pytest.raises(DeadlineExceeded) as e:
        d.check("train")
    assert e.value.stage == "train"


def test_no_deadline_never_expires():
    c = Clock()
    for d in (Deadline(None, clock=c), Deadline(0, clock=c)):
        c.t = 1e9
        d.check("x")
        assert not d.expired()


def test_stale_outputs_are_deleted_before_a_fold_trains(tmp_path):
    from llm_preference_audit.lora_cost import clear_stale_outputs

    out, adapter = tmp_path / "out", tmp_path / "adapter"
    out.mkdir()
    adapter.mkdir()
    for n in ("oof_fold0.parquet", "log_fold0.json", "oof_fold1.parquet", "probe.json"):
        (out / n).write_text("old")
    removed = clear_stale_outputs("fold", out, adapter, fold=0)
    assert sorted(p.name for p in removed) == ["log_fold0.json", "oof_fold0.parquet"]
    assert (out / "oof_fold1.parquet").exists() and (out / "probe.json").exists()
    assert not (out / "oof_fold0.parquet").exists()


def test_stale_outputs_for_probe_and_full(tmp_path):
    from llm_preference_audit.lora_cost import clear_stale_outputs

    out, adapter = tmp_path / "out", tmp_path / "adapter"
    out.mkdir()
    adapter.mkdir()
    (out / "probe.json").write_text("old")
    (out / "log_full.json").write_text("old")
    (adapter / "adapter_model.safetensors").write_text("old")
    assert [p.name for p in clear_stale_outputs("probe", out, adapter)] == ["probe.json"]
    names = sorted(p.name for p in clear_stale_outputs("full", out, adapter))
    assert names == ["adapter_model.safetensors", "log_full.json"]
