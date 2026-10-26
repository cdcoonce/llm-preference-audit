"""Rung 6b decisions that need no torch: attention choice, discovery, ladder, projection, G0."""

import json

import pytest

from llm_preference_audit import lora_g0 as g


def _good_probe(**over):
    probe = {
        "non_finite_loss": None,
        "train": {"optimizer_steps": 200, "requested_steps": 200, "scaler_skipped_steps": 3},
        "infer": {"finite": True},
        "peak_gpu_gib": 11.0,
        "gpu_total_gib": 14.6,
        "micro_batch_chosen": 4,
        "projection": {"best_fold0_hours": 6.0},
    }
    probe.update(over)
    return probe


# ---- G0 verdict


def test_g0_passes_a_clean_probe():
    v = g.g0_verdict(_good_probe())
    assert v["g0_pass"] is True and v["reasons"] == [] and all(v["checks"].values())


def test_g0_fails_on_a_non_finite_loss_even_when_everything_else_is_fine():
    v = g.g0_verdict(_good_probe(non_finite_loss="non-finite loss at step 7"))
    assert v["g0_pass"] is False and v["checks"]["loss_finite"] is False
    assert any("non-finite" in r for r in v["reasons"])


@pytest.mark.parametrize(
    ("skipped", "ok"), [(0, True), (9, True), (10, False), (60, False)]
)  # 5% of 200 steps is 10: the share must be strictly below
def test_g0_scaler_skip_threshold(skipped, ok):
    p = _good_probe()
    p["train"]["scaler_skipped_steps"] = skipped
    assert g.g0_verdict(p)["checks"]["scaler_skips_ok"] is ok


def test_g0_memory_threshold_and_missing_micro_batch():
    assert g.g0_verdict(_good_probe(peak_gpu_gib=13.5))["checks"]["fits_memory"] is True
    assert g.g0_verdict(_good_probe(peak_gpu_gib=14.0))["checks"]["fits_memory"] is False
    v = g.g0_verdict(_good_probe(micro_batch_chosen=None))
    assert v["checks"]["fits_memory"] is False and not v["g0_pass"]


@pytest.mark.parametrize(("hours", "ok"), [(9.99, True), (10.0, True), (10.01, False)])
def test_g0_ten_hour_cap(hours, ok):
    v = g.g0_verdict(_good_probe(projection={"best_fold0_hours": hours}))
    assert v["checks"]["projected_fold0_within_cap"] is ok and v["g0_pass"] is ok


def test_g0_incomplete_or_empty_probe_never_passes():
    p = _good_probe()
    p["train"]["optimizer_steps"] = 120  # stopped by the deadline
    assert not g.g0_verdict(p)["g0_pass"]
    v = g.g0_verdict({})
    assert v["g0_pass"] is False and not any(v["checks"].values())


# ---- projection arithmetic


def test_projection_single_gpu_arithmetic():
    n = 10_000
    p = g.project_kaggle(
        n_cv=n, train_sps_single=2.0, infer_sps_single=10.0, load_seconds=100, prep_seconds=50
    )
    f = p["single"]["fold0"]
    assert f["train_seconds"] == pytest.approx(n * 0.8 * 0.9 / 2.0)  # 3600 s
    assert f["eval_seconds"] == pytest.approx(2 * 2 * n * 0.8 * 0.1 / 10.0)  # 320 s
    assert f["predict_seconds"] == pytest.approx(2 * n * 0.2 / 10.0)  # 400 s
    assert f["seconds"] == pytest.approx(50 + 100 + 3600 + 320 + 400)
    assert p["single"]["all_cv_train_seconds"] == pytest.approx(50 + 100 + n / 2.0)
    assert p["ddp"] is None and p["best_fold0_mode"] == "single"


def test_projection_picks_the_faster_of_single_and_ddp():
    kw = {"n_cv": 10_000, "train_sps_single": 2.0, "infer_sps_single": 10.0, "load_seconds": 0}
    fast = g.project_kaggle(**kw, train_sps_ddp=3.6)
    assert fast["best_fold0_mode"] == "ddp"
    assert fast["best_fold0_hours"] == pytest.approx(fast["ddp"]["fold0"]["hours"])
    assert fast["ddp"]["fold0"]["predict_seconds"] == pytest.approx(2 * 10_000 * 0.2 / 20.0)
    assert fast["assumptions"]["ddp_inference_measured"] is False
    slow = g.project_kaggle(**kw, train_sps_ddp=1.0)  # DDP slower than one GPU: single wins
    assert slow["best_fold0_mode"] == "single"
    assert slow["best_fold0_hours"] == pytest.approx(slow["single"]["fold0"]["hours"])


def test_projection_rejects_non_positive_rates():
    with pytest.raises(ValueError):
        g.project_kaggle(n_cv=1, train_sps_single=0, infer_sps_single=1, load_seconds=0)


# ---- attention choice


def test_attention_choice_logic():
    tol = g.attention_tolerance("fp16-autocast")
    assert tol == 2e-2 and g.attention_tolerance("bf16") == 1e-3
    assert g.choose_attention(0.019, 3.0, tol) == "sdpa"
    assert g.choose_attention(0.021, 3.0, tol) == "eager"
    assert g.choose_attention(0.0, 0.01, tol) == "eager"  # logits too small to see a difference
    assert g.choose_attention(float("nan"), 3.0, tol) == "eager"
    assert g.choose_attention(0.019, 3.0, g.attention_tolerance("bf16")) == "eager"


# ---- discovery and ladder


def _write_cfg(d, model_type, weights=True):
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"model_type": model_type}))
    if weights:
        (d / "model.safetensors").write_text("x")


def test_discover_model_dir_finds_qwen3_by_config_not_path(tmp_path):
    _write_cfg(tmp_path / "other" / "gemma", "gemma2")
    _write_cfg(tmp_path / "qwen-3" / "transformers" / "1.7b" / "1", "qwen3")
    assert g.discover_model_dir(tmp_path).name == "1"


def test_discover_model_dir_errors(tmp_path):
    _write_cfg(tmp_path / "g", "gemma2")
    with pytest.raises(SystemExit, match="no config.json"):
        g.discover_model_dir(tmp_path)
    _write_cfg(tmp_path / "a", "qwen3")
    _write_cfg(tmp_path / "b", "qwen3")
    with pytest.raises(SystemExit, match="more than one"):
        g.discover_model_dir(tmp_path)


def test_discover_prefers_the_directory_with_weights(tmp_path):
    _write_cfg(tmp_path / "bare", "qwen3", weights=False)
    _write_cfg(tmp_path / "full", "qwen3")
    assert g.discover_model_dir(tmp_path).name == "full"


class _Oom(Exception):
    pass


def test_fallback_steps_down_only_on_oom_and_records_attempts():
    seen = []

    def fn(size):
        seen.append(size)
        if size > 2:
            raise _Oom
        return f"ran {size}"

    res, size, attempts = g.run_with_fallback(
        fn, g.MICRO_BATCH_LADDER, lambda e: isinstance(e, _Oom)
    )
    assert (res, size, seen) == ("ran 2", 2, [4, 2])
    assert attempts == [
        {"size": 4, "oom": True},
        {"size": 2, "oom": False},
    ]
    res, size, attempts = g.run_with_fallback(lambda s: _raise(_Oom), (4, 2, 1), lambda e: True)
    assert res is None and size is None and len(attempts) == 3
    with pytest.raises(ZeroDivisionError):
        g.run_with_fallback(lambda s: 1 / 0, (4, 2, 1), lambda e: False)


def _raise(exc):
    raise exc


# ---- the template is the plain rung-6 one, never Qwen3's chat template


def test_template_segments_carry_no_chat_or_thinking_markup():
    from llm_preference_audit.lora_inputs import SEGMENTS

    joined = "".join(SEGMENTS)
    for marker in ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|endoftext|>"):
        assert marker not in joined
