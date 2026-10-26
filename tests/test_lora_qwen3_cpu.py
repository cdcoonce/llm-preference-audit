"""Rung 6b CPU checks on a tiny from-scratch Qwen3: padding invariance and pooling, the fp32 head
and norm, dtype assertions, the GradScaler path (unscale before clip, skip counting), the
first-overflow report, and the Kaggle probe end to end (including a 2-process gloo DDP stage).

Skipped unless torch, transformers and peft are installed (`make ci` has none of them):
  uv run --with torch --with transformers --with peft --with sentencepiece --with protobuf \\
      --with accelerate python -m pytest tests/test_lora_qwen3_cpu.py
fp16 autocast does not exist on CPU, so `--precision fp16-autocast` is emulated with bf16 there
(`precision_dtypes`); the fp16 dtypes themselves are checked by pretending the device is CUDA.
"""

import argparse
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from llm_preference_audit import lora_kaggle as lk
from llm_preference_audit import lora_train as lt
from llm_preference_audit.lora_inputs import Template

VOCAB, PAD = 300, 0
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
CUDA = torch.device("cuda")  # constructing the device object needs no GPU


def _args(**kw):
    base = {
        "lora_r": 4, "lora_alpha": 8, "lora_dropout": 0.0, "precision": "bf16",
        "fp32_head_norm": False, "tiny": True, "tiny_arch": "qwen3", "model": lt.DEFAULT_MODEL,
        "model_dir": "", "t": 64, "micro_batch": 4, "infer_batch": 4, "lr": 1e-3,
        "force_scaler": False, "deadline": None,
    }  # fmt: skip
    return argparse.Namespace(**(base | kw))


def _tiny(dtype=torch.float32, attn="eager"):
    return lt.build_tiny_model(VOCAB, PAD, 64, attn, dtype, arch="qwen3")


# ---------------------------------------------------------------- the head: padding and pooling


def _alone_vs_padded(attn):
    model = _tiny(attn=attn).eval()
    torch.manual_seed(1)
    ids = torch.randint(1, VOCAB, (3, 20))
    lens = [20, 13, 7]
    padded = torch.full((3, 20), PAD)
    mask = torch.zeros((3, 20), dtype=torch.long)
    for r, n in enumerate(lens):
        padded[r, :n] = ids[r, :n]
        mask[r, :n] = 1
    with torch.no_grad():
        batch = model(input_ids=padded, attention_mask=mask).logits
        alone = [model(input_ids=ids[r : r + 1, :n]).logits[0] for r, n in enumerate(lens)]
    return max(float((alone[r] - batch[r]).abs().max()) for r in range(3))


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_qwen3_right_padding_does_not_change_the_pooled_logits(attn):
    assert _alone_vs_padded(attn) < 1e-4


def test_qwen3_pools_the_last_non_pad_token_not_the_last_position():
    model = _tiny().eval()
    ids = torch.randint(1, VOCAB, (1, 12))
    short = torch.cat([ids[:, :7], torch.full((1, 5), PAD)], dim=1)
    mask = torch.cat([torch.ones(1, 7), torch.zeros(1, 5)], dim=1).long()
    with torch.no_grad():
        a = model(input_ids=short, attention_mask=mask).logits
        b = model(input_ids=ids[:, :7]).logits
        c = model(input_ids=ids).logits
    assert torch.allclose(a, b, atol=1e-5) and not torch.allclose(a, c, atol=1e-3)
    assert type(model).__name__ == "Qwen3ForSequenceClassification"


# ---------------------------------------------------------------- precision and dtypes


def test_precision_dtypes_per_device_and_default_is_the_gemma_behaviour():
    cpu = torch.device("cpu")
    assert lt.precision_dtypes(_args(), cpu) == (torch.bfloat16, torch.bfloat16)
    assert lt.precision_dtypes(_args(precision="fp32"), cpu) == (torch.float32, None)
    assert lt.precision_dtypes(_args(precision="fp16-autocast"), CUDA) == (torch.float16,) * 2
    assert lt.precision_dtypes(_args(precision="fp16-autocast"), cpu) == (torch.bfloat16,) * 2
    assert lt.dtype_description(_args(), cpu) == "base bf16; LoRA+head fp32 masters; bf16 autocast"


def test_fp32_head_stays_fp32_inside_an_autocast_region():
    plain = torch.nn.Linear(8, 3)
    head = lt.Fp32Head(8, 3)
    x = torch.randn(2, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert plain(x).dtype == torch.bfloat16  # what the guard exists to prevent
        assert head(x.bfloat16()).dtype == torch.float32


@pytest.mark.parametrize("fp32_head_norm", [False, True])
def test_fp16_autocast_keeps_lora_and_head_fp32_and_the_base_fp16(monkeypatch, fp32_head_norm):
    monkeypatch.setattr(lt, "pick_device", lambda: CUDA)  # fp16 dtypes without a GPU
    args = _args(precision="fp16-autocast", fp32_head_norm=fp32_head_norm)
    base = lt.build_tiny_model(VOCAB, PAD, 64, "eager", torch.float16, arch="qwen3")
    assert {p.dtype for p in base.parameters()} == {torch.float16}
    if fp32_head_norm:
        lt.make_head_norm_fp32(base)
    pm = lt.wrap_lora(base, args)
    census = lt.assert_precision_dtypes(pm, args, CUDA)
    assert census["trainable"] == ["torch.float32"]
    trainable = {n: p for n, p in pm.named_parameters() if p.requires_grad}
    assert any("lora_A" in n for n in trainable) and any("score" in n for n in trainable)
    assert {p.dtype for p in trainable.values()} == {torch.float32}
    norm = dict(pm.named_parameters())
    final_norm = next(p for n, p in norm.items() if n.endswith("model.norm.weight"))
    layer_norm = next(p for n, p in norm.items() if n.endswith("0.input_layernorm.weight"))
    assert final_norm.dtype == (torch.float32 if fp32_head_norm else torch.float16)
    assert layer_norm.dtype == torch.float16  # only the FINAL norm moves
    assert ("torch.float32" in census["frozen"]) == fp32_head_norm


def test_dtype_assertion_rejects_an_fp16_trainable_parameter(monkeypatch):
    monkeypatch.setattr(lt, "pick_device", lambda: CUDA)
    args = _args(precision="fp16-autocast")
    pm = lt.wrap_lora(
        lt.build_tiny_model(VOCAB, PAD, 64, "eager", torch.float16, arch="qwen3"), args
    )
    victim = next(p for n, p in pm.named_parameters() if "lora_A" in n)
    victim.data = victim.data.half()
    with pytest.raises(SystemExit, match="must be fp32"):
        lt.assert_precision_dtypes(pm, args, CUDA)


# ---------------------------------------------------------------- the training loop


def _synthetic(n=64, seed=0):
    rng = np.random.default_rng(seed)

    def toks():
        return [
            rng.integers(3, VOCAB, size=int(rng.integers(5, 14))).astype(np.int32) for _ in range(n)
        ]

    p, a, b = toks(), toks(), toks()
    meta = pd.DataFrame(
        {
            "id": np.arange(n),
            "y": rng.integers(0, 3, n),
            "n_prompt": [len(x) for x in p],
            "n_a": [len(x) for x in a],
            "n_b": [len(x) for x in b],
        }
    )
    template = Template((), ((5, 6), (7,), (8,), (9, 10)))
    return lt.Encoded(meta, p, a, b, n, {}), template


def _train(steps=3, seed=1, pre=None, **kw):
    enc, template = _synthetic()
    args = _args(**kw)
    model = lt.wrap_lora(_tiny(torch.bfloat16), args)
    if pre:
        pre(model)
    tok = types.SimpleNamespace(pad_token_id=PAD)
    torch.manual_seed(seed)
    idx = np.arange(len(enc.meta))
    return model, lt.train(
        model, enc, template, idx, None, tok, args, seed, torch.device("cpu"), steps
    )


def test_unscale_happens_before_the_clip():
    """With the scaler on, clip_grad_norm_ must see TRUE gradients: the recorded norm equals the one
    from a scaler-off run (the 2**16 scale is a power of two, so the match is exact up to rounding).
    Without unscale_, the clip would see 65536x larger gradients."""
    _, (_, _, off) = _train(precision="fp16-autocast", force_scaler=False)
    _, (_, _, on) = _train(precision="fp16-autocast", force_scaler=True)
    assert off["scaler_enabled"] is False and on["scaler_enabled"] is True
    assert on["grad_norm_max"] == pytest.approx(off["grad_norm_max"], rel=1e-3)
    assert on["grad_norm_last"] == pytest.approx(off["grad_norm_last"], rel=1e-3)
    assert on["mean_train_loss"] == pytest.approx(off["mean_train_loss"], rel=1e-4)


def test_scaler_skipped_steps_are_counted_and_the_skipped_step_does_not_move_weights():
    def poison(model):
        state = {"n": 0}

        def hook(g):
            state["n"] += 1
            return g * float("inf") if state["n"] <= 4 else g  # first optimizer step only

        next(
            p for n, p in model.named_parameters() if "lora_B" in n and p.requires_grad
        ).register_hook(hook)

    _, (_, _, stats) = _train(steps=3, precision="fp16-autocast", force_scaler=True, pre=poison)
    assert stats["scaler_skipped_steps"] == 1
    assert stats["optimizer_steps"] == 3 and stats["requested_steps"] == 3
    assert stats["scaler_final_scale"] < 2.0**16  # backed off after the skip
    _, (_, _, clean) = _train(steps=3, precision="fp16-autocast", force_scaler=True)
    assert clean["scaler_skipped_steps"] == 0


@pytest.mark.parametrize("layer", [0, 1])
def test_non_finite_loss_stops_the_run_and_names_the_first_overflowing_module(layer):
    def inject(model):
        target = next(m for n, m in model.named_modules() if n.endswith(f"layers.{layer}.mlp"))
        target.register_forward_hook(lambda m, i, o: o * float("inf"))

    with pytest.raises(lt.NonFiniteLoss) as e:
        _train(pre=inject)
    report = e.value.first_overflow
    assert report["reproduced"] is True
    assert report["first_overflow_module"].endswith(f"layers.{layer}.mlp")
    assert report["step"] == 0 and report["hooked_modules"] > 10
    assert report["first_overflow_module"] in str(e.value)


def test_report_on_a_clean_model_finds_nothing_and_removes_its_hooks():
    model, _ = _train(steps=1)
    ids = torch.randint(1, VOCAB, (2, 9))
    mask = torch.ones_like(ids)
    n_hooks = sum(len(m._forward_hooks) for m in model.modules())
    report = lt.first_overflow_report(model, ids, mask, _args(), torch.device("cpu"))
    assert report["reproduced"] is False and report["first_overflow_module"] is None
    assert report["rerun_logits_finite"] is True and model.training
    assert sum(len(m._forward_hooks) for m in model.modules()) == n_hooks


# ---------------------------------------------------------------- the Kaggle probe, end to end


@pytest.fixture
def tiny_ready():
    if not (RAW / "train.csv").exists():
        pytest.skip("data/raw/train.csv not present")
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(lt.TINY_QWEN3_TOKENIZER)
    except Exception:  # noqa: BLE001  offline
        pytest.skip("Qwen3 tokenizer not available")


def _qprobe_argv(tmp_path, *extra):
    return [
        "--mode", "qprobe", "--tiny", "--tiny-arch", "qwen3", "--precision", "fp16-autocast",
        "--fp32-head-norm", "--max-rows", "300", "--t", "128", "--probe-steps", "3",
        "--probe-infer-rows", "16", "--qattn-rows", "8", "--ddp-steps", "2",
        "--data-dir", str(RAW), "--out-dir", str(tmp_path / "out"), "--git-sha", "abc",
        "--tag", "t", *extra,
    ]  # fmt: skip


def test_qwen3_tokenizer_has_a_pad_and_no_bos_and_the_template_builds(tiny_ready):
    tok = lt.make_tokenizer(_args(tiny=True))
    assert tok.pad_token_id is not None and tok.bos_token_id is None
    template = lt.tokenizer_template(tok)
    assert template.bos == () and template.overhead > 0


def test_qprobe_writes_every_stage_and_a_g0_verdict(tmp_path, tiny_ready):
    code = lk.main(_qprobe_argv(tmp_path, "--no-ddp"))
    assert code == 0
    probe = json.loads((tmp_path / "out" / "probe.json").read_text())
    assert probe["complete"] is True
    assert probe["stages_done"] == ["setup", "attention", "train", "infer", "ddp", "verdict"]
    assert probe["precision"] == "fp16-autocast" and probe["fp32_head_norm"] is True
    assert probe["attention_check"]["tolerance"] == 2e-2
    assert probe["attention_decision"] in ("sdpa", "eager")
    assert probe["micro_batch_chosen"] == 4 and probe["micro_batch_attempts"] == [
        {"size": 4, "oom": False}
    ]
    assert probe["train"]["optimizer_steps"] == 3 == probe["train"]["requested_steps"]
    assert probe["train"]["scaler_skipped_steps"] == 0 and probe["non_finite_loss"] is None
    assert probe["infer"]["finite"] and probe["infer"]["rows"] == 16
    assert probe["ddp"] == {"skipped": "--no-ddp"}
    assert probe["projection"]["best_fold0_hours"] > 0 and probe["projection"]["ddp"] is None
    assert set(probe["g0"]["checks"]) >= {"loss_finite", "scaler_skips_ok", "fits_memory"}
    assert probe["g0"]["g0_pass"] is False  # a CPU run has no GPU memory figure: not established
    assert "fits_memory" in " ".join(probe["g0"]["reasons"]) or probe["g0"]["reasons"]


def test_qprobe_with_an_expired_deadline_leaves_partial_evidence(tmp_path, tiny_ready):
    code = lk.main(_qprobe_argv(tmp_path, "--deadline-s", "0.001"))
    assert code == lt.DEADLINE_EXIT
    probe = json.loads((tmp_path / "out" / "probe.json").read_text())
    assert probe["error"] == "deadline" and probe["complete"] is False
    assert probe["g0"]["g0_pass"] is False


def test_qprobe_records_the_first_overflow_and_fails_g0(tmp_path, tiny_ready, monkeypatch):
    real = lt.wrap_lora

    def inject(model, args):
        pm = real(model, args)
        target = next(m for n, m in pm.named_modules() if n.endswith("layers.1.self_attn"))
        target.register_forward_hook(
            lambda m, i, o: tuple(x * float("inf") for x in o if x is not None)
        )
        return pm

    monkeypatch.setattr(lt, "wrap_lora", inject)
    code = lk.main(_qprobe_argv(tmp_path, "--no-ddp", "--attn", "eager"))
    assert code == 0  # a numerical failure is a recorded result, not a crash
    probe = json.loads((tmp_path / "out" / "probe.json").read_text())
    assert probe["non_finite_loss"] and probe["train"] == {}
    assert probe["first_overflow"]["first_overflow_module"].endswith("layers.1.self_attn")
    assert probe["g0"]["g0_pass"] is False and probe["g0"]["checks"]["loss_finite"] is False
    assert "skipped" in probe["ddp"]


def test_qprobe_ddp_stage_runs_two_gloo_processes(tmp_path, tiny_ready):
    code = lk.main(_qprobe_argv(tmp_path, "--ddp-cpu-procs", "2", "--attn", "eager"))
    assert code == 0
    probe = json.loads((tmp_path / "out" / "probe.json").read_text())
    ddp = probe["ddp"]
    assert ddp["ok"] is True and ddp["world_size"] == 2 and ddp["attempts"][0]["returncode"] == 0
    train = ddp["result"]["train"]
    assert (
        train["world_size"] == 2 and train["accumulation"] == 2
    )  # 16 = 4 rows x 2 accum x 2 ranks
    assert train["optimizer_steps"] == 2 == train["requested_steps"]
    assert ddp["speedup_vs_single"] > 0 and probe["projection"]["ddp"] is not None
    assert not list(Path(tmp_path / "out").glob("*.pkl"))  # token ids never land in the output dir


def test_dataset_text_ids_never_contain_the_pad_id_for_literal_special_tokens():
    """A response containing "<|endoftext|>" must not put Qwen's pad id inside the sequence."""
    from transformers import AutoTokenizer

    from llm_preference_audit.lora_train import text_ids

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    ids = text_ids(tok, ["synthetic answer <|endoftext|> more <|im_start|>", "plain text"])
    assert tok.pad_token_id not in ids[0]
    assert len(ids) == 2 and all(len(x) > 0 for x in ids)
