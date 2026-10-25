"""CPU checks on a tiny from-scratch Gemma2. Skipped unless torch, transformers and peft are
installed (`make ci` has none of them; run with `uv run --with torch --with transformers
--with peft --with sentencepiece --with protobuf pytest tests/test_lora_tiny_cpu.py`)."""

import argparse

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from llm_preference_audit import lora_train as lt

VOCAB, PAD = 300, 0


def _logits_alone_vs_padded(attn, dtype):
    model = lt.build_tiny_model(VOCAB, PAD, 64, attn, dtype).eval()
    torch.manual_seed(1)
    ids = torch.randint(1, VOCAB, (3, 20))
    lens = [20, 13, 7]
    padded = torch.full((3, 20), PAD)
    mask = torch.zeros((3, 20), dtype=torch.long)
    for r, n in enumerate(lens):
        padded[r, :n] = ids[r, :n]
        mask[r, :n] = 1
    with torch.no_grad():
        batch = model(input_ids=padded, attention_mask=mask).logits.float()
        alone = [model(input_ids=ids[r : r + 1, :n]).logits.float()[0] for r, n in enumerate(lens)]
    return max(float((alone[r] - batch[r]).abs().max()) for r in range(3))


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_right_padding_does_not_change_the_pooled_logits(attn):
    assert _logits_alone_vs_padded(attn, torch.float32) < 1e-4


def test_pooling_uses_the_last_non_pad_token_not_the_last_position():
    model = lt.build_tiny_model(VOCAB, PAD, 64, "eager", torch.float32).eval()
    ids = torch.randint(1, VOCAB, (1, 12))
    short = torch.cat([ids[:, :7], torch.full((1, 5), PAD)], dim=1)
    mask = torch.cat([torch.ones(1, 7), torch.zeros(1, 5)], dim=1).long()
    with torch.no_grad():
        a = model(input_ids=short, attention_mask=mask).logits
        b = model(input_ids=ids[:, :7]).logits
        c = model(input_ids=ids).logits
    assert torch.allclose(a, b, atol=1e-5) and not torch.allclose(a, c, atol=1e-3)


def test_lora_wiring_dtypes_and_head():
    base = lt.build_tiny_model(VOCAB, PAD, 64, "eager", torch.bfloat16)
    assert {p.dtype for p in base.parameters()} == {torch.bfloat16}
    args = argparse.Namespace(lora_r=16, lora_alpha=32, lora_dropout=0.05)
    pm = lt.wrap_lora(base, args)
    train = {n: p for n, p in pm.named_parameters() if p.requires_grad}
    assert {p.dtype for p in train.values()} == {torch.float32}
    assert any("score" in n for n in train) and any("lora_A" in n for n in train)
    frozen = [p for p in pm.parameters() if not p.requires_grad]
    assert frozen and {p.dtype for p in frozen} == {torch.bfloat16}
    for target in lt.LORA_TARGETS:
        assert any(f"{target}.lora_A" in n for n in train), target


def test_best_state_roundtrip_touches_only_trainable_params():
    base = lt.build_tiny_model(VOCAB, PAD, 64, "eager", torch.bfloat16)
    pm = lt.wrap_lora(base, argparse.Namespace(lora_r=4, lora_alpha=8, lora_dropout=0.0))
    state = lt.trainable_state(pm)
    frozen_before = {n: p.clone() for n, p in pm.named_parameters() if not p.requires_grad}
    with torch.no_grad():
        for p in pm.parameters():
            if p.requires_grad:
                p.add_(1.0)
    lt.load_trainable_state(pm, state)
    for n, p in pm.named_parameters():
        if p.requires_grad:
            assert torch.equal(p.cpu(), state[n])
        else:
            assert torch.equal(p, frozen_before[n])
