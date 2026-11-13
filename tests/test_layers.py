import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from brushgpt.layers import (
    GPT,
    GPTConfig,
    RMSNorm,
    SelfAttention,
    SelfAttentionCache,
    apply_rope,
    rope_tables,
    swiglu_hidden,
)

TINY = GPTConfig(vocab_size=50, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)


def reference(module: str):
    """A module of the research code this package was extracted from (path in $SELL_ANYTHING_ML)."""
    root = os.environ.get("SELL_ANYTHING_ML")
    if not root:
        pytest.skip("SELL_ANYTHING_ML is not set")
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(module)


# ------------------------------------------------------------------------------------------------- unit tests
def test_rmsnorm_gives_unit_root_mean_square():
    x = torch.randn(3, 5, 16) * 7
    y = RMSNorm(16)(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(3, 5), atol=1e-4)


def test_swiglu_hidden_is_eight_thirds_rounded_to_64():
    assert swiglu_hidden(512) == 1408
    assert swiglu_hidden(384) == 1024
    assert swiglu_hidden(64) % 64 == 0


def test_rope_keeps_lengths_and_depends_only_on_the_offset():
    cos, sin = rope_tables(8, 32)
    q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)

    def at(x, p):
        return apply_rope(x, cos[p : p + 1], sin[p : p + 1])

    assert torch.allclose(at(q, 5).norm(), q.norm(), atol=1e-5)
    assert torch.allclose((at(q, 7) * at(k, 3)).sum(), (at(q, 14) * at(k, 10)).sum(), atol=1e-5)


def test_self_attention_cache_matches_a_full_causal_pass():
    torch.manual_seed(0)
    attn = SelfAttention(32, 4).eval()
    x = torch.randn(2, 6, 32)
    cos, sin = rope_tables(8, 6)
    full = attn(x, cos, sin, causal=True)
    cache = SelfAttentionCache(size=10)
    steps = [attn(x[:, :3], cos[:3], sin[:3], causal=True, cache=cache)]
    steps += [attn(x[:, t : t + 1], cos[t : t + 1], sin[t : t + 1], causal=True, cache=cache) for t in range(3, 6)]
    assert torch.allclose(torch.cat(steps, 1), full, atol=1e-6)


def test_gpt_is_causal():
    model = GPT(TINY).eval()
    a = torch.randint(0, 50, (1, 10))
    b = a.clone()
    b[0, 7] = (b[0, 7] + 1) % 50
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.equal(la[0, :7], lb[0, :7]) and not torch.allclose(la[0, 7], lb[0, 7])


def test_gpt_cache_reproduces_the_full_forward():
    model = GPT(TINY).eval()
    idx = torch.randint(0, 50, (2, 12))
    full, _ = model(idx)
    cache = model.empty_cache()
    parts = [model(idx[:, :5], cache=cache)[0]]
    parts += [model(idx[:, t : t + 1], cache=cache, start_pos=t)[0] for t in range(5, 12)]
    assert torch.allclose(torch.cat(parts, 1), full, atol=1e-5)


def test_gpt_masked_only_loss_equals_the_masked_loss():
    model = GPT(TINY).eval()
    idx, tgt = torch.randint(0, 50, (2, 12)), torch.randint(0, 50, (2, 12))
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[:, 4:9] = True
    _, full = model(idx, tgt, mask)
    _, fast = model(idx, tgt, mask, masked_only=True)
    assert torch.allclose(full, fast, atol=1e-6)


def test_gpt_head_is_tied_to_the_embedding():
    model = GPT(TINY)
    assert model.head.weight is model.embed.weight
    assert model.num_params() == sum(p.numel() for p in model.parameters())


def test_config_round_trip_and_old_dicts():
    assert GPTConfig.from_dict(TINY.to_dict()) == TINY
    assert GPTConfig.from_dict({**TINY.to_dict(), "abs_pos": False}) == TINY
    with pytest.raises(ValueError):
        GPTConfig.from_dict({**TINY.to_dict(), "abs_pos": True})


# ----------------------------------------------------------------------------------- equivalence with the reference
def _pair(dropout: float = 0.0):
    llm = reference("sellanything_ml.llm.model")
    cfg = GPTConfig(**{**TINY.to_dict(), "dropout": dropout})
    torch.manual_seed(0)
    ref = llm.GPT(llm.GPTConfig(**cfg.to_dict()))
    torch.manual_seed(0)
    return ref, GPT(cfg)


def test_same_seed_gives_the_reference_weights_with_the_same_keys():
    ref, mine = _pair()
    a, b = ref.state_dict(), mine.state_dict()
    assert list(a) == list(b)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_logits_loss_and_cache_match_the_reference():
    ref, mine = _pair()
    ref.eval(), mine.eval()
    idx, tgt = torch.randint(0, 50, (3, 16)), torch.randint(0, 50, (3, 16))
    mask = torch.rand(3, 16) < 0.5
    for kw in ({}, {"targets": tgt}, {"targets": tgt, "loss_mask": mask}):
        (la, sa), (lb, sb) = ref(idx, **kw), mine(idx, **kw)
        assert torch.equal(la, lb)
        assert (sa is None and sb is None) or torch.equal(sa, sb)
    assert torch.equal(ref(idx, tgt, mask, masked_only=True)[1], mine(idx, tgt, mask, masked_only=True)[1])
    rc, mc = [{} for _ in ref.blocks], mine.empty_cache()
    assert torch.equal(ref(idx[:, :9], cache=rc)[0], mine(idx[:, :9], cache=mc)[0])
    for t in range(9, 16):
        assert torch.equal(
            ref(idx[:, t : t + 1], cache=rc, start_pos=t)[0], mine(idx[:, t : t + 1], cache=mc, start_pos=t)[0]
        )


def test_reference_state_dict_loads_unchanged():
    llm = reference("sellanything_ml.llm.model")
    torch.manual_seed(3)
    ref = llm.GPT(llm.GPTConfig(**TINY.to_dict())).eval()
    mine = GPT(TINY).eval()
    mine.load_state_dict(ref.state_dict())
    idx = torch.randint(0, 50, (2, 10))
    assert torch.equal(ref(idx)[0], mine(idx)[0])


def test_training_steps_match_the_reference():
    ref, mine = _pair(dropout=0.1)
    losses = []
    for model in (ref, mine):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
        gen = torch.Generator().manual_seed(1)
        torch.manual_seed(2)
        run = []
        for _ in range(4):
            idx = torch.randint(0, 50, (4, 16), generator=gen)
            tgt = torch.randint(0, 50, (4, 16), generator=gen)
            mask = torch.rand(4, 16, generator=gen) < 0.6
            _, loss = model(idx, tgt, mask, masked_only=True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run.append(loss.item())
        losses.append(run)
    assert losses[0] == losses[1]


def test_real_eye_checkpoint_matches_the_reference():
    llm = reference("sellanything_ml.llm.model")
    root = Path(os.environ["SELL_ANYTHING_ML"])
    ckpt, data = root / "runs/eyegpt/E1/best.pt", root / "data/processed/eye_1110/data.npz"
    if not (ckpt.exists() and data.exists()):
        pytest.skip("the eye model's checkpoint or data is not there")
    ck = torch.load(ckpt, map_location="cpu")
    ref = llm.GPT(llm.GPTConfig(**ck["config"])).eval()
    ref.load_state_dict(ck["model"])
    mine = GPT(GPTConfig.from_dict(ck["config"])).eval()
    mine.load_state_dict(ck["model"])
    z = np.load(data)
    n = int(z["lengths"][:4].max())
    idx = torch.from_numpy(z["tokens"][:4, :n].astype(np.int64))
    with torch.no_grad():
        a, b = ref(idx)[0], mine(idx)[0]
    assert torch.equal(a, b)
