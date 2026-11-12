"""Transformer building blocks shared by the models, and a decoder-only GPT.

The parts follow the Llama recipe rather than the original GPT-2 one:

| GPT-2             | here                  | why                                                           |
|-------------------|-----------------------|---------------------------------------------------------------|
| LayerNorm         | RMSNorm               | no mean subtraction and no bias; cheaper, works as well        |
| learned positions | rotary embeddings     | attention sees relative offsets; nothing to learn              |
| GELU MLP (4d)     | SwiGLU (about 8/3 d)  | a gated MLP; better quality at the same parameter count        |

Two attention families live here:

* `SelfAttention` and `CrossAttention`: full multi-head attention with rotary positions, used by the Brush GPT
  encoder-decoder (`brushgpt.model`). Cross-attention applies the rotary embedding to the encoder's positions too, so
  a decoder position and an encoder position are compared through their offset. During generation the decoder keeps
  its keys and values in fixed-size buffers (`SelfAttentionCache`), and the encoder's projected keys and values are
  computed once (`CrossAttentionCache`).
* `GroupedQueryAttention`: causal self-attention in which groups of query heads share one key/value head, used by the
  decoder-only `GPT` (the eye part model is built on it).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

ROPE_THETA = 10000.0
MASKED_ROUND = 256  # GPT.forward(masked_only=True) pads the number of scored positions to a multiple of this


# ---------------------------------------------------------------------------------------------------- basic parts
class RMSNorm(nn.Module):
    """Scale each vector to unit root-mean-square, then by a learned per-feature gain (LayerNorm without the mean)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def swiglu_hidden(d_model: int) -> int:
    """Hidden width of a SwiGLU MLP: 8/3 of d_model (a 4x MLP's parameter count), rounded up to a multiple of 64."""
    return int(math.ceil(8 * d_model / 3 / 64) * 64)


class SwiGLU(nn.Module):
    """Gated MLP: down(silu(gate(x)) * up(x)). The gate decides how much of each hidden unit passes."""

    def __init__(self, d_model: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or swiglu_hidden(d_model)
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def rope_tables(head_dim: int, seq_len: int, theta: float = ROPE_THETA,
                device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of the rotation angle of every (position, feature pair): two (seq_len, head_dim / 2) tables."""
    freqs = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    angles = torch.outer(torch.arange(seq_len, device=device).float(), freqs)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each (even, odd) feature pair of x (B, H, T, D) by its position's angle (cos, sin: (T, D / 2))."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


# ---------------------------------------------------------------------- encoder-decoder attention (Brush GPT)
@dataclass
class SelfAttentionCache:
    """One decoder layer's keys and values of every position fed so far, in buffers `size` positions long."""

    size: int
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None
    length: int = 0

    def extend(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append the new positions' k, v (B, H, T, D) and return the keys and values of every position so far."""
        if self.keys is None:
            batch, heads, _, dim = k.shape
            self.keys = k.new_zeros(batch, heads, self.size, dim)
            self.values = v.new_zeros(batch, heads, self.size, dim)
        n, t = self.length, k.size(2)
        self.keys[:, :, n:n + t] = k
        self.values[:, :, n:n + t] = v
        self.length = n + t
        return self.keys[:, :, :n + t], self.values[:, :, :n + t]


@dataclass
class CrossAttentionCache:
    """One decoder layer's keys and values of the encoder output, projected once per generation."""

    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None


@dataclass
class DecoderLayerCache:
    """What one decoder layer keeps between generation steps."""

    self_attn: SelfAttentionCache
    cross_attn: CrossAttentionCache

    @classmethod
    def empty(cls, size: int) -> DecoderLayerCache:
        return cls(SelfAttentionCache(size), CrossAttentionCache())


class _RotaryAttention(nn.Module):
    """Projections shared by self- and cross-attention: queries, keys and values (one matrix), and the output."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads, self.dropout = n_heads, dropout
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def _queries(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return apply_rope(self.q(x).view(b, t, self.n_heads, -1).transpose(1, 2), cos, sin)

    def _keys_values(self, x: torch.Tensor, cos: torch.Tensor,
                     sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        k, v = self.kv(x).view(b, t, 2, self.n_heads, -1).permute(2, 0, 3, 1, 4)
        return apply_rope(k, cos, sin), v

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
        b, _, t, _ = q.shape
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, t, -1))


class SelfAttention(_RotaryAttention):
    """Multi-head self-attention with rotary positions; causal in the decoder, bidirectional in the encoder."""

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, causal: bool,
                cache: SelfAttentionCache | None = None) -> torch.Tensor:
        """x (B, T, d) -> (B, T, d). With a cache, x holds the newest positions and attends to every earlier one."""
        q = self._queries(x, cos, sin)
        k, v = self._keys_values(x, cos, sin)
        if cache is not None:
            k, v = cache.extend(k, v)
            causal = causal and x.size(1) > 1  # a single new position may see everything before it
        return self._attend(q, k, v, causal)


class CrossAttention(_RotaryAttention):
    """Multi-head attention from the decoder's positions to the encoder's output (`memory`), never causal."""

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, memory: torch.Tensor,
                memory_cos: torch.Tensor, memory_sin: torch.Tensor,
                cache: CrossAttentionCache | None = None) -> torch.Tensor:
        """x (B, T, d), memory (B, S, d) -> (B, T, d). cos/sin are x's rotary tables, memory_cos/sin the memory's."""
        q = self._queries(x, cos, sin)
        if cache is not None and cache.keys is not None:
            k, v = cache.keys, cache.values
        else:
            k, v = self._keys_values(memory, memory_cos, memory_sin)
            if cache is not None:
                cache.keys, cache.values = k, v
        return self._attend(q, k, v, causal=False)


class EncoderBlock(nn.Module):
    """Pre-norm encoder layer: bidirectional self-attention, then the MLP, each added to the residual stream."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(d_model), SelfAttention(d_model, n_heads, dropout)
        self.mlp_norm, self.mlp = RMSNorm(d_model), SwiGLU(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x), cos, sin, causal=False))
        return x + self.dropout(self.mlp(self.mlp_norm(x)))


class DecoderBlock(nn.Module):
    """Pre-norm decoder layer: causal self-attention, cross-attention to the encoder, then the MLP."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(d_model), SelfAttention(d_model, n_heads, dropout)
        self.cross_norm, self.cross_attn = RMSNorm(d_model), CrossAttention(d_model, n_heads, dropout)
        self.mlp_norm, self.mlp = RMSNorm(d_model), SwiGLU(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, memory: torch.Tensor,
                memory_cos: torch.Tensor, memory_sin: torch.Tensor,
                cache: DecoderLayerCache | None = None) -> torch.Tensor:
        self_cache = cross_cache = None
        if cache is not None:
            self_cache, cross_cache = cache.self_attn, cache.cross_attn
        x = x + self.dropout(self.attn(self.attn_norm(x), cos, sin, causal=True, cache=self_cache))
        x = x + self.dropout(self.cross_attn(self.cross_norm(x), cos, sin, memory, memory_cos, memory_sin,
                                             cross_cache))
        return x + self.dropout(self.mlp(self.mlp_norm(x)))


# ------------------------------------------------------------------------------------------- decoder-only GPT
@dataclass
class GPTConfig:
    """Size and shape of a decoder-only GPT; stored in every checkpoint (`to_dict`)."""

    vocab_size: int = 8192
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    max_seq_len: int = 1024
    rope_theta: float = ROPE_THETA
    dropout: float = 0.0  # on the attention weights, training only

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_dim(self) -> int:
        return swiglu_hidden(self.d_model)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GPTConfig:
        """A config from a checkpoint's dict. Older checkpoints also carry `abs_pos` (a learned absolute position
        embedding), which this model does not have; such checkpoints load only when it is off."""
        d = dict(d)
        if d.pop("abs_pos", False):
            raise ValueError("checkpoints with a learned absolute position embedding are not supported")
        return cls(**d)


@dataclass
class GrowingKVCache:
    """One GPT layer's keys and values of every position fed so far, grown by concatenation."""

    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None

    def extend(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys is not None:
            k = torch.cat([self.keys, k], dim=2)
            v = torch.cat([self.values, v], dim=2)
        self.keys, self.values = k, v
        return k, v


class GroupedQueryAttention(nn.Module):
    """Causal self-attention with rotary positions; each of the n_kv_heads key/value heads serves a group of
    n_heads / n_kv_heads query heads (Ainslie et al., 2023), which shrinks the cache kept during generation."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.kv = nn.Linear(cfg.d_model, 2 * cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.out = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cache: GrowingKVCache | None = None) -> torch.Tensor:
        """x (B, T, d) -> (B, T, d). With a cache, x holds the newest positions and attends to every earlier one."""
        b, t, _ = x.shape
        c = self.cfg
        q = self.q(x).view(b, t, c.n_heads, c.head_dim).transpose(1, 2)
        k, v = self.kv(x).view(b, t, 2, c.n_kv_heads, c.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.extend(k, v)
        if c.n_kv_heads != c.n_heads:
            k = k.repeat_interleave(c.n_heads // c.n_kv_heads, dim=1)
            v = v.repeat_interleave(c.n_heads // c.n_kv_heads, dim=1)
        causal = cache is None or t > 1
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=c.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, t, -1))


class GPTBlock(nn.Module):
    """One pre-norm GPT layer: attention, then the MLP, each added to the residual stream."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = GroupedQueryAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.ffn_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cache: GrowingKVCache | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        return x + self.mlp(self.mlp_norm(x))


def init_gpt2_style(model: nn.Module, n_layers: int) -> None:
    """GPT-2 initialisation: every linear and embedding weight from N(0, 0.02^2), then the two projections that write
    into the residual stream (attention `out`, MLP `down`) scaled down by sqrt(2 * n_layers)."""
    for m in model.modules():
        _init_normal(m)
    _scale_residual_projections(model, n_layers)


def _scale_residual_projections(model: nn.Module, n_layers: int) -> None:
    for name, p in model.named_parameters():
        if name.endswith("out.weight") or name.endswith("down.weight"):
            nn.init.normal_(p, std=0.02 / math.sqrt(2 * n_layers))


class GPT(nn.Module):
    """Decoder-only transformer: token embedding, the blocks, a final norm, and an output layer tied to the
    embedding (one table both reads and writes tokens)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(GPTBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(_init_normal)
        _scale_residual_projections(self, cfg.n_layers)

    def num_params(self) -> int:
        """Parameter count, the tied embedding and output table counted once."""
        return sum(p.numel() for p in self.parameters())

    def empty_cache(self) -> list[GrowingKVCache]:
        """A fresh key/value cache for generation, one entry per layer."""
        return [GrowingKVCache() for _ in self.blocks]

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, loss_mask: torch.Tensor | None = None,
                cache: list[GrowingKVCache] | None = None, start_pos: int = 0,
                masked_only: bool = False) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """idx (B, T) -> (logits (B, T, V), loss). The loss is the mean cross-entropy over the positions where
        loss_mask is 1 (every position without a mask), or None without targets.

        masked_only (with targets and a loss mask): the output layer runs only on the scored positions and no logits
        are returned; the same loss with far less memory when the scored part of each sequence is small.
        cache and start_pos (generation): idx holds the positions start_pos onward; earlier ones are in the cache."""
        t = idx.size(1)
        cos = self.rope_cos[start_pos:start_pos + t]
        sin = self.rope_sin[start_pos:start_pos + t]
        x = self.embed(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, None if cache is None else cache[i])
        if masked_only and targets is not None and loss_mask is not None:
            return None, self._masked_loss(x, targets, loss_mask)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        if loss_mask is None:
            return logits, loss.mean()
        mask = loss_mask.reshape(-1).float()
        return logits, (loss * mask).sum() / mask.sum().clamp(min=1)

    def _masked_loss(self, x: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """The masked mean cross-entropy computed on the scored positions only. Their count is padded with unscored
        ones to a multiple of MASKED_ROUND, so the accelerator sees a few shapes rather than a new one every batch."""
        weight = loss_mask.reshape(-1).float()
        k = min(weight.numel(), math.ceil(int((weight > 0).sum()) / MASKED_ROUND) * MASKED_ROUND)
        keep = torch.topk(weight, k).indices
        picked = self.head(self.norm(x.reshape(-1, x.size(-1))[keep]))
        loss = F.cross_entropy(picked, targets.reshape(-1)[keep], reduction="none")
        return (loss * weight[keep]).sum() / weight.sum().clamp(min=1)


def _init_normal(m: nn.Module) -> None:
    if isinstance(m, nn.Linear | nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)
