"""Brush GPT: an encoder-decoder transformer that writes a painting as a sequence of stroke tokens.

The encoder reads the condition, a handful of tokens (the task token, then the attributes: left eye colour, right
eye colour, base coat, coat pattern), with every token seeing every other. The decoder writes the painting causally,
one token at a time, each layer also attending to the encoder's output. Both use RMSNorm, SwiGLU and rotary
positions (`brushgpt.layers`).

A painting is a sequence of *slots* (marker tokens: sketch-plan outlines, parts, layers, the palette, EOS) and
*strokes* (x, y, angle, size, paint; a palette entry is a single colour token). The model never chooses among the
whole vocabulary: at every position it chooses among the tokens of one *choice set*:

    choice 0       a stroke may start here: one of the slot tokens or an x token
    choice c >= 1  stroke field c (1 y, 2 angle, 3 size, 4 paint, 5 palette colour)

The logits of a choice set are the decoder's hidden state times that set's rows of the (tied) embedding table.
Training uses the data's own field of every target to know its choice set; sampling asks a `Grammar` which set comes
next. `StrokeGrammar` is the grammar of the painter's token format; any object with the same three methods can
replace it.

Condition dropout and classifier-free guidance (Ho & Salimans, 2021): a model trained with some conditions replaced
by a "null" token (the last vocabulary id) can sample with logits uncond + g * (cond - uncond); g = 1 is the
conditional model alone, larger g follows the condition more closely.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch import nn

from .layers import DecoderBlock, DecoderLayerCache, EncoderBlock, RMSNorm, init_gpt2_style, rope_tables

STROKE_FIELDS = 5  # x, y, angle, size, paint: the fields of one stroke, in order
COLOUR_CHOICE = 5  # the choice set of a palette entry (after the five stroke fields)


@dataclass(frozen=True)
class TokenLayout:
    """Where each kind of token sits in the vocabulary.

    slots           the marker tokens that may stand where a stroke could start, EOS last
    field_offsets   first id of each field: x, y, angle, size, paint, and (with a palette) colour
    field_sizes     number of tokens of each field
    plan_markers    markers that open a sketch-plan outline: (x, y) points follow, not strokes
    palette         (marker, number of colour tokens after it) for each palette marker
    """

    bos: int
    slots: tuple[int, ...]
    field_offsets: tuple[int, ...]
    field_sizes: tuple[int, ...]
    plan_markers: tuple[int, ...] = ()
    palette: tuple[tuple[int, int], ...] = ()

    @property
    def eos(self) -> int:
        return self.slots[-1]

    @property
    def n_choices(self) -> int:
        """Choice 0 (slot or x), then one choice per field after x."""
        return len(self.field_offsets)

    def stroke_start_ids(self, device: torch.device | str) -> torch.Tensor:
        """The tokens of choice 0, in the order their logits are laid out: the slots, then every x token."""
        x0, nx = self.field_offsets[0], self.field_sizes[0]
        return torch.cat([torch.tensor(self.slots, device=device), torch.arange(x0, x0 + nx, device=device)])

    def target_choices(self, tokens: torch.Tensor, fields: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """For target tokens and their data fields (0 a marker, 1 x, 2 y, ... as numbered in the data): each
        target's choice set and its index among that set's tokens.

        A marker that is not one of `slots` gets index 0 (the first slot): it can be fed to the decoder but is
        never a target the model can write."""
        choice = torch.where(fields <= 1, torch.zeros_like(fields), fields - 1)
        offsets = torch.tensor(self.field_offsets, device=tokens.device)
        index = tokens - offsets[(fields - 1).clamp(min=0)]
        slots = torch.tensor(self.slots, device=tokens.device)
        is_slot = fields == 0
        index[is_slot] = (tokens[is_slot, None] == slots).float().argmax(-1)
        index[fields == 1] += len(self.slots)
        return choice, index


@dataclass
class GrammarState:
    """Where each painting being sampled is in the token format."""

    choice: torch.Tensor  # (B,) the choice set of the next token
    in_plan: torch.Tensor  # (B,) inside a sketch-plan outline: (x, y) points only
    palette_left: torch.Tensor  # (B,) colour tokens still to write after a palette marker


class Grammar(Protocol):
    """Which choice set each sampled painting draws its next token from."""

    def start(self, batch: int, device: torch.device | str) -> Any: ...

    def choice(self, state: Any) -> torch.Tensor: ...

    def advance(self, state: Any, tokens: torch.Tensor) -> Any: ...


class StrokeGrammar:
    """The painter's token format. Where a stroke may start (choice 0) an x token starts a stroke and y, angle, size,
    paint follow; a slot token leaves the next position at choice 0. After a plan-outline marker each point is (x, y)
    only, until the next slot token. After a palette marker come its colour tokens, then choice 0 again."""

    def __init__(self, layout: TokenLayout):
        self.layout = layout

    def start(self, batch: int, device: torch.device | str) -> GrammarState:
        zeros = torch.zeros(batch, dtype=torch.long, device=device)
        return GrammarState(zeros, torch.zeros(batch, dtype=torch.bool, device=device), zeros.clone())

    def choice(self, state: GrammarState) -> torch.Tensor:
        return state.choice

    def advance(self, state: GrammarState, tokens: torch.Tensor) -> GrammarState:
        lay, step, dev = self.layout, state.choice, tokens.device
        is_x = (step == 0) & (tokens >= lay.field_offsets[0])
        is_slot = (step == 0) & ~is_x
        plan = torch.tensor(lay.plan_markers, dtype=torch.long, device=dev)
        in_plan = torch.where(is_slot, torch.isin(tokens, plan), state.in_plan)
        in_palette = step == COLOUR_CHOICE
        nxt = torch.where(is_x, torch.ones_like(step), torch.where(step == 0, step, (step + 1) % STROKE_FIELDS))
        nxt = torch.where(in_plan & (nxt == 2), torch.zeros_like(nxt), nxt)  # an outline point is x, y only
        left = state.palette_left
        if lay.palette:
            markers = torch.tensor([m for m, _ in lay.palette], dtype=torch.long, device=dev)
            counts = torch.tensor([k for _, k in lay.palette], dtype=torch.long, device=dev)
            opens = is_slot & torch.isin(tokens, markers)
            left = torch.where(opens, counts[(tokens[:, None] == markers).float().argmax(1)], left - in_palette.long())
            nxt = torch.where(
                opens | (in_palette & (left > 0)),
                torch.full_like(nxt, COLOUR_CHOICE),
                torch.where(in_palette, torch.zeros_like(nxt), nxt),
            )
        return GrammarState(nxt, in_plan, left)


@dataclass
class BrushGPTConfig:
    """Size and shape of Brush GPT; stored in every checkpoint."""

    vocab_size: int  # including the null token when null_token is set
    d_model: int = 512
    n_heads: int = 8
    enc_layers: int = 4
    dec_layers: int = 3
    max_len: int = 16384  # longest position the rotary tables cover
    dropout: float = 0.0  # training only: embeddings, attention weights, residual branches
    null_token: bool = False  # the last vocabulary id means "no attributes" (condition dropout, guidance)

    @property
    def null_id(self) -> int | None:
        return self.vocab_size - 1 if self.null_token else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BrushGPT(nn.Module):
    def __init__(self, cfg: BrushGPTConfig, layout: TokenLayout):
        super().__init__()
        self.cfg, self.layout = cfg, layout
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_dropout = nn.Dropout(cfg.dropout)
        self.encoder = nn.ModuleList(EncoderBlock(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.enc_layers))
        self.encoder_norm = RMSNorm(cfg.d_model)
        self.decoder = nn.ModuleList(DecoderBlock(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.dec_layers))
        self.norm = RMSNorm(cfg.d_model)
        cos, sin = rope_tables(cfg.d_model // cfg.n_heads, cfg.max_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        init_gpt2_style(self, cfg.enc_layers + cfg.dec_layers)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------------------------------------ forward
    def encode(self, cond: torch.Tensor) -> torch.Tensor:
        """cond (B, C) -> the encoder's output (B, C, d)."""
        t = cond.size(1)
        x = self.embed_dropout(self.embed(cond))
        for block in self.encoder:
            x = block(x, self.cos[:t], self.sin[:t])
        return self.encoder_norm(x)

    def decode(
        self, tokens: torch.Tensor, memory: torch.Tensor, start: int = 0, cache: list[DecoderLayerCache] | None = None
    ) -> torch.Tensor:
        """The decoder's final hidden states (B, T, d) for tokens (B, T) at positions start, start + 1, ...;
        memory is the encoder's output. With a cache, earlier positions are taken from it."""
        t = tokens.size(1)
        cos, sin = self.cos[start : start + t], self.sin[start : start + t]
        memory_cos, memory_sin = self.cos[: memory.size(1)], self.sin[: memory.size(1)]
        x = self.embed_dropout(self.embed(tokens))
        for i, block in enumerate(self.decoder):
            x = block(x, cos, sin, memory, memory_cos, memory_sin, None if cache is None else cache[i])
        return self.norm(x)

    def choice_logits(self, h: torch.Tensor, choice: int) -> torch.Tensor:
        """Logits (..., n) of one choice set's n tokens for hidden states h (..., d), from the tied embedding."""
        if choice == 0:
            return h @ self.embed.weight[self.layout.stroke_start_ids(h.device)].T
        o, n = self.layout.field_offsets[choice], self.layout.field_sizes[choice]
        return h @ self.embed.weight[o : o + n].T

    # --------------------------------------------------------------------------------------------------- loss
    def loss(
        self, cond: torch.Tensor, tokens: torch.Tensor, fields: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One painting (cond (C,), tokens (L,), their data fields (L,)) with teacher forcing.

        Returns the mean cross-entropy over the L tokens and the closing EOS, each scored among its own choice set,
        and per target (L + 1,) its loss and whether the most likely token is the right one."""
        lay = self.layout
        targets = torch.cat([tokens, tokens.new_tensor([lay.eos])])
        target_fields = torch.cat([fields, fields.new_tensor([0])])
        n = targets.size(0)
        memory = self.encode(cond[None])
        h = self.decode(torch.cat([tokens.new_tensor([lay.bos]), tokens])[None], memory)[0]
        choice, index = lay.target_choices(targets, target_fields)
        total = h.new_zeros(())
        nll = torch.zeros(n, device=h.device)
        right = torch.zeros(n, dtype=torch.bool, device=h.device)
        for c in range(lay.n_choices):
            k = choice == c
            if not k.any():
                continue
            logits = self.choice_logits(h[k], c).float()
            ce = F.cross_entropy(logits, index[k], reduction="none")
            nll[k] = ce.detach()
            right[k] = logits.argmax(-1) == index[k]
            total = total + ce.sum()
        return total / n, nll, right

    # ----------------------------------------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        max_tokens: int,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        uncond: torch.Tensor | None = None,
        guidance: float = 1.0,
        grammar: Grammar | None = None,
    ) -> torch.Tensor:
        """Paint one painting per condition (cond (B, C)), token by token with a key/value cache, until EOS or
        max_tokens. Returns (B, max_tokens): the tokens up to and including EOS, then -1.

        Each token is drawn from its choice set's distribution at `temperature`; nothing is picked among samples.
        uncond (B, C), the same conditions with the attributes replaced by the null token, turns on classifier-free
        guidance with weight `guidance`; the conditional and unconditional rows run side by side and are fed the
        same token."""
        grammar = grammar or StrokeGrammar(self.layout)
        b, dev = cond.size(0), cond.device
        guided = uncond is not None and guidance != 1.0
        rows = torch.cat([cond, uncond]) if guided else cond
        cache = [DecoderLayerCache.empty(max_tokens + 8) for _ in self.decoder]
        memory = self.encode(rows)
        h = self.decode(torch.full((rows.size(0), 1), self.layout.bos, device=dev), memory, 0, cache)[:, -1]
        pos = 1
        eos = self.layout.eos
        out = torch.full((b, max_tokens), -1, dtype=torch.long, device=dev)
        start_ids = self.layout.stroke_start_ids(dev)
        state = grammar.start(b, dev)
        done = torch.zeros(b, dtype=torch.bool, device=dev)
        for i in range(max_tokens):
            tok = self._draw(
                h, grammar.choice(state), start_ids, (guidance if guided else None, temperature), generator
            )
            state = grammar.advance(state, tok)
            out[:, i] = torch.where(done, torch.full_like(tok, -1), tok)
            done |= tok == eos
            if bool(done.all()):
                break
            fed = torch.where(done, torch.full_like(tok, eos), tok)
            h = self.decode((torch.cat([fed, fed]) if guided else fed)[:, None], memory, pos, cache)[:, -1]
            pos += 1
        return out

    def _draw(
        self,
        h: torch.Tensor,
        choice: torch.Tensor,
        start_ids: torch.Tensor,
        how: tuple[float | None, float],
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """One token per painting from its choice set; how = (guidance or None, temperature). With guidance, h holds
        the conditional rows, then the unconditional ones. The choice sets present are drawn in increasing order,
        which fixes how the generator is used."""
        guidance, temperature = how
        b = choice.size(0)
        tok = torch.empty(b, dtype=torch.long, device=h.device)
        for c in choice.unique().tolist():
            k = choice == c
            logits = self.choice_logits(h[:b][k], c).float()
            if guidance is not None:
                free = self.choice_logits(h[b:][k], c).float()
                logits = free + guidance * (logits - free)
            pick = torch.multinomial((logits / temperature).softmax(-1), 1, generator=generator)[:, 0]
            tok[k] = start_ids[pick] if c == 0 else pick + self.layout.field_offsets[c]
        return tok


def drop_attributes(cond: torch.Tensor, null_id: int, keep: int = 1) -> torch.Tensor:
    """cond (..., C) with every token after the first `keep` (the task token) replaced by the null token."""
    out = cond.clone()
    out[..., keep:] = null_id
    return out


# ---------------------------------------------------------------------------------------------- checkpoints
_REFERENCE_KEYS = [
    (r"^enc\.", "encoder."),
    (r"^dec\.", "decoder."),
    (r"^enc_norm\.", "encoder_norm."),
    (r"\.n1\.", ".attn_norm."),
    (r"\.nc\.", ".cross_norm."),
    (r"\.xattn\.", ".cross_attn."),
    (r"\.n2\.", ".mlp_norm."),
]


def convert_reference_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename the parameters of a checkpoint written by the research code this package was extracted from:
    enc/dec -> encoder/decoder, n1 -> attn_norm, nc -> cross_norm, xattn -> cross_attn, n2 -> mlp_norm."""
    out = {}
    for key, value in state.items():
        for pattern, repl in _REFERENCE_KEYS:
            key = re.sub(pattern, repl, key)
        out[key] = value
    return out


def config_from_reference(d: dict[str, Any]) -> BrushGPTConfig:
    """BrushGPTConfig of a research-code checkpoint (an encoder-decoder writing its own layout)."""
    if d.get("arch") != "encdec" or d.get("layout") != "free":
        raise ValueError(f"not a Brush GPT checkpoint: arch={d.get('arch')}, layout={d.get('layout')}")
    return BrushGPTConfig(
        vocab_size=d["vocab"],
        d_model=d["d_model"],
        n_heads=d["n_heads"],
        enc_layers=d["enc_layers"],
        dec_layers=d["dec_layers"],
        max_len=d["max_len"],
        dropout=d.get("dropout", 0.0),
        null_token=d.get("cond_drop", 0.0) > 0,
    )


def load_checkpoint(checkpoint: dict[str, Any], layout: TokenLayout, device: torch.device | str = "cpu") -> BrushGPT:
    """A BrushGPT from a loaded checkpoint dict, in this package's format or the research code's."""
    if "arch" in checkpoint["config"]:
        cfg, state = config_from_reference(checkpoint["config"]), convert_reference_state_dict(checkpoint["model"])
    else:
        cfg, state = BrushGPTConfig(**checkpoint["config"]), checkpoint["model"]
    model = BrushGPT(cfg, layout).to(device)
    model.load_state_dict(state)
    return model
