"""The eye part model: a decoder-only GPT (`layers.GPT`) that paints one eye's six eye-stage layers after a given
prefix (`brushgpt.eye_data`): the side, the eye colour asked for, the outline's size, the block-in colour around the
eye, and the sketch plan's outline and pupil in the eye's own frame.

Training. The loss counts only the body (the strokes and EOS), never the prefix. For condition dropout a share of the
eyes have their colour replaced by NULL, which teaches the unconditional model that classifier-free guidance needs
(Ho and Salimans, 2022): logits = uncond + g * (cond - uncond); g = 1 is the conditional model alone. The validation
eyes are every VAL_EVERY-th training cat (by a hash of its name), painting A as it is; they choose `best.pt`, so the
held cats are only ever scored, never used to choose.

Sampling follows the stroke grammar: a layer marker, then x, y, angle, size and paint per stroke, layers in order,
then EOS. A token of the wrong field cannot be drawn; this is the format, not a choice among outputs. One sample per
eye, nothing picked.
"""

from __future__ import annotations

import json
import math
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO

import numpy as np
import torch
import torch.nn.functional as F

from .eye_data import OUTLINE_POINTS, PUPIL_POINTS
from .layers import GPT, GPTConfig

VAL_EVERY = 25
OUTLINE_AT = slice(6, 6 + 2 * OUTLINE_POINTS)  # the outline's (x, y) tokens in a prefix
PUPIL_MARK = 6 + 2 * OUTLINE_POINTS  # PUPIL or NO_PUPIL
PUPIL_AT = slice(PUPIL_MARK + 1, PUPIL_MARK + 1 + 2 * PUPIL_POINTS)
COLOUR_AT = 2  # the asked eye colour's position in a prefix


def resolve_device(name: str = "auto") -> torch.device:
    """cuda, mps or cpu; "auto" takes the first one available."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ------------------------------------------------------------------ data
class EyeData:
    """A built eye data file: the sequences, the vocabulary (read from the file) and which eyes train, validate and
    are held out."""

    def __init__(self, path: str | Path):
        z = np.load(path)
        self.tokens = z["tokens"].astype(np.int64)
        self.lengths = z["lengths"].astype(np.int64)
        self.specials = [str(s) for s in z["specials"]]
        self.sp = {n: i for i, n in enumerate(self.specials)}
        self.vocab, self.prefix = int(z["vocab"]), int(z["prefix"])
        self.fields = [str(f) for f in z["fields"]]
        o, self.range = len(self.specials), {}
        for f, n in zip(self.fields, (int(v) for v in z["sizes"]), strict=True):
            self.range[f] = (o, o + n)
            o += n
        if o != self.vocab:
            raise ValueError("the special tokens and fields do not add up to the vocabulary")
        self.cat_id = z["cat_id"]
        self.cats = [str(c) for c in z["cats"]]
        split = np.array([str(s) for s in z["split"]])[self.cat_id]
        self.version = np.array([str(v) for v in z["version"]])
        self.side, self.eye = z["side"], np.array([str(e) for e in z["eye"]])
        val_cat = np.array([zlib.crc32(c.encode()) % VAL_EVERY == 0 for c in self.cats])[self.cat_id]
        self.train = np.flatnonzero((split == "train") & ~val_cat)
        self.val = np.flatnonzero((split == "train") & val_cat & (self.version == "A"))
        self.held = np.flatnonzero((split == "held") & (self.version == "A"))  # as painted: no versions, no mirrors
        self.by_cat = {int(c): self.train[self.cat_id[self.train] == c] for c in np.unique(self.cat_id[self.train])}

    def batch(
        self, ix: np.ndarray, device: torch.device, cond_drop: float = 0.0, gen: np.random.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(inputs, targets, loss mask) of these eyes, padded to the longest; the mask counts the body only. With
        cond_drop, that share of the eyes have their colour replaced by NULL."""
        n = int(self.lengths[ix].max())
        t = torch.from_numpy(self.tokens[ix, :n].copy())
        if cond_drop:
            drop = torch.from_numpy(gen.random(len(ix)) < cond_drop)
            t[drop, COLOUR_AT] = self.sp["NULL"]
        pos = torch.arange(n - 1)[None]
        mask = (pos >= self.prefix - 1) & (pos < torch.from_numpy(self.lengths[ix])[:, None] - 1)
        return t[:, :-1].to(device), t[:, 1:].to(device), mask.to(device)

    def sample_ix(self, n: int, gen: np.random.Generator, balance_cats: bool) -> np.ndarray:
        """n training eyes: uniformly, or (balance_cats) a cat first, then one of its eyes."""
        if not balance_cats:
            return gen.choice(self.train, n)
        cats = list(self.by_cat)
        return np.array([gen.choice(self.by_cat[cats[k]]) for k in gen.integers(0, len(cats), n)])

    def prefixes(self, rows: np.ndarray, eye_as: str = "") -> torch.Tensor:
        """These eyes' prefixes; eye_as: every eye asked in that colour instead (the swap test)."""
        pre = torch.from_numpy(self.tokens[rows, : self.prefix].copy())
        if eye_as:
            pre[:, COLOUR_AT] = self.sp[f"EYE_{eye_as}"]
        return pre


def without_colour(prefix: torch.Tensor, null_id: int) -> torch.Tensor:
    """The prefixes with the asked colour replaced by NULL (the unconditional half of guidance)."""
    out = prefix.clone()
    out[:, COLOUR_AT] = null_id
    return out


def model_config(
    data: EyeData, d_model: int = 384, n_layers: int = 8, n_heads: int = 6, dropout: float = 0.1
) -> GPTConfig:
    """The GPT for this data: half as many key/value heads as query heads, positions for the longest eye."""
    return GPTConfig(
        vocab_size=data.vocab,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=max(1, n_heads // 2),
        max_seq_len=int(data.lengths.max()) + 8,
        dropout=dropout,
    )


# ------------------------------------------------------------------ training
@dataclass
class EyeTrainConfig:
    steps: int = 20000
    batch: int = 32
    lr: float = 6e-4
    warmup: int = 300
    weight_decay: float = 0.1
    cond_drop: float = 0.15
    balance_cats: bool = False
    eval_every: int = 500
    eval_limit: int = 0  # score only the first N validation and held eyes (a quick check)
    log_every: int = 50
    sample_every: int = 2000  # also sample validation eyes and check their geometry (0: never)
    sample_n: int = 16
    stop_after_min: float = 0.0
    seed: int = 0


def lr_at(step: int, steps: int, lr: float, warmup: int) -> float:
    """Linear warm-up, then a cosine down to a tenth of the peak."""
    if step < warmup:
        return lr * (step + 1) / warmup
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))


@torch.no_grad()
def mean_body_loss(
    model: GPT, data: EyeData, rows: np.ndarray, device: torch.device, batch: int, limit: int = 0
) -> float:
    """Mean loss per body token over these eyes (only the first `limit` with a limit)."""
    model.eval()
    rows = rows[:limit] if limit else rows
    total = count = 0.0
    for k in range(0, len(rows), batch):
        x, y, m = data.batch(rows[k : k + batch], device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="none")
        total += float((loss * m.reshape(-1)).sum())
        count += float(m.sum())
    model.train()
    return total / max(count, 1)


def save_checkpoint(path: Path, model: GPT, step: int, val: float, held: float) -> None:
    torch.save(
        {"model": model.state_dict(), "config": model.cfg.to_dict(), "step": step, "val": val, "held": held}, path
    )


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu", min_seq_len: int = 0) -> GPT:
    """A trained eye model, in evaluation mode; min_seq_len: room for at least that many positions."""
    ck = torch.load(path, map_location="cpu")
    cfg = GPTConfig.from_dict(ck["config"])
    cfg.max_seq_len = max(cfg.max_seq_len, min_seq_len)
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    return model.eval()


def train(
    data: EyeData, model_cfg: GPTConfig, cfg: EyeTrainConfig, out: str | Path, device: torch.device | str = "cpu"
) -> GPT:
    """Train from scratch with AdamW (no weight decay on norms), gradient clipping at 1 and float16 on CUDA. Writes
    info.json, log.jsonl (training loss every log_every steps; validation and held loss every eval_every steps) and
    best.pt (the lowest validation loss)."""
    device = torch.device(device)
    torch.manual_seed(cfg.seed)
    gen = np.random.default_rng(cfg.seed)
    model = GPT(model_cfg).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": rest, "weight_decay": 0.0}],
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    info = {
        "train": asdict(cfg),
        "config": model_cfg.to_dict(),
        "params": model.num_params(),
        "train_eyes": len(data.train),
        "val_eyes": len(data.val),
        "held_eyes": len(data.held),
        "train_cats": len(data.by_cat),
        "device": str(device),
    }
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    amp = device.type == "cuda"  # float16 with loss scaling (a T4 has no fast bfloat16)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, t0, seen = float("inf"), time.time(), 0
    with open(out / "log.jsonl", "a") as log:
        for step in range(cfg.steps + 1):
            if step % cfg.eval_every == 0 or step == cfg.steps:
                best = _evaluate(model, data, cfg, device, out, log, step, best, seen / max(time.time() - t0, 1e-9))
                if step == cfg.steps or (cfg.stop_after_min and time.time() - t0 > 60 * cfg.stop_after_min):
                    break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg.steps, cfg.lr, cfg.warmup)
            x, y, m = data.batch(data.sample_ix(cfg.batch, gen, cfg.balance_cats), device, cfg.cond_drop, gen)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                _, loss = model(x, y, m, masked_only=True)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            seen += int(m.sum())
            if step % cfg.log_every == 0:
                log.write(
                    json.dumps({"step": step, "loss": round(loss.item(), 5), "lr": round(opt.param_groups[0]["lr"], 7)})
                    + "\n"
                )
    return model


def _evaluate(
    model: GPT,
    data: EyeData,
    cfg: EyeTrainConfig,
    device: torch.device,
    out: Path,
    log: IO[str],
    step: int,
    best: float,
    tokens_per_s: float,
) -> float:
    """The validation and held losses now; best.pt when the validation loss is the lowest so far. Returns the
    lowest."""
    v = mean_body_loss(model, data, data.val, device, cfg.batch, cfg.eval_limit)
    h = mean_body_loss(model, data, data.held, device, cfg.batch, cfg.eval_limit)
    rec = {"step": step, "val": round(v, 5), "held": round(h, 5), "tokens_per_s": round(tokens_per_s)}
    if v < best:
        best = v
        save_checkpoint(out / "best.pt", model, step, v, h)
        rec["best"] = True
    if cfg.sample_every and step % cfg.sample_every == 0 and step:
        rec.update(geometry_check(model, data, device, cfg.sample_n, 2.0, cfg.seed))
    print(json.dumps(rec), flush=True)
    log.write(json.dumps(rec) + "\n")
    log.flush()
    return best


# ------------------------------------------------------------------ sampling
class EyeGrammar:
    """Which tokens may come next. States: 0 a layer just opened (x); 1-4 inside a stroke (y, angle, size, paint);
    5 the start (a layer marker); 6 a stroke ended (another stroke's x, a later layer's marker, or EOS). A layer's
    marker is never allowed again once that layer or a later one has been opened."""

    START, AFTER_STROKE = 5, 6

    def __init__(self, data: EyeData, device: torch.device | str):
        v = data.vocab
        self.layer_ids = torch.tensor([data.sp[n] for n in data.specials if n.startswith("LAYER_")], device=device)

        def mask(*fields: str, layers: bool = False, eos: bool = False) -> torch.Tensor:
            m = torch.zeros(v, dtype=torch.bool, device=device)
            for f in fields:
                a, b = data.range[f]
                m[a:b] = True
            if layers:
                m[self.layer_ids] = True
            if eos:
                m[data.sp["EOS"]] = True
            return m

        self.masks = torch.stack(
            [
                mask("x"),
                mask("y"),
                mask("angle"),
                mask("size"),
                mask("paint"),
                mask(layers=True),
                mask("x", layers=True, eos=True),
            ]
        )
        self.eos = data.sp["EOS"]
        self.layer_of = torch.full((v,), -1, dtype=torch.long, device=device)
        self.layer_of[self.layer_ids] = torch.arange(len(self.layer_ids), device=device)

    def allowed(self, state: torch.Tensor, layer: torch.Tensor) -> torch.Tensor:
        """(B, V): the tokens each sequence may draw next."""
        m = self.masks[state].clone()
        lay = self.layer_of[None, :]
        return m & ~((lay >= 0) & (lay <= layer[:, None]))

    def advance(self, state: torch.Tensor, layer: torch.Tensor, tok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        is_layer = self.layer_of[tok] >= 0
        layer = torch.where(is_layer, self.layer_of[tok], layer)
        nxt = torch.where(
            is_layer,
            torch.zeros_like(state),
            torch.where(
                state == 4,
                torch.full_like(state, self.AFTER_STROKE),
                torch.where(state == self.AFTER_STROKE, torch.ones_like(state), state + 1),
            ),
        )
        return nxt, layer


@torch.no_grad()
def generate(
    model: GPT,
    prefix: torch.Tensor,
    null_prefix: torch.Tensor | None,
    grammar: EyeGrammar,
    max_new: int,
    temperature: float,
    guidance: float,
    gen: torch.Generator,
) -> torch.Tensor:
    """Sample a body for each prefix (B, PREFIX), with a key/value cache; null_prefix: the same prefixes with the
    colour NULL, for guidance (both halves run as one batch). Sampling runs in float32. Returns (B, steps); a
    sequence that has ended is padded with EOS."""
    model.eval()
    b = prefix.size(0)
    guided = null_prefix is not None and guidance != 1.0
    cache = model.empty_cache()
    logits, _ = model(torch.cat([prefix, null_prefix]) if guided else prefix, cache=cache)
    pos = prefix.size(1)
    state = torch.full((b,), EyeGrammar.START, dtype=torch.long, device=prefix.device)
    layer = torch.full((b,), -1, dtype=torch.long, device=prefix.device)
    done = torch.zeros(b, dtype=torch.bool, device=prefix.device)
    out = []
    for _ in range(max_new):
        lg = logits[:, -1].float()
        if guided:
            lg = lg[b:] + guidance * (lg[:b] - lg[b:])
        lg = lg.masked_fill(~grammar.allowed(state, layer), float("-inf")) / max(temperature, 1e-6)
        tok = torch.multinomial(lg.softmax(-1), 1, generator=gen)[:, 0]
        tok = torch.where(done, torch.full_like(tok, grammar.eos), tok)
        out.append(tok)
        done |= tok == grammar.eos
        if bool(done.all()) or pos + 1 >= model.cfg.max_seq_len:
            break
        state, layer = grammar.advance(state, layer, tok)
        step = tok[:, None]
        logits, _ = model(torch.cat([step, step]) if guided else step, cache=cache, start_pos=pos)
        pos += 1
    return torch.stack(out, 1)


@dataclass
class EyeSamples:
    """Sampled eyes: each one's row in the data file, its prefix and its body (padded with EOS)."""

    rows: np.ndarray
    prefix: np.ndarray
    bodies: np.ndarray
    guidance: float
    temperature: float
    seed: int
    eye_as: str = ""

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **asdict(self))

    @classmethod
    def load(cls, path: str | Path) -> EyeSamples:
        z = np.load(path)
        return cls(
            z["rows"],
            z["prefix"],
            z["bodies"],
            float(z["guidance"]),
            float(z["temperature"]),
            int(z["seed"]),
            str(z["eye_as"]) if "eye_as" in z.files else "",
        )


def sample_held(
    model: GPT,
    data: EyeData,
    n_cats: int = 32,
    guidance: float = 1.0,
    temperature: float = 1.0,
    eye_as: str = "",
    max_new: int = 1200,
    batch: int = 32,
    seed: int = 0,
) -> EyeSamples:
    """Paint the first n_cats held cats' eyes (0: every held eye), one sample per eye, `batch` eyes at a time from
    one random generator. eye_as: every eye asked in that colour (the swap test)."""
    device = model.embed.weight.device
    rows = data.held[: 2 * n_cats] if n_cats else data.held
    pre = data.prefixes(rows, eye_as)
    null = without_colour(pre, data.sp["NULL"])
    grammar = EyeGrammar(data, device)
    gen = torch.Generator(device=device).manual_seed(seed)
    bodies: list[np.ndarray] = []
    for k in range(0, len(rows), batch):
        got = generate(
            model,
            pre[k : k + batch].to(device),
            null[k : k + batch].to(device),
            grammar,
            max_new,
            temperature,
            guidance,
            gen,
        )
        bodies += [r.cpu().numpy() for r in got]
    width = max(len(b) for b in bodies)
    padded = np.stack([np.pad(b, (0, width - len(b)), constant_values=data.sp["EOS"]) for b in bodies])
    return EyeSamples(rows, pre.numpy(), padded, guidance, temperature, seed, eye_as)


# ------------------------------------------------------------------ the geometry check
def inside(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Which points lie inside a polygon (ray casting)."""
    x, y = pts[:, 0:1], pts[:, 1:2]
    x0, y0 = poly[:, 0][None], poly[:, 1][None]
    x1, y1 = np.roll(poly[:, 0], -1)[None], np.roll(poly[:, 1], -1)[None]
    cross = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / np.where(y1 == y0, 1e-9, y1 - y0) + x0)
    return cross.sum(1) % 2 == 1


def stroke_bins(body: np.ndarray, data: EyeData) -> tuple[np.ndarray, np.ndarray]:
    """A body's stroke positions (eye-frame bins) and each stroke's layer."""
    x0, y0 = data.range["x"][0], data.range["y"][0]
    xy, lay, j, i = [], [], -1, 0
    while i < len(body) and body[i] != data.sp["EOS"]:
        n = data.specials[body[i]] if body[i] < len(data.specials) else ""
        if n.startswith("LAYER_"):
            j, i = int(n[6:]), i + 1
            continue
        if i + 5 <= len(body) and data.range["x"][0] <= body[i] < data.range["x"][1]:
            xy.append((body[i] - x0, body[i + 1] - y0))
            lay.append(j)
        i += 5
    return np.array(xy, np.float64).reshape(-1, 2), np.array(lay)


def geometry(prefix: np.ndarray, body: np.ndarray, truth: np.ndarray, data: EyeData) -> dict[str, float]:
    """A sampled eye against the painter's, in eye-frame bins: strokes (sample / painter), the share of stroke
    positions inside the prefix's outline, and whether a pupils-layer stroke lies inside the prefix's pupil."""
    off = [data.range["x"][0], data.range["y"][0]]
    outline = prefix[OUTLINE_AT].reshape(-1, 2) - off
    pupil = prefix[PUPIL_AT].reshape(-1, 2) - off
    xy, lay = stroke_bins(body, data)
    n_true = len(stroke_bins(truth, data)[0])
    pup = xy[lay == 3]
    no_pupil = prefix[PUPIL_MARK] == data.sp["NO_PUPIL"]
    return {
        "strokes_ratio": len(xy) / max(n_true, 1),
        "inside": float(inside(xy, outline).mean()) if len(xy) else 0.0,
        "pupil_in_pupil": float("nan") if no_pupil else float(len(pup) > 0 and inside(pup, pupil).any()),
    }


@torch.no_grad()
def geometry_check(
    model: GPT, data: EyeData, device: torch.device, n: int, guidance: float, seed: int
) -> dict[str, float]:
    """Sample n validation eyes (one each, temperature 1) and average `geometry` against the painter's."""
    rows = data.val[:n]
    pre = data.prefixes(rows).to(device)
    out = (
        generate(
            model,
            pre,
            without_colour(pre, data.sp["NULL"]),
            EyeGrammar(data, device),
            int(data.lengths.max()) - data.prefix + 8,
            1.0,
            guidance,
            torch.Generator(device=device).manual_seed(seed),
        )
        .cpu()
        .numpy()
    )
    model.train()
    g = [
        geometry(data.tokens[r, : data.prefix], out[k], data.tokens[r, data.prefix : data.lengths[r]], data)
        for k, r in enumerate(rows)
    ]
    return {f"sample_{k}": round(float(np.nanmean([x[k] for x in g])), 3) for k in g[0]}
