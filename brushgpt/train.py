"""Training and held-out sampling for Brush GPT.

The data file (.npz) holds one row per painting: its tokens and their fields (padded with -1), its condition tokens
(left eye colour, right eye colour, base coat, coat pattern), its cat and version. Splits are by cat, so a held cat's
paintings are never trained on. A training cat may have several paintings (the original, its mirror image, eye-colour
versions); a held cat is scored on its original painting only.

Each optimiser step accumulates the gradients of `accum` paintings drawn at random (one painting per forward pass;
paintings are several thousand tokens long). With `balance_cats` a cat is drawn first, then one of its paintings, so a
cat with many versions is not seen more often. With `cond_drop` a painting's attributes are replaced by the null
token that often, which teaches the unconditional model used by classifier-free guidance.

The learning rate warms up linearly, then follows a cosine down to a tenth of its peak. Every `eval_every` steps the
held loss is measured; `best.pt` keeps the weights with the lowest held loss.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .model import BrushGPT, BrushGPTConfig, TokenLayout, drop_attributes

SLOT_PREFIXES = ("LAYER", "PART_", "PLAN_", "POSE_", "PAL_")
EYE_LAYERS = range(2, 8)  # LAYER2 ... LAYER7: the layers of the painter's eye stage
SWAP_TEST_PAINTINGS = 32


# --------------------------------------------------------------------------------------------------------- data
def layout_from_specials(specials: list[str], field_names: list[str], field_sizes: list[int],
                         palette_k: tuple[int, int] = (0, 0)) -> TokenLayout:
    """The TokenLayout of a vocabulary whose first ids are the named special tokens, followed by each field's tokens.
    palette_k: the number of colours after the whole-face palette marker and after each eye's palette marker."""
    index = {n: i for i, n in enumerate(specials)}
    slots = [i for i, n in enumerate(specials) if n.startswith(SLOT_PREFIXES) or n in ("EYE_L", "EYE_R")]
    offsets, o = [], len(specials)
    for size in field_sizes:
        offsets.append(o)
        o += size
    plan = [i for i, n in enumerate(specials) if n.startswith("PLAN_") and n not in ("PLAN_POSE", "PLAN_END")]
    k_all, k_eye = palette_k
    palette = [(index[n], k) for n, k in (("PAL_ALL", k_all), ("PAL_EYE_L", k_eye), ("PAL_EYE_R", k_eye))
               if n in index]
    if len(field_names) != len(field_sizes):
        raise ValueError("one size per field")
    return TokenLayout(bos=index["BOS"], slots=(*slots, index["EOS"]), field_offsets=tuple(offsets),
                       field_sizes=tuple(int(s) for s in field_sizes), plan_markers=tuple(plan),
                       palette=tuple(palette))


class PaintingData:
    """A built data file, read into memory; `example(i)` returns one painting as tensors on `device`."""

    def __init__(self, path: str | Path, device: torch.device | str = "cpu"):
        z = np.load(path)
        self.device = device
        self.tokens, self.fields, self.cond = z["tokens"], z["fields"], z["cond"]
        self.cat_id, self.versions = z["cat_id"], z["versions"]
        split = [str(s) for s in z["split"]]  # per cat
        self.specials = [str(s) for s in z["specials"]]
        self.vocab = int(z["vocab"])
        palette_k = tuple(int(v) for v in z["palette_k"]) if "palette_k" in z.files else (0, 0)
        self.layout = layout_from_specials(self.specials, [str(f) for f in z["field_names"]],
                                           [int(s) for s in z["field_sizes"]], palette_k)
        if self.layout.field_offsets[-1] + self.layout.field_sizes[-1] != self.vocab:
            raise ValueError("the special tokens and fields do not add up to the vocabulary")
        self.task_token = self.specials.index("TASK_GEN") if "TASK_GEN" in self.specials else None
        kind = [str(k) for k in z["kind"]] if "kind" in z.files else ["gen"] * len(self.tokens)
        rows = [i for i in range(len(self.tokens)) if kind[i] == "gen"]
        self.train = [i for i in rows if split[self.cat_id[i]] == "train"]
        self.held = [i for i in rows if split[self.cat_id[i]] == "held" and str(self.versions[i]) == "A"]
        self.train_by_cat: dict[int, list[int]] = {}
        for i in self.train:
            self.train_by_cat.setdefault(int(self.cat_id[i]), []).append(i)
        self._init_swap_test()

    @property
    def n_attributes_kept(self) -> int:
        """Leading condition tokens that condition dropout keeps (the task token)."""
        return 1 if self.task_token is not None else 0

    def longest(self) -> int:
        return int((self.tokens >= 0).sum(1).max())

    def example(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(condition, tokens, fields) of painting i; the condition starts with the generation task token."""
        t = self.tokens[i]
        n = int((t >= 0).sum())
        tokens = torch.from_numpy(t[:n].astype(np.int64)).to(self.device)
        fields = torch.from_numpy(self.fields[i][:n].astype(np.int64)).to(self.device)
        cond = torch.from_numpy(self.cond[i].astype(np.int64)).to(self.device)
        if self.task_token is not None:
            cond = torch.cat([cond.new_tensor([self.task_token]), cond])
        return cond, tokens, fields

    # The swap test asks whether the model uses its eye-colour condition: the loss of the eye layers' paint tokens
    # with both eye colours replaced by another colour, minus the loss with the true ones (near 0: unused).
    def _init_swap_test(self) -> None:
        sp = self.specials
        self.eye_layers = [sp.index(f"LAYER{k}") for k in EYE_LAYERS if f"LAYER{k}" in sp]
        colours = [i for i, n in enumerate(sp) if n.startswith("COND_")
                   and not n.startswith(("COND_BASE_", "COND_PAT_")) and n != "COND_unclear"]
        self.colour_swap = dict(zip(colours, colours[1:] + colours[:1], strict=True))
        if "COND_unclear" in sp and colours:
            self.colour_swap[sp.index("COND_unclear")] = colours[0]
        self._markers = [i for i, n in enumerate(sp) if n.startswith(("LAYER", "PAL_"))]
        self._eye_markers = self.eye_layers + [sp.index(n) for n in ("PAL_EYE_L", "PAL_EYE_R") if n in sp]

    def eyes_swapped(self, cond: torch.Tensor) -> torch.Tensor:
        """cond with both eye colours replaced by another colour."""
        c = cond.clone()
        o = self.n_attributes_kept
        for e in (o, o + 1):
            c[e] = self.colour_swap.get(int(c[e]), int(c[e]))
        return c

    def eye_paint(self, tokens: torch.Tensor, fields: torch.Tensor) -> torch.Tensor:
        """Which positions carry an eye's colour: the paint token of a stroke in an eye layer, and each colour of an
        eye's palette."""
        pos = torch.arange(len(tokens), device=tokens.device)
        markers = torch.tensor(self._markers, device=tokens.device)
        eyes = torch.tensor(self._eye_markers, device=tokens.device)
        is_marker = torch.isin(tokens, markers) & (fields == 0)
        last = torch.cummax(torch.where(is_marker, pos, torch.full_like(pos, -1)), 0).values
        return ((fields == 5) | (fields == 6)) & (last >= 0) & torch.isin(tokens[last.clamp(min=0)], eyes)


# ------------------------------------------------------------------------------------------------------ training
@dataclass
class TrainConfig:
    steps: int = 3000
    accum: int = 8                 # paintings per optimiser step
    lr: float = 3e-4
    warmup: int = 200
    weight_decay: float = 0.1
    cond_drop: float = 0.0         # share of paintings trained with their attributes replaced by the null token
    balance_cats: bool = False     # draw a cat first, then one of its paintings
    eval_every: int = 250
    eval_paintings: int = 60       # held paintings scored per evaluation
    stop_after_min: float = 0.0    # stop (and save last.pt) after this many minutes; 0: run every step
    seed: int = 0


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warm-up to cfg.lr, then a cosine from cfg.lr down to cfg.lr / 10 at the last step."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * 0.1 + 0.9 * cfg.lr * 0.5 * (1 + math.cos(math.pi * p))


def autocast_for(device: torch.device | str) -> Callable[[], AbstractContextManager]:
    """float16 autocast on CUDA, plain float32 elsewhere."""
    if torch.device(device).type == "cuda":
        return lambda: torch.autocast("cuda", dtype=torch.float16)
    return lambda: torch.autocast("cpu", enabled=False)


def evaluate(model: BrushGPT, data: PaintingData, items: list[int],
             amp: Callable[[], AbstractContextManager]) -> dict[str, float]:
    """Held loss and argmax accuracy over every token of each painting and its EOS, and (on the first
    SWAP_TEST_PAINTINGS) the swap test: eye_paint_nll with the true eye colours, eye_swap_gap the increase with them
    swapped."""
    model.eval()
    right = n = 0
    nll = eye_true = eye_swap = 0.0
    n_eye = 0
    with torch.no_grad(), amp():
        for k, i in enumerate(items):
            cond, tokens, fields = data.example(i)
            _, token_nll, token_right = model.loss(cond, tokens, fields)
            right += int(token_right.sum())
            n += len(token_nll)
            nll += float(token_nll.sum())
            if k < SWAP_TEST_PAINTINGS and data.eye_layers:
                m = torch.cat([data.eye_paint(tokens, fields), torch.zeros(1, dtype=torch.bool, device=tokens.device)])
                _, swapped_nll, _ = model.loss(data.eyes_swapped(cond), tokens, fields)
                eye_true += float(token_nll[m].sum())
                eye_swap += float(swapped_nll[m].sum())
                n_eye += int(m.sum())
    model.train()
    out = {"gen_acc": round(right / max(n, 1), 5), "gen_nll": round(nll / max(n, 1), 5), "gen_tokens": n}
    if n_eye:
        out |= {"eye_paint_nll": round(eye_true / n_eye, 5), "eye_swap_gap": round((eye_swap - eye_true) / n_eye, 5)}
    return out


class _PaintingSampler:
    """Draws training paintings: uniformly, or a cat first and then one of its paintings."""

    def __init__(self, data: PaintingData, balance_cats: bool, seed: int):
        self.data, self.balance_cats = data, balance_cats
        self.cats = sorted(data.train_by_cat)
        self.rng = np.random.default_rng(seed)

    def next(self) -> int:
        self.rng.random()  # one uniform draw per painting precedes the choice; it fixes the order a seed gives
        if self.balance_cats:
            paintings = self.data.train_by_cat[self.cats[self.rng.integers(len(self.cats))]]
            return paintings[self.rng.integers(len(paintings))]
        return self.data.train[self.rng.integers(len(self.data.train))]

    def drop_condition(self, p: float) -> bool:
        return p > 0 and self.rng.random() < p


def save_checkpoint(path: Path, model: BrushGPT, train_cfg: TrainConfig, step: int) -> None:
    torch.save({"config": model.cfg.to_dict(), "train": asdict(train_cfg), "model": model.state_dict(),
                "step": step}, path)


def train(data: PaintingData, model_cfg: BrushGPTConfig, cfg: TrainConfig, out: str | Path,
          device: torch.device | str = "cpu") -> BrushGPT:
    """Train Brush GPT on data's training paintings; write info.json, log.jsonl, last.pt and best.pt to out."""
    if cfg.cond_drop > 0 and model_cfg.null_id is None:
        raise ValueError("condition dropout needs a model with a null token")
    torch.manual_seed(cfg.seed)
    model = BrushGPT(model_cfg, data.layout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    use_amp = torch.device(device).type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp = autocast_for(device)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sampler = _PaintingSampler(data, cfg.balance_cats, cfg.seed)
    held = [data.held[i] for i in np.random.default_rng(1).permutation(len(data.held))[:cfg.eval_paintings]]
    info = {"config": model_cfg.to_dict(), "train": asdict(cfg), "params": model.num_params(),
            "train_paintings": len(data.train), "train_cats": len(sampler.cats), "held_paintings": len(data.held),
            "eval_paintings": len(held), "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if use_amp else None}
    print(json.dumps(info), flush=True)
    (out / "info.json").write_text(json.dumps(info, indent=1))
    best = math.inf
    t0, seen = time.time(), 0
    with open(out / "log.jsonl", "a") as log:
        for step in range(cfg.steps):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg)
            total = 0.0
            for _ in range(cfg.accum):
                cond, tokens, fields = data.example(sampler.next())
                if sampler.drop_condition(cfg.cond_drop):
                    cond = drop_attributes(cond, model_cfg.null_id, data.n_attributes_kept)
                with amp():
                    loss, _, _ = model.loss(cond, tokens, fields)
                scaler.scale(loss / cfg.accum).backward()
                total += loss.item() / cfg.accum
                seen += len(tokens)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            row = {"step": step + 1, "loss": round(total, 5), "lr": lr_at(step, cfg),
                   "seconds": round(time.time() - t0, 1)}
            if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
                row |= evaluate(model, data, held, amp)
                row["tokens_per_s"] = round(seen / (time.time() - t0), 1)
                save_checkpoint(out / "last.pt", model, cfg, step + 1)
                if row["gen_nll"] < best:
                    best = row["gen_nll"]
                    shutil.copyfile(out / "last.pt", out / "best.pt")
                    row["best"] = True
                print(json.dumps(row), flush=True)
            if (step + 1) % 10 == 0 or "gen_nll" in row:
                log.write(json.dumps(row) + "\n")
                log.flush()
            if cfg.stop_after_min and time.time() - t0 > 60 * cfg.stop_after_min:
                save_checkpoint(out / "last.pt", model, cfg, step + 1)
                print(json.dumps({"stopped_at_step": step + 1, "minutes": round((time.time() - t0) / 60, 1)}),
                      flush=True)
                break
    return model


# ------------------------------------------------------------------------------------------------------ sampling
def sample_held(model: BrushGPT, data: PaintingData, n_cats: int, batch: int = 8, temperature: float = 1.0,
                guidance: float = 1.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Paint the first n_cats held cats (in cat order) from their own conditions, one painting each, `batch` at a
    time, up to 1.3 times the data's longest painting. Returns (tokens (n, length) padded with -1, data rows)."""
    if guidance != 1.0 and model.cfg.null_id is None:
        raise ValueError("guidance needs a model trained with condition dropout")
    device = model.embed.weight.device
    items = sorted(data.held, key=lambda i: int(data.cat_id[i]))[:n_cats]
    max_tokens = int(1.3 * data.longest()) + 64
    generator = torch.Generator(device=device).manual_seed(seed)
    out: list[np.ndarray] = []
    model.eval()
    for k in range(0, len(items), batch):
        cond = torch.stack([data.example(i)[0] for i in items[k:k + batch]])
        uncond = drop_attributes(cond, model.cfg.null_id, data.n_attributes_kept) if guidance != 1.0 else None
        with torch.autocast("cuda", dtype=torch.float16, enabled=cond.is_cuda):
            got = model.sample(cond, max_tokens, temperature, generator, uncond, guidance)
        out += list(got.cpu().numpy())
    width = max(data.tokens.shape[1], max_tokens)
    tokens = np.array([np.pad(r, (0, width - len(r)), constant_values=-1) for r in out], np.int32)
    return tokens, np.array(items)
