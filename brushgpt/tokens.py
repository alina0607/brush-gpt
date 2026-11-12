"""Paintings as token sequences: the vocabulary, encoding a painting, and decoding tokens back to layers of strokes.

A painting is the painter's 21 layers (`painter.STAGES`; stage 2 per eye). Its sequence is, in order:

  plan      (optional) the sketch plan the painter painted from: PLAN_POSE and the head's yaw, pitch and roll
            (POSE_k, 16 bins over -90..90 degrees), then each outline as its marker (PLAN_HEAD, PLAN_EAR, PLAN_EYE,
            PLAN_PUPIL, PLAN_NOSE, PLAN_MOUTH) and its points as (x, y) tokens, then PLAN_END
  palette   (optional) PAL_ALL and 16 colour tokens: the colours of every stroke outside the eyes, by k-means
  strokes   either layer by layer (LAYER<k> and its strokes; stage 2 as EYE_L, the left eye's layers, EYE_R, the
            right's), or part by part (PART_<name>, then each of the painter's layers that painted there, in painting
            order, as LAYER<k> and its strokes there)

A stroke is five tokens: x and y (256 bins over the 512 px canvas), angle (32 over the full turn), size (6 x 6
log-spaced bins of length and width over the layer's range) and paint (12^3 colour bins x 8 opacity bins over the
layer's range). With a palette, paint is a slot x opacity instead: slots 0-15 are the palette's colours, and an eye
stroke uses slot 16 + its own colour bin, so an eye's colour is never rounded to the fur's palette.

Part order follows how a painter finishes one part before the next: ears, forehead, eyes, nose, cheeks, mouth, chin.
A stroke belongs to an ear when it starts inside that ear's triangle, otherwise to the nearest part anchor (set from
the eyes and the mouth). Inside a part, strokes go in serpentine rows 32 px tall; an eye layer's strokes go round the
eye from straight up, clockwise, and outwards; the ear's back, fill and rim keep the painter's own order (they are
chains along the ear's edge). Only the order changes: decoding puts every stroke back in its layer.

The vocabulary is the special tokens (markers, condition and task tokens), then each field's tokens in the order
x, y, angle, size, paint (and colour, with a palette). The condition tokens are the attribute names of
`brushgpt.attributes`; the task tokens are kept so that every build shares the same ids.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import numpy as np
import torch
from scipy.cluster.vq import kmeans2, vq

from .attributes import COAT_BASES, COAT_PATTERNS, EYE_COLOURS, UNCLEAR
from .brushes import StrokeParams
from .cat import INPUT_SIDE, ear_triangles
from .painter import EYE_STAGE, SIDE, STAGES, replay

# bins
NXY, NANGLE, NSIZE, NRGB, NALPHA = 256, 32, 6, 12, 8
FIELDS = ("x", "y", "angle", "size", "paint")
SIZES = {"x": NXY, "y": NXY, "angle": NANGLE, "size": NSIZE * NSIZE, "paint": NRGB ** 3 * NALPHA}
LAYERS = [(k, j) for k in sorted(STAGES) for j in range(len(STAGES[k]))]          # (stage, layer), 21
TASKS = ("gen", "eyes", "coat")

# part order
PARTS = ("ear L", "ear R", "forehead", "eye L", "eye R", "nose", "cheek L", "cheek R", "mouth", "chin")
PART_ROW = 32                   # px: inside a part, strokes go in serpentine rows this tall
CHAIN_LAYERS = {(7, 0), (7, 1), (7, 2)}   # ear back, ear fill, ear rim: chains along the ear, kept in painting order
# part anchors outside the ears: (from which point, eye distances towards the mouth, half eye distances sideways)
ANCHORS = {"forehead": ("mid", -0.9, 0.0), "eye L": ("eye L", 0.0, 0.0), "eye R": ("eye R", 0.0, 0.0),
           "nose": ("mid", 0.55, 0.0), "cheek L": ("eye L", 0.9, -0.6), "cheek R": ("eye R", 0.9, 0.6),
           "mouth": ("mouth", 0.0, 0.0), "chin": ("mouth", 0.6, 0.0)}
EAR_POINTS = ((3, 4, 5), (6, 7, 8))       # each ear's face points

# sketch plan
POSE_BINS = 16
PLAN_RUNS = ("HEAD", "EAR", "EYE", "PUPIL", "NOSE", "MOUTH")
PUPIL_POINTS = 6

# palette
PALETTE_K = 16
PALETTE_SLOTS = PALETTE_K + NRGB ** 3
LUMA = np.array([0.299, 0.587, 0.114])

BASE_SPECIALS = (["BOS", "EOS", "SEP", "EYE_L", "EYE_R"] + [f"LAYER{i}" for i in range(len(LAYERS))]
                 + [f"COND_{n}" for n in EYE_COLOURS] + [f"COND_BASE_{n}" for n in (UNCLEAR, *COAT_BASES)]
                 + [f"COND_PAT_{n}" for n in (UNCLEAR, *COAT_PATTERNS)] + [f"TASK_{k.upper()}" for k in TASKS])
PART_SPECIALS = [f"PART_{p.replace(' ', '_').upper()}" for p in PARTS]
PLAN_SPECIALS = (["PLAN_POSE"] + [f"POSE_{k}" for k in range(POSE_BINS)] + [f"PLAN_{r}" for r in PLAN_RUNS]
                 + ["PLAN_END"])
PALETTE_SPECIALS = ["PAL_ALL"]

PaintedLayer = tuple[int, int, StrokeParams, "int | None"]   # (stage, layer j, strokes, eye side or None)
Ranges = dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]  # per layer: lowest and highest (length, width, opacity)


# ------------------------------------------------------------------ the vocabulary
@dataclass(frozen=True)
class Vocabulary:
    """A build's vocabulary: the special tokens, then the stroke fields (with a palette: paint is slot x opacity and
    a sixth field, colour, writes the palette's colours)."""

    specials: tuple[str, ...]

    @classmethod
    def build(cls, part_order: bool = False, plan: bool = False, palette: bool = False) -> Vocabulary:
        """The base specials, then the part markers, the plan's tokens and the palette marker when asked (appended,
        so the condition and task ids are the same in every build)."""
        return cls(tuple(BASE_SPECIALS + (PART_SPECIALS if part_order else []) + (PLAN_SPECIALS if plan else [])
                         + (PALETTE_SPECIALS if palette else [])))

    @cached_property
    def ids(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.specials)}

    @property
    def has_palette(self) -> bool:
        return "PAL_ALL" in self.ids

    @property
    def has_plan(self) -> bool:
        return "PLAN_POSE" in self.ids

    @property
    def has_parts(self) -> bool:
        return PART_SPECIALS[0] in self.ids

    @cached_property
    def field_sizes(self) -> dict[str, int]:
        if not self.has_palette:
            return dict(SIZES)
        return {**SIZES, "paint": PALETTE_SLOTS * NALPHA, "colour": NRGB ** 3}

    @property
    def field_names(self) -> list[str]:
        return list(self.field_sizes)

    @cached_property
    def offsets(self) -> dict[str, int]:
        """Each field's first token id."""
        out, o = {}, len(self.specials)
        for f, n in self.field_sizes.items():
            out[f] = o
            o += n
        return out

    @property
    def size(self) -> int:
        return len(self.specials) + sum(self.field_sizes.values())

    @property
    def palette_k(self) -> np.ndarray:
        """Colours after the whole-face palette marker and after each eye's palette marker (none)."""
        return np.array([PALETTE_K if self.has_palette else 0, 0])

    def condition(self, names: list[str]) -> list[int]:
        """Condition token ids of attribute names (left eye, right eye, base coat, coat pattern)."""
        left, right, base, pattern = names
        return [self.ids[f"COND_{left}"], self.ids[f"COND_{right}"], self.ids[f"COND_BASE_{base}"],
                self.ids[f"COND_PAT_{pattern}"]]

    def condition_names(self, ids: list[int]) -> list[str]:
        """condition() backwards."""
        return [self.specials[i].removeprefix("COND_").removeprefix("BASE_").removeprefix("PAT_") for i in ids]


# ------------------------------------------------------------------ bins
def bin_index(v: np.ndarray, lo: float, hi: float, n: int, log: bool = False) -> np.ndarray:
    """Values to n bins over [lo, hi] (log-spaced with `log`); outside values go to the end bins."""
    if hi <= lo:
        return np.zeros(len(v), np.int64)
    f = (np.log(np.maximum(v, 1e-9)) - np.log(lo)) / (np.log(hi) - np.log(lo)) if log else (v - lo) / (hi - lo)
    return np.clip(np.floor(f * n), 0, n - 1).astype(np.int64)


def bin_value(i: np.ndarray, lo: float, hi: float, n: int, log: bool = False) -> np.ndarray:
    """Each bin's centre."""
    q = (i + 0.5) / n
    if hi <= lo:
        return np.full(len(i), lo, np.float32)
    return (np.exp(np.log(lo) + q * (np.log(hi) - np.log(lo))) if log else lo + q * (hi - lo)).astype(np.float32)


def ranges_to_array(ranges: Ranges) -> np.ndarray:
    """(21, 6) float32: each layer's lows then highs, NaN for a layer without a range."""
    return np.array([np.concatenate(ranges[lj]) if lj in ranges else np.full(6, np.nan) for lj in LAYERS], np.float32)


def ranges_from_array(arr: np.ndarray) -> Ranges:
    return {lj: (arr[i, :3], arr[i, 3:]) for i, lj in enumerate(LAYERS) if not np.isnan(arr[i, 0])}


def colour_bins(rgb: np.ndarray) -> np.ndarray:
    """Colours (n, 3; 0-1) as colour bin ids 0 .. NRGB^3 - 1."""
    c = np.clip(np.floor(rgb * NRGB), 0, NRGB - 1).astype(int)
    return (c[:, 0] * NRGB + c[:, 1]) * NRGB + c[:, 2]


def bin_colours(ids: np.ndarray) -> np.ndarray:
    """Each colour bin's centre colour."""
    ids = np.asarray(ids)
    return (np.stack([ids // NRGB ** 2, (ids // NRGB) % NRGB, ids % NRGB], 1) + 0.5) / NRGB


# ------------------------------------------------------------------ strokes
def stroke_tokens(layers: list[PaintedLayer], ranges: Ranges, vocab: Vocabulary) -> tuple[np.ndarray, np.ndarray]:
    """Layers in layer order: each layer's marker and its strokes (stage 2 opened per eye by EYE_L / EYE_R). Returns
    the token ids and each position's field (0 a marker, 1-5 x, y, angle, size, paint)."""
    toks, fields, side_now = [], [], None
    off = vocab.offsets
    for st, j, p, side in layers:
        if side is not None and side != side_now:
            toks.append(vocab.ids["EYE_L" if side == 0 else "EYE_R"])
            fields.append(0)
            side_now = side
        toks.append(vocab.ids[f"LAYER{LAYERS.index((st, j))}"])
        fields.append(0)
        lo, hi = ranges[(st, j)]
        xy = p.xy.numpy()
        x, y = bin_index(xy[:, 0], 0, SIDE, NXY), bin_index(xy[:, 1], 0, SIDE, NXY)
        a = bin_index(np.remainder(p.angle.numpy(), 2 * np.pi), 0, 2 * np.pi, NANGLE)
        lw = p.lw.numpy()
        size = bin_index(lw[:, 0], lo[0], hi[0], NSIZE, True) * NSIZE + bin_index(lw[:, 1], lo[1], hi[1], NSIZE, True)
        c = np.stack([bin_index(p.colour.numpy()[:, i], 0, 1, NRGB) for i in range(3)], 1)
        al = bin_index(p.alpha.numpy(), lo[2], hi[2], NALPHA)
        paint = ((c[:, 0] * NRGB + c[:, 1]) * NRGB + c[:, 2]) * NALPHA + al
        block = np.stack([x + off["x"], y + off["y"], a + off["angle"], size + off["size"], paint + off["paint"]], 1)
        toks += block.reshape(-1).tolist()
        fields += [1, 2, 3, 4, 5] * len(x)
    return np.array(toks, np.int32), np.array(fields, np.int8)


def decode_strokes(raw: np.ndarray, layer: tuple[int, int], ranges: Ranges) -> StrokeParams:
    """One layer's strokes from their five field values (n, 5; paint = colour bin x opacity), at the bins' centres."""
    lo, hi = ranges[layer]
    r = np.asarray(raw, np.int64).reshape(-1, 5)
    x, y = bin_value(r[:, 0], 0, SIDE, NXY), bin_value(r[:, 1], 0, SIDE, NXY)
    ang = bin_value(r[:, 2], 0, 2 * np.pi, NANGLE)
    lw = np.stack([bin_value(r[:, 3] // NSIZE, lo[0], hi[0], NSIZE, True),
                   bin_value(r[:, 3] % NSIZE, lo[1], hi[1], NSIZE, True)], 1)
    al = r[:, 4] % NALPHA
    c = r[:, 4] // NALPHA
    col = np.stack([bin_value(c // (NRGB * NRGB), 0, 1, NRGB), bin_value((c // NRGB) % NRGB, 0, 1, NRGB),
                    bin_value(c % NRGB, 0, 1, NRGB)], 1)
    return StrokeParams(*(torch.from_numpy(np.ascontiguousarray(v, np.float32)) for v in
                          (np.stack([x, y], 1), lw, ang, col, bin_value(al, lo[2], hi[2], NALPHA))))


# ------------------------------------------------------------------ palette
class Palette(NamedTuple):
    """A painting's palette: PALETTE_K colour bin ids, and for each layer each stroke's slot (None for an eye
    layer: its strokes keep their colour bins)."""

    colours: np.ndarray
    slots: list[np.ndarray | None]


def palette_colours(rgb: np.ndarray, k: int) -> np.ndarray:
    """k colour bins for these stroke colours: k-means (seed 0) on the colours, each centre to its bin, dark to light;
    with fewer distinct colours than k the list repeats; with no strokes, mid grey."""
    if len(rgb) == 0:
        return np.full(k, colour_bins(np.full((1, 3), 0.5))[0])
    cent, _ = kmeans2(rgb.astype(np.float64), min(k, len(np.unique(rgb, axis=0))), seed=0, minit="++")
    ids = np.unique(colour_bins(cent))
    ids = ids[np.argsort(bin_colours(ids) @ LUMA, kind="stable")]
    return np.resize(ids, k)


def palette_of(layers: list[PaintedLayer]) -> Palette:
    """The palette of every stroke outside the eyes; each such stroke takes its nearest palette colour."""
    fur = [e for e, (st, *_) in enumerate(layers) if st != EYE_STAGE]
    colours = palette_colours(np.concatenate([layers[e][2].colour.numpy() for e in fur] or [np.zeros((0, 3))]),
                              PALETTE_K)
    own = bin_colours(colours)
    return Palette(colours, [vq(p.colour.numpy().astype(np.float64), own)[0] if st != EYE_STAGE else None
                             for st, _, p, _ in layers])


def palette_tokens(colours: np.ndarray, vocab: Vocabulary) -> tuple[np.ndarray, np.ndarray]:
    """The palette marker and its colours as colour tokens (field 6)."""
    return (np.array([vocab.ids["PAL_ALL"]] + [int(c) + vocab.offsets["colour"] for c in colours], np.int32),
            np.array([0] + [6] * len(colours), np.int8))


# ------------------------------------------------------------------ part order
def part_map(points: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """For a cat's nine face points (256 grid), a function giving each stroke position (canvas px) its part
    (PARTS): inside an ear's triangle that ear (the one more to the image's left is "ear L"; it wins where the two
    overlap), otherwise the nearest anchor (ANCHORS)."""
    k = SIDE / INPUT_SIDE
    eyes = points[:2][np.argsort(points[:2, 0])] * k
    base = {"eye L": eyes[0], "eye R": eyes[1], "mid": eyes.mean(0), "mouth": points[2] * k}
    down, side = base["mouth"] - base["mid"], (eyes[1] - eyes[0]) / 2
    names = np.array(list(ANCHORS), object)
    at = np.array([base[b] + a * down + c * side for b, a, c in ANCHORS.values()])
    ears = []
    for e, ear in enumerate(EAR_POINTS):
        one = points.copy()
        one[list(EAR_POINTS[1 - e])] = one[list(ear)]      # both triangles this ear's: a mask of this ear alone
        ears.append((points[list(ear), 0].mean(), ear_triangles(one, SIDE, k)))
    ears.sort(key=lambda m: -m[0])                         # the right ear first, so the left ear wins an overlap

    def parts(xy: np.ndarray) -> np.ndarray:
        out = names[np.argmin(((xy[:, None] - at[None]) ** 2).sum(-1), 1)]
        ix = np.clip(xy.round().astype(int), 0, SIDE - 1)
        for name, (_, mask) in zip(("ear R", "ear L"), ears, strict=True):
            out[mask[ix[:, 1], ix[:, 0]]] = name
        return out
    return parts


def serpentine(xy: np.ndarray) -> np.ndarray:
    """The order of positions in serpentine rows PART_ROW px tall (left to right, then right to left, top down)."""
    r = (xy[:, 1] // PART_ROW).astype(int)
    return np.lexsort((np.where(r % 2 == 0, xy[:, 0], -xy[:, 0]), r))


def eye_centres(layers: list[PaintedLayer]) -> dict[bool, np.ndarray]:
    """Each eye's centre (False the left eye, True the right): the mean of its stage-2 strokes."""
    xy: dict[bool, list[np.ndarray]] = {False: [], True: []}
    for st, _, p, side in layers:
        if st == EYE_STAGE:
            xy[side == 1].append(p.xy.numpy())
    return {k: np.concatenate(v).mean(0) if v else np.full(2, SIDE / 2) for k, v in xy.items()}


def around(xy: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """The order of an eye layer's strokes: by angle from straight up, clockwise, then from the centre out."""
    d = xy - centre
    return np.lexsort((np.hypot(d[:, 0], d[:, 1]), np.arctan2(d[:, 0], -d[:, 1]) % (2 * np.pi)))


def part_runs(layers: list[PaintedLayer], parts: Callable[[np.ndarray], np.ndarray]
              ) -> dict[str, list[tuple[int, np.ndarray]]]:
    """For each part, its (layer index, stroke indices in writing order)."""
    runs: dict[str, list[tuple[int, np.ndarray]]] = {}
    centre = eye_centres(layers)
    for e, (st, j, p, side) in enumerate(layers):
        xy = p.xy.numpy()
        if st == EYE_STAGE:
            runs.setdefault("eye R" if side == 1 else "eye L", []).append((e, around(xy, centre[side == 1])))
            continue
        part = parts(xy)
        for name in set(part):
            k = np.flatnonzero(part == name)
            runs.setdefault(name, []).append((e, k if (st, j) in CHAIN_LAYERS else k[serpentine(xy[k])]))
    return runs


def run_tokens(layers: list[PaintedLayer], e: int, k: np.ndarray, ranges: Ranges, vocab: Vocabulary,
               pal: Palette | None) -> tuple[np.ndarray, np.ndarray]:
    """Strokes k of layer e as its LAYER marker and strokes; with a palette, each paint token becomes slot x
    opacity (the palette slot, or for an eye layer PALETTE_K + its colour bin)."""
    st, j, p, _ = layers[e]
    t, f = stroke_tokens([(st, j, StrokeParams(*(v[torch.from_numpy(k)] for v in p)), None)], ranges, vocab)
    if pal is not None:
        at = f == 5
        colour, alpha = np.divmod(t[at] - vocab.offsets["paint"], NALPHA)
        slot = PALETTE_K + colour if pal.slots[e] is None else pal.slots[e][k]
        t[at] = vocab.offsets["paint"] + slot * NALPHA + alpha
    return t, f


def part_tokens(layers: list[PaintedLayer], parts: Callable[[np.ndarray], np.ndarray], ranges: Ranges,
                vocab: Vocabulary, pal: Palette | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Layers in part order: each part (PARTS order) opens with its marker, then each layer that painted there in
    painting order, as its LAYER marker and its strokes there."""
    runs = part_runs(layers, parts)
    toks, fields = [], []
    for name in (n for n in PARTS if n in runs):
        toks.append(vocab.ids[f"PART_{name.replace(' ', '_').upper()}"])
        fields.append(0)
        for e, k in sorted(runs[name], key=lambda r: r[0]):
            t, f = run_tokens(layers, e, k, ranges, vocab, pal)
            toks += t.tolist()
            fields += f.tolist()
    return np.array(toks, np.int32), np.array(fields, np.int8)


# ------------------------------------------------------------------ sketch plan
def plan_tokens(plan: Mapping[str, np.ndarray], flip: bool, vocab: Vocabulary) -> tuple[np.ndarray, np.ndarray]:
    """The sketch plan (yaw, pitch, roll in degrees; head outlines (m, 3: x, y, is-ear); eye outlines; pupil
    ellipses (cx, cy, a, b, angle); nose outline; mouth line; 256 grid) as tokens: the pose, then each outline as its
    marker and its points as (x, y) tokens. The head's outline is split where it turns from head to ear; the eyes and
    pupils go left to right; the mouth keeps every other point. flip: the plan of the painting mirrored left to
    right (x -> 256 - x, yaw and roll negated)."""
    ids, off = vocab.ids, vocab.offsets
    fx = (lambda v: INPUT_SIDE - v) if flip else (lambda v: v)
    toks, fields = [ids["PLAN_POSE"]], [0]
    for a, s in ((plan["yaw"], -1 if flip else 1), (plan["pitch"], 1), (plan["roll"], -1 if flip else 1)):
        toks.append(ids[f"POSE_{int(np.clip((s * float(a) + 90) / 180 * POSE_BINS, 0, POSE_BINS - 1))}"])
        fields.append(0)

    def run(marker: str, xy: np.ndarray) -> None:
        if len(xy) == 0:
            return
        toks.append(ids[f"PLAN_{marker}"])
        fields.append(0)
        g = np.clip(np.floor(np.c_[fx(xy[:, 0]), xy[:, 1]]), 0, NXY - 1).astype(int)
        for x, y in g:
            toks.extend([x + off["x"], y + off["y"]])
            fields.extend([1, 2])

    for h in plan["head"]:
        h = np.asarray(h, np.float64)
        for seg in np.split(h, np.flatnonzero(np.diff(h[:, 2])) + 1):
            run("EAR" if seg[0, 2] else "HEAD", seg[:, :2])
    for e in sorted(np.asarray(plan["eyes"], np.float64), key=lambda e: fx(e[:, 0]).mean()):
        run("EYE", e)
    pupils = [q for q in np.asarray(plan["pupils"], np.float64) if np.isfinite(q).all()]
    for q in sorted(pupils, key=lambda q: fx(q[0])):
        t = np.linspace(0, 2 * np.pi, PUPIL_POINTS, endpoint=False)
        c, s = np.cos(q[4]), np.sin(q[4])
        run("PUPIL", q[:2] + np.c_[q[2] * np.cos(t) * c - q[3] * np.sin(t) * s,
                                   q[2] * np.cos(t) * s + q[3] * np.sin(t) * c])
    run("NOSE", np.asarray(plan["nose"], np.float64))
    run("MOUTH", np.asarray(plan["mouth"], np.float64)[::2])
    toks.append(ids["PLAN_END"])
    fields.append(0)
    return np.array(toks, np.int32), np.array(fields, np.int8)


# ------------------------------------------------------------------ a whole painting
def painting_tokens(layers: list[PaintedLayer], ranges: Ranges, vocab: Vocabulary,
                    parts: Callable[[np.ndarray], np.ndarray] | None = None,
                    plan: Mapping[str, np.ndarray] | None = None, flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """One painting's tokens and fields in the vocabulary's format: the plan (when the vocabulary has one; `flip`
    for a mirrored painting), the palette (when it has one), then the strokes in part order (`parts`, the cat's
    part_map) or in layer order."""
    if vocab.has_palette and parts is None:
        raise ValueError("a palette is written with part order: give the cat's part map")
    pal = palette_of(layers) if vocab.has_palette else None
    t, f = part_tokens(layers, parts, ranges, vocab, pal) if parts else stroke_tokens(layers, ranges, vocab)
    if pal is not None:
        pt, pf = palette_tokens(pal.colours, vocab)
        t, f = np.concatenate([pt, t]), np.concatenate([pf, f])
    if vocab.has_plan:
        if plan is None:
            raise ValueError("the vocabulary has a sketch plan: give the cat's plan")
        pt, pf = plan_tokens(plan, flip, vocab)
        t, f = np.concatenate([pt, t]), np.concatenate([pf, f])
    return np.asarray(t, np.int32), np.asarray(f, np.int8)


# ------------------------------------------------------------------ decoding
def _read_marker(name: str, state: dict) -> None:
    """A special token's effect on where the strokes that follow go."""
    if name.startswith(("PLAN_", "POSE_")):
        state.update(inplan=name != "PLAN_END", layer=None)
    elif name.startswith("PART_"):
        state.update(inplan=False, layer=None, side=int(name == "PART_EYE_R"))
    elif name in ("EYE_L", "EYE_R"):
        state["side"] = int(name == "EYE_R")
    elif name.startswith("LAYER"):
        state.update(inplan=False, layer=LAYERS[int(name[5:])])


def _is_stroke(block: np.ndarray, vocab: Vocabulary) -> bool:
    """Five tokens that are x, y, angle, size, paint in that order."""
    off, sizes = vocab.offsets, vocab.field_sizes
    return len(block) == 5 and all(0 <= v - off[f] < sizes[f] for v, f in zip(block, FIELDS, strict=True))


def _read_palette(toks: np.ndarray, i: int, vocab: Vocabulary, palette: dict[int, int]) -> int:
    """The colour tokens after the palette marker into `palette` (slot -> colour bin); returns the next index."""
    for m in range(PALETTE_K):
        if i >= len(toks) or not 0 <= toks[i] - vocab.offsets["colour"] < NRGB ** 3:
            break
        palette[m] = int(toks[i] - vocab.offsets["colour"])
        i += 1
    return i


def _stroke_values(block: np.ndarray, vocab: Vocabulary, palette: dict[int, int] | None) -> np.ndarray | None:
    """A stroke's five tokens as field values (paint = colour bin x opacity: with a palette, its slot's colour, or
    slot - PALETTE_K); None when its palette slot was never written."""
    raw = block - np.array([vocab.offsets[f] for f in FIELDS])
    if palette is not None:
        slot, alpha = divmod(int(raw[4]), NALPHA)
        if slot < PALETTE_K and slot not in palette:
            return None
        raw[4] = (palette[slot] if slot < PALETTE_K else slot - PALETTE_K) * NALPHA + alpha
    return raw


def layers_from_tokens(toks: np.ndarray, ranges: Ranges, vocab: Vocabulary) -> list[tuple[int, int, StrokeParams]]:
    """Tokens of any of these formats back to the painter's layers in painting order (stage, j, strokes): the plan
    is skipped, the palette is read and each slot replaced by its colour, and every stroke goes back to its layer
    (stage 2: to its eye); then the layers are laid in the painter's order (stage 2: the left eye's layers, then
    the right's). A sampled painting may break off or be malformed: whatever cannot be read as a stroke inside a
    layer is skipped (so is a stroke whose palette slot was never written, and a layer without ranges), and so is
    everything from EOS on."""
    toks = np.asarray(toks)
    eos = vocab.ids["EOS"]
    stop = int(np.argmax(toks == eos)) if (toks == eos).any() else len(toks)
    groups: dict[tuple[int, int, int], list[np.ndarray]] = {}
    state: dict = {"layer": None, "side": 0, "inplan": False}
    palette: dict[int, int] | None = {} if vocab.has_palette else None
    n_specials = len(vocab.specials)
    i = 0
    while i < stop:
        if toks[i] < n_specials:
            name = vocab.specials[int(toks[i])]
            if name == "PAL_ALL":
                i = _read_palette(toks[:stop], i + 1, vocab, palette)
                continue
            _read_marker(name, state)
            i += 1
            continue
        layer = state["layer"]
        if state["inplan"] or layer is None or not _is_stroke(toks[i:i + 5], vocab):
            i += 1
            continue
        raw = _stroke_values(toks[i:i + 5], vocab, palette)
        i += 5
        if raw is not None:
            groups.setdefault((layer[0], state["side"] if layer[0] == EYE_STAGE else 0, layer[1]), []).append(raw)
    return [(st, j, decode_strokes(np.stack(groups[(st, s, j)]), (st, j), ranges))
            for st, s, j in sorted(k for k in groups if (k[0], k[2]) in ranges)]


def render(layers: list[tuple[int, int, StrokeParams]]) -> torch.Tensor:
    """Layers (stage, j, strokes) painted on white paper with their layers' brushes."""
    with torch.no_grad():
        return replay(layers)
