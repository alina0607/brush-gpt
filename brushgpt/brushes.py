"""Differentiable painterly brushes and the compositing that lays their strokes on a canvas.

A stroke is five values: centre, size (length and width), angle, colour and opacity (`StrokeParams`). How it looks
beyond those values (the bristles, the bend, the wavering sides, the dry breaks) comes from a hash of the stroke's
binned position and angle (`seed_of`), so a model that writes a stroke's tokens always draws the same stroke. Every
brush is differentiable in all five values, which is what lets the painter fit strokes to a photo by gradient descent.

Brush kinds:

  wet      wet acrylic or oil: lanes of bristles of uneven strength, each touching down and lifting off at its own
           point, wavering frayed sides, paint drying out towards the tail; picks up `PICKUP` of the colour under it
  drag     the same bristles carrying no paint: they pull the paint under the stroke's head along the stroke, so the
           edges of shapes are drawn out into fur
  stick    a soft carbon stick that catches the paper's grain, more so the fainter the stroke
  pencil   a thin hard line, tapered at both ends, slightly bent
  whisker  a thin line, full width at its root and fine to a point, hanging down more and more towards its tip
  rake     several fine hairs side by side in one stroke, fanning out a little and fading towards their tips
  liner    an inking line of constant width with round ends; short liners laid end to end draw a smooth outline
  dot      a small ellipse with a soft rim

Strokes are composited in order (later over earlier) with the "over" operator, each evaluated only inside its own
square window around its centre:

    out = sum_i a_i c_i prod_{j>i} (1 - a_j) + canvas prod_j (1 - a_j)

Wet strokes mix in, and drag strokes pull, the canvas as it was before the layer (`start`), so all strokes of a
layer act together while they are fitted.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

SIDE = 512                   # the painter's canvas, pixels
KINDS = ("wet", "drag", "stick", "pencil", "whisker", "rake", "liner", "dot")

TOKEN_XY_BINS, TOKEN_ANGLE_BINS = 256, 32   # the token grid a stroke's seed is taken from
BRISTLE_PX = 1.5             # one bristle lane per this many px of width
GRAIN_PX = 3.0               # the paper's grain: cells of the coarser octave
PICKUP = 0.25                # wet: share of the colour under the brush mixed into its paint
HEAD = 0.30                  # wet, drag: the bristles touch down within this share of the length
SIDE_WAVER = 0.18            # wet, drag: how far a side may come in (share of the half width)
STICK_EDGE = 1.15            # stick: the soft edge reaches 0 at this many half-sizes from the centre
SIDE_STEP = 18.0             # px between the wavering side's samples along the stroke
FRAY = 0.35                  # wet, drag: the outer share of the half width where bristles drop out one by one
RAKE_HAIRS, RAKE_HAIR_PX, RAKE_FAN = 6, 0.6, 0.15   # rake: hairs a stroke, a hair's width at its root (px), how far
                                                  # the hairs fan out at the tip (share of the stroke's width)
RAKE_BEND, RAKE_BEND_OWN = 0.12, 0.02             # rake: the hairs' common bend and each hair's own (share of length)
RAKE_START, RAKE_END, RAKE_FADE = 0.05, 0.85, 0.45   # rake: a hair starts within the first 5%, ends past 85% of the
                                                  # length, and fades out over its last 45%
DOT_EDGE = 0.5               # dot: px over which its rim fades
WHISKER_DROOP = (0.06, 0.16) # whisker: how far its tip hangs below the straight line, a share of its length
SUBPIXELS = ((-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25))   # 2x2 samples a pixel for fine brushes
HASH_KEYS = (73856093, 19349663, 83492791, 2654435761)


class StrokeParams(NamedTuple):
    """A layer's strokes in canvas units."""

    xy: torch.Tensor             # (n, 2) centres, pixels
    lw: torch.Tensor             # (n, 2) length and width, pixels
    angle: torch.Tensor          # (n,) radians
    colour: torch.Tensor         # (n, 3) RGB in 0-1
    alpha: torch.Tensor          # (n,) opacity


@dataclass(frozen=True)
class Brush:
    """What laying a layer needs to know besides its strokes: the brush kind and the layer's nominal stroke size
    (which sets the window each stroke is evaluated in)."""

    kind: str
    length: float
    width: float
    pickup: float = PICKUP


def logit(p: np.ndarray | torch.Tensor) -> torch.Tensor:
    p = torch.as_tensor(p, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


# ------------------------------------------------------------------ seeds and noise
def _token_bin(x: np.ndarray, lo: float, hi: float, n: int) -> np.ndarray:
    return np.clip(((np.asarray(x, float) - lo) / (hi - lo) * n).astype(int), 0, n - 1)


def seed_of(xy: np.ndarray, angle: np.ndarray, side: int = SIDE) -> torch.Tensor:
    """A stroke's seed from its token bins (x, y, angle), so the same tokens always draw the same stroke."""
    xb = _token_bin(xy[:, 0], 0, side, TOKEN_XY_BINS).astype(np.int64)
    yb = _token_bin(xy[:, 1], 0, side, TOKEN_XY_BINS).astype(np.int64)
    ab = _token_bin(np.mod(angle, math.pi), 0, math.pi, TOKEN_ANGLE_BINS).astype(np.int64)
    return torch.from_numpy(((xb * 73856093) ^ (yb * 19349663) ^ (ab * 83492791)) % 1000003).float()


def stroke_seeds(p: StrokeParams, side: int = SIDE) -> torch.Tensor:
    """The seeds of a layer's strokes, on the strokes' device."""
    return seed_of(p.xy.detach().cpu().numpy(), p.angle.detach().cpu().numpy(), side).to(p.xy.device)


def hash01(*keys: torch.Tensor) -> torch.Tensor:
    """A value in [0, 1) from integer-valued tensors: the keys mixed in 64-bit integers and finished with murmur3's
    32-bit mix. Integer arithmetic only, so every device gives exactly the same value."""
    k = sum(t.round().long() * c for t, c in zip(keys, HASH_KEYS, strict=False)) & 0xFFFFFFFF
    k = k ^ (k >> 16)
    k = (k * 0x85EBCA6B) & 0xFFFFFFFF
    k = k ^ (k >> 13)
    k = (k * 0xC2B2AE35) & 0xFFFFFFFF
    k = k ^ (k >> 16)
    return k.float() / 4294967296.0


def _smooth(f: torch.Tensor) -> torch.Tensor:
    return f * f * (3 - 2 * f)


def noise1(seed: torch.Tensor, x: torch.Tensor, salt: float, extra: torch.Tensor | None = None) -> torch.Tensor:
    """Smooth 1-D value noise along x (integer steps), per seed (and per `extra`, e.g. a bristle or a side)."""
    x0 = torch.floor(x)
    f = _smooth(x - x0)
    e = torch.zeros_like(x0) if extra is None else extra
    a = hash01(seed, x0, torch.full_like(x0, salt), e)
    b = hash01(seed, x0 + 1, torch.full_like(x0, salt), e)
    return a + (b - a) * f


def paper_grain(grid: torch.Tensor) -> torch.Tensor:
    """The paper's grain, fixed on the canvas: two octaves of smooth value noise, the finer one on a turned lattice
    so no square cells show; values 0-1, the shape of grid[..., 0]."""
    def octave(q: torch.Tensor, salt: float) -> torch.Tensor:
        q0 = torch.floor(q)
        f = _smooth(q - q0)
        k = torch.full_like(q0[..., 0], salt)
        c = [[hash01(q0[..., 0] + i, q0[..., 1] + j, k) for i in (0, 1)] for j in (0, 1)]
        top = c[0][0] + (c[0][1] - c[0][0]) * f[..., 0]
        bottom = c[1][0] + (c[1][1] - c[1][0]) * f[..., 0]
        return top + (bottom - top) * f[..., 1]
    turn = torch.tensor([[0.8, -0.6], [0.6, 0.8]], device=grid.device)
    return 0.55 * octave(grid / GRAIN_PX, 9.0) + 0.45 * octave((grid @ turn) / (GRAIN_PX / 2.3), 11.0)


def root(x: torch.Tensor, power: float) -> torch.Tensor:
    """x ** power for x >= 0 and 0 < power < 1, with slope 0 instead of infinite where x is 0 (an infinite slope
    there can turn a tapered stroke's gradient into NaN)."""
    return torch.where(x > 0, x.clamp(min=1e-12) ** power, torch.zeros_like(x))


# ------------------------------------------------------------------ the brushes
def frame(p: StrokeParams, grid: torch.Tensor, bend: torch.Tensor | None = None):
    """Each pixel in each stroke's own frame: u along (head at -h, tail at +h), v across (bent), the half length h,
    the half width w and s (0 at the head, 1 at the tail), all (n, H, W). grid is (H, W, 2), shared by all strokes,
    or (n, H, W, 2), one window per stroke."""
    g = grid if grid.dim() == 4 else grid[None]
    d = g - p.xy[:, None, None, :]
    c, s_ = p.angle.cos()[:, None, None], p.angle.sin()[:, None, None]
    u = d[..., 0] * c + d[..., 1] * s_
    v = -d[..., 0] * s_ + d[..., 1] * c
    h, w = (p.lw[:, 0] / 2)[:, None, None], (p.lw[:, 1] / 2)[:, None, None]
    if bend is not None:
        v = v - bend[:, None, None] * h * (u / h) ** 2
    return u, v, h, w, ((u + h) / (2 * h)).clamp(0, 1)


def bristle_body(seed: torch.Tensor, u, v, h, w, s, solid: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """The wet or drag stroke's coverage (n, H, W) and the wet paint's shade.

    The outline narrows towards both ends and each side swells slowly on its own. Inside are lanes of bristles of
    uneven strength, the outer ones weaker and breaking off one by one in the outer FRAY of the width; each bristle
    touches down within the first HEAD of the length and lifts off in the last 45%, and the paint runs dry towards
    the tail. solid (wet paint): the body is nearly opaque and the streaks show as small differences in value
    (shade); otherwise (drag) the lanes' strength varies more, so the pull is streaky."""
    sd = seed[:, None, None]
    taper = (1 - 0.35 * s ** 2) * (1 - 0.35 * (1 - s) ** 2) / 0.9
    side = torch.where(v < 0, 0.0, 1.0)
    waver = 1 - SIDE_WAVER * (0.7 * noise1(sd, (u + h) / SIDE_STEP, 8.0, side)
                              + 0.3 * noise1(sd, (u + h) / (SIDE_STEP / 3.1), 10.0, side))
    half = w * taper * waver
    outline = torch.sigmoid((h - u.abs()) / 0.7) * torch.sigmoid((half - v.abs()) / 1.2)
    lanes = (2 * w / BRISTLE_PX).clamp(min=3)
    across = v / (2 * w * taper.clamp(min=0.05))                      # -0.5 .. 0.5 across the narrowing stroke
    t = (across + 0.5) * lanes
    lane = torch.floor(t)
    strength = noise1(sd, t, 1.0)
    strength = (0.8 + 0.2 * strength) if solid else (0.35 + 0.65 * strength)
    strength = strength * (1 - 0.45 * (2 * across.abs()).clamp(0, 1) ** 3)       # outer bristles carry less
    edge = ((2 * across.abs() - (1 - FRAY)) / FRAY).clamp(0, 1)                    # 0 inside, 1 at the side
    keep = torch.sigmoid((noise1(sd, (u + h) / 6.0, 12.0, lane) - edge) * 12 + 3)  # outer bristles break off
    start, end = HEAD * noise1(sd, t, 6.0), 0.55 + 0.45 * noise1(sd, t, 2.0)
    run = torch.sigmoid((end - s) * 2 * h / 1.5) * torch.sigmoid((s - start) * 2 * h / 1.5)
    tail = (-0.4 + 1.3 * ((s - 0.45) / 0.55).clamp(0, 1) ** 2) if solid else (-0.1 + 0.7 * s ** 2)
    head = -0.4 + 0.9 * ((HEAD - s) / HEAD).clamp(0, 1) ** 2                       # a little dry where it touches down
    floor = torch.maximum(tail, head)
    dry = torch.sigmoid((noise1(sd, (u + h) / 2.5, 3.0, lane) - floor) * 10)
    shade = 1 + 0.18 * (noise1(sd, t, 4.0) - 0.5) if solid else torch.ones_like(t)
    return outline * strength * keep * run * dry, shade


def whisker(p: StrokeParams, seed: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """The whisker's coverage (n, H, W): rooted at the stroke's start (u = -h), its centre line hanging down by
    droop * length * f**2 at f of the way to the tip, its width w at the root narrowing to 0 at the tip."""
    lo, hi = WHISKER_DROOP
    droop = lo + (hi - lo) * hash01(seed, torch.full_like(seed, 41.0))
    down = torch.where(p.angle.cos() >= 0, 1.0, -1.0)                     # +v is down when the stroke points right
    out = 0
    for dx, dy in SUBPIXELS:
        u, v, h, w, _ = frame(p, grid + torch.tensor([dx, dy], device=grid.device))
        f = ((u + h) / (2 * h)).clamp(0, 1)
        centre = (droop * down)[:, None, None] * 2 * h * f ** 2
        half = w * root(1 - f, 0.8)
        out = out + torch.sigmoid((h - u.abs()) / 0.5) * torch.sigmoid((half - (v - centre).abs()) / 0.35)
    return out / 4


def rake(p: StrokeParams, seed: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """The rake's coverage (n, H, W): RAKE_HAIRS hairs side by side, rooted near the stroke's start, fanning out by
    RAKE_FAN of the width at the tip and bending together (RAKE_BEND, plus a little of their own) so they run
    alongside without crossing; each fine at its end and fading out over its last RAKE_FADE. The hairs build up as
    paint does, 1 - prod(1 - hair)."""
    out = 0
    sd = seed[:, None, None]
    for dx, dy in SUBPIXELS:
        u, v, h, w, s = frame(p, grid + torch.tensor([dx, dy], device=grid.device))
        keep = 1.0
        for j in range(RAKE_HAIRS):
            jj = torch.full_like(sd, float(j))
            across = (j + 0.5) / RAKE_HAIRS - 0.5 + 0.3 * (hash01(sd, jj, torch.full_like(sd, 51.0)) - 0.5) / RAKE_HAIRS
            bend = RAKE_BEND * (2 * hash01(sd, torch.full_like(sd, 59.0)) - 1) \
                + RAKE_BEND_OWN * (2 * hash01(sd, jj, torch.full_like(sd, 53.0)) - 1)
            start = RAKE_START * hash01(sd, jj, torch.full_like(sd, 55.0))
            end = RAKE_END + (1 - RAKE_END) * hash01(sd, jj, torch.full_like(sd, 57.0))
            f = ((s - start) / (end - start)).clamp(0, 1)
            centre = 2 * w * across * (1 + RAKE_FAN * f) + bend * 2 * h * f ** 2
            half = 0.5 * RAKE_HAIR_PX * root(1 - f, 0.7) + 0.15
            fade = ((1 - f) / RAKE_FADE).clamp(max=1)
            hair = torch.sigmoid((half - (v - centre).abs()) / 0.35) * torch.sigmoid((s - start) * 2 * h / 0.7) \
                * torch.sigmoid((end - s) * 2 * h / 0.7) * fade
            keep = keep * (1 - hair)
        out = out + (1 - keep) * torch.sigmoid((h - u.abs()) / 0.5)       # nothing past the stroke's ends
    return out / 4


def liner(p: StrokeParams, grid: torch.Tensor) -> torch.Tensor:
    """The liner's coverage (n, H, W): constant width, round ends."""
    out = 0
    for dx, dy in SUBPIXELS:
        u, v, h, w, _ = frame(p, grid + torch.tensor([dx, dy], device=grid.device))
        run = (u.abs() - (h - w).clamp(min=0)).clamp(min=0)              # 0 along the middle, round past the ends
        out = out + torch.sigmoid((w - torch.sqrt(run ** 2 + v ** 2 + 1e-8)) / 0.35)
    return out / 4


def dot(p: StrokeParams, grid: torch.Tensor) -> torch.Tensor:
    """The dot's coverage (n, H, W): an ellipse of the stroke's length and width with a soft rim."""
    out = 0
    for dx, dy in SUBPIXELS:
        u, v, h, w, _ = frame(p, grid + torch.tensor([dx, dy], device=grid.device))
        r = torch.sqrt((u / h) ** 2 + (v / w) ** 2 + 1e-8)
        out = out + torch.sigmoid((1 - r) * torch.minimum(h, w) / DOT_EDGE)
    return out / 4


def coverage(kind: str, p: StrokeParams, seed: torch.Tensor, grid: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Each stroke's coverage (n, H, W) in 0-1, before its opacity, and the wet brush's shade (None for others)."""
    if kind not in KINDS:
        raise ValueError(f"unknown brush kind {kind!r}")
    if kind == "whisker":
        return whisker(p, seed, grid), None
    if kind == "rake":
        return rake(p, seed, grid), None
    if kind == "liner":
        return liner(p, grid), None
    if kind == "dot":
        return dot(p, grid), None
    bend = (hash01(seed, torch.full_like(seed, 5.0)) - 0.5) * (0.3 if kind != "stick" else 0.0)
    u, v, h, w, s = frame(p, grid, bend)
    if kind in ("wet", "drag"):
        cov, shade = bristle_body(seed, u, v, h, w, s, kind == "wet")
        return cov, (shade if kind == "wet" else None)
    if kind == "stick":
        r = torch.sqrt((u / h) ** 2 + (v / w) ** 2 + 1e-8)
        body = _smooth(((STICK_EDGE - r) / (STICK_EDGE - 0.55)).clamp(0, 1))      # soft, exactly 0 past STICK_EDGE
        g = paper_grain(grid if grid.dim() == 4 else grid[None])
        depth = (1 - p.alpha).clamp(0.15, 0.85)[:, None, None]                  # the paper shows through faint strokes
        return body * (1 - depth + depth * torch.sigmoid((g - (1 - body)) * 6)), None
    taper = root((1 - (2 * s - 1) ** 2).clamp(min=0), 0.5)                   # pencil: thin at both ends
    return torch.sigmoid((h - u.abs()) / 0.5) * torch.sigmoid((w * taper - v.abs()) / 0.35), None


def drag_source(canvas: torch.Tensor, p: StrokeParams, seed: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """What each drag stroke pulls into each pixel (n, H, W, 3): the canvas sampled back towards the stroke's head by
    0.45-0.75 of the pixel's distance from it, depending on the bristle, so the paint is pulled out to ragged
    lengths. grid as in `frame`; canvas (side, side, 3)."""
    side = canvas.shape[0]
    g = grid if grid.dim() == 4 else grid[None].expand(len(p.xy), -1, -1, -1)
    u, v, h, w, _ = frame(p, g)
    lane = (v / (2 * w) + 0.5) * (2 * w / BRISTLE_PX).clamp(min=3)
    pull = 0.45 + 0.3 * noise1(seed[:, None, None], lane, 13.0)
    back = torch.stack([p.angle.cos(), p.angle.sin()], -1)[:, None, None, :]
    src = g - (pull * (u + h).clamp(min=0))[..., None] * back
    img = canvas.permute(2, 0, 1)[None].expand(len(p.xy), -1, -1, -1)
    return F.grid_sample(img, src / side * 2 - 1, align_corners=False, padding_mode="border").permute(0, 2, 3, 1)


# ------------------------------------------------------------------ windowed compositing
def windows(p: StrokeParams, brush: Brush) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Every stroke's square window of pixels: ix, iy (n, side * side of the window) and the window's half side,
    large enough for a stroke 1.5 times the layer's nominal size."""
    reach = STICK_EDGE if brush.kind == "stick" else 1.0                      # the stick's soft edge reaches further
    half = int(math.ceil(reach * math.hypot(0.75 * brush.length, 0.75 * brush.width) + 6))
    off = torch.arange(-half, half + 1, device=p.xy.device)
    oy, ox = torch.meshgrid(off, off, indexing="ij")
    ix = p.xy[:, 0].detach().floor().long()[:, None] + ox.reshape(1, -1)
    iy = p.xy[:, 1].detach().floor().long()[:, None] + oy.reshape(1, -1)
    return ix, iy, half


def lay(canvas: torch.Tensor, start: torch.Tensor, p: StrokeParams, seed: torch.Tensor, brush: Brush) -> torch.Tensor:
    """A layer's strokes laid over the canvas in order, each only in its window; each pixel's paint colour comes from
    the brush (wet mixes in `start`, drag pulls `start`). Differentiable in the strokes.

    The window entries are grouped by pixel (sorted by pixel, then stroke) and laid into a (pixels touched x most
    strokes on one pixel) matrix of log(1 - a); a reversed cumulative sum along each row gives every entry's
    prod_{j>i} (1 - a_j) in one step."""
    h, w = canvas.shape[:2]
    ix, iy, half = windows(p, brush)
    n, m = ix.shape
    side = 2 * half + 1
    grid = torch.stack([ix + 0.5, iy + 0.5], -1).float().reshape(n, side, side, 2)
    cov, shade = coverage(brush.kind, p, seed, grid)
    valid = ((ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)).float()
    a = (cov.reshape(n, m) * p.alpha[:, None] * valid).clamp(max=1 - 1e-6)
    pix = (iy.clamp(0, h - 1) * w + ix.clamp(0, w - 1))
    under = start.reshape(-1, 3)[pix]                                             # (n, m, 3)
    if brush.kind == "drag":
        col = drag_source(start, p, seed, grid).reshape(n, m, 3)
    elif brush.kind == "wet":
        col = ((1 - brush.pickup) * p.colour[:, None, :] + brush.pickup * under) * shade.reshape(n, m, 1)
    else:
        col = p.colour[:, None, :].expand(n, m, 3)
    keep_box = a > 1e-4                                                           # only entries that paint anything
    if not bool(keep_box.any()):                                                  # e.g. a nearly clear stroke
        return canvas + 0 * a.sum()
    pix_k, a_k, col_k = pix[keep_box], a[keep_box], col.clamp(0, 1)[keep_box]
    dev = canvas.device
    stroke = torch.arange(n, device=dev)[:, None].expand(n, m)[keep_box]
    order = torch.argsort(pix_k * n + stroke)
    px_s, a_s, col_s = pix_k[order], a_k[order], col_k[order]
    first = torch.ones_like(px_s, dtype=torch.bool)
    first[1:] = px_s[1:] != px_s[:-1]
    seg = torch.cumsum(first.long(), 0) - 1
    starts = torch.nonzero(first).squeeze(1)
    rank = torch.arange(len(px_s), device=dev) - starts[seg]
    log_keep = torch.zeros(len(starts), int(rank.max()) + 1, device=dev).index_put((seg, rank), torch.log1p(-a_s))
    later = torch.flip(torch.cumsum(torch.flip(log_keep, [1]), 1), [1]) - log_keep  # sum_{j>i} log(1 - a_j)
    paint = torch.zeros(h * w, 3, device=dev).index_add(0, px_s, (a_s * torch.exp(later[seg, rank]))[:, None] * col_s)
    keep = torch.zeros(h * w, device=dev).index_add(0, px_s[starts], log_keep.sum(1))
    return (canvas.reshape(-1, 3) * torch.exp(keep)[:, None] + paint).reshape(h, w, 3)


def lay_layer(canvas: torch.Tensor, p: StrokeParams, brush: Brush) -> torch.Tensor:
    """One finished layer laid on the canvas (no gradient), reading that canvas as its start; on the strokes'
    device, the result back on the CPU."""
    dev = p.xy.device
    with torch.no_grad():
        return lay(canvas.to(dev), canvas.to(dev), p, stroke_seeds(p, canvas.shape[0]), brush).cpu()


def render(layers: Iterable[tuple[Brush, StrokeParams]], side: int = SIDE) -> torch.Tensor:
    """Layers laid one after another on white paper, each reading the canvas it is laid on: how a painting is
    replayed from its strokes (or from decoded tokens). (side, side, 3) in 0-1."""
    canvas = torch.ones(side, side, 3)
    with torch.no_grad():
        for brush, p in layers:
            canvas = lay(canvas, canvas, p, stroke_seeds(p, side), brush)
    return canvas
