"""The data painter: fits brush strokes to a cat photo in seven stages, the order a painter works in.

  1 block-in    a soft stick, then large wet strokes along the fur, fitted to the photo blurred 6 px: the average
                values and the movement
  2 eyes        one run of strokes per eye, inside out: the eye's base colour, the irises, the shadow the lids cast
                just inside the outline, the pupil (one dot on the plan's ellipse, coloured as the photo's darkest
                there), the lid line (liners laid end to end along the plan's smoothed outline), the catchlights
  3 form        medium wet strokes: the big shapes' light and shade
  4 edges       drag strokes on the edges of values and of the head, pulling the paint out into fur, free to grow up
                to `band` px past the outline
  5 markings    small wet strokes
                (stages 3-5 leave the eyes alone)
  6 nose/mouth  the nose (wet) and the mouth line (pencil) inside the plan's outlines
  7 details     the ears (the dark back of the ear along its inner edge, a fill of the ear's edge band, a clean rim
                along its outline, the shadow inside it, its white hairs traced from the photo), the dark dots on
                the whisker pads, the whiskers (rooted on the photo's light lines on the pads, running away from
                between the nose and the mouth, free to run past the head) and the long hairs above the eyes

Each layer places its strokes (where the painting is furthest from the photo, on edges, inside a part of the sketch
plan, or where a rule, the plan or the photo puts them), starts them along the fur and fits position, size, angle
(within the layer's bound around the fur), colour and opacity by gradient descent through the brushes' renderer:
L1 to the photo on the cat, nothing on the uncertain pixels, and changing a protected pixel (the paper, or anything
outside the layer's part) costs `protect` times the change.

Eyes apart (`paint_apart`): every stage but the eyes is fitted to the cat seen without its eyes, on a canvas that
never had eyes, and laid on the real canvas too; the eyes are painted one at a time. So the strokes outside one eye
depend on nothing inside it, and `paint_eye_versions` repaints only the eyes for versions of a photo that differ
inside them (such as recoloured eyes).

Output per stage: stage<k>.npz with the canvas after the stage ("canvas", float32 (512, 512, 3)) and, for every
layer j that painted anything, "<j>_xy", "<j>_lw", "<j>_angle", "<j>_colour", "<j>_alpha" (float32, pixels, radians,
0-1) and "<j>_kind" (the brush); stage 2 also has "<j>_side" (int8, 0 the image's left eye, 1 the right). Stage 2's
strokes are stored per layer, left eye first; replaying them per eye (`painting_layers`) gives the same canvas.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.feature import blob_log, hessian_matrix, hessian_matrix_eigvals
from skimage.morphology import skeletonize
from skimage.transform import probabilistic_hough_line

from .brushes import Brush, StrokeParams, hash01, lay, lay_layer, logit, render, seed_of, stroke_seeds
from .cat import EYE_APART_GROW, CatInputs, ear_triangles, eye_side, one_eye, prepare_cat, without_eyes
from .fur import orientation

SIDE = 512
ITERS = 150  # Adam steps per layer
LR = 0.05
PROTECT = 10.0
LID_SMOOTH = 4.0  # px: the plan's eye outline smoothed along its length before the lid is laid on it
SHADOW_IN, SHADOW_STEP = 4.0, 5.0  # the lid's shadow: px in from the eye outline, px between its strokes
LID_STEP = 6.0  # px between lid liners along the eye outline (each twice as long: they overlap)
CATCH_MIN, CATCH_MAX = 0.70, 2  # a catchlight: at least this light and 0.3 over the eye's median; at most this
# many an eye
CATCH_GLINT, CATCH_SIZE = 0.25, 0.15  # ... and this much over the 9 px median about it; at most this share of the
# eye's width across
EAR_EMPHASIS = 0.3  # the ear's shadow and back this much darker than the photo's darkest there
WHISKER_BAND = 70  # px past the head where whiskers are free (no loss, no protection)
WHISKER_NEAR = 10  # px about a whisker's root where its colour is read from the photo
WHISKER_TOP = 0.95  # its colour: the light-line pixels at or above this quantile of lightness
DOT_RING, DOT_CONTRAST, DOT_MIN, DOT_MAX = 0.30, 0.10, 4, 40  # whisker dots: on a muzzle at least this light
# (blurred), this much darker than around them;
# fewer than DOT_MIN are none, at most DOT_MAX
DOT_DARK = 0.3  # a dark colour read from the photo: its pixels at or below this quantile of lightness
DOT_ROUND = 0.35  # a dot is round: its two curvatures within this ratio (a whisker's shadow is not)
BROW_MIN, BROW_MAX = 2, 4  # brow whiskers a side
BROW_UP, BROW_SIDE = 0.6, 0.05  # brow whiskers root at most this far above the eye, this far beyond its width
TINT_BLUR = 2.0  # px: the paint under the ear's rim, blurred
VARY = (0.6, 1.4)  # strokes with a set direction: each one's length and width, from its hash
EAR_BACK_IN, EAR_FILL_IN, EAR_RIM_IN = 6.0, 4.0, 1.0  # px in from the outline: the ear's back, its fill, its rim
TRACE_TOP, TRACE_LEN, TRACE_MAX = 80, 7, 0.3  # traced ear hairs: ridge percentile, shortest hair (px), longest
# (share of the ear's height)
TRACE_CAP, TRACE_CLAIM = 60, 2  # at most this many hairs a cat; px round a traced hair taken by it
TRACE_WIDTH = (0.15, 1.5, 3.0)  # a hair clump's width: share of its length, at least, at most (px)
TRACE_TAIL, TRACE_HALF = 0.5, 0.5  # a traced hair goes on past its tip (at most this share of its
# length) while the photo there stays this share as light
TRACE_REACH = 1.3  # the clump's stroke this many times the hair's length (the rake
# fades out over its end)
EYE_STAGE = 2


@dataclass(frozen=True)
class Stage:
    """One layer of a stage: its brush, how many strokes, their nominal size and opacity range, and where and how
    they are placed and fitted."""

    name: str
    kind: str  # the brush (brushes.KINDS)
    n: int  # strokes (fixed places: as many as there are places)
    length: float  # px; strokes fit within size_range times this
    width: float
    alpha: tuple[float, float]
    turn_deg: float | None  # how far from the fur's (or the set) direction; None: free, starting along the
    # photo's local structure
    place: str  # how strokes are placed: error, edges, zone, ear_dark, whiskers, or a fixed rule
    # (FIXED_PLACES)
    only: str | None = None  # a part (eye, eye_rim, pupil, nose, mouth, ears, ear_inner) outside which
    # changing the canvas costs `protect`
    blur: float = 0.0  # fit to the photo blurred this many px on the cat
    protect: float = PROTECT  # what changing a protected pixel costs, times the change
    inset: int = 0  # strokes start at least this many px inside the head
    band: int = 0  # px outside the head that are free (no loss, no protection)
    fixed_place: bool = False  # positions are not fitted
    no_ears: bool = False  # no strokes start on the ears
    tint: float | None = None  # colour = the paint just inside the ear's outline plus at most this much a channel
    radial: bool = False  # the place is the stroke's root; the stroke runs away from it
    photo_colour: str | None = None  # colour read from the photo near the place, not fitted: "light" (the brightest of
    # the light-line pixels), "dark" (the darkest pixels), "bright" (a catchlight's)
    avoid: tuple[str, ...] = ()  # parts the strokes must not paint over (protected as the paper is)
    size_range: tuple[float, float] = (0.5, 1.5)  # strokes' length and width, times the stage's
    light_top: float = WHISKER_TOP  # photo_colour "light": the light-line pixels at or above this quantile
    emphasis: float = 0.0  # photo colours pushed this share of the way to black ("dark") or white ("light")
    fit_size: bool = True  # False: the sizes the places came with are kept

    @property
    def brush(self) -> Brush:
        return Brush(self.kind, self.length, self.width)


STAGES = {
    1: [
        Stage(
            "block-in: stick", "stick", 40, 200, 110, (0.3, 0.9), 40, "error", blur=6.0, protect=40.0, inset=6, band=10
        ),
        Stage("block-in: wet", "wet", 120, 90, 32, (0.5, 1.0), 25, "error", blur=6.0, protect=100.0, inset=6, band=10),
    ],
    2: [
        Stage("iris base", "wet", 30, 24, 10, (0.7, 1.0), None, "zone", "eye"),
        Stage("irises", "wet", 100, 12, 5, (0.6, 1.0), None, "zone", "eye"),
        Stage(
            "lid shadow", "wet", 0, 16, 8, (0.15, 0.8), 8, "lid_shadow", "eye", fixed_place=True, size_range=(0.7, 1.3)
        ),
        Stage("pupils", "dot", 0, 24, 6, (0.8, 1.0), 10, "pupil_plan", "pupil", fixed_place=True, photo_colour="dark"),
        Stage(
            "lids",
            "liner",
            0,
            2 * LID_STEP,
            3,
            (0.6, 1.0),
            5,
            "lid",
            "eye_rim",
            fixed_place=True,
            size_range=(0.85, 1.15),
            photo_colour="dark",
        ),
        Stage(
            "highlights",
            "dot",
            0,
            6,
            6,
            (0.85, 1.0),
            None,
            "catchlight",
            "eye",
            fixed_place=True,
            photo_colour="bright",
        ),
    ],
    3: [Stage("form", "wet", 160, 70, 24, (0.4, 1.0), 20, "error", avoid=("eye",))],
    4: [Stage("edges", "drag", 140, 60, 18, (0.4, 1.0), 15, "edges", band=14, avoid=("eye",), no_ears=True)],
    5: [Stage("markings", "wet", 160, 34, 11, (0.3, 0.9), 20, "error", avoid=("eye",))],
    6: [
        Stage("nose", "wet", 40, 14, 6, (0.6, 1.0), None, "zone", "nose"),
        Stage("mouth", "pencil", 14, 12, 1.8, (0.6, 1.0), None, "zone", "mouth"),
    ],
    7: [
        Stage(
            "ear back",
            "wet",
            0,
            18,
            9,
            (0.6, 1.0),
            10,
            "ear_back",
            "ears",
            fixed_place=True,
            photo_colour="dark",
            emphasis=EAR_EMPHASIS,
            size_range=(0.8, 1.2),
        ),
        Stage("ear fill", "wet", 0, 16, 9, (0.7, 1.0), 10, "ear_fill", None, fixed_place=True, size_range=(0.8, 1.2)),
        Stage(
            "ear rim",
            "liner",
            0,
            2 * LID_STEP,
            2.4,
            (0.8, 1.0),
            5,
            "ear_rim",
            "ears",
            fixed_place=True,
            size_range=(0.9, 1.1),
            tint=0.03,
        ),
        Stage(
            "ear shadow",
            "wet",
            40,
            14,
            6,
            (0.4, 1.0),
            None,
            "ear_dark",
            "ear_inner",
            photo_colour="dark",
            emphasis=EAR_EMPHASIS,
        ),
        Stage(
            "ear tufts",
            "rake",
            0,
            20,
            4,
            (0.4, 0.75),
            5,
            "ear_traced",
            "ear_inner",
            fixed_place=True,
            photo_colour="light",
            light_top=0.98,
            emphasis=0.25,
            size_range=(0.25, 3.0),
            fit_size=False,
        ),
        Stage(
            "muzzle dots",
            "dot",
            0,
            4,
            4,
            (0.6, 1.0),
            None,
            "dots",
            fixed_place=True,
            photo_colour="dark",
            avoid=("nose", "mouth"),
        ),
        Stage(
            "whiskers",
            "whisker",
            28,
            140,
            1.6,
            (0.7, 1.0),
            30,
            "whiskers",
            band=WHISKER_BAND,
            radial=True,
            photo_colour="light",
            avoid=("nose", "mouth"),
            fixed_place=True,
        ),
        Stage(
            "brow whiskers",
            "whisker",
            0,
            120,
            1.4,
            (0.7, 1.0),
            10,
            "brow",
            band=WHISKER_BAND,
            radial=True,
            photo_colour="light",
            fixed_place=True,
        ),
    ],
}
FIXED_PLACES = (
    "dots",
    "pupil_plan",
    "lid",
    "lid_shadow",
    "catchlight",
    "brow",
    "ear_back",
    "ear_fill",
    "ear_rim",
    "ear_traced",
)


def layer_brush(stage: int, j: int) -> Brush:
    """The brush of layer j of a stage."""
    return STAGES[stage][j].brush


# ------------------------------------------------------------------ image helpers
def luminance(img: np.ndarray) -> np.ndarray:
    return img[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=img.dtype)


def bright_lines(lum: np.ndarray, s: int) -> np.ndarray:
    """Per pixel, how much brighter it is than the brighter of its two sides s px away, best of four directions."""
    p = np.pad(lum, s, mode="edge")
    h, w = lum.shape
    out = np.zeros_like(lum)
    for dy, dx in ((0, s), (s, 0), (s, s), (s, -s)):
        a = p[s + dy : s + dy + h, s + dx : s + dx + w]
        b = p[s - dy : s - dy + h, s - dx : s - dx + w]
        out = np.maximum(out, lum - np.maximum(a, b))
    return out.clip(0)


def light_lines(lum: np.ndarray) -> np.ndarray:
    """Thin light lines (one or two px wide)."""
    return np.maximum(bright_lines(lum, 1), bright_lines(lum, 2))


def ridges(lum: np.ndarray) -> np.ndarray:
    """How much each pixel is the middle of a thin light line (the Hessian's stronger eigenvalue, 0.8 and 1.2 px)."""
    ridge = np.zeros_like(lum)
    for sg in (0.8, 1.2):
        l2 = hessian_matrix_eigvals(hessian_matrix(lum, sigma=sg, order="rc", use_gaussian_derivatives=True))[1]
        ridge = np.maximum(ridge, np.clip(-l2, 0, None) * sg**2)
    return ridge


# ------------------------------------------------------------------ strokes being fitted
@dataclass
class Strokes:
    """A layer's free parameters while it is fitted; `stroke_params` maps them to canvas units."""

    xy: torch.Tensor  # logits of the centre in [0, SIDE)
    size: torch.Tensor  # logits within size_range times the stage's (length, width)
    angle: torch.Tensor  # free: radians; bound: the unbounded turn from the fur (or from `base`)
    colour: torch.Tensor
    alpha: torch.Tensor
    base: torch.Tensor | None = None  # a direction set by the place (radians) instead of the fur's
    under: torch.Tensor | None = None  # a colour set by the place (photo colours, or the paint the ear rim tints)

    def tensors(self) -> list[torch.Tensor]:
        return [self.xy, self.size, self.angle, self.colour, self.alpha]


def stroke_params(s: Strokes, st: Stage, cat: dict) -> StrokeParams:
    xy = torch.sigmoid(s.xy) * SIDE
    lo_s, hi_s = st.size_range
    lw = (lo_s + (hi_s - lo_s) * torch.sigmoid(s.size)) * torch.tensor([st.length, st.width], device=s.size.device)
    if st.turn_deg is None:
        angle = s.angle
    else:
        if s.base is not None:
            along = s.base
        else:  # the fur's line at the centre
            g = (xy.detach() / SIDE * 2 - 1).view(1, 1, -1, 2)
            f = F.grid_sample(cat["field"].permute(2, 0, 1)[None], g, align_corners=False)[0, :, 0].T
            along = 0.5 * torch.atan2(f[:, 1], f[:, 0])
        angle = along + math.radians(st.turn_deg) * torch.tanh(s.angle)
    if st.radial:  # xy is the root: the stroke's centre half its length on
        xy = xy + 0.5 * lw[:, :1] * torch.stack([angle.cos(), angle.sin()], -1)
    lo, hi = st.alpha
    if s.under is None:
        colour = torch.sigmoid(s.colour)
    elif st.photo_colour:
        colour = s.under
    else:
        assert st.tint is not None  # `under` without photo colour is the tint's paint
        colour = (s.under + st.tint * torch.tanh(s.colour)).clamp(0, 1)
    return StrokeParams(xy, lw, angle, colour, lo + (hi - lo) * torch.sigmoid(s.alpha))


# ------------------------------------------------------------------ where strokes start
def goal_of(st: Stage, cat: dict) -> torch.Tensor:
    """What a layer is fitted to: the photo, or the photo blurred st.blur px within the cat only (a normalised blur,
    so the white background does not bleed in), the background left white."""
    if not st.blur:
        return cat["target"]
    m = (cat["tri"] == 1).astype(np.float32)
    img = cat["target"].numpy()
    den = ndimage.gaussian_filter(m, st.blur) + 1e-6
    num = np.stack([ndimage.gaussian_filter(img[..., c] * m, st.blur) for c in range(3)], -1)
    return torch.from_numpy(np.where(m[..., None] > 0, num / den[..., None], img).astype(np.float32))


def place_map(st: Stage, cat: dict, canvas: torch.Tensor, goal: torch.Tensor) -> np.ndarray:
    """How likely each pixel is to receive a stroke of a layer whose places are drawn at random."""
    tri, target = cat["tri"], goal.numpy()
    inside = ndimage.binary_erosion(tri == 1, iterations=st.inset) if st.inset else tri == 1
    if st.no_ears:
        inside = inside & ~cat["ears"]
    for zone in st.avoid:
        inside = inside & ~ndimage.binary_dilation(cat[zone], iterations=3)
    on_cat = inside.astype(np.float32)
    err = np.abs(target - canvas.numpy()).sum(-1) * on_cat
    if st.place == "error":  # where the painting is furthest from the photo
        return err
    lum = luminance(target)
    if st.place == "edges":  # edges of values and of the head, where the painting misses
        g = ndimage.gaussian_gradient_magnitude(lum, 2.0)
        rim = ndimage.binary_dilation(tri == 1, iterations=4) & ~ndimage.binary_erosion(tri == 1, iterations=4)
        m = (g / (g.max() + 1e-6) + 0.5 * rim) * ndimage.binary_dilation(tri == 1, iterations=2)
        if st.no_ears:
            m = m * ~cat["ears"]
        return m * (0.3 + err / (err.max() + 1e-6))
    if st.place == "zone":  # the part's misses
        return np.abs(target - canvas.numpy()).sum(-1) * cat[st.only]
    if st.place == "ear_dark":  # the ear's shadows: its darkest 30% in the photo
        dark = cat["ear_inner"] & (lum <= np.quantile(lum[cat["ear_inner"]], 0.3))
        return (np.abs(target - canvas.numpy()).sum(-1) + 0.05) * dark + 1e-12
    if st.place == "whiskers":  # light lines on the whisker pads
        light = light_lines(lum)
        pads = cat["whisker_pad"] * ndimage.binary_erosion(tri == 1, iterations=4)
        return (light + 0.05 * light.max()) * pads + 1e-12
    raise ValueError(f"unknown place {st.place!r}")


def muzzle_dots(cat: dict) -> tuple[np.ndarray, np.ndarray]:
    """The dark dots where the whiskers root: the photo's small dark blobs (LoG, radius 1-3.5 px) on the dot pads
    where the muzzle is light (blurred lightness at least DOT_RING), at least DOT_CONTRAST darker than a ring about
    them and round (the darkness curves both ways, not along a line); the DOT_MAX clearest, in rows of about 6 px,
    top to bottom and left to right; none if fewer than DOT_MIN. (centres (n, 2), radii (n,))."""
    lum = luminance(cat["photo"])
    zone = (
        cat["dot_pad"]
        & ndimage.binary_erosion(cat["tri"] == 1, iterations=2)
        & ~ndimage.binary_dilation(cat["nose"] | cat["mouth"], iterations=3)
        & (ndimage.gaussian_filter(lum, 4.0) >= DOT_RING)
    )
    if not zone.any():
        return np.zeros((0, 2)), np.zeros(0)
    ys, xs = np.nonzero(zone)
    y0, y1, x0, x1 = max(ys.min() - 8, 0), ys.max() + 9, max(xs.min() - 8, 0), xs.max() + 9
    sub = lum[y0:y1, x0:x1]
    yy, xx = np.mgrid[: sub.shape[0], : sub.shape[1]]
    curv = {}
    found = []
    for y, x, sg in blob_log(1 - sub, min_sigma=0.8, max_sigma=2.5, num_sigma=5, threshold=0.02):
        if not zone[int(y) + y0, int(x) + x0]:
            continue
        if sg not in curv:  # the dark side's curvatures at the blob's scale
            curv[sg] = hessian_matrix_eigvals(hessian_matrix(sub, sigma=sg, order="rc", use_gaussian_derivatives=True))
        l1, l2 = curv[sg][0][int(y), int(x)], curv[sg][1][int(y), int(x)]
        if min(l1, l2) <= 0 or min(l1, l2) < DOT_ROUND * max(l1, l2):
            continue
        r = max(1.0, sg * math.sqrt(2))
        dd = np.hypot(yy - y, xx - x)
        contrast = sub[(dd > r + 1) & (dd < r + 4)].mean() - sub[dd <= r].mean()
        if contrast >= DOT_CONTRAST:
            found.append((contrast, x + x0 + 0.5, y + y0 + 0.5, r))
    found = sorted(found, reverse=True)[:DOT_MAX] if len(found) >= DOT_MIN else []
    found.sort(key=lambda f: (round(f[2] / 6), f[1]))
    return np.array([f[1:3] for f in found]).reshape(-1, 2), np.array([f[3] for f in found])


def brow_roots(cat: dict) -> list[np.ndarray]:
    """Where a brow whisker may root, one mask a side: above the eye's outline, within its width (and BROW_SIDE eye
    distances either way), from 2 px up to BROW_UP eye distances above its top, on the head and off the ear's
    triangle grown 3 px."""
    p = cat["points"]
    d = float(np.linalg.norm(p[0] - p[1]))
    mid = 0.5 * (p[0, 0] + p[1, 0])
    yy, xx = np.mgrid[:SIDE, :SIDE] + 0.5
    ear = ndimage.binary_dilation(ear_triangles(p * 256 / SIDE, SIDE, SIDE / 256, tight=True), iterations=3)
    zones = []
    for e in (0, 1):
        eye = cat["eye"] & ((xx < mid) if p[e, 0] < mid else (xx >= mid))
        if eye.any():
            ys, xs = np.nonzero(eye)
            x0, x1, top = xs.min(), xs.max(), ys.min()
        else:
            x0, x1, top = p[e, 0] - 0.25 * d, p[e, 0] + 0.25 * d, p[e, 1] - 0.15 * d
        zones.append(
            (xx >= x0 - BROW_SIDE * d)
            & (xx <= x1 + BROW_SIDE * d)
            & (yy <= top - 2)
            & (yy >= top - BROW_UP * d)
            & (cat["tri"] == 1)
            & ~ear
        )
    return zones


def brow_whiskers(cat: dict) -> tuple[np.ndarray, np.ndarray]:
    """The long hairs above the eyes: the photo's light lines (Hessian ridges, the strongest 5% above each eye) found
    as straight segments at least 0.2 eye distances long, rooted (their lower end) in brow_roots, rising and not
    leaning in towards the middle; the longest BROW_MAX a side, no two rooted within 8 px at angles within 12 degrees.
    A side with fewer than BROW_MIN gets the rest by rule: rooted in brow_roots, running towards its ear's tip.
    (roots (n, 2), angles (n,) radians from the root along the hair)."""
    ridge = ridges(luminance(cat["photo"]))
    p = cat["points"]
    d = float(np.linalg.norm(p[0] - p[1]))
    mid = 0.5 * (p[0, 0] + p[1, 0])
    yy, xx = np.mgrid[:SIDE, :SIDE]
    roots, angles = [], []
    for (e, tip), ok in zip(((0, 4), (1, 7)), brow_roots(cat), strict=True):
        ex, ey = p[e]
        out = 1.0 if ex > mid else -1.0
        zone = (yy < ey - 0.15 * d) & (yy > ey - 1.4 * d) & (np.abs(xx - ex) < 0.9 * d)
        lines = probabilistic_hough_line(
            skeletonize((ridge > np.percentile(ridge[zone], 95)) & zone),
            threshold=10,
            line_length=int(0.2 * d),
            line_gap=5,
            rng=0,
        )
        cands = []
        for a, b in lines:
            lo, hi = np.array(a, float), np.array(b, float)
            if a[1] <= b[1]:  # lo: the lower end (larger y), the root
                lo, hi = hi, lo
            dx, dy = hi - lo
            length = math.hypot(dx, dy)
            if ok[int(lo[1]), int(lo[0])] and dy < -0.3 * length and dx * out >= -0.2 * length:
                cands.append((length, lo + 0.5, math.atan2(dy, dx)))
        kept: list[tuple[np.ndarray, float]] = []
        for _length, lo, ang in sorted(cands, key=lambda c: -c[0]):
            if all(
                np.linalg.norm(lo - k[0]) > 8 or abs(math.remainder(ang - k[1], 2 * math.pi)) > math.radians(12)
                for k in kept
            ):
                kept.append((lo, ang))
        kept = kept[:BROW_MAX]
        oy, ox = np.nonzero(ok)
        for k in range(len(kept), BROW_MIN if len(ox) else 0):  # by rule, from the brow
            want = np.array([ex + out * (0.05 + 0.1 * k) * d, oy.min() + 0.6 * (oy.max() - oy.min())])
            j = int(np.argmin((ox + 0.5 - want[0]) ** 2 + (oy + 0.5 - want[1]) ** 2))
            lo = np.array([ox[j] + 0.5, oy[j] + 0.5])
            to = p[tip] - lo
            kept.append((lo, math.atan2(to[1], to[0]) + math.radians(8 * (k - 0.5))))
        roots += [k[0] for k in kept]
        angles += [k[1] for k in kept]
    return np.array(roots).reshape(-1, 2), np.array(angles)


def lid_places(cat: dict, inset: float = 0.0, step: float = LID_STEP) -> tuple[np.ndarray, np.ndarray]:
    """Places `step` px apart along each eye outline, each along the outline's tangent, moved `inset` px in towards
    the eye's middle; the outline resampled a pixel apart and smoothed along its length first (LID_SMOOTH px)."""
    xy, ang = [], []
    for o in cat["eye_outlines"]:
        q = np.r_[o, o[:1]]
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        step1 = np.arange(0, seg[-1], 1.0)
        dense = np.c_[np.interp(step1, seg, q[:, 0]), np.interp(step1, seg, q[:, 1])]
        dense = ndimage.gaussian_filter1d(dense, LID_SMOOTH, axis=0, mode="wrap")
        q = np.r_[dense, dense[:1]]
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        at = np.arange(0, seg[-1], step)
        p = np.c_[np.interp(at, seg, q[:, 0]), np.interp(at, seg, q[:, 1])]
        t = np.roll(p, -1, 0) - np.roll(p, 1, 0)
        if inset:
            to_mid = p.mean(0) - p
            p = p + inset * to_mid / (np.linalg.norm(to_mid, axis=1, keepdims=True) + 1e-9)
        xy.append(p)
        ang.append(np.arctan2(t[:, 1], t[:, 0]))
    return np.concatenate(xy), np.concatenate(ang)


def catchlights(cat: dict) -> tuple[np.ndarray, np.ndarray]:
    """The photo's catchlights: in each eye, the lightest small glints (at least CATCH_MIN, 0.3 over the eye's median
    lightness and CATCH_GLINT over the 9 px median about them; 2 px to CATCH_SIZE of the eye's width across), the
    CATCH_MAX lightest. (centres (n, 2), diameters (n,))."""
    lum = luminance(cat["photo"])
    glint = lum - ndimage.median_filter(lum, 9) >= CATCH_GLINT
    xy, dia = [], []
    for o in cat["eye_outlines"]:
        im = Image.new("L", (SIDE, SIDE), 0)
        ImageDraw.Draw(im).polygon([tuple(q) for q in o], fill=255)
        eye = ndimage.binary_erosion(np.asarray(im) > 127, iterations=1)
        if eye.sum() < 20:
            continue
        thr = max(CATCH_MIN, float(np.median(lum[eye])) + 0.3)
        width = float(np.ptp(o[:, 0]))
        lab, n = ndimage.label(eye & (lum >= thr) & glint)
        blobs = []
        for j in range(1, n + 1):
            m = lab == j
            if m.sum() >= 2 and 2 * math.sqrt(m.sum() / math.pi) <= CATCH_SIZE * width:
                ys, xs = np.nonzero(m)
                w = lum[m]
                blobs.append(
                    (
                        float(w.max()),
                        np.array([np.average(xs, weights=w), np.average(ys, weights=w)]) + 0.5,
                        float(np.clip(2 * math.sqrt(m.sum() / math.pi), 2.5, 9.0)),
                    )
                )
        for _, c, dd in sorted(blobs, key=lambda b: -b[0])[:CATCH_MAX]:
            xy.append(c)
            dia.append(dd)
    return np.array(xy).reshape(-1, 2), np.array(dia)


# ------------------------------------------------------------------ the ears
def follow(start: np.ndarray, way: int, along: np.ndarray, walk: np.ndarray, steps: int) -> list[np.ndarray]:
    """Places 1 px apart from start along a hair (the angles `along`, first `way` 1 or -1), until it leaves `walk`."""
    q, d, run = start.copy(), None, []
    for _ in range(steps):
        t = along[int(q[1]), int(q[0])]
        v = np.array([math.cos(t), math.sin(t)]) * (way if d is None else 1)
        v = -v if d is not None and v @ d < 0 else v
        q, d = q + v, v
        if not (0 <= q[0] < SIDE and 0 <= q[1] < SIDE and walk[int(q[1]), int(q[0])]):
            break
        run.append(q.copy())
    return run


def hair_direction(lum: np.ndarray) -> np.ndarray:
    """The photo's own hair direction at each pixel (radians, either way along it): the structure tensor of the
    lightness (gradients at 1 px, the tensor blurred 2.5 px), across the strongest change."""
    gx, gy = ndimage.gaussian_filter(lum, 1.0, order=(0, 1)), ndimage.gaussian_filter(lum, 1.0, order=(1, 0))
    jxx, jyy, jxy = (ndimage.gaussian_filter(t, 2.5) for t in (gx * gx, gy * gy, gx * gy))
    return 0.5 * np.arctan2(2 * jxy, jxx - jyy) + math.pi / 2


def tip_of(root: np.ndarray, tip: np.ndarray, path: np.ndarray, over: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """A traced hair's tip moved on along it, 1 px at a time (at most TRACE_TAIL of its length), while the photo stays
    at least TRACE_HALF as much lighter than about it (`over`: lightness over its 3 px blur, the lightest of 3 lines
    1 px apart across) as the traced hair is at its median, inside the ear: the trace stops where the hair is no
    longer a clear ridge, but a hair fades out towards its tip."""

    def light(q: np.ndarray, across: np.ndarray) -> float:
        at = np.clip((q + np.outer((-1, 0, 1), across)).astype(int), 0, SIDE - 1)
        return float(over[at[:, 1], at[:, 0]].max()) if inner[at[1, 1], at[1, 0]] else -np.inf

    ix = np.clip(path.astype(int), 0, SIDE - 1)
    keep = TRACE_HALF * np.median(over[ix[:, 1], ix[:, 0]])
    u = (tip - root) / (np.linalg.norm(tip - root) + 1e-9)
    across, step = np.array([-u[1], u[0]]), 0
    while step < TRACE_TAIL * np.linalg.norm(tip - root) and light(tip + (step + 1) * u, across) >= keep:
        step += 1
    return tip + step * u


def ear_traced(cat: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ears' white hairs traced from the photo: the thin light lines inside the ears (ridges above TRACE_TOP
    percent of the ears', lighter than their median); from the strongest one not yet taken, followed both ways along
    the photo's own direction until the light line ends or TRACE_MAX of the ear's height, and the TRACE_CLAIM px round
    it taken; up to TRACE_CAP, those TRACE_LEN px or longer kept. Each becomes one clump rooted at its end nearer the
    ear's base, as long as the hair (its faint tip too: tip_of), TRACE_WIDTH wide; painted from the base up.
    (centres (n, 2), angles (n,) from the root, sizes (n, 2) px)"""
    p = cat["points"]
    lum = luminance(cat["photo"])
    ridge = ridges(lum)
    inner = cat["ear_inner"]
    hair = (ridge > np.percentile(ridge[inner], TRACE_TOP)) & inner & (lum > np.median(lum[inner]))
    over = lum - ndimage.gaussian_filter(lum, 3.0)
    walk = ndimage.binary_dilation(hair, iterations=1)
    along = hair_direction(lum)
    height = np.mean([np.linalg.norm(p[t] - 0.5 * (p[a] + p[b])) for a, t, b in ((3, 4, 5), (6, 7, 8))])
    strength, free = np.where(hair, ridge, 0.0), np.ones((SIDE, SIDE), bool)
    mid = 0.5 * (p[0, 0] + p[1, 0])
    xy: list[np.ndarray] = []
    ang, size, up_at = [], [], []
    while len(xy) < TRACE_CAP:
        k = int(np.argmax(np.where(free, strength, 0)))
        if not free.flat[k] or strength.flat[k] <= 0:
            break
        start = np.array([k % SIDE + 0.5, k // SIDE + 0.5])
        steps = int(TRACE_MAX * height)
        path = np.array(follow(start, -1, along, walk, steps)[::-1] + [start] + follow(start, 1, along, walk, steps))
        taken = np.zeros((SIDE, SIDE), bool)
        ix = np.clip(path.astype(int), 0, SIDE - 1)
        taken[ix[:, 1], ix[:, 0]] = True
        free &= ~ndimage.binary_dilation(taken, iterations=TRACE_CLAIM)
        a, b = path[0], path[-1]
        length = float(np.linalg.norm(b - a))
        if length < TRACE_LEN:
            continue
        b_in, tip, b_out = (5, 4, 3) if a[0] + b[0] < 2 * mid else (6, 7, 8)
        base = 0.5 * (p[b_in] + p[b_out])
        up = p[tip] - base
        if (a - base) @ up > (b - base) @ up:  # root at the end nearer the base
            a, b = b, a
        b = tip_of(a, b, path, over, inner)
        length = float(np.linalg.norm(b - a))
        lo, share, hi = TRACE_WIDTH
        xy.append(a + 0.5 * TRACE_REACH * (b - a))
        ang.append(math.atan2(b[1] - a[1], b[0] - a[0]))
        size.append((TRACE_REACH * length, float(np.clip(share * length, lo, hi))))
        up_at.append(((a - base) @ up) / (up @ up))
    order = np.argsort(up_at, kind="stable")
    return np.array(xy).reshape(-1, 2)[order], np.array(ang)[order], np.array(size).reshape(-1, 2)[order]


def ear_edges(cat: dict, inset: float, step: float, inner_only: bool) -> tuple[np.ndarray, np.ndarray]:
    """Places `step` px apart along the painter's outline where it runs round an ear (the ear's triangle grown, above
    the eyes), along the outline, moved `inset` px in towards the ear's middle; inner_only: only the edge from the
    base by the face to the tip, where the dark back of the ear shows from the front."""
    c = cat["contour"]
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    at = np.arange(0, seg[-1], step)
    pts = np.c_[np.interp(at, seg, c[:, 0]), np.interp(at, seg, c[:, 1])]
    t = np.roll(pts, -1, 0) - np.roll(pts, 1, 0)
    ang = np.arctan2(t[:, 1], t[:, 0])
    p = cat["points"]
    ix, iy = np.clip(pts.astype(int), 0, SIDE - 1).T
    on_ear = ndimage.binary_dilation(cat["ears"], iterations=4)[iy, ix] & (pts[:, 1] < p[:2, 1].min())
    keep, out_xy = np.zeros(len(pts), bool), pts.copy()
    for b_in, tip, b_out in ((5, 4, 3), (6, 7, 8)):
        mid = p[[b_in, tip, b_out]].mean(0)
        near = on_ear & (np.linalg.norm(pts - mid, axis=1) < 1.2 * np.linalg.norm(p[tip] - mid))
        if inner_only:

            def dist(a, b):
                ab = b - a
                f = np.clip(((pts - a) @ ab) / (ab @ ab), 0, 1)
                return np.linalg.norm(pts - (a + f[:, None] * ab), axis=1)

            near &= dist(p[b_in], p[tip]) < dist(p[b_out], p[tip])
        to_mid = mid - pts[near]
        out_xy[near] = pts[near] + inset * to_mid / (np.linalg.norm(to_mid, axis=1, keepdims=True) + 1e-9)
        keep |= near
    return out_xy[keep], ang[keep]


def fixed_places(st: Stage, cat: dict) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """The places a rule, the plan or the photo sets: (xy (n, 2), angles or None, sizes (n, 2) px or None)."""
    if st.place == "dots":
        xy, radius = muzzle_dots(cat)
        return xy, None, np.c_[2 * radius, 2 * radius]
    if st.place == "pupil_plan":  # one stroke a pupil: the plan's ellipse
        p = np.array(cat["pupils"]).reshape(-1, 5)
        return p[:, :2], p[:, 4], np.c_[2 * p[:, 2], 2 * p[:, 3]]
    if st.place == "lid":
        xy, ang = lid_places(cat)
        return xy, ang, None
    if st.place == "lid_shadow":
        xy, ang = lid_places(cat, SHADOW_IN, SHADOW_STEP)
        return xy, ang, None
    if st.place in ("ear_back", "ear_fill", "ear_rim"):
        inset, step = {
            "ear_back": (EAR_BACK_IN, 5.0),
            "ear_fill": (EAR_FILL_IN, 4.0),
            "ear_rim": (EAR_RIM_IN, LID_STEP),
        }[st.place]
        xy, ang = ear_edges(cat, inset, step, st.place == "ear_back")
        return xy, ang, None
    if st.place == "ear_traced":
        return ear_traced(cat)
    if st.place == "catchlight":
        xy, dia = catchlights(cat)
        return xy, None, np.c_[dia, dia]
    if st.place == "brow":
        xy, ang = brow_whiskers(cat)
        return xy, ang, None
    raise ValueError(f"unknown fixed place {st.place!r}")


def photo_colours(st: Stage, cat: dict, xy: np.ndarray, sizes: np.ndarray | None) -> np.ndarray:
    """Colours read from the photo at each place: "light", the brightest (st.light_top) of the light-line pixels
    within WHISKER_NEAR px (a whisker is thinner than a pixel, so the rest of a light line is mixed with the fur);
    "dark", the darkest DOT_DARK of a dot's pixels (lids: of the rim about the point); "bright", the lightest 30% of
    a catchlight's."""
    img = cat["photo"]
    lum = luminance(img)
    light = light_lines(lum) * (cat["tri"] == 1)
    yy, xx = np.mgrid[:SIDE, :SIDE] + 0.5
    cols = []
    for i, (x, y) in enumerate(xy):
        if st.photo_colour in ("dark", "bright"):
            r = 0.5 * sizes[i].min() if sizes is not None else st.width
            near = (xx - x) ** 2 + (yy - y) ** 2 <= (r + 1) ** 2
            if st.photo_colour == "dark":
                pick = lum <= np.quantile(lum[near], DOT_DARK)
            else:
                pick = lum >= np.quantile(lum[near], 0.7)
            cols.append(img[near & pick].mean(0))
            continue
        near = ((xx - x) ** 2 + (yy - y) ** 2 < WHISKER_NEAR**2) & (light > 0)
        if near.sum() < 3:
            near = (xx - x) ** 2 + (yy - y) ** 2 < WHISKER_NEAR**2
        cols.append(img[near & (lum >= np.quantile(lum[near], st.light_top))].mean(0))
    return np.array(cols, np.float32)


# ------------------------------------------------------------------ fitting one layer
def fit_stage(
    st: Stage,
    cat: dict,
    canvas: torch.Tensor,
    rng: np.random.Generator,
    device: torch.device | str = "cpu",
    iters: int | None = None,
) -> tuple[StrokeParams, torch.Tensor, Stage] | None:
    """Place one layer's strokes, fit them to the photo and lay them on the canvas. Returns the fitted strokes (on the
    CPU), the canvas after them and the layer with its real stroke count, or None when the layer has no places in
    this photo (no whisker dots, say). Fitting runs on `device`; the places are drawn from `rng`."""
    device = torch.device(device)
    target = goal_of(st, cat)
    base, sizes = None, None
    if st.place in FIXED_PLACES:  # places set by rule, plan or photo
        xy_np, ang, sizes = fixed_places(st, cat)
        if not len(xy_np):
            return None
        base = None if ang is None else torch.tensor(ang, dtype=torch.float32)
        st = dataclasses.replace(st, n=len(xy_np))
        px, py = np.clip(xy_np.astype(int), 0, SIDE - 1).T
        idx = py * SIDE + px
        xy = torch.tensor(xy_np, dtype=torch.float32)
    else:  # places drawn from the layer's place map
        p_map = place_map(st, cat, canvas, target).flatten() + 1e-9
        idx = rng.choice(p_map.size, st.n, replace=False, p=p_map / p_map.sum())
        idx = idx[np.argsort(idx // SIDE, kind="stable")]  # painted top to bottom
        py, px = idx // SIDE, idx % SIDE
        xy = torch.tensor(np.c_[px, py] + 0.5, dtype=torch.float32)
        if st.radial:  # rooted at the place, running away from the muzzle
            away = xy.numpy() - np.broadcast_to(cat["whisker_origin"], xy.shape)
            base = torch.tensor(np.arctan2(away[:, 1], away[:, 0]), dtype=torch.float32)
    small = F.interpolate(target.permute(2, 0, 1)[None], size=(256, 256), mode="area")[0].permute(1, 2, 0)
    start_angle = orientation(small)  # free strokes start along the photo's local structure
    a0 = F.interpolate(start_angle[None, None], size=(SIDE, SIDE), mode="nearest")[0, 0].flatten()[torch.as_tensor(idx)]
    size = torch.zeros(st.n, 2)
    if base is not None:  # strokes with a set direction vary in size by their hash
        seed = seed_of(xy.numpy(), base.numpy(), SIDE)
        lo, hi = VARY
        vary = torch.stack([lo + (hi - lo) * hash01(seed, torch.full_like(seed, salt)) for salt in (23.0, 29.0)], -1)
        size = logit((vary - 0.5).clamp(0.02, 0.98))
    if sizes is not None:  # each its own size (dots, pupils, catchlights, ear hairs)
        f = torch.tensor(sizes / np.array([st.length, st.width]), dtype=torch.float32)
        lo_s, hi_s = st.size_range
        size = logit(((f - lo_s) / (hi_s - lo_s)).clamp(0.02, 0.98))
    under: torch.Tensor | None = None
    if st.tint is not None:  # the ear's rim: the paint just inside it
        ear_mid = [cat["points"][[5, 4, 3]].mean(0), cat["points"][[6, 7, 8]].mean(0)]
        to_mid = np.array([ear_mid[0] if x < cat["centre"][0] else ear_mid[1] for x in xy.numpy()[:, 0]]) - xy.numpy()
        at = xy.numpy() + 4 * to_mid / (np.linalg.norm(to_mid, axis=1, keepdims=True) + 1e-9)
        soft = np.stack([ndimage.gaussian_filter(canvas[..., c].numpy(), TINT_BLUR) for c in range(3)], -1)
        ux, uy = np.clip(at, 0, SIDE - 1).astype(int).T
        under = torch.from_numpy(soft[uy, ux].astype(np.float32))
    if st.photo_colour:  # coloured as the photo at the place
        rgb = photo_colours(st, cat, xy.numpy(), sizes)
        if st.emphasis:  # stronger than the photo, as a painter would
            rgb = rgb * (1 - st.emphasis) if st.photo_colour == "dark" else rgb + st.emphasis * (1 - rgb)
        under = torch.from_numpy(rgb.astype(np.float32))
    s = Strokes(
        xy=logit(xy / SIDE),
        size=size,
        angle=a0.clone() if st.turn_deg is None else torch.zeros(st.n),
        colour=(
            logit(target[torch.as_tensor(py), torch.as_tensor(px)].clamp(0.02, 0.98))
            if under is None
            else torch.zeros(st.n, 3)
        ),
        alpha=torch.zeros(st.n),
        base=base,
        under=under,
    )
    s = Strokes(*(t.to(device) for t in s.tensors()), *(None if t is None else t.to(device) for t in (base, under)))
    target, cat = target.to(device), {**cat, "field": cat["field"].to(device)}
    weight, protect = fit_weights(st, cat, device)
    fitted = (
        ([s.size] if st.fit_size else [])
        + [s.angle, s.alpha]
        + ([] if st.photo_colour else [s.colour])
        + ([] if st.fixed_place else [s.xy])
    )
    for t in fitted:
        t.requires_grad_(True)
    opt = torch.optim.Adam(fitted, lr=LR)
    start = canvas.detach().to(device)
    brush = st.brush
    for _ in range(ITERS if iters is None else iters):
        p = stroke_params(s, st, cat)
        out = lay(start, start, p, stroke_seeds(p), brush)
        loss = ((out - target).abs() * weight).mean() + st.protect * ((out - start).abs().mean(-1) * protect).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = stroke_params(s, st, cat)
        bad = [f for f, t in zip(p._fields, p, strict=True) if not torch.isfinite(t).all()]
        if bad:
            raise FloatingPointError(f"{st.name}: strokes not finite after fitting ({', '.join(bad)}) on {device}")
        out = lay(start, start, p, stroke_seeds(p), brush)
        return StrokeParams(*(t.detach().cpu() for t in p)), out.cpu(), st


def fit_weights(st: Stage, cat: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The loss's weights: (SIDE, SIDE, 1) 1 where the photo counts (not the uncertain pixels, nor the free band past
    the head) and (SIDE, SIDE) 1 where changing the canvas is penalised (the paper, outside the layer's part, the
    parts it avoids)."""
    tri = cat["tri"].copy()
    if st.band:
        tri[ndimage.binary_dilation(tri == 1, iterations=st.band) & (tri == 0)] = 2
    weight = torch.from_numpy((tri != 2).astype(np.float32))[..., None].to(device)
    outside = (tri == 0) if st.only is None else ~ndimage.binary_dilation(cat[st.only], iterations=2)
    for zone in st.avoid:
        outside = outside | ndimage.binary_dilation(cat[zone], iterations=3)
    return weight, torch.from_numpy(outside.astype(np.float32)).to(device)


# ------------------------------------------------------------------ the seven stages, eyes apart
Layer = tuple[int, int, StrokeParams, Stage]  # (stage, layer index j, strokes, layer)


@dataclass
class StageResult:
    """A painted stage: the canvas before and after it and its layers in the order they were fitted."""

    stage: int
    before: torch.Tensor
    canvas: torch.Tensor
    layers: list[Layer]
    seconds: float


def paint_eyes_apart(
    cat: dict, canvas: torch.Tensor, layers: list[Layer], device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Stage 2: the left eye's layers, then the right's, each eye with its own random places and its share of the
    strokes by its area (fitted strokes appended to `layers`)."""
    eyes = [one_eye(cat, side) for side in (0, 1)]
    area = np.array([e["eye"].sum() for e in eyes], np.float64)
    for side, eye in enumerate(eyes):
        rng = np.random.default_rng([EYE_STAGE, side])
        for j, st in enumerate(STAGES[EYE_STAGE]):
            n = st.n if st.fixed_place else round(st.n * area[side] / max(area.sum(), 1))
            res = fit_stage(dataclasses.replace(st, n=n), eye, canvas, rng, device) if st.fixed_place or n else None
            if res is not None:
                p, canvas, st2 = res
                layers.append((EYE_STAGE, j, p, st2))
    return canvas


def paint_without_eyes(
    stage: int,
    seen: dict,
    canvas: torch.Tensor,
    plain: torch.Tensor,
    layers: list[Layer],
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """A stage other than the eyes: fitted to `seen` on the canvas without eyes (`plain`), laid on both canvases."""
    rng = np.random.default_rng(stage)
    for j, st in enumerate(STAGES[stage]):
        res = fit_stage(st, seen, plain, rng, device)
        if res is None:
            continue
        p, plain, st2 = res
        canvas = lay_layer(canvas, StrokeParams(*(t.to(device) for t in p)), st2.brush)
        layers.append((stage, j, p, st2))
    return canvas, plain


def paint_apart(
    cat: dict,
    inputs: CatInputs,
    device: torch.device | str = "cpu",
    on_stage: Callable[[StageResult], None] | None = None,
) -> tuple[torch.Tensor, list[StageResult]]:
    """All seven stages with the eyes apart: every stage but the eyes is fitted to without_eyes(cat) on a canvas
    that never had eyes (and laid on the real canvas too); the eyes are painted one at a time. on_stage is called
    with each stage as it finishes. (the final canvas, the stages)"""
    seen = without_eyes(cat, inputs)
    canvas = plain = torch.ones(SIDE, SIDE, 3)
    results = []
    for stage in sorted(STAGES):
        torch.manual_seed(stage)
        t0, before = time.time(), canvas
        layers: list[Layer] = []
        if stage == EYE_STAGE:
            canvas = paint_eyes_apart(cat, canvas, layers, device)
        else:
            canvas, plain = paint_without_eyes(stage, seen, canvas, plain, layers, device)
        results.append(StageResult(stage, before, canvas, layers, round(time.time() - t0, 1)))
        if on_stage is not None:
            on_stage(results[-1])
    return canvas, results


def paint_eye_versions(
    versions: list[dict], inputs: CatInputs, device: torch.device | str = "cpu"
) -> list[tuple[torch.Tensor, list[Layer]]]:
    """paint_apart for several versions of one cat that differ only inside the eyes (e.g. recoloured eyes): the
    stages other than the eyes read nothing inside the eyes, so they are fitted once and every version gets only its
    own eyes, laid layer by layer as paint_apart lays them, so each version's result equals paint_apart's.
    versions: prepared cats; inputs: the first version's. [(canvas, layers)] by version"""
    cat = versions[0]
    zone = ndimage.binary_dilation(cat["eye"], iterations=EYE_APART_GROW)
    source = (cat["tri"] == 1) & ~zone
    for v in versions[1:]:
        for k in ("photo", "target"):
            if not np.array_equal(np.asarray(v[k])[source], np.asarray(cat[k])[source]):
                raise ValueError(f"a version differs outside the eyes: {k}")
    seen = without_eyes(cat, inputs)
    canvas = plain = torch.ones(SIDE, SIDE, 3)
    before: list[Layer] = []
    after: list[Layer] = []
    at_eyes = None
    for stage in sorted(STAGES):
        if stage == EYE_STAGE:
            at_eyes = canvas
            continue
        torch.manual_seed(stage)
        canvas, plain = paint_without_eyes(stage, seen, canvas, plain, after if at_eyes is not None else before, device)
    assert at_eyes is not None  # EYE_STAGE is one of STAGES
    out = []
    for v in versions:
        torch.manual_seed(EYE_STAGE)
        eyes: list[Layer] = []
        canvas = paint_eyes_apart(v, at_eyes, eyes, device)
        for _, _, p, st in after:
            canvas = lay_layer(canvas, StrokeParams(*(t.to(device) for t in p)), st.brush)
        out.append((canvas, before + eyes + after))
    return out


# ------------------------------------------------------------------ saving and replaying
def stage_arrays(result: StageResult, points: np.ndarray) -> dict[str, np.ndarray]:
    """A painted stage as stage<k>.npz stores it: the canvas after it and, per layer j that painted anything, its
    strokes (both eyes' strokes of a stage 2 layer together, left eye first, with "<j>_side")."""
    saved: dict[str, np.ndarray] = {"canvas": result.canvas.numpy()}
    for j, st in enumerate(STAGES[result.stage]):
        ps = [p for _, jj, p, _ in result.layers if jj == j]
        if not ps:
            continue
        p = StrokeParams(*(torch.cat(f) for f in zip(*ps, strict=True)))
        for f, v in zip(p._fields, p, strict=True):
            saved[f"{j}_{f}"] = v.numpy()
        saved[f"{j}_kind"] = np.array(st.kind)
        if result.stage == EYE_STAGE:
            saved[f"{j}_side"] = eye_side(p.xy.numpy(), points)
    return saved


def stage_layers(arrays: Mapping[str, np.ndarray], stage: int) -> list[tuple[int, int, StrokeParams]]:
    """A stage's layers in painting order from its stored arrays; stage 2 one eye at a time (every layer of the
    left eye, then of the right), as it was painted."""
    files = set(arrays.keys())
    layers = []
    for j in range(len(STAGES[stage])):
        if f"{j}_xy" in files:
            p = StrokeParams(*(torch.from_numpy(np.asarray(arrays[f"{j}_{f}"])) for f in StrokeParams._fields))
            side = np.asarray(arrays[f"{j}_side"]) if f"{j}_side" in files else None
            layers.append((j, p, side))
    if stage != EYE_STAGE:
        return [(stage, j, p) for j, p, _ in layers]
    out = []
    for eye in (0, 1):
        for j, p, side in layers:
            keep = torch.from_numpy(side == eye)
            if keep.any():
                out.append((stage, j, StrokeParams(*(t[keep] for t in p))))
    return out


def painting_layers(folder: Path | str) -> list[tuple[int, int, StrokeParams]]:
    """Every layer of a saved painting (stage1.npz .. stage7.npz in `folder`) in painting order."""
    out = []
    for stage in sorted(STAGES):
        with np.load(Path(folder) / f"stage{stage}.npz") as z:
            out += stage_layers(z, stage)
    return out


def replay(layers: list[tuple[int, int, StrokeParams]], start: torch.Tensor | None = None) -> torch.Tensor:
    """Layers (stage, j, strokes) laid in order, on white paper unless `start` is given."""
    if start is None:
        return render([(layer_brush(k, j), p) for k, j, p in layers], SIDE)
    canvas = start
    for k, j, p in layers:
        canvas = lay_layer(canvas, p, layer_brush(k, j))
    return canvas


def l1_on_cat(canvas: torch.Tensor, cat: dict) -> float:
    """Mean absolute difference to the target on the head."""
    return float((canvas - cat["target"]).abs()[torch.from_numpy(cat["tri"] == 1)].mean())


def save_stage(folder: Path, result: StageResult, cat: dict, device: str) -> dict:
    """Write stage<k>.npz and append the stage's row to log.jsonl (strokes, seconds, L1 on the head before and after;
    for stage 2, how far replaying it one eye at a time lands from the fitted canvas). Returns the row."""
    arrays = stage_arrays(result, cat["points"])
    # mypy checks **arrays against savez_compressed's allow_pickle: bool; the keys are array names
    np.savez_compressed(folder / f"stage{result.stage}.npz", **arrays)  # type: ignore[arg-type]
    row = {
        "stage": result.stage,
        "layers": [s.name for s in STAGES[result.stage]],
        "strokes": sum(len(p.xy) for _, _, p, _ in result.layers),
        "seconds": result.seconds,
        "device": device,
        "l1_cat_before": round(l1_on_cat(result.before, cat), 4),
        "l1_cat_after": round(l1_on_cat(result.canvas, cat), 4),
    }
    if result.stage == EYE_STAGE:
        end = replay(stage_layers(arrays, EYE_STAGE), result.before)
        row["eye_order_max_diff"] = float((end - result.canvas).abs().max())
    with open(folder / "log.jsonl", "a") as log:
        log.write(json.dumps(row) + "\n")
    return row


def paint_cat(
    inputs: CatInputs,
    folder: Path | str,
    device: torch.device | str = "cpu",
    on_row: Callable[[dict], None] | None = None,
) -> torch.Tensor:
    """Paint one cat's seven stages with the eyes apart into `folder` (stage1.npz .. stage7.npz, log.jsonl).
    Returns the finished canvas."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "log.jsonl").unlink(missing_ok=True)
    cat = prepare_cat(inputs)

    def write(result: StageResult) -> None:
        row = save_stage(folder, result, cat, str(device))
        if on_row is not None:
            on_row(row)

    canvas, _ = paint_apart(cat, inputs, device, write)
    return canvas
