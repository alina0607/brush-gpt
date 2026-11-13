"""The cat as the painter reads it: the photo, the head's outline and the face's parts as masks on a 512 px canvas.

Inputs (`CatInputs`), all per cat:

  cut-out   the photo cut to a square box around the face at 256 px, white where the subject mask says background
            (`target`), a tri map (1 cat, 2 uncertain, 0 background), the nine face points on that grid and the box
            in the photo's pixels
  photo     the original photo (the box is cut from it again at 512 px)
  plan      the sketch plan: each eye's outline, each pupil's ellipse, the nose's outline, the mouth's line (256 grid)

The head's outline. A subject mask keeps the whole cat, so the head is cut from it with a rounded face shape built
from the nine points (a superellipse from the ear bases to below the chin, plus generous ear triangles). Where the
body goes on past that shape, the line is moved to the photo's colour edge when a clear one is near, then smoothed.
The painter's outline is that head with its thin spurs and notches removed and its contour smoothed (less on the
ears, so their tips stay sharp). The photo is not trusted right at the cut-out's edge (it carries the background's
colour); there, and out to a band past the outline, the target colour is filled in from the trusted fur nearby.

The plan's parts become masks (eyes, the lids' rim, pupils, nose, mouth) and two zones in the face's own frame: the
whisker pads beside the nose and mouth, and the smaller pads nearer the nose where the dark whisker dots are.

Eyes apart. Every stage but the eyes is fitted to the cat seen without its eyes (`without_eyes`: the photo inside
the eyes filled from the fur around them, and the fur field recomputed the same way), so the strokes outside an eye
never depend on what is inside it and an eye can be repainted alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from matplotlib.path import Path as Polygon
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.measure import find_contours

from .fur import fur_field_at

SIDE = 512  # the painter's canvas
INPUT_SIDE = 256  # the cut-out's and the face points' grid
EARS = ((3, 4, 5), (6, 7, 8))  # each ear's face points: base, tip, base
# the head shape built from the face points (face frame: eye distances, y towards the mouth)
HEAD_CHIN, HEAD_SIDE, HEAD_POWER = 0.45, 0.10, 2.2  # chin below the mouth, cheeks beside the ear bases, roundness
EAR_SCALE, EAR_PAD = 1.3, 0.08  # how generous the ear triangles are (the mask gives the ear's real edge)
# moving the head's line onto the photo's colour edge where the body goes on past it
SAMPLES = 240  # points along the head shape's outline
IN_REACH, OUT_REACH = 0.15, 0.25  # how far the line may move in and out, eye distances
STEP_HALF = 0.08  # the colour step's half window, eye distances
CLEAR_Q, MIN_EDGE = 0.90, 0.12  # an edge is clear above the face's CLEAR_Q texture step and MIN_EDGE (RGB 0-1)
MEDIAN, SIGMA = 9, 4.0  # smoothing along the line, in samples
# the painter's outline and target
OPEN = 0.06  # spurs and notches narrower than this are removed (eye distances; not the ears)
SMOOTH_FACE, SMOOTH_EAR = 0.06, 0.012  # Gaussian along the contour (eye distances)
TRUST = 0.05  # the photo is not trusted this close to the cut-out's edge (eye distances)
BAND = 0.20  # the target is extended this far past the outline, for fur (eye distances)
FILL_SCALES, FILL_ENOUGH = (2.0, 4.0, 8.0, 16.0, 32.0), 0.3  # px at 512; share of trusted pixels a scale needs
EAR_SHRINK = 0.85  # the inside of an ear: its triangle shrunk about its middle
# the plan's parts
RIM_PX, PUPIL_GROW, MOUTH_PX = 3, 1.2, 3  # the lids' band either side of the eye outline, pupils' margin, mouth
PAD_TOP, PAD_BELOW, PAD_GAP, PAD_OUT = 0.10, 0.10, 0.03, 0.75  # whisker pads, eye distances (see plan_zones)
DOTS_BELOW, DOTS_GAP, DOTS_OUT = 0.08, 0.04, 0.45  # the whisker dots' pads, nearer the nose
# eyes apart
EYE_APART_GROW = 6  # px about the eye outlines whose photo the other stages never read
EYE_FILL_BLUR = 4.0  # px: the filled eyes smoothed


@dataclass
class CatInputs:
    """One cat's inputs to the painter (see the module docstring)."""

    target: np.ndarray  # (256, 256, 3) uint8, the cut-out on white
    tri: np.ndarray  # (256, 256) uint8: 1 cat, 2 uncertain, 0 background
    points: np.ndarray  # (9, 2) face points on the 256 grid
    box: tuple[int, int, int, int]  # the cut-out's box in the photo (left, top, right, bottom)
    photo: Image.Image  # the original photo, RGB
    plan: dict[str, np.ndarray]  # eyes (k, m, 2), pupils (k, 6), nose (m, 2), mouth (m, 2) on the 256 grid


def load_inputs(cutout: Path | str, photo: Path | str, plan: Path | str) -> CatInputs:
    """CatInputs from the cut-out .npz (target, tri, points, box), the photo file and the sketch plan .npz."""
    z = np.load(cutout, allow_pickle=True)
    p = np.load(plan, allow_pickle=True)
    return CatInputs(
        target=z["target"],
        tri=z["tri"],
        points=z["points"],
        box=cast(tuple[int, int, int, int], tuple(int(v) for v in z["box"])),
        photo=Image.open(photo).convert("RGB"),
        plan={k: p[k] for k in ("eyes", "pupils", "nose", "mouth")},
    )


def cutout_meta(cutout: Path | str) -> dict:
    """The cut-out's metadata (its photo's source and id)."""
    return json.loads(str(np.load(cutout, allow_pickle=True)["meta"]))


# ------------------------------------------------------------------ the face's frame and the head shape
def face_frame(p: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The face's frame: eye distance, the eyes' midpoint, the unit x along the eyes and y towards the mouth, and the
    points in that frame (eye distances)."""
    d = float(np.linalg.norm(p[0] - p[1]))
    mid = p[:2].mean(0)
    ex = (p[1] - p[0]) / d
    ey = np.array([-ex[1], ex[0]])
    if np.dot(p[2] - mid, ey) < 0:
        ey = -ey
    return d, mid, ex, ey, np.stack([(p - mid) @ ex, (p - mid) @ ey], -1) / d


def head_arc(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The head shape's superellipse as SAMPLES points (px) and their outward unit normals."""
    d, mid, ex, ey, P = face_frame(p)
    bases = P[[3, 5, 6, 8]]
    a = max(float(np.abs(bases[:, 0]).max()), 1.0) + HEAD_SIDE
    top, bot = float(bases[:, 1].min()), float(P[2, 1]) + HEAD_CHIN
    cy, b, k = (top + bot) / 2, (bot - top) / 2, HEAD_POWER
    t = np.linspace(0, 2 * np.pi, SAMPLES, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    qx, qy = a * np.sign(c) * np.abs(c) ** (2 / k), cy + b * np.sign(s) * np.abs(s) ** (2 / k)
    gx = np.sign(qx) * np.abs(qx / a) ** (k - 1) / a
    gy = np.sign(qy - cy) * np.abs((qy - cy) / b) ** (k - 1) / b
    xy = mid + (qx[:, None] * ex + qy[:, None] * ey) * d
    n = gx[:, None] * ex + gy[:, None] * ey
    return xy, n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def ear_triangles(p: np.ndarray, side: int, scale: float, tight: bool = False) -> np.ndarray:
    """The head shape's two ear triangles rasterised at `side` (points on the 256 grid, times `scale`); tight: the
    triangles base, tip, base themselves, not enlarged."""
    d, mid, ex, ey, P = face_frame(p)
    yy, xx = np.mgrid[:side, :side] + 0.5
    rel = np.stack([xx / scale - mid[0], yy / scale - mid[1]], -1)
    q = np.stack([rel @ ex / d, rel @ ey / d], -1).reshape(-1, 2)
    out = np.zeros(side * side, bool)
    for ear in EARS:
        T = P[list(ear)]
        c = T.mean(0)
        v = T - c
        grow, pad = (1.0, 0.0) if tight else (EAR_SCALE, EAR_PAD)
        out |= Polygon(c + v * grow + v / np.linalg.norm(v, axis=1, keepdims=True) * pad).contains_points(q)
    return out.reshape(side, side)


def step_strength(
    img: np.ndarray, xy: np.ndarray, n: np.ndarray, reach: tuple[float, float], half: float
) -> tuple[np.ndarray, np.ndarray]:
    """Along each point's normal, every offset in [-reach0, reach1] (0.5 px apart): the RGB distance between the mean
    colour over `half` px before and after it. (offsets, strength (points, offsets))."""
    off = np.arange(-reach[0], reach[1] + 1e-6, 0.5)
    k = max(1, int(round(half / 0.5)))
    s = np.arange(off[0] - half, off[-1] + half + 1e-6, 0.5)
    pts = xy[:, None, :] + s[None, :, None] * n[:, None, :]
    at = [pts[..., 1] - 0.5, pts[..., 0] - 0.5]
    col = np.stack([ndimage.map_coordinates(img[..., c], at, order=1, mode="nearest") for c in range(3)], -1)
    cum = np.concatenate([np.zeros((len(xy), 1, 3)), np.cumsum(col, 1)], 1)
    i = np.arange(len(off)) + k  # the sample at each offset
    before = (cum[:, i] - cum[:, i - k]) / k
    after = (cum[:, i + k] - cum[:, i]) / k
    return off, np.linalg.norm(after - before, axis=-1)


def head_line(c: CatInputs) -> dict:
    """The head's line on the 256 grid: the head shape's outline, moved onto the photo's colour edge where the body
    goes on past it. Returns the line (SAMPLES, 2), the eye distance, which pixels are cat and which lie inside the
    photo (the box may reach past its edge)."""
    tri, p, img = c.tri, c.points.astype(np.float64), c.target.astype(np.float32) / 255
    box = c.box
    w, h = c.photo.size
    k = INPUT_SIDE / (box[2] - box[0])
    yy, xx = np.mgrid[:INPUT_SIDE, :INPUT_SIDE] + 0.5
    px, py = box[0] + xx / k, box[1] + yy / k
    in_photo = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    cat = (tri != 0) & in_photo
    d = float(np.linalg.norm(p[0] - p[1]))
    xy, n = head_arc(p)

    def look(m: np.ndarray, q: np.ndarray) -> np.ndarray:
        return m[np.clip(q[:, 1].astype(int), 0, INPUT_SIDE - 1), np.clip(q[:, 0].astype(int), 0, INPUT_SIDE - 1)]

    ear = ear_triangles(p, INPUT_SIDE, 1.0)
    free = look(cat, xy + 2 * n) & look(cat, xy - 2 * n) & ~look(ear, xy)  # the body goes on past the arc
    blur = ndimage.gaussian_filter(img, (1.0, 1.0, 0))
    off, strength = step_strength(blur, xy, n, (IN_REACH * d, OUT_REACH * d), STEP_HALF * d)
    rng = np.random.default_rng(0)  # the face's own texture
    inside = ndimage.binary_erosion((tri == 1) & ~ear, iterations=int(STEP_HALF * d) + 2)
    ys, xs = np.nonzero(inside)
    pick = rng.choice(len(xs), min(400, len(xs)), replace=False)
    ang = rng.uniform(0, np.pi, len(pick))
    _, tex = step_strength(
        blur, np.c_[xs[pick], ys[pick]] + 0.5, np.c_[np.cos(ang), np.sin(ang)], (0.0, 0.0), STEP_HALF * d
    )
    clear = max(MIN_EDGE, float(np.quantile(tex[:, 0], CLEAR_Q)))
    best = strength.argmax(1)
    s_best, edge = off[best], strength[np.arange(len(xy)), best]
    took = free & (edge >= clear)
    offset = np.zeros(len(xy))
    lab, runs = ndimage.label(free)
    if free[0] and free[-1] and runs > 1:  # the loop is closed: join the ends
        lab[lab == lab[-1]] = lab[0]
    for r in np.unique(lab[lab > 0]):
        idx = np.nonzero(lab == r)[0]
        if len(idx) == SAMPLES:
            order = idx
        else:  # walk the run in order around the loop
            gap = np.nonzero(np.diff(idx) > 1)[0]
            order = idx if not len(gap) else np.r_[idx[gap[0] + 1 :], idx[: gap[0] + 1]]
        known = np.r_[0.0, s_best[order], 0.0]
        have = np.r_[True, took[order], True]
        t = np.arange(len(order) + 2)
        filled = np.interp(t, t[have], known[have])[1:-1]
        smooth = ndimage.gaussian_filter1d(ndimage.median_filter(filled, MEDIAN, mode="nearest"), SIGMA, mode="nearest")
        ends = np.minimum(np.arange(len(order)) + 1, np.arange(len(order))[::-1] + 1) / (2 * SIGMA)
        offset[order] = smooth * np.clip(ends, 0, 1) if len(order) < SAMPLES else smooth
    return {"line": xy + offset[:, None] * n, "d": d, "cat": cat, "in_photo": in_photo, "points": p}


def _upsample_mask(m: np.ndarray, side: int) -> np.ndarray:
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((side, side), Image.Resampling.BILINEAR)) > 127


def head_tri(r: dict, side: int) -> np.ndarray:
    """The head's tri map at `side`: 1 the head (the mask inside the line or an ear), 2 the head beyond the photo,
    0 the rest (background and body)."""
    k = side / INPUT_SIDE
    im = Image.new("L", (side, side), 0)
    ImageDraw.Draw(im).polygon([tuple(q) for q in r["line"] * k], fill=255)
    shape = (np.asarray(im) > 127) | ear_triangles(r["points"], side, k)
    tri = np.zeros((side, side), np.uint8)
    tri[shape & ~_upsample_mask(r["in_photo"], side)] = 2
    tri[shape & _upsample_mask(r["cat"], side)] = 1
    return tri


def photo_crop(c: CatInputs, side: int) -> np.ndarray:
    """The photo cut to the box at `side` (float 0-1, white beyond the photo, not cut out)."""
    box = c.box
    sheet = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (255, 255, 255))
    sheet.paste(c.photo, (-box[0], -box[1]))
    return np.asarray(sheet.resize((side, side), Image.Resampling.LANCZOS), np.float32) / 255


def disk(radius: float) -> np.ndarray:
    k = int(np.ceil(radius))
    yy, xx = np.mgrid[-k : k + 1, -k : k + 1]
    return xx**2 + yy**2 <= radius**2


def painter_outline(r: dict, side: int) -> dict:
    """The painter's smooth outline at `side`: its contour (n, 2) px, one point a pixel; the head inside it; its tri
    map (1 head, 2 head beyond the photo, 0 the rest)."""
    k = side / INPUT_SIDE
    d = r["d"] * k
    in_photo = _upsample_mask(r["in_photo"], side)
    shape = head_tri(r, side) != 0  # beyond the photo only a cut-off ear tip, as its own triangle
    shape &= in_photo | ear_triangles(r["points"], side, k, tight=True)
    ear = ear_triangles(r["points"], side, k)
    tidy = ndimage.binary_closing(ndimage.binary_opening(shape, disk(OPEN * d)), disk(OPEN * d))
    tidy = (tidy & ~ear) | (shape & ear)
    lab, n = ndimage.label(tidy)
    tidy = ndimage.binary_fill_holes(lab == 1 + int(np.argmax(ndimage.sum(tidy, lab, range(1, n + 1)))))
    c = max(find_contours(np.pad(tidy, 1).astype(float), 0.5), key=len)[:, ::-1] - 1
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    at = np.arange(0, seg[-1], 1.0)
    xy = np.c_[np.interp(at, seg, c[:, 0]), np.interp(at, seg, c[:, 1])]
    w = ndimage.map_coordinates(
        ndimage.gaussian_filter(ear.astype(np.float32), 0.05 * d), [xy[:, 1] - 0.5, xy[:, 0] - 0.5], order=1
    )[:, None]
    smooth = lambda s_: ndimage.gaussian_filter1d(xy, s_ * d, axis=0, mode="wrap")  # noqa: E731
    xy = (1 - w) * smooth(SMOOTH_FACE) + w * smooth(SMOOTH_EAR)
    im = Image.new("L", (side, side), 0)
    ImageDraw.Draw(im).polygon([tuple(q) for q in xy], fill=255)
    head = np.asarray(im) > 127
    tri = np.zeros((side, side), np.uint8)
    tri[head & ~in_photo] = 2
    tri[head & in_photo] = 1
    return {"contour": xy, "tri": tri, "head": head, "d": d}


def painter_target(c: CatInputs, r: dict, o: dict, photo: np.ndarray, side: int) -> np.ndarray:
    """The photo where it is trusted (inside the outline, cat, at least TRUST from the cut-out's background); inside
    the outline and out to BAND past it elsewhere, the trusted colours around it (the finest blur scale with
    FILL_ENOUGH of them); white beyond."""
    in_photo = _upsample_mask(r["in_photo"], side)
    bg = _upsample_mask(c.tri == 0, side)
    bad = bg | ~in_photo
    trusted = o["head"] & ~bad & (ndimage.distance_transform_edt(~bad) >= TRUST * o["d"])
    region = ndimage.binary_dilation(o["head"], disk(BAND * o["d"])) & in_photo
    out = np.where(trusted[..., None], photo, 1.0).astype(np.float32)
    todo = region & ~trusted
    t = trusted.astype(np.float32)
    for s_ in FILL_SCALES:
        s_ *= side / 512
        den = ndimage.gaussian_filter(t, s_)
        num = np.stack([ndimage.gaussian_filter(photo[..., c] * t, s_) for c in range(3)], -1)
        ok = todo & (den >= (FILL_ENOUGH if s_ < FILL_SCALES[-1] * side / 512 else 1e-6))
        out[ok] = num[ok] / den[ok, None]
        todo &= ~ok
    return out


# ------------------------------------------------------------------ the plan's parts
def ear_inner(points: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Inside the ears: each ear's triangle (base, tip, base) shrunk EAR_SHRINK about its middle, on the head."""
    yy, xx = np.mgrid[:SIDE, :SIDE] + 0.5
    q = np.c_[xx.ravel(), yy.ravel()]
    out = np.zeros(SIDE * SIDE, bool)
    for ear in EARS:
        t = points[list(ear)]
        c = t.mean(0)
        out |= Polygon(c + EAR_SHRINK * (t - c)).contains_points(q)
    return out.reshape(SIDE, SIDE) & (tri == 1)


def plan_zones(plan: dict[str, np.ndarray], points256: np.ndarray) -> dict:
    """The sketch plan's parts as masks at SIDE: eye (inside the eye outlines), eye_rim (RIM_PX either side of
    them), pupil (the pupils' ellipses grown PUPIL_GROW, inside the eyes), iris (the eyes less the pupils), nose
    (inside its outline), mouth (MOUTH_PX about its line); whisker_pad: in the face's frame, from PAD_TOP below the
    nose's top to PAD_BELOW below the mouth point, from PAD_GAP beside the nose's widest point out to PAD_OUT, less
    the nose and the mouth grown 4 px; whisker_origin, between the nose and the mouth point, which the whiskers run
    away from; dot_pad: from the nose's middle height to DOTS_BELOW below the mouth point, DOTS_GAP to DOTS_OUT from
    the middle, less the nose and the mouth grown 3 px. Also the eye outlines and pupil ellipses (cx, cy, semi-axes,
    angle) in canvas pixels. A part the plan lacks is an empty mask."""
    k = SIDE / INPUT_SIDE
    yy, xx = np.mgrid[:SIDE, :SIDE] + 0.5
    q = np.c_[xx.ravel(), yy.ravel()]

    def poly(xy: np.ndarray) -> np.ndarray:
        if len(xy) < 3:
            return np.zeros((SIDE, SIDE), bool)
        return Polygon(xy * k).contains_points(q).reshape(SIDE, SIDE)

    eye = np.zeros((SIDE, SIDE), bool)
    for e in plan["eyes"]:
        eye |= poly(e)
    rim = ndimage.binary_dilation(eye, iterations=RIM_PX) & ~ndimage.binary_erosion(eye, iterations=RIM_PX)
    pupil = np.zeros((SIDE, SIDE), bool)
    for c in plan["pupils"]:
        if np.all(np.isfinite(c)):
            (cx, cy), (ra, rb), a = c[:2] * k, c[2:4] * k * PUPIL_GROW, c[4]
            u = (xx - cx) * np.cos(a) + (yy - cy) * np.sin(a)
            v = -(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a)
            pupil |= (u / max(ra, 1)) ** 2 + (v / max(rb, 1)) ** 2 <= 1
    im = Image.new("L", (SIDE, SIDE), 0)
    if len(plan["mouth"]) >= 2:
        ImageDraw.Draw(im).line([tuple(p) for p in plan["mouth"] * k], fill=255, width=2 * MOUTH_PX + 1)
    nose, mouth = poly(plan["nose"]), np.asarray(im) > 127
    d, mid, ex, ey, P = face_frame(points256.astype(np.float64))
    face = lambda xy: ((np.asarray(xy) - mid) @ np.c_[ex, ey]) / d  # noqa: E731
    nf = face(plan["nose"]) if len(plan["nose"]) >= 3 else np.array([[0.0, 0.6 * P[2, 1]], [0.15, 0.6 * P[2, 1]]])
    rel = face(np.c_[xx.ravel(), yy.ravel()] / k)
    qx, qy = np.abs(rel[:, 0]).reshape(SIDE, SIDE), rel[:, 1].reshape(SIDE, SIDE)
    pad = (
        (qy >= nf[:, 1].min() + PAD_TOP)
        & (qy <= P[2, 1] + PAD_BELOW)
        & (qx >= np.abs(nf[:, 0]).max() + PAD_GAP)
        & (qx <= PAD_OUT)
    )
    pad &= ~ndimage.binary_dilation(nose | mouth, iterations=4)
    origin = 0.5 * ((np.asarray(plan["nose"]).mean(0) if len(plan["nose"]) >= 3 else points256[2]) + points256[2]) * k
    dots = (qy >= nf[:, 1].mean()) & (qy <= P[2, 1] + DOTS_BELOW) & (qx >= DOTS_GAP) & (qx <= DOTS_OUT)
    dots &= ~ndimage.binary_dilation(nose | mouth, iterations=3)
    pupils = [c[:5] * np.r_[k, k, k, k, 1] for c in plan["pupils"] if np.all(np.isfinite(c[:5]))]
    return {
        "eye": eye,
        "eye_rim": rim,
        "pupil": pupil & eye,
        "nose": nose,
        "mouth": mouth,
        "whisker_pad": pad,
        "whisker_origin": origin,
        "dot_pad": dots,
        "iris": eye & ~ndimage.binary_dilation(pupil, iterations=1),
        "eye_outlines": [np.asarray(e) * k for e in plan["eyes"]],
        "pupils": pupils,
    }


# ------------------------------------------------------------------ the prepared cat
def prepare_cat(c: CatInputs) -> dict:
    """The cat at SIDE px as the painter reads it:

    target     (SIDE, SIDE, 3) float tensor: the photo inside the painter's outline (filled near the cut-out's
               edge), white outside
    photo      (SIDE, SIDE, 3) float array: the photo cut to the box, not cut out
    tri        (SIDE, SIDE) uint8: 1 head, 2 head beyond the photo, 0 the rest
    contour    the painter's outline (n, 2) px
    points     the nine face points at SIDE
    field      the fur field (SIDE, SIDE, 2), doubled-angle vectors
    ears       the ear triangles (enlarged); ear_inner, the inside of the ears on the head
    centre     the face's centre (between the eyes' midpoint and the mouth)
    and the plan's parts (plan_zones)."""
    r = head_line(c)
    o = painter_outline(r, SIDE)
    photo = photo_crop(c, SIDE)
    target = painter_target(c, r, o, photo, SIDE)
    img = np.where(o["head"][..., None], target, 1.0).astype(np.float32)
    points = c.points * SIDE / INPUT_SIDE
    field = fur_field_at(torch.from_numpy(c.target.astype(np.float32) / 255), c.points, SIDE)
    return {
        "target": torch.from_numpy(img),
        "tri": o["tri"],
        "points": points,
        "field": field,
        "contour": o["contour"],
        "photo": photo,
        "ears": ear_triangles(c.points, SIDE, SIDE / INPUT_SIDE),
        "centre": 0.5 * (points[:2].mean(0) + points[2]),
        "ear_inner": ear_inner(points, o["tri"]),
        **plan_zones(c.plan, c.points),
    }


# ------------------------------------------------------------------ eyes apart
def eye_side(xy: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Which eye each point belongs to, by the nearer eye centre: 0 the eye on the image's left, 1 the right."""
    eyes = points[:2][np.argsort(points[:2, 0])]
    return ((xy[:, None, :] - eyes[None]) ** 2).sum(-1).argmin(1).astype(np.int8)


def fill_eyes(img: np.ndarray, zone: np.ndarray, source: np.ndarray) -> np.ndarray:
    """img with the zone filled from the source pixels only (each zone pixel takes the nearest source pixel's colour,
    then smoothed EYE_FILL_BLUR px), so nothing inside the zone reaches the result."""
    iy, ix = ndimage.distance_transform_edt(~source, return_distances=False, return_indices=True)
    near = img[iy, ix]
    soft = np.stack([ndimage.gaussian_filter(near[..., c], EYE_FILL_BLUR) for c in range(3)], -1)
    return np.where(zone[..., None], soft, img).astype(np.float32)


def without_eyes(cat: dict, c: CatInputs) -> dict:
    """The cat as every stage but the eyes sees it: the photo and the target inside the eyes (grown EYE_APART_GROW
    px) filled from the head outside them, and the fur field computed from the cut-out filled the same way. Changing
    an eye of the photo changes nothing here."""
    zone = ndimage.binary_dilation(cat["eye"], iterations=EYE_APART_GROW)
    source = (cat["tri"] == 1) & ~zone
    t256 = c.target.astype(np.float32) / 255

    def shrink(m: np.ndarray) -> np.ndarray:
        return np.asarray(
            Image.fromarray(m.astype(np.uint8) * 255).resize((INPUT_SIDE, INPUT_SIDE), Image.Resampling.BOX)
        )

    zone256, src256 = shrink(zone) > 0, shrink(source) == 255
    field = fur_field_at(torch.from_numpy(fill_eyes(t256, zone256, src256)), c.points, SIDE)
    return {
        **cat,
        "target": torch.from_numpy(fill_eyes(cat["target"].numpy(), zone, source)),
        "photo": fill_eyes(np.asarray(cat["photo"], np.float32), zone, source),
        "field": field,
    }


def one_eye(cat: dict, side: int) -> dict:
    """The cat with only one eye's parts (0: the image's left eye, 1: the right; by the nearer eye centre)."""
    yy, xx = np.mgrid[:SIDE, :SIDE] + 0.5
    mine = (eye_side(np.c_[xx.ravel(), yy.ravel()], cat["points"]) == side).reshape(SIDE, SIDE)
    keep = lambda xy: eye_side(np.asarray(xy, np.float64).reshape(1, 2), cat["points"])[0] == side  # noqa: E731
    return {
        **cat,
        **{k: cat[k] & mine for k in ("eye", "eye_rim", "pupil", "iris")},
        "eye_outlines": [o for o in cat["eye_outlines"] if keep(o.mean(0))],
        "pupils": [p for p in cat["pupils"] if keep(p[:2])],
    }
