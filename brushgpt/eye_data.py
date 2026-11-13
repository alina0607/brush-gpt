"""The eye part model's data: one token sequence per eye, written in the eye's own frame.

The whole-face model paints the block-in and writes a sketch plan with both eyes' outlines and pupils, but its eyes
often come out too small, off their outline, in speckled colours or without a pupil. The eye part model repaints one
eye's six eye-stage layers (iris base, irises, lid shadow, pupils, lids, highlights) from what the whole-face model
produced: the plan's outline and pupil of that eye, the block-in's colour around it, and the eye colour asked for.
It never sees the photo. Its training data is the data painter's own eye strokes; nothing is repainted.

One sequence per eye:

  prefix  (given, not learned; always PREFIX tokens long, so a batch of eyes is sampled together)
          BOS, SIDE_L / SIDE_R, EYE_<colour> (the condition; NULL when dropped for guidance), RADIUS_<k> (the
          outline's size), the ring colour (the block-in's mean colour in a ring around the outline), OUTLINE and its
          16 points, PUPIL and 6 points on the pupil ellipse (NO_PUPIL and 6 points at the centre when the plan has
          none), START
  body    (learned) for each of the six layers present, LAYER_<j> and its strokes (x, y, angle, size, paint), then EOS

Coordinates are the eye's own frame: the centre is the outline's mean point and the unit r is the outline's largest
distance from the centre (median 31 px); x and y are NXY bins over [-SPAN, SPAN] r (0.73 px for the median eye).
Stroke length and width are in units of r too (log bins per layer, spanning the training cats' strokes). Strokes
outside the frame are dropped and counted. A layer's strokes go around the eye: by angle from straight up,
clockwise, then from the centre out. The plan's points are put on the 2 px grid the whole-face model writes them on,
so the eye model sees the same geometry in training as behind the whole-face model.

Each eye is also written mirrored (x -> -x, angle -> pi - angle, the side swapped, the order rebuilt). A cat's eyes
come from its painting A and from each of its eye-colour versions (`painter.paint_eye_versions`); an eye version
leaves the other eye as painted, so repeated sequences of one cat are written once.

The split is by cat, given (cats it does not name train) or drawn as in `dataset.split_cats`; a held cat whose photo
is the same as a training cat's is moved to train.

Output (data.npz): tokens (int16, padded with PAD), lengths, cat_id, version, side, eye, centre, r, flip per eye;
cats and split per cat; ranges (6 layers x lo/hi x length, width, opacity), specials, fields, sizes, vocab, prefix and
span. stats.json: counts and token lengths.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from .attributes import EYE_COLOURS, UNCLEAR
from .brushes import StrokeParams
from .cat import INPUT_SIDE
from .dataset import CatSource, cat_versions, finite, has_finite_canvas, read_plan, split_cats
from .painter import EYE_STAGE, SIDE
from .tokens import NRGB, PaintedLayer, around, bin_colours, bin_index, bin_value, colour_bins

LAYER_NAMES = ("iris base", "irises", "lid shadow", "pupils", "lids", "highlights")
NXY, SPAN = 128, 1.5  # eye-frame bins over [-SPAN, SPAN] r: nearly every stroke but the irises' within 1 r
NANGLE, NSIZE, NALPHA = 32, 8, 8
NRADIUS, R_LO, R_HI = 16, 12.0, 72.0  # the outline's r in px, log bins (5-95%: 24-39 px)
RING = (1.3, 2.2)  # the ring around the outline the block-in colour is read in (units of r)
OUTLINE_POINTS, PUPIL_POINTS = 16, 6
SPECIALS = (
    ["PAD", "BOS", "START", "EOS", "NULL", "SIDE_L", "SIDE_R", "OUTLINE", "PUPIL", "NO_PUPIL"]
    + [f"EYE_{n}" for n in EYE_COLOURS]
    + [f"RADIUS_{k}" for k in range(NRADIUS)]
    + [f"LAYER_{j}" for j in range(len(LAYER_NAMES))]
)
SP = {n: i for i, n in enumerate(SPECIALS)}
FIELDS = ("x", "y", "angle", "size", "paint", "ring")
SIZES = {"x": NXY, "y": NXY, "angle": NANGLE, "size": NSIZE * NSIZE, "paint": NRGB**3 * NALPHA, "ring": NRGB**3}
OFFSET = {f: len(SPECIALS) + sum(SIZES[g] for g in FIELDS[:k]) for k, f in enumerate(FIELDS)}
VOCAB = len(SPECIALS) + sum(SIZES.values())
PREFIX = 6 + 2 * OUTLINE_POINTS + 1 + 2 * PUPIL_POINTS + 1  # 52
STROKE_FIELDS = FIELDS[:5]
NO_COAT = {"base": {"label": UNCLEAR, "confident": False}, "pattern": {"label": UNCLEAR, "confident": False}}

EyeRanges = dict[int, np.ndarray]  # per eye layer j: [[lo], [hi]] of length / r, width / r and opacity
EyeLayers = list[tuple[int, StrokeParams]]  # one eye's layers: (j, strokes in canvas px)


# ------------------------------------------------------------------ the eye's frame
@dataclass(frozen=True, eq=False)
class Frame:
    """An eye's frame: its centre (canvas px), its unit r (px) and whether x is mirrored."""

    centre: np.ndarray
    r: float
    flip: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "centre", np.asarray(self.centre, np.float64))
        object.__setattr__(self, "r", float(self.r))

    def local(self, xy: np.ndarray) -> np.ndarray:
        """Canvas px to the eye's frame."""
        u = (np.asarray(xy, np.float64) - self.centre) / self.r
        return u * [-1, 1] if self.flip else u

    def canvas(self, u: np.ndarray) -> np.ndarray:
        """The eye's frame to canvas px."""
        return self.centre + (u * [-1, 1] if self.flip else u) * self.r

    def mirrored(self) -> Frame:
        return Frame(self.centre, self.r, True)


def xy_bins(u: np.ndarray) -> np.ndarray:
    return bin_index(u.reshape(-1), -SPAN, SPAN, NXY).reshape(u.shape)


def xy_values(i: np.ndarray) -> np.ndarray:
    return bin_value(np.asarray(i).reshape(-1), -SPAN, SPAN, NXY).reshape(np.shape(i))


# ------------------------------------------------------------------ the plan's eyes
def on_plan_grid(p: np.ndarray) -> np.ndarray:
    """Plan points (256 grid) as the whole-face model writes them (the floor of each coordinate) and reads them back
    (the bin's centre), in canvas px."""
    return (np.clip(np.floor(p), 0, INPUT_SIDE - 1) + 0.5) * SIDE / INPUT_SIDE


def pupil_points(q: np.ndarray) -> np.ndarray:
    """PUPIL_POINTS points on a plan pupil ellipse (cx, cy, a, b, angle on the 256 grid), canvas px on the plan
    grid."""
    t = np.linspace(0, 2 * np.pi, PUPIL_POINTS, endpoint=False)
    c, s = np.cos(q[4]), np.sin(q[4])
    p = q[:2] + np.c_[q[2] * np.cos(t) * c - q[3] * np.sin(t) * s, q[2] * np.cos(t) * s + q[3] * np.sin(t) * c]
    return on_plan_grid(p)


@dataclass(frozen=True, eq=False)
class EyeGeometry:
    """One eye of the sketch plan: its frame, its outline and its pupil points (canvas px; None without a pupil)."""

    frame: Frame
    outline: np.ndarray
    pupil: np.ndarray | None


def frame_of(outline: np.ndarray) -> Frame:
    c = outline.mean(0)
    return Frame(c, float(np.max(np.linalg.norm(outline - c, axis=1))))


def plan_eyes(plan: Mapping[str, np.ndarray]) -> list[EyeGeometry] | None:
    """Both eyes of a sketch plan (`dataset.read_plan`), the image's left eye (side 0) first; None without two eye
    outlines. An eye's pupil is the first finite plan pupil whose centre lies within r of the eye's centre."""
    eyes = sorted((on_plan_grid(np.asarray(e, np.float64)) for e in plan["eyes"]), key=lambda e: e[:, 0].mean())
    if len(eyes) != 2:
        return None
    pupils = [np.asarray(q, np.float64) for q in plan["pupils"]]
    out = []
    for e in eyes:
        frame = frame_of(e)
        near = [
            q
            for q in pupils
            if np.isfinite(q).all() and np.linalg.norm(q[:2] * SIDE / INPUT_SIDE - frame.centre) < frame.r
        ]
        out.append(EyeGeometry(frame, e, pupil_points(near[0]) if near else None))
    return out


def ring_colour(canvas: np.ndarray, frame: Frame) -> int:
    """The canvas's mean colour in the ring RING around the eye, as a colour bin."""
    yy, xx = np.mgrid[0:SIDE, 0:SIDE]
    d = np.hypot(xx + 0.5 - frame.centre[0], yy + 0.5 - frame.centre[1]) / frame.r
    m = (d >= RING[0]) & (d <= RING[1])
    c = canvas[m].mean(0) if m.any() else np.full(3, 0.5)
    return int(colour_bins(np.clip(c, 0, 1)[None])[0])


# ------------------------------------------------------------------ tokens
def prefix_tokens(
    side: int, eye: str | None, frame: Frame, outline: np.ndarray, pupil: np.ndarray | None, ring: int
) -> list[int]:
    """The given part of an eye's sequence (PREFIX tokens). eye None: NULL (no colour asked)."""
    rk = int(bin_index(np.array([frame.r]), R_LO, R_HI, NRADIUS, True)[0])
    t = [
        SP["BOS"],
        SP["SIDE_R" if side else "SIDE_L"],
        SP[f"EYE_{eye}"] if eye else SP["NULL"],
        SP[f"RADIUS_{rk}"],
        OFFSET["ring"] + ring,
        SP["OUTLINE"],
    ]
    for x, y in xy_bins(frame.local(outline)):
        t += [OFFSET["x"] + int(x), OFFSET["y"] + int(y)]
    t.append(SP["NO_PUPIL"] if pupil is None else SP["PUPIL"])
    for x, y in xy_bins(np.zeros((PUPIL_POINTS, 2)) if pupil is None else frame.local(pupil)):
        t += [OFFSET["x"] + int(x), OFFSET["y"] + int(y)]
    t.append(SP["START"])
    assert len(t) == PREFIX
    return t


def body_tokens(layers: EyeLayers, frame: Frame, ranges: EyeRanges) -> tuple[list[int], int]:
    """An eye's layers as LAYER_<j> and their strokes, then EOS; and how many strokes fell outside the frame."""
    t, dropped = [], 0
    for j, p in layers:
        u = frame.local(p.xy.numpy())
        keep = (np.abs(u) < SPAN).all(1)
        dropped += int((~keep).sum())
        if not keep.any():
            continue
        u = u[keep]
        k = around(u, np.zeros(2))
        lw = p.lw.numpy()[keep][k] / frame.r
        ang = np.remainder(p.angle.numpy()[keep][k], 2 * np.pi)
        if frame.flip:
            ang = np.remainder(np.pi - ang, 2 * np.pi)
        lo, hi = ranges[j]
        size = bin_index(lw[:, 0], lo[0], hi[0], NSIZE, True) * NSIZE + bin_index(lw[:, 1], lo[1], hi[1], NSIZE, True)
        paint = colour_bins(p.colour.numpy()[keep][k]) * NALPHA + bin_index(
            p.alpha.numpy()[keep][k], lo[2], hi[2], NALPHA
        )
        g = xy_bins(u[k])
        block = np.stack(
            [
                g[:, 0] + OFFSET["x"],
                g[:, 1] + OFFSET["y"],
                bin_index(ang, 0, 2 * np.pi, NANGLE) + OFFSET["angle"],
                size + OFFSET["size"],
                paint + OFFSET["paint"],
            ],
            1,
        ).reshape(-1)
        t += [SP[f"LAYER_{j}"]] + block.tolist()
    return t + [SP["EOS"]], dropped


def strokes_from_tokens(toks: np.ndarray, frame: Frame, ranges: EyeRanges) -> EyeLayers:
    """A body back to strokes in canvas px, [(j, strokes)] in layer order. A sampled body may be malformed: tokens
    that are not a stroke inside a layer are skipped, and so is everything from EOS on."""
    out: dict[int, list[np.ndarray]] = {}
    bounds = [(OFFSET[f], OFFSET[f] + SIZES[f]) for f in STROKE_FIELDS]
    j, i = None, 0
    while i < len(toks):
        t = int(toks[i])
        if t == SP["EOS"]:
            break
        name = SPECIALS[t] if t < len(SPECIALS) else ""
        if name.startswith("LAYER_"):
            j = int(name[6:])
            i += 1
            continue
        blk = toks[i : i + 5]
        if j is None or len(blk) < 5 or not all(a <= v < b for v, (a, b) in zip(blk, bounds, strict=True)):
            i += 1
            continue
        out.setdefault(j, []).append(np.asarray(blk, np.int64) - [OFFSET[f] for f in STROKE_FIELDS])
        i += 5
    res = []
    for j in sorted(out):
        r = np.array(out[j])
        lo, hi = ranges[j]
        ang = bin_value(r[:, 2], 0, 2 * np.pi, NANGLE)
        if frame.flip:
            ang = np.remainder(np.pi - ang, 2 * np.pi)
        lw = (
            np.stack(
                [
                    bin_value(r[:, 3] // NSIZE, lo[0], hi[0], NSIZE, True),
                    bin_value(r[:, 3] % NSIZE, lo[1], hi[1], NSIZE, True),
                ],
                1,
            )
            * frame.r
        )
        values = (
            frame.canvas(xy_values(r[:, :2])),
            lw,
            ang,
            bin_colours(r[:, 4] // NALPHA),
            bin_value(r[:, 4] % NALPHA, lo[2], hi[2], NALPHA),
        )
        res.append((j, StrokeParams(*(torch.from_numpy(np.ascontiguousarray(v, np.float32)) for v in values))))
    return res


# ------------------------------------------------------------------ paintings
def eye_layers(layers: list[PaintedLayer], side: int) -> EyeLayers:
    """One eye's eye-stage layers [(j, strokes)] of a painting's layers (`dataset.version_layers`)."""
    return [(j, p) for st, j, p, s in layers if st == EYE_STAGE and s == side and len(p.xy)]


def with_eyes(
    painted: list[tuple[int, int, StrokeParams]], eyes: list[tuple[int, int, StrokeParams]]
) -> list[tuple[int, int, StrokeParams]]:
    """A painting's layers (stage, j, strokes) with its eye stage replaced by these eyes' layers."""
    at = next((i for i, x in enumerate(painted) if x[0] >= EYE_STAGE), len(painted))
    rest = [x for x in painted if x[0] != EYE_STAGE]
    return rest[:at] + eyes + rest[at:]


@dataclass(eq=False)
class EyeRow:
    """One eye of one painting, before binning."""

    version: str
    side: int
    eye: str
    geometry: EyeGeometry
    ring: int
    layers: EyeLayers = field(default_factory=list)

    @property
    def frame(self) -> Frame:
        return self.geometry.frame

    def mirrored(self) -> EyeRow:
        """The eye flipped left to right in its own frame: the other side."""
        g = self.geometry
        return EyeRow(
            f"{self.version}~m",
            1 - self.side,
            self.eye,
            EyeGeometry(g.frame.mirrored(), g.outline, g.pupil),
            self.ring,
            self.layers,
        )

    def tokens(self, ranges: EyeRanges) -> tuple[np.ndarray, int]:
        """The eye's sequence (prefix and body), and how many of its strokes fell outside the frame."""
        g = self.geometry
        body, dropped = body_tokens(self.layers, g.frame, ranges)
        return np.array(
            prefix_tokens(self.side, self.eye, g.frame, g.outline, g.pupil, self.ring) + body, np.int16
        ), dropped


def cat_eye_rows(
    source: CatSource, eye_versions: Path | None = None, eye_targets: Mapping[str, np.ndarray] | None = None
) -> list[EyeRow] | None:
    """Every eye of one cat's paintings (A, then its eye versions; each the left eye, then the right): the colour
    read on the photo the painting was fitted to, the plan's geometry, the block-in's ring colour and the eye's
    layers. None when the plan has not two eyes."""
    eyes = plan_eyes(read_plan(source.plan))
    if eyes is None:
        return None
    with np.load(source.painting / "stage1.npz") as z:
        block_in = z["canvas"]
    rings = [ring_colour(block_in, g.frame) for g in eyes]
    rows = []
    for version, layers, names in cat_versions(source, NO_COAT, eye_versions, eye_targets):
        layers, _ = finite(layers)
        rows += [
            EyeRow(version, side, names[side], eyes[side], rings[side], eye_layers(layers, side)) for side in (0, 1)
        ]
    return rows


def layer_ranges(cats: Iterable[list[EyeRow]]) -> EyeRanges:
    """Each eye layer's lowest and highest stroke length / r, width / r and opacity over these cats' paintings A."""
    r: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for rows in cats:
        for row in rows:
            if row.version != "A":
                continue
            for j, p in row.layers:
                v = np.c_[p.lw.numpy() / row.frame.r, p.alpha.numpy()]
                lo, hi = r.get(j, (np.full(3, np.inf), np.full(3, -np.inf)))
                r[j] = (np.minimum(lo, v.min(0)), np.maximum(hi, v.max(0)))
    return {j: np.stack(v) for j, v in r.items()}


@dataclass
class EyeSequences:
    """Every eye of every cat as tokens, with what each one is."""

    tokens: list[np.ndarray] = field(default_factory=list)
    meta: list[tuple] = field(default_factory=list)  # (cat index, version, side, eye, centre x, centre y, r, flip)
    counts: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(
            ("strokes", "strokes_outside_frame", "eyes_without_strokes", "repeated_sequences_dropped"), 0
        )
    )


def sequences(cats: list[list[EyeRow]], ranges: EyeRanges) -> EyeSequences:
    """Every eye of every cat, as it is and mirrored; a cat's repeated sequences are written once."""
    out = EyeSequences()
    for ci, rows in enumerate(cats):
        seen: set[bytes] = set()
        for row in rows:
            if not row.layers:
                out.counts["eyes_without_strokes"] += 1
                continue
            for r in (row, row.mirrored()):
                t, dropped = r.tokens(ranges)
                if t.tobytes() in seen:
                    out.counts["repeated_sequences_dropped"] += 1
                    continue
                seen.add(t.tobytes())
                out.counts["strokes"] += sum(len(p.xy) for _, p in r.layers)
                out.counts["strokes_outside_frame"] += dropped
                out.tokens.append(t)
                f = r.frame
                out.meta.append((ci, r.version, r.side, r.eye, *f.centre, f.r, f.flip))
    return out


# ------------------------------------------------------------------ which cats
def reviewed_for_eyes(assignments: Path | str) -> set[str]:
    """The cats a review kept for the eye model: rows of a CSV (source, id, eyes) whose eyes column is "used"."""
    with open(assignments, newline="") as f:
        return {f"{r['source']}__{r['id']}" for r in csv.DictReader(f) if r["eyes"] == "used"}


def duplicates_to_train(split: Mapping[str, str], duplicates: Iterable[tuple[str, str]]) -> tuple[dict[str, str], int]:
    """The split with every pair of same-photo cats that it puts on different sides moved to train (so no held cat
    has its photo in training), and how many held cats were moved."""
    out, moved = dict(split), 0
    for a, b in duplicates:
        if a in out and b in out and out[a] != out[b]:
            for n in (a, b):
                moved += out[n] == "held"
                out[n] = "train"
    return out, moved


# ------------------------------------------------------------------ the build
_WORKER: dict = {}


def _init_worker(eye_versions: Path | None, eye_targets: Mapping[str, np.ndarray] | None) -> None:
    _WORKER.update(eye_versions=eye_versions, eye_targets=eye_targets)


def _one_cat(source: CatSource) -> tuple[str, list[EyeRow] | None | str]:
    try:
        return source.name, cat_eye_rows(source, _WORKER["eye_versions"], _WORKER["eye_targets"])
    except (FileNotFoundError, KeyError, ValueError) as e:  # a missing or broken input file: named in stats.json
        return source.name, repr(e)


def build(
    cats: list[CatSource],
    out: Path | str,
    split: Mapping[str, str] | None = None,
    duplicates: Iterable[tuple[str, str]] = (),
    eye_versions: Path | None = None,
    eye_targets: Mapping[str, np.ndarray] | None = None,
    limit: int = 0,
    threads: int = 1,
    log: Callable[[dict], None] | None = None,
) -> dict:
    """Build out/data.npz and out/stats.json from the painted cats (in name order; cats without a finished, finite
    painting are left out; `limit`: only the first ones). The bins of each layer's stroke size and opacity span the
    training cats' paintings A."""
    out = Path(out)
    duplicates = list(duplicates)
    painted = sorted((c for c in cats if (c.painting / "stage7.npz").exists()), key=lambda c: c.name)
    painted = [c for c in painted if has_finite_canvas(c.painting)]
    painted = painted[:limit] if limit else painted
    names = [c.name for c in painted]
    split = split_cats(names, duplicates) if split is None else split
    split, moved = duplicates_to_train({n: split.get(n, "train") for n in names}, duplicates)
    got: list[tuple[str, list[EyeRow] | None | str]] = []
    if threads > 1:
        with Pool(threads, initializer=_init_worker, initargs=(eye_versions, eye_targets)) as pool:
            for c in pool.imap(_one_cat, painted, chunksize=4):
                got.append(c)
                if log is not None and len(got) % 200 == 1:
                    log({"cats": len(got), "of": len(painted)})
    else:
        _init_worker(eye_versions, eye_targets)
        for source in painted:
            got.append(_one_cat(source))
            if log is not None and len(got) % 200 == 1:
                log({"cats": len(got), "of": len(painted)})
    left_out = {
        n: rows if isinstance(rows, str) else "no two plan eyes" for n, rows in got if not isinstance(rows, list)
    }
    kept = [(n, rows) for n, rows in got if isinstance(rows, list)]
    ranges = layer_ranges(rows for n, rows in kept if split[n] == "train")
    seqs = sequences([rows for _, rows in kept], ranges)
    seqs.counts["held_cats_moved_to_train_duplicates"] = moved
    toks, m = seqs.tokens, list(zip(*seqs.meta, strict=True))
    width = max(len(t) for t in toks)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "data.npz",
        tokens=np.stack([np.pad(t, (0, width - len(t)), constant_values=SP["PAD"]) for t in toks]),
        lengths=np.array([len(t) for t in toks], np.int32),
        cat_id=np.array(m[0], np.int32),
        version=np.array(m[1]),
        side=np.array(m[2], np.int8),
        eye=np.array(m[3]),
        centre=np.c_[m[4], m[5]].astype(np.float32),
        r=np.array(m[6], np.float32),
        flip=np.array(m[7]),
        cats=np.array([n for n, _ in kept]),
        split=np.array([split[n] for n, _ in kept]),
        ranges=np.stack([ranges[j] for j in range(len(LAYER_NAMES))]),
        specials=np.array(SPECIALS),
        fields=np.array(FIELDS),
        sizes=np.array([SIZES[f] for f in FIELDS]),
        vocab=np.array(VOCAB),
        prefix=np.array(PREFIX),
        span=np.array(SPAN),
    )
    lens = np.array([len(t) for t in toks])
    kinds, counts = np.unique([v.split(":")[0].removesuffix("~m") for v in m[1]], return_counts=True)
    stats = {
        "cats": len(kept),
        "held_cats": sum(split[n] == "held" for n, _ in kept),
        "left_out": left_out,
        "eyes": len(toks),
        "versions": {str(k): int(v) for k, v in zip(kinds, counts, strict=True)},
        **seqs.counts,
        "tokens_min_median_max": [int(lens.min()), int(np.median(lens)), int(lens.max())],
        "vocab": VOCAB,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def load_eye_ranges(data: Path | str | Mapping[str, np.ndarray]) -> EyeRanges:
    """The bins' ranges of a built data file (or its loaded arrays)."""
    if isinstance(data, Mapping):
        arr = np.asarray(data["ranges"])
    else:
        with np.load(data) as z:
            arr = z["ranges"]
    return {j: arr[j] for j in range(len(LAYER_NAMES))}
