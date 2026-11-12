"""The judge: how well generated paintings follow their condition, on held-out cats, one sample per cat, nothing
picked. Every painting is decoded from its tokens and painted with the painter's brushes on the CPU, then read by the
same readers that wrote the training conditions (`brushgpt.attributes`): the eyes by the colour of each iris region
of the cat's photo, the coat by the learned coat reader. An attribute "reads right" when the name read on the canvas
is the name asked for; attributes asked as "unclear" are not counted.

Three measures:

whole face   `measure_whole_face`: each sampled painting of the whole-face model and the painter's own painting of
             the same cat (decoded from its tokens, so the binning cancels), every attribute read against the
             condition. The painter's row is the most a model can reach with these readers.
eye alone    `measure_eyes_alone`: the painter's painting with its eye stage replaced by the eye model's eyes, drawn
             from the painter's own plan; next to the painter's own painting.
system       `measure_system`: the whole-face model's paintings with their eyes repainted: the eye model reads each
             eye's outline and pupil from the whole-face model's own plan, the block-in colour from its first stage
             and the colour from its condition. Reported next to the whole-face model alone, so the whole-face
             model's weakness is never hidden behind the part model.

Per eye: strokes, the share of stroke positions inside the eye's outline, and the pupil contrast (mean luminance
inside the outline but outside the pupil, minus inside the pupil; a dark pupil gives a high value).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from matplotlib.path import Path as Polygon
from PIL import Image, ImageDraw, ImageFont

from .attributes import ATTRIBUTES, UNCLEAR, eye_names, eye_view, painting_attributes
from .brushes import StrokeParams
from .cat import INPUT_SIDE, load_inputs
from .dataset import CatSource, finite, version_layers
from .eye_data import (
    OFFSET,
    OUTLINE_POINTS,
    PUPIL_POINTS,
    SP,
    SPECIALS,
    EyeLayers,
    EyeRanges,
    Frame,
    eye_layers,
    frame_of,
    load_eye_ranges,
    prefix_tokens,
    ring_colour,
    strokes_from_tokens,
    with_eyes,
    xy_values,
)
from .eye_model import EyeData, EyeGrammar, generate, without_colour
from .layers import GPT
from .painter import EYE_STAGE, SIDE
from .tokens import Vocabulary, layers_from_tokens, ranges_from_array, render

LUMA = np.array([0.299, 0.587, 0.114])
PIXELS = np.c_[np.mgrid[0:SIDE, 0:SIDE][1].ravel() + 0.5, np.mgrid[0:SIDE, 0:SIDE][0].ravel() + 0.5]
EYE_MEASURES = ("strokes", "inside", "pupil_contrast")
CROP_SIZE = (352, 160)

Layers = list[tuple[int, int, StrokeParams]]
CoatReading = Callable[[list[np.ndarray]], list[dict]]
Pictures = list[list[tuple[str, Image.Image]]]


class CatViews:
    """What the readers need of each cat (`attributes.eye_view`), loaded once per cat from its inputs."""

    def __init__(self, sources: Mapping[str, CatSource]):
        self.sources = sources
        self._views: dict[str, dict] = {}

    def __getitem__(self, name: str) -> dict:
        if name not in self._views:
            s = self.sources[name]
            self._views[name] = eye_view(load_inputs(s.cutout, s.photo, s.plan))
        return self._views[name]


class WholeFaceData:
    """The whole-face model's data file, as far as judging its samples needs it."""

    def __init__(self, path: str | Path):
        z = np.load(path)
        self.tokens, self.cond, self.cat_id = z["tokens"], z["cond"], z["cat_id"]
        self.cats = [str(c) for c in z["cats"]]
        self.versions = [str(v) for v in z["versions"]]
        self.vocab = Vocabulary(tuple(str(s) for s in z["specials"]))
        if self.vocab.size != int(z["vocab"]):
            raise ValueError("the data's special tokens do not match a vocabulary of this package")
        self.ranges = ranges_from_array(z["ranges"])

    def cat(self, row: int) -> str:
        return self.cats[int(self.cat_id[row])]

    def asked(self, row: int) -> list[str]:
        """The four attribute names a painting was conditioned on (a leading task token skipped)."""
        ids = [int(c) for c in self.cond[row] if not self.vocab.specials[int(c)].startswith("TASK_")]
        return self.vocab.condition_names(ids)

    def layers(self, tokens: np.ndarray) -> Layers:
        return layers_from_tokens(tokens[tokens >= 0], self.ranges, self.vocab)


def read_whole_samples(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """A whole-face samples file's (tokens, data rows)."""
    z = np.load(path)
    return z["tokens"], z["rows"]


# ------------------------------------------------------------------ the whole-face model
def judge_painting(view: Mapping, asked: Sequence[str], pictures: Mapping[str, np.ndarray],
                   read_coat: CoatReading) -> dict[str, dict]:
    """Each painting's attributes read on its canvas, and per attribute whether it reads as asked and whether it
    was asked at all."""
    names = painting_attributes(view, list(pictures.values()), read_coat)
    return {who: {"names": n, "right": [g == asked[i] for i, g in enumerate(n)], "asked": [a != UNCLEAR for a in asked]}
            for who, n in zip(pictures, names, strict=True)}


def measure_whole_face(tokens: np.ndarray, rows: np.ndarray, data: WholeFaceData, views: CatViews,
                       read_coat: CoatReading, log: Callable[[dict], None] | None = None) -> dict:
    """Each sampled painting (tokens, padded with -1; rows: the data row it was conditioned on) against the
    painter's own painting of that row."""
    eos = data.vocab.ids["EOS"]
    out = []
    for k, row in enumerate(rows):
        row = int(row)
        name = data.cat(row)
        got = tokens[k][tokens[k] >= 0]
        asked = data.asked(row)
        rec = {"cat": name, "version": data.versions[row], "tokens": int(len(got)),
               "ends_with_eos": bool(len(got) and got[-1] == eos), "painter_tokens": int((data.tokens[row] >= 0).sum()),
               "cond": asked}
        pictures = {"painter": render(data.layers(data.tokens[row])).numpy(),
                    "model": render(data.layers(got)).numpy()}
        rec |= judge_painting(views[name], asked, pictures, read_coat)
        out.append(rec)
        if log is not None:
            log({"k": k, "cat": name, **{w: rec[w]["names"] for w in pictures}})
    return {"n": len(out), "rows": out, **whole_face_summary(out)}


def whole_face_summary(rows: list[dict]) -> dict:
    """Per attribute, read right / asked, for the painter and the model; and how many samples ended with EOS."""
    res: dict = {}
    for who in ("painter", "model"):
        right = np.array([r[who]["right"] for r in rows]) & np.array([r[who]["asked"] for r in rows])
        n = np.array([r[who]["asked"] for r in rows]).sum(0)
        res[who] = {"gen_read_right": {a: f"{int(right[:, i].sum())}/{int(n[i])}" for i, a in enumerate(ATTRIBUTES)}}
    res["gen_ends_with_eos"] = f"{sum(r['ends_with_eos'] for r in rows)}/{len(rows)}"
    return res


# ------------------------------------------------------------------ eye measures
def eye_measures(canvas: np.ndarray, layers: EyeLayers, outline: np.ndarray, pupil: np.ndarray | None) -> dict:
    """One eye's strokes, the share of them inside its outline, and its pupil contrast."""
    shape = Polygon(outline)
    xy = np.concatenate([p.xy.numpy() for _, p in layers]) if layers else np.zeros((0, 2))
    in_outline = shape.contains_points(PIXELS).reshape(SIDE, SIDE)
    res = {"strokes": len(xy), "inside": float(shape.contains_points(xy).mean()) if len(xy) else float("nan"),
           "pupil_contrast": float("nan")}
    if pupil is not None:
        in_pupil = Polygon(pupil).contains_points(PIXELS).reshape(SIDE, SIDE)
        lum = canvas @ LUMA
        if in_pupil.any() and (in_outline & ~in_pupil).any():
            res["pupil_contrast"] = float(lum[in_outline & ~in_pupil].mean() - lum[in_pupil].mean())
    return res


def eye_summary(rows: list[dict], who: Sequence[str]) -> dict:
    """Per painter or model: eyes read right (of the asked colours that are not unclear), eyes read unclear, and the
    eye measures' means."""
    out = {}
    for w in who:
        asked = [(a, n) for r in rows for a, n in zip(r["asked"], r["names"][w], strict=True) if a != UNCLEAR]
        eyes = [e for r in rows for e in r["eyes"].get(w, [])]
        means = {k: round(float(np.nanmean([e[k] for e in eyes])), 3) for k in EYE_MEASURES}
        out[w] = {"read_right": f"{sum(a == n for a, n in asked)}/{len(asked)}",
                  "read_unclear": sum(n == UNCLEAR for r in rows for n in r["names"][w]), **(means if eyes else {})}
    return out


def prefix_geometry(prefix: np.ndarray, frame: Frame) -> tuple[np.ndarray, np.ndarray | None]:
    """The outline and pupil points (canvas px) written in an eye's prefix."""
    off = [OFFSET["x"], OFFSET["y"]]
    o = prefix[6:6 + 2 * OUTLINE_POINTS].reshape(-1, 2) - off
    k = 6 + 2 * OUTLINE_POINTS
    p = prefix[k + 1:k + 1 + 2 * PUPIL_POINTS].reshape(-1, 2) - off
    pupil = None if prefix[k] == SP["NO_PUPIL"] else frame.canvas(xy_values(p))
    return frame.canvas(xy_values(o)), pupil


def painter_layers(source: CatSource) -> tuple[list, Layers]:
    """A cat's painting A: its layers with their eye sides, and as (stage, j, strokes)."""
    layers, _ = finite(version_layers(source.painting))
    return layers, [(st, j, p) for st, j, p, _ in layers]


# ------------------------------------------------------------------ the eye model alone
def measure_eyes_alone(samples: Mapping, eye_data: Mapping, sources: Mapping[str, CatSource],
                       views: CatViews, n_cats: int = 32) -> tuple[dict, Pictures]:
    """The eye model's eyes (an eye samples file: rows, prefix, bodies, guidance) painted into the painter's own
    paintings, against those paintings, for the first n_cats cats of the samples."""
    ranges = load_eye_ranges(eye_data)
    cats = [str(c) for c in eye_data["cats"]]
    by_cat: dict[int, dict[int, int]] = {}
    for k, r in enumerate(samples["rows"]):
        by_cat.setdefault(int(eye_data["cat_id"][r]), {})[int(eye_data["side"][r])] = k
    rows, pics = [], []
    for c, sides in list(by_cat.items())[:n_cats]:
        name = cats[c]
        layers, painted = painter_layers(sources[name])
        p0 = render(painted).numpy()
        eyes: Layers = []
        model_eyes, painter_eyes, outlines = [], [], []
        for side in (0, 1):
            k = sides[side]
            r = int(samples["rows"][k])
            frame = Frame(eye_data["centre"][r], float(eye_data["r"][r]), bool(eye_data["flip"][r]))
            mine = strokes_from_tokens(samples["bodies"][k], frame, ranges)
            eyes += [(EYE_STAGE, j, p) for j, p in mine]
            outline, pupil = prefix_geometry(samples["prefix"][k], frame)
            outlines.append(outline)
            model_eyes.append((mine, outline, pupil))
            painter_eyes.append((eye_layers(layers, side), outline, pupil))
        p1 = render(with_eyes(painted, eyes)).numpy()
        asked = [SPECIALS[int(samples["prefix"][sides[side]][2])].removeprefix("EYE_") for side in (0, 1)]
        view = views[name]
        row = {"cat": name, "asked": asked, "names": {"painter": eye_names(view, p0), "eye model": eye_names(view, p1)},
               "eyes": {"painter": [eye_measures(p0, *e) for e in painter_eyes],
                        "eye model": [eye_measures(p1, *e) for e in model_eyes]}}
        rows.append(row)
        g = float(samples["guidance"])
        pics.append([(f"{name[-6:]} painter {'/'.join(row['names']['painter'])}", crop(p0, outlines)),
                     (f"eye model g{g:g}: {'/'.join(row['names']['eye model'])} (asked {'/'.join(asked)})",
                      crop(p1, outlines))])
    return {"rows": rows, "summary": eye_summary(rows, ["painter", "eye model"])}, pics


# ------------------------------------------------------------------ the whole system
def plan_from_tokens(tokens: np.ndarray, vocab: Vocabulary) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """A whole-face painting's own sketch plan: its eye outlines and pupil points (each with at least 3 points), in
    canvas px."""
    sp, off = vocab.specials, vocab.offsets
    runs: dict[str, list] = {"EYE": [], "PUPIL": []}
    cur, i = None, 0
    while i < len(tokens):
        if tokens[i] < len(sp):
            n = sp[int(tokens[i])]
            if n == "PLAN_END":
                break
            cur = n[5:] if n[5:] in runs else None
            if cur:
                runs[cur].append([])
            i += 1
            continue
        if cur and i + 1 < len(tokens):
            runs[cur][-1].append((tokens[i] - off["x"], tokens[i + 1] - off["y"]))
        i += 2

    def to_canvas(pts: list) -> np.ndarray:
        return (np.asarray(pts, np.float64) + 0.5) * SIDE / INPUT_SIDE
    return [to_canvas(e) for e in runs["EYE"] if len(e) >= 3], [to_canvas(p) for p in runs["PUPIL"] if len(p) >= 3]


def eye_prefixes(layers: Layers, outlines: list[np.ndarray], pupils: list[np.ndarray], asked: list[str]
                 ) -> tuple[list[list[int]], list[Frame], list[tuple[np.ndarray, np.ndarray | None]]]:
    """Both eyes' prefixes read from a whole-face painting (its plan's outlines and pupils, the ring colour on its
    block-in), their frames, and each eye's (outline, pupil)."""
    block_in = render([x for x in layers if x[0] < EYE_STAGE]).numpy()
    pre, frames, geo = [], [], []
    for side, o in enumerate(outlines):
        frame = frame_of(o)
        near = [p for p in pupils if np.linalg.norm(p.mean(0) - frame.centre) < frame.r]
        pupil = near[0] if near else None
        pre.append(prefix_tokens(side, asked[side], frame, o, pupil, ring_colour(block_in, frame)))
        frames.append(frame)
        geo.append((o, pupil))
    return pre, frames, geo


def own_eyes(layers: Layers, frames: list[Frame]) -> list[EyeLayers]:
    """The whole-face model's own eye layers [(j, strokes)] per eye, each layer given to the nearer eye."""
    own: list[EyeLayers] = [[], []]
    for st, j, p in layers:
        if st == EYE_STAGE and len(p.xy):
            m = p.xy.numpy().mean(0)
            own[int(np.linalg.norm(m - frames[1].centre) < np.linalg.norm(m - frames[0].centre))].append((j, p))
    return own


def measure_system(tokens: np.ndarray, rows: np.ndarray, whole: WholeFaceData, model: GPT, eye_data: EyeData,
                   eye_ranges: EyeRanges, views: CatViews, guidance: float = 2.0, temperature: float = 1.0,
                   max_new: int = 1200, n_cats: int = 32, seed: int = 0) -> tuple[dict, Pictures]:
    """The whole-face model's samples (tokens, data rows) as they are and with both eyes repainted by the eye model
    (one sample per eye from one random generator, nothing picked). A painting whose plan has not two 16-point
    outlines keeps its own eyes and is counted."""
    device = model.embed.weight.device
    grammar = EyeGrammar(eye_data, device)
    gen = torch.Generator(device=device).manual_seed(seed)
    out, pics = [], []
    for k in range(min(n_cats, len(tokens))):
        row = int(rows[k])
        name = whole.cat(row)
        asked = whole.asked(row)[:2]
        t = tokens[k][tokens[k] >= 0]
        layers = whole.layers(t)
        outlines, pupils = plan_from_tokens(t, whole.vocab)
        pupils = [p for p in pupils if len(p) == PUPIL_POINTS]
        view = views[name]
        p0 = render(layers).numpy()
        rec: dict = {"cat": name, "asked": asked, "names": {"whole-face model": eye_names(view, p0)}, "eyes": {}}
        if len(outlines) != 2 or any(len(o) != OUTLINE_POINTS for o in outlines):
            rec["names"]["whole + eye model"] = rec["names"]["whole-face model"]
            rec["plan_not_two_outlines"] = [len(o) for o in outlines]
            out.append(rec)
            continue
        outlines = sorted(outlines, key=lambda e: e[:, 0].mean())
        pre, frames, geo = eye_prefixes(layers, outlines, pupils, asked)
        pre_t = torch.tensor(pre, device=device)
        bodies = generate(model, pre_t, without_colour(pre_t, eye_data.sp["NULL"]), grammar, max_new, temperature,
                          guidance, gen).cpu().numpy()
        mine = [strokes_from_tokens(bodies[side], frames[side], eye_ranges) for side in (0, 1)]
        p1 = render(with_eyes(layers, [(EYE_STAGE, j, p) for m in mine for j, p in m])).numpy()
        rec["names"]["whole + eye model"] = eye_names(view, p1)
        own = own_eyes(layers, frames)
        rec["eyes"] = {"whole-face model": [eye_measures(p0, own[i], *geo[i]) for i in (0, 1)],
                       "whole + eye model": [eye_measures(p1, mine[i], *geo[i]) for i in (0, 1)]}
        out.append(rec)
        pics.append([(f"{name[-6:]} whole-face {'/'.join(rec['names']['whole-face model'])}", crop(p0, outlines)),
                     (f"+ eye model g{guidance:g}: {'/'.join(rec['names']['whole + eye model'])} "
                      f"(asked {'/'.join(asked)})", crop(p1, outlines))])
    summary = eye_summary(out, ["whole-face model", "whole + eye model"])
    summary["plan_not_two_outlines"] = sum("plan_not_two_outlines" in r for r in out)
    return {"rows": out, "summary": summary}, pics


# ------------------------------------------------------------------ contact sheets
def crop(canvas: np.ndarray, outlines: list[np.ndarray]) -> Image.Image:
    """Both eyes of a canvas, with a margin, at CROP_SIZE."""
    p = np.concatenate(outlines)
    x0, y0 = p.min(0) - 25
    x1, y1 = p.max(0) + 25
    cx, cy, h = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, 2.2 * (y1 - y0)) / 2
    img = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    return img.crop((int(cx - h), int(cy - h / 2.2), int(cx + h), int(cy + h / 2.2))).resize(CROP_SIZE, Image.LANCZOS)


def contact_sheet(rows: Pictures, path: str | Path) -> None:
    """Labelled crops, one row per cat."""
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:                                    # Pillow before 10.1: the fixed bitmap font
        font = ImageFont.load_default()
    w, h = CROP_SIZE
    out = Image.new("RGB", (w * max(len(r) for r in rows), (h + 4) * len(rows)), "white")
    draw = ImageDraw.Draw(out)
    for i, r in enumerate(rows):
        for j, (label, im) in enumerate(r):
            out.paste(im, (j * w, i * (h + 4)))
            draw.text((j * w + 4, i * (h + 4) + 2), label, fill=(220, 0, 0), font=font)
    out.save(path)
