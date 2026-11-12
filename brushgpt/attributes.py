"""What a cat looks like, in the words a condition token carries: each eye's colour, the coat's base colour and its
pattern. The same readers build the training condition (read on the photo a painting was fitted to) and judge a
generated painting (read on its canvas), so a painting "reads right" when it would have been given the condition
that was asked for.

Eye colour. An eye's colour is the mean CIELAB colour of its iris pixels between the 35th and 95th percentiles of
lightness (the pupil and the catchlights left out), named by hue: unclear when it is too dark (L < 22) or too grey
(chroma < 8); blue, green or yellow by hue angle; otherwise orange when light (L >= 55) and copper when dark. Left and
right are the image's left and right.

Eye recolouring. An eye-colour version of a photo moves each asked eye to the colour real cat eyes of that colour
have at the eye's own lightness, keeping the eye's own light and shade: every pixel keeps its lightness and its
colour (a, b) moves by the same step, a pixel with less colour than the eye (the pupil, a catchlight) moving less.
The painter repaints only the eyes of such a photo (`painter.paint_eye_versions`).

Coat. The coat's base colour and pattern are read by a learned reader: a logistic regression on CLIP image features
(and, for the base colour, colour statistics of the head), trained on hand-labelled photos and paintings. An attribute
the reader is not confident of is "unclear" rather than a guess.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import lab2rgb, rgb2lab

from .cat import INPUT_SIDE, SIDE, CatInputs, photo_crop, plan_zones

UNCLEAR = "unclear"
EYE_COLOURS = ("unclear", "blue", "green", "yellow", "orange", "copper")
COAT_BASES = ("white", "black", "grey", "orange", "cream", "brown")
COAT_PATTERNS = ("solid", "tabby", "spots", "bicolour", "tortie_calico", "pointed")
ATTRIBUTES = ("left_eye", "right_eye", "base", "pattern")

# recolouring an eye
RECOLOURED = ("yellow", "blue", "green", "orange", "copper")   # the colours an eye can be recoloured to
GROW, FEATHER, HUE_NEAR = 2, 0.8, 60.0  # px past the plan's eye outline where a pixel of the iris's hue (within
                                        # HUE_NEAR degrees) is recoloured too; the soft edge (px)
NEAR_L, NEAR_MIN = 5.0, 30              # real eyes within NEAR_L of the eye's lightness (at least NEAR_MIN of them)
VIVID = ("yellow",)                     # colours whose target is the more saturated half of the real eyes
WRITE_GROW = 2                          # px about the eye outline a recoloured eye is written into

# the learned coat reader
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711])
READER_SETUP = {"base": ("both", 32.0), "pattern": ("clip", 8.0)}   # features and the regression's C
CONFIDENT = {"base": 0.6, "pattern": 0.5}                            # the least probability that is not "unclear"
COLOUR_WEIGHT = 0.1     # the standardised colour statistics' weight next to the unit-length CLIP features


# ------------------------------------------------------------------ eye colour
def lch(lab: np.ndarray) -> tuple[float, float, float]:
    """Lightness, chroma and hue angle (degrees) of one Lab colour."""
    L, a, b = (float(x) for x in lab)
    return L, math.hypot(a, b), math.degrees(math.atan2(b, a))


def eye_name(lab: np.ndarray) -> str:
    """The colour name of one eye's Lab colour (EYE_COLOURS)."""
    L, C, h = lch(lab)
    if L < 22 or C < 8:
        return UNCLEAR
    if h >= 180 or h < -30:
        return "blue"
    if 105 <= h < 180:
        return "green"
    if 75 <= h < 105:
        return "yellow"
    return "orange" if L >= 55 else "copper"


def eye_colour(lab: np.ndarray, iris: np.ndarray) -> np.ndarray:
    """An eye's Lab colour: the mean of its iris pixels between the 35th and 95th percentiles of lightness."""
    e = lab[iris]
    lo, hi = np.percentile(e[:, 0], [35, 95])
    return e[(e[:, 0] >= lo) & (e[:, 0] <= hi)].mean(0)


def image_left(cat: Mapping) -> np.ndarray:
    """The canvas pixels left of the midpoint between the two eye points."""
    mid = 0.5 * (cat["points"][0, 0] + cat["points"][1, 0])
    return (np.mgrid[:SIDE, :SIDE][1] + 0.5) < mid


def eye_masks(cat: Mapping) -> list[np.ndarray]:
    """The iris of the eye on the image's left, then of the eye on its right."""
    left = image_left(cat)
    return [cat["iris"] & left, cat["iris"] & ~left]


def eye_names(cat: Mapping, image: np.ndarray) -> list[str]:
    """Both eyes' colour names (left, right) read on an image of the cat (a photo or a canvas, RGB 0-1)."""
    lab = rgb2lab(np.clip(image, 0, 1))
    return [eye_name(eye_colour(lab, iris)) for iris in eye_masks(cat)]


def eye_view(inputs: CatInputs) -> dict:
    """What the eye readers need of a cat, at the painter's canvas size: the photo cut to the box ("photo"), the
    face points ("points") and the plan's eye masks ("eye", "iris", ...). The same as those keys of
    `cat.prepare_cat`, without the costly parts of it."""
    return {"photo": photo_crop(inputs, SIDE), "points": inputs.points * SIDE / INPUT_SIDE,
            **plan_zones(inputs.plan, inputs.points)}


# ------------------------------------------------------------------ eye recolouring
def real_eye_colours(rows: Path | str) -> dict[str, np.ndarray]:
    """The Lab colours of real cat eyes, by colour name: every labelled eye of a JSON-lines file whose rows carry
    "eye_left" and "eye_right" ({"name", "lab"})."""
    by: dict[str, list] = {n: [] for n in RECOLOURED}
    for line in Path(rows).read_text().splitlines():
        r = json.loads(line)
        for key in ("eye_left", "eye_right"):
            if r.get(key) and r[key]["name"] in by:
                by[r[key]["name"]].append(r[key]["lab"])
    return {n: np.array(v) for n, v in by.items()}


def vivid_half(eyes: np.ndarray) -> np.ndarray:
    """The real eyes at or above the median chroma of the eyes of the same colour within NEAR_L of their lightness."""
    chroma = np.hypot(eyes[:, 1], eyes[:, 2])
    keep = [c >= np.median(chroma[np.abs(eyes[:, 0] - lab[0]) <= NEAR_L]) for lab, c in zip(eyes, chroma, strict=True)]
    return eyes[np.array(keep)]


def eye_targets(rows: Path | str) -> dict[str, np.ndarray]:
    """The real eyes each colour is recoloured towards (the vivid half for VIVID colours)."""
    return {n: (vivid_half(v) if n in VIVID else v) for n, v in real_eye_colours(rows).items()}


def target_at(eyes: np.ndarray, lightness: float) -> tuple[np.ndarray, int]:
    """The colour (a, b) real eyes of one colour have at this lightness: the median over those within NEAR_L of it
    (widened until there are NEAR_MIN), and how many."""
    near = NEAR_L
    while (k := np.abs(eyes[:, 0] - lightness) <= near).sum() < NEAR_MIN and near < 100:
        near += NEAR_L
    return np.median(eyes[k, 1:], 0), int(k.sum())


def recolour(img: np.ndarray, cat: Mapping, targets: Sequence[np.ndarray]) -> np.ndarray:
    """Both eyes (left, right) of an image moved to the colours of `targets` (the real eyes of each colour), keeping
    each eye's own light and shade, on the plan's iris and GROW px about it where a pixel's hue is near the iris's;
    feathered FEATHER px."""
    lab = rgb2lab(np.clip(img, 0, 1))
    out = lab.copy()
    left = image_left(cat)
    hue, chroma = np.arctan2(lab[..., 2], lab[..., 1]), np.hypot(lab[..., 1], lab[..., 2])
    grown = ndimage.binary_dilation(cat["eye"], iterations=GROW)
    for iris, side, t in zip(eye_masks(cat), (left, ~left), targets, strict=True):
        own = eye_colour(lab, iris)
        step = target_at(t, own[0])[0] - own[1:]
        near = np.clip((np.cos(hue - math.atan2(own[2], own[1])) - math.cos(math.radians(HUE_NEAR)))
                       / (1 - math.cos(math.radians(HUE_NEAR))), 0, 1)
        region = np.where(iris, 1.0, near * (grown & side))
        w = ndimage.gaussian_filter(region, FEATHER) * np.clip(chroma / max(math.hypot(own[1], own[2]), 1e-3), 0, 1)
        out[..., 1:] += w[..., None] * step
    return np.clip(lab2rgb(out), 0, 1).astype(np.float32)


def recolour_eyes(img: np.ndarray, cat: Mapping, targets: Mapping[str, np.ndarray],
                  to: Sequence[str | None]) -> np.ndarray:
    """img with each eye named in `to` (left, right; None keeps that eye) recoloured to that colour, written inside
    that eye only (its outline grown WRITE_GROW px), so the rest of the image is unchanged."""
    both = recolour(img, cat, [targets[c or "blue"] for c in to])   # a kept eye's recolouring is never written
    left = image_left(cat)
    grown = ndimage.binary_dilation(cat["eye"], iterations=WRITE_GROW)
    write = np.zeros_like(grown)
    for half, c in zip((left, ~left), to, strict=True):
        if c:
            write |= grown & half
    return np.where(write[..., None], both, img).astype(np.float32)


# ------------------------------------------------------------------ coat
def coat_names(row: Mapping) -> list[str]:
    """A coat reading ({"base": {"label", "confident"}, "pattern": {...}}) as [base, pattern] names, "unclear" where
    the reader is not confident."""
    return [row[k]["label"] if row[k]["confident"] else UNCLEAR for k in ("base", "pattern")]


def read_coat_labels(path: Path | str) -> dict[str, dict]:
    """The coat readings of a JSON-lines file with one row per cat ({"name", "base", "pattern"}), by cat."""
    rows = (json.loads(x) for x in Path(path).read_text().splitlines() if x.strip())
    return {r["name"]: r for r in rows}


def to_uint8(img: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8))


def colour_stats(img: np.ndarray) -> np.ndarray:
    """Colour statistics of the head (the non-white pixels): the 10/25/50/75/90th percentiles of L*, a*, b* and
    chroma, and the shares of dark (L* < 25), white (L* > 80, chroma < 10) and warm (chroma > 15, hue 20-100 deg)
    pixels."""
    lab = rgb2lab(np.clip(img, 0, 1))[img.min(-1) < 0.999]
    c = np.hypot(lab[:, 1], lab[:, 2])
    h = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360
    q = np.percentile(np.c_[lab, c], [10, 25, 50, 75, 90], axis=0).ravel()
    return np.r_[q, (lab[:, 0] < 25).mean(), ((lab[:, 0] > 80) & (c < 10)).mean(),
                 ((c > 15) & (h > 20) & (h < 100)).mean()]


def reader_features(kind: str, clip: np.ndarray, colour: np.ndarray, colour_ref: np.ndarray) -> np.ndarray:
    """The regression's input: the CLIP features ("clip"), or those and the colour statistics standardised over the
    labelled photos ("both")."""
    if kind == "clip":
        return clip
    return np.c_[clip, (colour - colour_ref.mean(0)) / (colour_ref.std(0) + 1e-6) * COLOUR_WEIGHT]


class CoatReader:
    """The learned coat reader. Built from the labelled cats: `features` (.npz: "names", and CLIP features of each
    cat's "photo" and "painting"), `colours` (.npz: the same with colour statistics) and `labels` (JSON lines:
    {"name", "base", "pattern", "skip"}). Reading needs the CLIP model's weights already in the local Hugging Face
    cache; they are never downloaded.

    reader(images) -> one row per image, {"base": {"label", "p", "confident"}, "pattern": {...}}; images are RGB in
    [0, 1], the head on white."""

    def __init__(self, features: Path | str, colours: Path | str, labels: Path | str, clip_model: str = CLIP_MODEL):
        from sklearn.linear_model import LogisticRegression
        from transformers import CLIPModel

        z, c = np.load(features), np.load(colours)
        at = {str(n): i for i, n in enumerate(z["names"])}
        rows = [json.loads(x) for x in Path(labels).read_text().splitlines() if x.strip()]
        rows = [r for r in rows if not r.get("skip")]
        idx = np.array([at[r["name"]] for r in rows])
        self.colour_ref = c["photo"]
        self.models = {}
        for key, (kind, strength) in READER_SETUP.items():
            y = np.array([r[key] for r in rows])
            x = np.r_[reader_features(kind, z["photo"][idx], c["photo"][idx], self.colour_ref),
                      reader_features(kind, z["painting"][idx], c["painting"][idx], self.colour_ref)]
            model = LogisticRegression(C=strength, max_iter=5000, class_weight="balanced").fit(x, np.r_[y, y])
            self.models[key] = (kind, model)
        self.clip = CLIPModel.from_pretrained(clip_model, local_files_only=True).eval()

    def clip_features(self, images: Sequence[np.ndarray]) -> np.ndarray:
        import torch

        x = np.stack([((np.asarray(to_uint8(a).resize((224, 224), Image.BICUBIC)) / 255.0 - CLIP_MEAN) / CLIP_STD)
                      .transpose(2, 0, 1) for a in images])
        with torch.no_grad():
            e = self.clip.get_image_features(pixel_values=torch.tensor(x, dtype=torch.float32))
        e = e if isinstance(e, torch.Tensor) else e.pooler_output
        return torch.nn.functional.normalize(e, dim=-1).numpy()

    def __call__(self, images: Sequence[np.ndarray]) -> list[dict]:
        clip = self.clip_features(images)
        colour = np.array([colour_stats(a) for a in images], np.float32)
        rows = []
        for j in range(len(images)):
            row = {}
            for key, (kind, model) in self.models.items():
                pr = model.predict_proba(reader_features(kind, clip[j:j + 1], colour[j:j + 1], self.colour_ref))[0]
                k = int(pr.argmax())
                row[key] = {"label": str(model.classes_[k]), "p": round(float(pr[k]), 3),
                            "confident": bool(pr[k] >= CONFIDENT[key])}
            rows.append(row)
        return rows


# ------------------------------------------------------------------ judging a painting
def painting_attributes(cat: Mapping, images: Sequence[np.ndarray],
                        read_coat: Callable[[list[np.ndarray]], list[dict]]) -> list[list[str]]:
    """The four attribute names (ATTRIBUTES) read on each image of one cat: the eyes by colour, the coat by the
    learned reader."""
    coats = read_coat([np.clip(v, 0, 1) for v in images])
    return [eye_names(cat, img) + coat_names(c) for img, c in zip(images, coats, strict=True)]


def read_right(names: Sequence[str], asked: Sequence[str]) -> list[bool | None]:
    """Per attribute: whether the painting reads as asked; None where nothing was asked (the condition is unclear)."""
    return [None if a == UNCLEAR else n == a for n, a in zip(names, asked, strict=True)]
