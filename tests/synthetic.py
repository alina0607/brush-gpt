"""A small synthetic cat for tests that must run without any data: a round brown face with two ears, green eyes,
a pink nose and a mouth line, on a 256 px cut-out whose box is the whole photo."""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path as Polygon
from PIL import Image

from brushgpt.cat import CatInputs

POINTS = np.array(
    [[95, 120], [161, 120], [128, 182], [50, 95], [62, 28], [100, 72], [156, 72], [194, 28], [206, 95]], np.float32
)
EYE_RADII = (17.0, 11.0)


def ellipse(cx: float, cy: float, rx: float, ry: float, n: int = 16) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[cx + rx * np.cos(t), cy + ry * np.sin(t)]


def synthetic_inputs(eye_colour: tuple[float, float, float] = (0.35, 0.6, 0.25)) -> CatInputs:
    side = 256
    yy, xx = np.mgrid[:side, :side] + 0.5
    q = np.c_[xx.ravel(), yy.ravel()]
    inside = lambda poly: Polygon(poly).contains_points(q).reshape(side, side)  # noqa: E731
    head = ((xx - 128) / 92) ** 2 + ((yy - 140) / 84) ** 2 <= 1
    head |= inside(POINTS[[3, 4, 5]]) | inside(POINTS[[6, 7, 8]])
    rng = np.random.default_rng(0)
    grain = 0.08 * np.sin(xx / 3.0 + yy / 7.0)[..., None] + 0.03 * rng.standard_normal((side, side, 1))
    fur = np.array([0.55, 0.42, 0.30]) + grain
    img = np.where(head[..., None], fur, 1.0)
    eyes = [ellipse(POINTS[e][0], POINTS[e][1], *EYE_RADII) for e in (0, 1)]
    for e in (0, 1):
        cx, cy = POINTS[e]
        iris = ((xx - cx) / (0.75 * EYE_RADII[0])) ** 2 + ((yy - cy) / (0.75 * EYE_RADII[1])) ** 2 <= 1
        img[iris] = eye_colour
        img[((xx - cx) / 3) ** 2 + ((yy - cy) / 8) ** 2 <= 1] = 0.05
    nose = ellipse(128, 160, 10, 6, 8)
    img[inside(nose)] = (0.85, 0.55, 0.55)
    target = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
    tri = head.astype(np.uint8)
    plan = {
        "eyes": np.stack(eyes),
        "pupils": np.array([[95, 120, 3, 8, 0, 0], [161, 120, 3, 8, 0, 0]], float),
        "nose": nose,
        "mouth": np.array([[112, 186], [120, 190], [128, 186], [136, 190], [144, 186]], float),
    }
    return CatInputs(
        target=target, tri=tri, points=POINTS.copy(), box=(0, 0, side, side), photo=Image.fromarray(target), plan=plan
    )
