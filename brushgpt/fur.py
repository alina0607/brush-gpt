"""The direction the fur grows at every pixel of a cat's face, which the painter's strokes start along.

The field blends an anatomy template with the photo's own texture:

  template  from the nose outwards, drooping on the cheeks below it, along the eyes around them, and in each ear
            outwards from the middle of the ear's base
  photo     the structure tensor of the photo's lightness (the fine grain of the hairs), which bends the template
            where the two roughly agree (BEND x coherence x cos of the doubled angle between them)
  texture   around the eyes and the nose the photo shows texture rather than hairs; there the field is the photo's
            local edge direction alone

Directions are lines, not arrows (a stroke at angle t and at t + pi is the same stroke), so fields are stored as
doubled-angle unit vectors (cos 2t, sin 2t). The field is computed on the 256 px grid the face points are given on.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

FIELD_SIDE = 256                       # the grid the field is computed on (the face points' grid)
TEXTURE_EYE, TEXTURE_NOSE = 0.45, 0.30  # eye distances around the eyes and the nose where the photo shows texture
BEND = 0.8                             # how strongly the photo's grain bends the template


def gauss(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of an (H, W) or (H, W, C) image, edges reflected."""
    r = max(1, int(3 * sigma))
    k = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = img[None, None] if img.dim() == 2 else img.permute(2, 0, 1)[:, None]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return x[0, 0] if img.dim() == 2 else x[:, 0].permute(1, 2, 0)


def _sobel(g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8
    x = F.pad(g[None, None], (1, 1, 1, 1), mode="replicate")
    return F.conv2d(x, kx)[0, 0], F.conv2d(x, kx.transpose(2, 3))[0, 0]


def texture(img: torch.Tensor, window: float = 1.5) -> tuple[torch.Tensor, torch.Tensor]:
    """The fine-scale direction along the hair texture (across the lightness gradient) and its coherence in 0-1."""
    gx, gy = _sobel(img @ torch.tensor([0.299, 0.587, 0.114]))
    jxx, jyy, jxy = (gauss(t, window) for t in (gx * gx, gy * gy, gx * gy))
    theta = 0.5 * torch.atan2(2 * jxy, jxx - jyy) + math.pi / 2
    coh = torch.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    return theta, coh


def orientation(target: torch.Tensor) -> torch.Tensor:
    """Per-pixel angle along the image's local structure (across the smoothed gradient, 9 px window), (H, W)."""
    g = target.mean(-1)[None, None]
    kx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=g.dtype) / 8
    gx, gy = F.conv2d(g, kx, padding=1), F.conv2d(g, kx.transpose(2, 3), padding=1)
    jxx, jyy, jxy = (F.avg_pool2d(t, 9, 1, 4) for t in (gx * gx, gy * gy, gx * gy))
    return (0.5 * torch.atan2(2 * jxy, jxx - jyy) + math.pi / 2)[0, 0]


def doubled(theta: torch.Tensor) -> torch.Tensor:
    """Directions as (cos 2t, sin 2t): a line at t and at t + pi are the same."""
    return torch.stack([torch.cos(2 * theta), torch.sin(2 * theta)], -1)


def pixel_grid(side: int = FIELD_SIDE) -> torch.Tensor:
    """Pixel centres (side, side, 2) as (x, y)."""
    ys, xs = torch.meshgrid(torch.arange(side) + 0.5, torch.arange(side) + 0.5, indexing="ij")
    return torch.stack([xs, ys], -1)


def soft_triangle(grid: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, soft: float) -> torch.Tensor:
    """1 inside the triangle abc, 0 outside, with an edge `soft` px wide."""
    centroid = (a + b + c) / 3
    a, b, c = (centroid + (v - centroid) for v in (a, b, c))   # recentred on the centroid (differs only by rounding)
    dist = []
    for p, q, other in ((a, b, c), (b, c, a), (c, a, b)):
        e = q - p
        n = torch.stack([-e[1], e[0]]) / torch.linalg.norm(e)
        if torch.dot(other - p, n) < 0:
            n = -n
        dist.append((grid - p) @ n)
    return torch.sigmoid(torch.stack(dist).min(0).values / soft)


def anatomy_template(points: np.ndarray, grid: torch.Tensor) -> torch.Tensor:
    """The anatomy template as doubled-angle unit vectors (H, W, 2). points: the nine face points (eyes 0-1, mouth
    2, left ear 3-5 and right ear 6-8 as outer base, tip, inner base / inner base, tip, outer base)."""
    p = torch.as_tensor(points, dtype=torch.float32)
    mid = (p[0] + p[1]) / 2
    d = float(torch.linalg.norm(p[1] - p[0]))
    nose = mid + 0.7 * (p[2] - mid)
    right = (p[1] - p[0]) / d
    up = (mid - p[2]) / torch.linalg.norm(mid - p[2])
    rel = grid - nose
    v = rel / (torch.linalg.norm(rel, dim=-1, keepdim=True) + 1e-6)
    below = torch.sigmoid(-(rel @ up) / (0.3 * d))
    lateral = ((rel @ right).abs() / d).clamp(0, 1)
    v = v - (0.45 * below * lateral)[..., None] * up
    radial = doubled(torch.atan2(v[..., 1], v[..., 0]))
    eye_dir = doubled(torch.atan2(right[1], right[0])).expand_as(radial)
    w_eye = sum(torch.exp(-(torch.linalg.norm(grid - e, dim=-1) / (0.45 * d)) ** 2) for e in (p[0], p[1]))
    parts, weights = [radial, eye_dir], [torch.ones_like(w_eye), 2.0 * w_eye]
    for outer, tip, inner in ((p[3], p[4], p[5]), (p[8], p[7], p[6])):
        out = grid - (outer + inner) / 2
        parts.append(doubled(torch.atan2(out[..., 1], out[..., 0])))
        weights.append(6.0 * soft_triangle(grid, outer, tip, inner, soft=4.0))
    f = sum(w[..., None] * q for w, q in zip(weights, parts, strict=True))
    return f / (torch.linalg.norm(f, dim=-1, keepdim=True) + 1e-6)


def texture_zone(points: np.ndarray, side: int = FIELD_SIDE) -> np.ndarray:
    """Where the photo shows texture rather than hairs: within TEXTURE_EYE eye distances of an eye or TEXTURE_NOSE
    of the nose (0.7 of the way from the eyes' midpoint to the mouth). (side, side) bool."""
    yy, xx = np.mgrid[:side, :side] + 0.5
    d = float(np.linalg.norm(points[0] - points[1]))
    nose = points[:2].mean(0) + 0.7 * (points[2] - points[:2].mean(0))
    eye = np.min([np.hypot(xx - x, yy - y) for x, y in points[:2]], 0)
    return (eye < TEXTURE_EYE * d) | (np.hypot(xx - nose[0], yy - nose[1]) < TEXTURE_NOSE * d)


def fur_field(target: torch.Tensor, points: np.ndarray) -> torch.Tensor:
    """The fur's direction (FIELD_SIDE, FIELD_SIDE, 2) as doubled-angle unit vectors, from the photo `target`
    (FIELD_SIDE, FIELD_SIDE, 3) in 0-1 and the face points on that grid."""
    tmpl = anatomy_template(points, pixel_grid())
    theta, coh = texture(target)
    img = doubled(theta)
    w = (BEND * coh * (tmpl * img).sum(-1).clamp(min=0))[..., None]
    f = gauss((1 - w) * tmpl + w * img, 2.0)
    f = f / (torch.linalg.norm(f, dim=-1, keepdim=True) + 1e-6)
    t = torch.from_numpy(texture_zone(points).astype(np.float32))[..., None]
    return (1 - t) * f + t * doubled(orientation(target))


def fur_field_at(target: torch.Tensor, points: np.ndarray, side: int) -> torch.Tensor:
    """fur_field upsampled (bilinear) to a side x side canvas, (side, side, 2)."""
    g = fur_field(target, points)
    up = F.interpolate(g.permute(2, 0, 1)[None], size=(side, side), mode="bilinear", align_corners=False)
    return up[0].permute(1, 2, 0)
