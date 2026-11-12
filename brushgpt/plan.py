"""The sketch plan: what a painter decides and outlines before the first stroke.

From the cut-out (photo on white, tri map) and the nine face points, all on the 256 px grid:

  pose     yaw, pitch and roll of the head. A mean 3D cat head (the nine points) is recovered from the 2D points of
           6,000 cats by Tomasi-Kanade factorisation (scaled orthographic cameras, a rank-3 factorisation and a
           metric upgrade), so the template's depths come from data, not from hand-set numbers. A cat's pose is the
           scaled rotation that best maps the template onto its points (least squares), as Euler angles: roll about
           the view axis, then pitch, then yaw. The fitted template ships with the package (`load_head_template`).
  head     the cat's outline where it meets the background (tri 1 next to tri 0), in runs resampled every HEAD_STEP
           px. Where the outline meets the uncertain region (the chin and neck, tri 2) there is no line, because the
           head's end there is not known. Each point is flagged 1 inside a (slightly enlarged) ear triangle.
  eyes     each eye's outline from Segment Anything (SAM, `facebook/sam-vit-base`) on a zoomed crop around the eye,
           prompted from several points near the eye point; the most eye-like mask is kept.
  pupils   in each eye (its outline shrunk towards its centre, so the dark rim is not taken for the pupil), the dark
           blob around the darkest point, as an ellipse (centre, two radii, angle) from its second moments.
  nose     SAM prompted at the nose, with negative points on the eyes, the bridge of the nose and below the mouth.
  mouth    the strongest thin dark line across the muzzle below the nose, found by dynamic programming.

The plan is written as tokens ahead of a painting's strokes (`plan_tokens`): the pose as three angle bins, then each
outline as a short list of points on the 256 grid. `save_plan` writes the `.npz` the painter reads
(`brushgpt.cat.load_inputs`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.path import Path as Polygon
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.ndimage import label as connected_components
from scipy.spatial.transform import Rotation
from skimage.measure import find_contours

from .cat import face_frame

GRID = 256                          # the cut-out's and the face points' grid
TEMPLATE_FILE = Path(__file__).parent / "data" / "cat_head_template.npy"
SAM_MODEL = "facebook/sam-vit-base"
PLAN_VERSION = "sketch-plan-v1"
EARS = ((3, 4, 5), (6, 7, 8))       # each ear's face points: base, tip, base
EAR_GROW = 1.25                     # an ear triangle enlarged about its middle when flagging outline points
HEAD_STEP = 6.0                     # px between plan points along the head's outline
EYE_POINTS = 16                     # points per eye outline
NOSE_POINTS = 12                    # points per nose outline
EYE_CROP = 1.2                      # the eye crop's side for SAM, in eye distances
EYE_PROMPTS = ((0, 0), (0, 0.08), (0, 0.15), (0, -0.08))   # prompt centres (along, across the eye line), eye distances
NOSE_AT = 0.85                      # the nose prompt, as a fraction of the way from the eyes' midpoint to the mouth
BRIDGE_AT = 0.45                    # a negative prompt on the bridge of the nose (else SAM takes the whole bridge)
PUPIL_SHRINK = 0.75                 # the eye outline shrunk about its centre before looking for the pupil
MOUTH_HALF = 0.4                    # half the mouth's width searched, in eye distances
MOUTH_FLOOR = 0.02                  # a mouth column is strong when its line is at least this much darker than its sides
MOUTH_GAP = 3                       # weak columns in a row the mouth line may cross
ANGLE_BINS = 64                     # angle tokens over -90..90 degrees


# ------------------------------------------------------------------ pose
def fit_head_template(points: np.ndarray) -> np.ndarray:
    """A mean 3D head (3, 9) from many 2D views (F, 9, 2) of the nine face points, by Tomasi-Kanade factorisation.

    The centred views are stacked into a (2F, 9) measurement matrix and factorised at rank 3 into cameras and shape;
    the metric upgrade finds the 3x3 matrix that makes every camera's two rows orthogonal and of equal length. The
    template is expressed in the face's frame: eye distance 1, x along the eyes, y towards the mouth, z away from the
    viewer (the ears lie behind the mouth)."""
    centred = points - points.mean(1, keepdims=True)
    W = np.concatenate([centred[..., 0], centred[..., 1]], 0)          # (2F, 9): all x rows, then all y rows
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    cameras, shape = U[:, :3] * np.sqrt(s[:3]), np.sqrt(s[:3])[:, None] * Vt[:3]
    f = len(points)
    m, n = cameras[:f], cameras[f:]

    def quadratic(a: np.ndarray, b: np.ndarray) -> np.ndarray:         # a L b^T, linear in L's six entries
        return np.stack([a[:, 0] * b[:, 0], a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0],
                         a[:, 0] * b[:, 2] + a[:, 2] * b[:, 0], a[:, 1] * b[:, 1],
                         a[:, 1] * b[:, 2] + a[:, 2] * b[:, 1], a[:, 2] * b[:, 2]], 1)

    A = np.concatenate([quadratic(m, m) - quadratic(n, n), quadratic(m, n)], 0)
    ell = np.linalg.svd(A)[2][-1]
    L = np.array([[ell[0], ell[1], ell[2]], [ell[1], ell[3], ell[4]], [ell[2], ell[4], ell[5]]])
    if np.trace(L) < 0:
        L = -L
    vals, vecs = np.linalg.eigh(L)
    Q = vecs @ np.diag(np.sqrt(np.clip(vals, 1e-12, None)))
    shape = np.linalg.solve(Q, shape)                                   # (3, 9), metric up to a rotation
    ex = shape[:, 1] - shape[:, 0]
    d = np.linalg.norm(ex)
    ex /= d
    to_mouth = shape[:, 2] - shape[:, :2].mean(1)
    ey = to_mouth - ex * (ex @ to_mouth)
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    t = np.stack([ex, ey, ez]) @ (shape - shape[:, :2].mean(1, keepdims=True)) / d
    if t[2, [3, 4, 5, 6, 7, 8]].mean() < t[2, 2]:                       # a reflection: ears go behind the mouth
        t[2] = -t[2]
    return t


def load_head_template(path: Path | str = TEMPLATE_FILE) -> np.ndarray:
    """The mean 3D head fitted to the face points of 6,000 cats (AFHQ and the CAT dataset): (3, 9) float32."""
    return np.load(path)


@dataclass
class Pose:
    """A head pose: Euler angles in degrees, the scale (pixels per eye distance), the mean fitting error (eye
    distances) and the rotation matrix."""

    yaw: float
    pitch: float
    roll: float
    scale: float
    residual: float
    R: np.ndarray


def head_pose(template: np.ndarray, points: np.ndarray) -> Pose:
    """The scaled-orthographic fit of the template (3, 9) to one cat's points (9, 2): the least-squares affine
    camera, projected to the nearest scaled rotation."""
    w = (points - points.mean(0)).T                                     # (2, 9)
    s = template - template.mean(1, keepdims=True)
    affine = w @ np.linalg.pinv(s)                                      # (2, 3)
    u, sig, vt = np.linalg.svd(affine, full_matrices=False)
    r12 = u @ vt
    R = np.vstack([r12, np.cross(r12[0], r12[1])])
    scale = sig.mean()
    residual = np.linalg.norm(scale * r12 @ s - w, axis=0).mean() / np.linalg.norm(points[0] - points[1])
    roll, pitch, yaw = Rotation.from_matrix(R).as_euler("ZXY", degrees=True)
    return Pose(float(yaw), float(pitch), float(roll), float(scale), float(residual), R)


# ------------------------------------------------------------------ outlines
def arc_length(xy: np.ndarray) -> np.ndarray:
    """The length along a polyline at each of its points, starting at 0."""
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]


def resample_open(xy: np.ndarray, step: float) -> np.ndarray:
    """An open polyline resampled every `step` px along its length."""
    length = arc_length(xy)
    at = np.arange(0, length[-1] + 1e-6, step)
    return np.c_[np.interp(at, length, xy[:, 0]), np.interp(at, length, xy[:, 1])]


def resample_closed(xy: np.ndarray, n: int) -> np.ndarray:
    """A closed outline resampled to n points evenly spaced along its length."""
    p = np.r_[xy, xy[:1]]
    length = arc_length(p)
    at = np.linspace(0, length[-1], n, endpoint=False)
    return np.c_[np.interp(at, length, p[:, 0]), np.interp(at, length, p[:, 1])]


def in_ears(points: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Whether each point lies inside an ear triangle enlarged EAR_GROW about its middle."""
    out = np.zeros(len(xy), bool)
    for ear in EARS:
        t = points[list(ear)]
        out |= Polygon(t.mean(0) + EAR_GROW * (t - t.mean(0))).contains_points(xy)
    return out


def head_outline(tri: np.ndarray, points: np.ndarray) -> list[np.ndarray]:
    """The runs of the cat's outline that touch the background, each (n, 3): x, y and 1.0 where inside an ear."""
    near_background = binary_dilation(tri == 0, iterations=2)
    runs = []
    for contour in find_contours(np.pad(tri == 1, 1).astype(float), 0.5):
        xy = contour[:, ::-1] - 1                                       # (x, y) in pixels
        if len(xy) < 20:
            continue
        ix = np.clip(xy[:, 0].round().astype(int), 0, GRID - 1)
        iy = np.clip(xy[:, 1].round().astype(int), 0, GRID - 1)
        on = near_background[iy, ix]
        if on.all():
            pieces = [xy]
        else:                                                           # start off the background: no run wraps around
            k = int(np.argmin(on))
            xy, on = np.roll(xy, -k, 0), np.roll(on, -k)
            labels, n = connected_components(on)
            pieces = [xy[labels == j] for j in range(1, n + 1)]
        for piece in pieces:
            if arc_length(piece)[-1] < 3 * HEAD_STEP:
                continue
            q = resample_open(piece, HEAD_STEP)
            runs.append(np.c_[q, in_ears(points, q)])
    return runs


# ------------------------------------------------------------------ Segment Anything
@dataclass
class Sam:
    """A loaded Segment Anything model and its processor (Hugging Face `transformers`)."""

    model: Any
    processor: Any


def load_sam(name: str = SAM_MODEL, local_only: bool = False) -> Sam:
    """SAM on the CPU in eval mode. With `local_only` it is read from the Hugging Face cache only (an OSError when it
    is not there); otherwise it is downloaded on first use."""
    from transformers import SamModel, SamProcessor
    model = SamModel.from_pretrained(name, use_safetensors=True, local_files_only=local_only).eval()
    return Sam(model, SamProcessor.from_pretrained(name, local_files_only=local_only))


def sam_masks(sam: Sam, image: np.ndarray, positive: list, negative: list, embedding: Any = None) -> tuple:
    """SAM's three masks and their predicted IoU scores for point prompts on a uint8 image, and the image embedding
    (pass it back to prompt the same image again without re-encoding it)."""
    import torch
    prompts = [[[list(map(float, q)) for q in positive + negative]]]
    labels = [[[1] * len(positive) + [0] * len(negative)]]
    inputs = sam.processor(images=image, input_points=prompts, input_labels=labels, return_tensors="pt")
    with torch.no_grad():
        if embedding is None:
            embedding = sam.model.get_image_embeddings(inputs["pixel_values"])
        out = sam.model(image_embeddings=embedding, input_points=inputs["input_points"],
                        input_labels=inputs["input_labels"], multimask_output=True)
    masks = sam.processor.image_processor.post_process_masks(out.pred_masks, inputs["original_sizes"],
                                                             inputs["reshaped_input_sizes"])[0][0].numpy()
    return masks, out.iou_scores[0, 0].numpy(), embedding


def outer_contour(mask: np.ndarray) -> np.ndarray:
    """The longest contour of a mask, as (x, y) pixels."""
    return max(find_contours(np.pad(mask, 1).astype(float), 0.5), key=len)[:, ::-1] - 1


def is_eye_like(mask: np.ndarray, eye: np.ndarray, d: float) -> bool:
    """A mask passes as an eye (d: the eye distance in the mask's pixels; eye: the eye point there) when it has the
    area of a disc of 0.10-0.35 eye distances, no point farther than 0.4 eye distances from its centre, its centre
    within 0.25 eye distances of the eye point, and it fills at least 0.85 of its convex hull."""
    from skimage.morphology import convex_hull_image
    area = mask.sum()
    if not np.pi * (0.10 * d) ** 2 < area < np.pi * (0.35 * d) ** 2:
        return False
    ys, xs = np.nonzero(mask)
    centre = np.array([xs.mean(), ys.mean()])
    return not (np.hypot(xs - centre[0], ys - centre[1]).max() > 0.4 * d or np.linalg.norm(centre - eye) > 0.25 * d
                or area < 0.85 * convex_hull_image(mask).sum())


def eye_outlines(sam: Sam, target: np.ndarray, eyes: np.ndarray, d: float) -> list[np.ndarray]:
    """Each eye's outline (n, 2) on the canvas, from SAM on a crop of EYE_CROP eye distances around the eye point,
    zoomed to 256 px.

    The eye point is not always on the eye (it can sit on the fur just above it), so SAM is prompted from several
    centres near it (EYE_PROMPTS), each with two more positives 0.1 eye distances either side along the eye line and
    negatives 0.6 eye distances above and below the eye point. Of all the masks, the best-scoring eye-like one
    (`is_eye_like`) is kept; when none qualifies, an ellipse of 0.2 x 0.12 eye distances around the eye point."""
    ex = (eyes[1] - eyes[0]) / d
    ey = np.array([-ex[1], ex[0]])
    image = Image.fromarray((target.clip(0, 1) * 255).astype(np.uint8))
    out = []
    for e in eyes:
        side = EYE_CROP * d
        lo = e - side / 2
        zoom = GRID / side
        crop = np.asarray(image.transform((GRID, GRID), Image.EXTENT, (*lo, *(lo + side)), Image.BICUBIC,
                                          fillcolor=(255, 255, 255)))

        def to_crop(q: np.ndarray, lo: np.ndarray = lo, zoom: float = zoom) -> np.ndarray:
            return (q - lo) * zoom

        best, best_score, embedding = None, -1.0, None
        for u, v in EYE_PROMPTS:
            c = e + d * (u * ex + v * ey)
            positive = [to_crop(c), to_crop(c + 0.1 * d * ex), to_crop(c - 0.1 * d * ex)]
            negative = [to_crop(e - 0.6 * d * ey), to_crop(e + 0.6 * d * ey)]
            masks, scores, embedding = sam_masks(sam, crop, positive, negative, embedding)
            for mask, score in zip(masks, scores, strict=True):
                if score > best_score and is_eye_like(mask, to_crop(e), d * zoom):
                    best, best_score = mask, float(score)
        if best is None:
            t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
            out.append(e + d * (0.2 * np.cos(t)[:, None] * ex + 0.12 * np.sin(t)[:, None] * ey))
        else:
            out.append(outer_contour(best) / zoom + lo)
    return out


def nose_outline(sam: Sam, target: np.ndarray, points: np.ndarray, d: float) -> np.ndarray | None:
    """The nose's outline (NOSE_POINTS, 2), or None when SAM gives no mask of a nose's size (the area of a disc of
    0.06-0.25 eye distances). Positive prompts: the nose point (NOSE_AT of the way from the eyes to the mouth) and
    0.08 eye distances either side; negative: both eyes, the bridge of the nose and just below the mouth."""
    mid = points[:2].mean(0)
    nose = mid + NOSE_AT * (points[2] - mid)
    ex = (points[1] - points[0]) / d
    positive = [nose, nose + 0.08 * d * ex, nose - 0.08 * d * ex]
    negative = [points[0], points[1], mid + BRIDGE_AT * (points[2] - mid),
                points[2] + 0.15 * d * (points[2] - mid) / np.linalg.norm(points[2] - mid)]
    masks, scores, _ = sam_masks(sam, (target.clip(0, 1) * 255).astype(np.uint8), positive, negative)
    area = masks.reshape(len(masks), -1).sum(1)
    ok = (area > np.pi * (0.06 * d) ** 2) & (area < np.pi * (0.25 * d) ** 2)
    if not ok.any():
        return None
    return resample_closed(outer_contour(masks[int(np.argmax(np.where(ok, scores, -1)))]), NOSE_POINTS)


# ------------------------------------------------------------------ pupils and the mouth
def luminance(img: np.ndarray) -> np.ndarray:
    return img[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=img.dtype)


def dark_lines(lum: np.ndarray, s: int = 2) -> np.ndarray:
    """Per pixel, how much darker it is than the mean of its two sides s px away, in the best of four directions."""
    p = np.pad(lum, s, mode="edge")
    h, w = lum.shape
    out = np.zeros_like(lum)
    for dy, dx in ((0, s), (s, 0), (s, s), (s, -s)):
        a = p[s + dy:s + dy + h, s + dx:s + dx + w]
        b = p[s - dy:s - dy + h, s - dx:s - dx + w]
        out = np.maximum(out, (a + b) / 2 - lum)
    return out.clip(0)


@dataclass
class Pupil:
    """A pupil as an ellipse: centre (x, y), radii (major, minor; two standard deviations), the major axis's angle
    (radians) and the share of the eye it covers."""

    centre: np.ndarray
    radii: np.ndarray
    angle: float
    share: float

    def as_row(self) -> np.ndarray:
        return np.r_[self.centre, self.radii, self.angle, self.share]


def pupil_ellipse(lum: np.ndarray, outline: np.ndarray) -> Pupil | None:
    """The dark blob around the darkest (smoothed) point inside the eye outline (shrunk PUPIL_SHRINK about its
    centre), as an ellipse from the blob's second moments; None when there is no blob of at least three pixels."""
    yy, xx = np.mgrid[:GRID, :GRID] + 0.5
    c0 = outline.mean(0)
    outline = c0 + PUPIL_SHRINK * (outline - c0)
    inside = Polygon(outline).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(GRID, GRID)
    if inside.sum() < 12:
        return None
    smooth = gaussian_filter(lum, 1.0)
    vals = smooth[inside]
    lo, median = vals.min(), np.median(vals)
    labels, _ = connected_components(inside & (smooth < lo + 0.4 * (median - lo)))
    iy, ix = np.unravel_index(np.argmin(np.where(inside, smooth, np.inf)), smooth.shape)
    if labels[iy, ix] == 0:
        return None
    blob = labels == labels[iy, ix]
    if blob.sum() < 3:                                                  # one or two pixels: no shape to fit
        return None
    ys, xs = np.nonzero(blob)
    cov = np.cov(np.c_[xs, ys].T) + np.eye(2) * 1e-3
    vals, vecs = np.linalg.eigh(cov)
    if not np.all(np.isfinite(vals)):
        return None
    return Pupil(np.array([xs.mean(), ys.mean()]) + 0.5, 2 * np.sqrt(vals[::-1]),
                 float(np.arctan2(vecs[1, 1], vecs[0, 1])), float(blob.sum() / inside.sum()))


def to_face(points: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Points (n, 2) in the face's frame: eye distances from the eyes' midpoint, x along the eyes, y to the mouth."""
    d, mid, ex, ey, _ = face_frame(points)
    rel = xy - mid
    return np.stack([rel @ ex, rel @ ey], -1) / d


def mouth_line(lum: np.ndarray, points: np.ndarray, nose: np.ndarray | None, d: float) -> np.ndarray:
    """The mouth as one smooth curve (n, 2), possibly empty.

    Over the columns of the face's frame within MOUTH_HALF eye distances of the mouth point and the rows from the
    nose's lowest point to 0.3 eye distances below the mouth point, the path of strongest thin dark line that moves
    at most one row per column (dynamic programming), cut to the columns around the middle where the line is at least
    MOUTH_FLOOR strong (crossing at most MOUTH_GAP weak columns in a row)."""
    lines = dark_lines(lum)
    _, mid, ex, ey, _ = face_frame(points)
    m = to_face(points, points[2:3])[0]
    top = to_face(points, nose).max(0)[1] if nose is not None else m[1] - 0.15
    xs = np.arange(m[0] - MOUTH_HALF, m[0] + MOUTH_HALF, 1.5 / d)
    ys = np.arange(top, m[1] + 0.3, 0.5 / d)
    if len(ys) < 2:
        return np.zeros((0, 2))
    q = mid + d * (xs[:, None, None] * ex + ys[None, :, None] * ey)        # (columns, rows, 2) on the canvas
    ix, iy = np.clip(q[..., 0].astype(int), 0, GRID - 1), np.clip(q[..., 1].astype(int), 0, GRID - 1)
    v = lines[iy, ix]
    score, back = v[0].copy(), np.zeros(v.shape, int)
    for c in range(1, len(xs)):
        prev = np.stack([np.r_[-np.inf, score[:-1]], score, np.r_[score[1:], -np.inf]])
        k = np.argmax(prev, 0)
        back[c] = np.arange(len(ys)) + k - 1
        score = prev[k, np.arange(len(ys))] + v[c]
    path = [int(np.argmax(score))]
    for c in range(len(xs) - 1, 0, -1):
        path.append(back[c, path[-1]])
    path = path[::-1]
    strong = v[np.arange(len(xs)), path] >= MOUTH_FLOOR
    lo = hi = int(np.argmin(np.abs(xs - m[0])))
    while lo > 0 and strong[max(0, lo - MOUTH_GAP):lo].any():
        lo -= 1
    while hi < len(xs) - 1 and strong[hi + 1:hi + 1 + MOUTH_GAP].any():
        hi += 1
    return q[np.arange(lo, hi + 1), np.array(path[lo:hi + 1])]


# ------------------------------------------------------------------ the plan
def angle_token(degrees: float) -> str:
    return f"a{int(np.clip((degrees + 90) / 180 * ANGLE_BINS, 0, ANGLE_BINS - 1))}"


def point_tokens(xy: np.ndarray) -> list[str]:
    return [t for x, y in np.clip(np.round(xy), 0, GRID - 1).astype(int) for t in (f"x{x}", f"y{y}")]


def plan_tokens(pose: Pose, head: list[np.ndarray], eyes: list[np.ndarray], pupils: list[Pupil | None],
                nose: np.ndarray | None, mouth: np.ndarray) -> list[str]:
    """The plan as words: the pose's three angles in ANGLE_BINS bins over -90..90 degrees; each outline opened by its
    kind and closed by [end], every point as two grid words (x, y on the 256 grid); a pupil as its centre, two radii
    (whole pixels, at most 63) and its angle. Missing parts are left out."""
    words = ["[plan]", "[pose]", angle_token(pose.yaw), angle_token(pose.pitch), angle_token(pose.roll)]
    for run in head:
        words += ["[head]"] + point_tokens(run[:, :2]) + ["[end]"]
    for eye in eyes:
        words += ["[eye]"] + point_tokens(eye) + ["[end]"]
    for p in pupils:
        if p is not None:
            words += (["[pupil]"] + point_tokens(p.centre[None]) + [f"r{int(min(r, 63))}" for r in p.radii[:2]]
                      + [angle_token(np.degrees(p.angle) % 180 - 90)])
    if nose is not None:
        words += ["[nose]"] + point_tokens(nose) + ["[end]"]
    if len(mouth):
        words += ["[mouth]"] + point_tokens(mouth) + ["[end]"]
    return words + ["[/plan]"]


@dataclass
class SketchPlan:
    """One cat's sketch plan on the 256 grid (see the module docstring)."""

    pose: Pose
    head: list[np.ndarray]           # runs (n, 3): x, y, in an ear
    eyes: list[np.ndarray]           # (EYE_POINTS, 2) each
    pupils: list[Pupil | None]       # one per eye
    nose: np.ndarray | None          # (NOSE_POINTS, 2)
    mouth: np.ndarray                # (n, 2), possibly empty
    tokens: list[str]


def make_plan(target: np.ndarray, tri: np.ndarray, points: np.ndarray, template: np.ndarray, sam: Sam) -> SketchPlan:
    """The sketch plan of one cat: the cut-out `target` (256, 256, 3) uint8, its tri map and its nine face points."""
    image = target.astype(np.float32) / 255
    points = points.astype(np.float64)
    d = float(np.linalg.norm(points[0] - points[1]))
    lum = luminance(image)
    pose = head_pose(template, points)
    head = head_outline(tri, points)
    eyes = [resample_closed(o, EYE_POINTS) for o in eye_outlines(sam, image, points[:2], d)]
    pupils = [pupil_ellipse(lum, e) for e in eyes]
    nose = nose_outline(sam, image, points, d)
    mouth = mouth_line(lum, points, nose, d)
    return SketchPlan(pose, head, eyes, pupils, nose, mouth, plan_tokens(pose, head, eyes, pupils, nose, mouth))


def save_plan(path: Path | str, plan: SketchPlan) -> None:
    """The plan as a compressed .npz: yaw, pitch, roll, residual, R; head (an object array of runs); eyes (k, m, 2);
    pupils (k, 6) as centre, radii, angle, share (NaN for an eye without a pupil); nose (m, 2) (empty when missing);
    mouth (m, 2); tokens; version. The file is written next to `path` first and then renamed into place."""
    path = Path(path)
    pose = plan.pose
    pupils = [p.as_row() if p is not None else np.full(6, np.nan) for p in plan.pupils]
    head = np.empty(len(plan.head), dtype=object)                      # 1-D even when the runs have equal lengths
    for i, run in enumerate(plan.head):
        head[i] = run
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, yaw=pose.yaw, pitch=pose.pitch, roll=pose.roll, residual=pose.residual, R=pose.R,
                        head=head, eyes=np.array(plan.eyes), pupils=np.array(pupils),
                        nose=plan.nose if plan.nose is not None else np.zeros((0, 2)), mouth=plan.mouth,
                        tokens=np.array(plan.tokens), version=np.array(PLAN_VERSION))
    tmp.rename(path)
