import functools
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from matplotlib.path import Path as Polygon
from scipy.spatial.transform import Rotation

from brushgpt.plan import (
    ANGLE_BINS,
    EYE_POINTS,
    HEAD_STEP,
    Pose,
    Pupil,
    SketchPlan,
    fit_head_template,
    head_outline,
    head_pose,
    load_head_template,
    load_sam,
    make_plan,
    mouth_line,
    plan_tokens,
    pupil_ellipse,
    resample_closed,
    save_plan,
)

# real cats: (repaint input shard, name); the last has no nose in its plan
REAL_CATS = ((0, "afhq__flickr_cat_000014"), (11, "cat__CAT_DATASET_01__CAT_00__00000001_017"),
             (10, "afhq__pixabay_cat_004772"), (11, "cat__CAT_DATASET_01__CAT_00__00000002_008"))
TEMPLATE_SHARDS = range(12)          # the 6,000 cats the shipped template was fitted to


def reference(module: str):
    """A module of the research code this package was extracted from (path in $SELL_ANYTHING_ML)."""
    root = os.environ.get("SELL_ANYTHING_ML")
    if not root:
        pytest.skip("SELL_ANYTHING_ML is not set")
    for path in (root, os.path.join(root, "tools")):
        if path not in sys.path:
            sys.path.append(path)
    return importlib.import_module(module)


@functools.cache
def sam():
    try:
        return load_sam(local_only=True)
    except (OSError, ImportError) as e:
        pytest.skip(f"SAM is not in the local Hugging Face cache: {e}")


def ellipse(cx: float, cy: float, rx: float, ry: float, n: int = 16) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[cx + rx * np.cos(t), cy + ry * np.sin(t)]


POINTS = np.array([[95, 120], [161, 120], [128, 182], [50, 95], [62, 28], [100, 72],
                   [156, 72], [194, 28], [206, 95]], float)


# ------------------------------------------------------------------ unit tests
def test_shipped_template_is_in_the_face_frame():
    t = load_head_template()
    assert t.shape == (3, 9)
    assert np.allclose(t[:, 0], (-0.5, 0, 0), atol=1e-6) and np.allclose(t[:, 1], (0.5, 0, 0), atol=1e-6)
    assert t[1, 2] > 0.5                                         # the mouth below the eyes
    assert t[1, [4, 7]].max() < -1.0                             # the ear tips above them
    assert t[2, [3, 4, 5, 6, 7, 8]].mean() > t[2, 2]             # the ears behind the mouth


def random_views(template: np.ndarray, n: int, seed: int = 0) -> tuple[np.ndarray, list]:
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-30, 30, (n, 3))                        # roll, pitch, yaw
    views = []
    for a in angles:
        R = Rotation.from_euler("ZXY", a, degrees=True).as_matrix()
        views.append((rng.uniform(40, 80) * R[:2] @ template).T + rng.uniform(80, 170, 2))
    return np.array(views), angles


def test_factorisation_recovers_a_known_head_from_its_views():
    truth = load_head_template()
    views, _ = random_views(truth, 200)
    assert np.allclose(fit_head_template(views), truth, atol=1e-6)


def test_pose_recovers_the_angles_and_scale_of_a_projected_head():
    template = load_head_template().astype(np.float64)
    R = Rotation.from_euler("ZXY", (10.0, -15.0, 25.0), degrees=True).as_matrix()
    points = (60.0 * R[:2] @ template).T + (128, 120)
    pose = head_pose(template, points)
    assert (pose.roll, pose.pitch, pose.yaw) == pytest.approx((10.0, -15.0, 25.0), abs=1e-6)
    assert pose.scale == pytest.approx(60.0) and pose.residual < 1e-9
    assert np.allclose(pose.R, R)


def test_resample_closed_spaces_points_evenly():
    square = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    q = resample_closed(square, 8)
    assert len(q) == 8
    assert np.allclose(np.linalg.norm(np.diff(np.r_[q, q[:1]], axis=0), axis=1), 2.0)


def test_head_outline_skips_the_uncertain_chin_and_flags_the_ears():
    yy, xx = np.mgrid[:256, :256] + 0.5
    q = np.c_[xx.ravel(), yy.ravel()]
    cat = ((xx - 128) / 80) ** 2 + ((yy - 140) / 70) ** 2 <= 1
    for ear in ((3, 4, 5), (6, 7, 8)):
        cat |= Polygon(POINTS[list(ear)]).contains_points(q).reshape(256, 256)
    tri = cat.astype(np.uint8)
    tri[(yy > 180) & cat] = 2                                    # the chin: uncertain
    tri[(yy > 180) & ~cat & (np.abs(xx - 128) < 90)] = 2
    runs = head_outline(tri, POINTS)
    assert runs
    xy = np.concatenate([r[:, :2] for r in runs])
    assert xy[:, 1].max() < 186                                  # no line along the uncertain region
    for r in runs:
        chords = np.linalg.norm(np.diff(r[:, :2], axis=0), axis=1)    # HEAD_STEP apart along the pixel contour
        assert np.all((chords > 0.5 * HEAD_STEP) & (chords <= HEAD_STEP + 1e-9))   # corners cut the chord
        assert set(np.unique(r[:, 2])) <= {0.0, 1.0}
    flagged = np.concatenate(runs)
    assert flagged[flagged[:, 1] < 40, 2].all()                  # the ear tips
    assert not flagged[flagged[:, 1] > 150, 2].any()             # the cheeks


def test_pupil_is_the_dark_slit_inside_the_eye():
    yy, xx = np.mgrid[:256, :256] + 0.5
    lum = np.full((256, 256), 0.6, np.float32)
    slit = ((xx - 100) / 3) ** 2 + ((yy - 120) / 8) ** 2 <= 1
    lum[slit] = 0.05
    p = pupil_ellipse(lum, ellipse(100, 120, 18, 12))
    assert p is not None
    assert np.allclose(p.centre, (100, 120), atol=0.1)
    assert p.radii[0] > 2 * p.radii[1]                           # tall and thin
    assert abs(abs(p.angle) - np.pi / 2) < 0.1                   # the long axis is vertical
    assert pupil_ellipse(np.full((256, 256), 0.6, np.float32), ellipse(100, 120, 2, 2)) is None   # too small an eye


def test_mouth_follows_the_dark_line_below_the_nose():
    lum = np.full((256, 256), 0.7, np.float32)
    lum[190:192, 100:157] = 0.1                                  # a dark line just below the mouth point
    nose = ellipse(128, 165, 10, 6, 12)
    mouth = mouth_line(lum, POINTS, nose, float(np.linalg.norm(POINTS[0] - POINTS[1])))
    assert len(mouth) > 5
    assert np.all(np.abs(mouth[:, 1] - 191) <= 1.5)
    assert mouth[:, 0].min() < 115 and mouth[:, 0].max() > 140


def small_plan() -> SketchPlan:
    pose = Pose(0.0, -90.0, 95.0, 50.0, 0.01, np.eye(3))
    head = [np.c_[ellipse(128, 128, 80, 80, 6), np.r_[1, 1, 0, 0, 0, 0]], np.array([[10.4, 300.0, 0.0]] * 6)]
    eyes = [ellipse(95, 120, 15, 10, EYE_POINTS), ellipse(161, 120, 15, 10, EYE_POINTS)]
    pupils = [Pupil(np.array([95.5, 120.5]), np.array([80.0, 2.6]), np.pi / 2, 0.2), None]
    return SketchPlan(pose, head, eyes, pupils, None, np.array([[110.0, 190.0], [146.0, 190.0]]), [])


def test_plan_tokens_layout():
    p = small_plan()
    words = plan_tokens(p.pose, p.head, p.eyes, p.pupils, p.nose, p.mouth)
    assert words[:5] == ["[plan]", "[pose]", f"a{ANGLE_BINS // 2}", "a0", f"a{ANGLE_BINS - 1}"]
    assert words[-1] == "[/plan]" and "[nose]" not in words
    assert words.count("[head]") == 2 and words.count("[eye]") == 2 and words.count("[pupil]") == 1
    head2 = words[words.index("[head]", 3 + 2 * 6 + 2):][:3]
    assert head2 == ["[head]", "x10", "y255"]                    # rounded and clipped to the grid
    i = words.index("[pupil]")
    assert words[i:i + 6] == ["[pupil]", "x96", "y120", "r63", "r2", "a32"]   # radii capped at 63
    assert words[words.index("[mouth]"):][:6] == ["[mouth]", "x110", "y190", "x146", "y190", "[end]"]


def test_saved_plan_has_the_painters_keys(tmp_path):
    p = small_plan()
    p.head = [np.zeros((4, 3)), np.ones((4, 3))]                 # equal lengths: must stay a 1-D list of runs
    p.tokens = ["[plan]", "[/plan]"]
    save_plan(tmp_path / "cat.npz", p)
    z = np.load(tmp_path / "cat.npz", allow_pickle=True)
    assert set(z.files) == {"yaw", "pitch", "roll", "residual", "R", "head", "eyes", "pupils", "nose", "mouth",
                            "tokens", "version"}
    assert z["head"].shape == (2,) and z["head"].dtype == object and z["head"][1].shape == (4, 3)
    assert z["eyes"].shape == (2, EYE_POINTS, 2) and z["nose"].shape == (0, 2)
    assert np.allclose(z["pupils"][0], (95.5, 120.5, 80.0, 2.6, np.pi / 2, 0.2)) and np.isnan(z["pupils"][1]).all()
    assert not list(tmp_path.glob("*.tmp.npz"))


def test_plan_of_a_synthetic_cat_finds_its_eyes():
    from synthetic import synthetic_inputs
    inputs = synthetic_inputs()
    plan = make_plan(inputs.target, inputs.tri, inputs.points, load_head_template(), sam())
    assert len(plan.eyes) == 2 and all(e.shape == (EYE_POINTS, 2) for e in plan.eyes)
    for eye, point in zip(plan.eyes, inputs.points[:2], strict=True):
        assert np.linalg.norm(eye.mean(0) - point) < 6
    assert all(p is not None for p in plan.pupils)
    assert plan.tokens[0] == "[plan]" and plan.tokens[-1] == "[/plan]"


# ------------------------------------------------------------------ equivalence with the reference
def ml_root() -> Path:
    reference("sketch_plan")
    return Path(os.environ["SELL_ANYTHING_ML"])


def input_shard(shard: int) -> Path:
    f = ml_root() / "data" / "processed" / "repaint" / "inputs" / f"shard_{shard:03d}.npz"
    if not f.exists():
        pytest.skip(f"no repaint input {f.name}")
    return f


@functools.cache
def real_cat(shard: int, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A cat's target, tri map and face points from the repaint inputs, as the reference's batch reads them."""
    z = np.load(input_shard(shard))
    names = [f"{s}__{i}" for s, i in zip(z["source"], z["id"], strict=True)]
    k = names.index(name)
    return z["target"][k], z["tri"][k], z["points"][k]


def stored_plan(shard: int, name: str):
    files = sorted((ml_root() / "data" / "processed" / "sketch_plan").glob(f"*/shard_{shard:03d}/{name}.npz"))
    if not files:
        pytest.skip(f"no stored plan for {name}")
    return np.load(files[0], allow_pickle=True)


def test_refitting_the_template_on_the_reference_input_gives_the_shipped_one():
    SP = reference("sketch_plan")
    points = np.concatenate([np.load(input_shard(k))["points"] for k in TEMPLATE_SHARDS])
    assert points.shape == (6000, 9, 2)
    mine = fit_head_template(points)
    assert np.array_equal(mine, SP.template(points))
    assert np.array_equal(mine, load_head_template())


@pytest.mark.parametrize(("shard", "name"), REAL_CATS)
def test_sam_free_parts_match_the_reference(shard, name):
    SP, E = reference("sketch_plan"), reference("eye_crop_styles")
    target, tri, points = real_cat(shard, name)
    points = points.astype(np.float64)
    template = load_head_template()
    d = float(np.linalg.norm(points[0] - points[1]))
    lum = E.luminance(target.astype(np.float32) / 255)
    ref_pose, pose = SP.pose(template, points), head_pose(template, points)
    for key in ("yaw", "pitch", "roll", "scale", "residual"):
        assert ref_pose[key] == getattr(pose, key), key
    assert np.array_equal(ref_pose["R"], pose.R)
    ref_head, head = SP.head_outline(tri, points), head_outline(tri, points)
    assert len(ref_head) == len(head) and all(np.array_equal(a, b) for a, b in zip(ref_head, head, strict=True))
    # pupils and the mouth from given outlines: an ellipse around each eye point and a nose above the mouth point
    eyes = [resample_closed(ellipse(*e, 0.2 * d, 0.12 * d, 60), EYE_POINTS) for e in points[:2]]
    for eye in eyes:
        r, m = SP.pupil(lum, eye), pupil_ellipse(lum, eye)
        assert (r is None) == (m is None)
        if m is not None:
            assert np.array_equal(np.r_[r["centre"], r["radii"], r["angle"], r["share"]], m.as_row())
    nose = ellipse(*(points[:2].mean(0) + 0.85 * (points[2] - points[:2].mean(0))), 0.1 * d, 0.06 * d, 12)
    for n in (nose, None):
        assert np.array_equal(SP.mouth_line(lum, points, n, d), mouth_line(lum, points, n, d))
    pupils = [pupil_ellipse(lum, e) for e in eyes]
    ref_pupils = [SP.pupil(lum, e) for e in eyes]
    mouth = mouth_line(lum, points, nose, d)
    assert plan_tokens(pose, head, eyes, pupils, nose, mouth) == SP.plan_tokens(ref_pose, ref_head, eyes, ref_pupils,
                                                                                  nose, mouth)


@functools.cache
def both_plans(shard: int, name: str):
    import torch
    SP = reference("sketch_plan")
    torch.set_num_threads(2)
    target, tri, points = real_cat(shard, name)
    template = load_head_template()
    model = sam()
    SP.EL._SAM.update(model=model.model, proc=model.processor)  # the reference's loader, already filled
    ref = SP.plan_one(template, target, tri, points.astype(np.float64))
    return ref, make_plan(target, tri, points, template, model)


@pytest.mark.parametrize(("shard", "name"), REAL_CATS)
def test_plan_matches_the_reference(shard, name):
    ref, mine = both_plans(shard, name)
    assert ref["pose"]["yaw"] == mine.pose.yaw and np.array_equal(ref["pose"]["R"], mine.pose.R)
    assert len(ref["head"]) == len(mine.head)
    assert all(np.array_equal(a, b) for a, b in zip(ref["head"], mine.head, strict=True))
    assert all(np.array_equal(a, b) for a, b in zip(ref["eyes"], mine.eyes, strict=True))
    for r, m in zip(ref["pupils"], mine.pupils, strict=True):
        assert (r is None) == (m is None)
        if m is not None:
            assert np.array_equal(np.r_[r["centre"], r["radii"], r["angle"], r["share"]], m.as_row())
    assert (ref["nose"] is None) == (mine.nose is None)
    if mine.nose is not None:
        assert np.array_equal(ref["nose"], mine.nose)
    assert np.array_equal(ref["mouth"], mine.mouth)
    assert ref["tokens"] == mine.tokens


@pytest.mark.parametrize(("shard", "name"), REAL_CATS[:3])
def test_saved_plan_matches_the_stored_reference_file(shard, name, tmp_path):
    stored = stored_plan(shard, name)
    _, mine = both_plans(shard, name)
    save_plan(tmp_path / f"{name}.npz", mine)
    z = np.load(tmp_path / f"{name}.npz", allow_pickle=True)
    assert set(z.files) == set(stored.files)
    for key in z.files:
        if key == "version":
            continue
        if key == "head":
            assert z[key].shape == stored[key].shape
            assert all(np.array_equal(a, b) for a, b in zip(z[key], stored[key], strict=True))
        else:
            assert z[key].dtype == stored[key].dtype, key
            assert np.array_equal(z[key], stored[key], equal_nan=z[key].dtype.kind == "f"), key
