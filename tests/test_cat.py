import dataclasses
import importlib
import json
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image
from synthetic import synthetic_inputs

from brushgpt.cat import (
    SIDE,
    eye_side,
    face_frame,
    fill_eyes,
    load_inputs,
    one_eye,
    prepare_cat,
    without_eyes,
)

REAL_CATS = ("afhq__flickr_cat_000014", "cat__CAT_DATASET_01__CAT_00__00000001_017")


def reference(module: str):
    """A module of the research code this package was extracted from (path in $SELL_ANYTHING_ML)."""
    root = os.environ.get("SELL_ANYTHING_ML")
    if not root:
        pytest.skip("SELL_ANYTHING_ML is not set")
    for path in (root, os.path.join(root, "tools")):
        if path not in sys.path:
            sys.path.append(path)
    return importlib.import_module(module)


def same(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, list):
        return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    return np.asarray(a).dtype == np.asarray(b).dtype and np.array_equal(a, b)


@pytest.fixture(scope="module")
def synthetic():
    inputs = synthetic_inputs()
    return inputs, prepare_cat(inputs)


# ------------------------------------------------------------------ unit tests
def test_face_frame_puts_the_eyes_on_the_x_axis_and_the_mouth_below():
    p = np.array([[10.0, 20.0], [30.0, 20.0], [20.0, 40.0]])
    d, mid, ex, ey, P = face_frame(p)
    assert d == 20 and np.allclose(mid, (20, 20)) and np.allclose(P[:2], [[-0.5, 0], [0.5, 0]])
    assert P[2, 1] == pytest.approx(1.0)


def test_eye_side_splits_by_the_nearer_eye():
    points = np.array([[300.0, 200.0], [200.0, 200.0]])          # listed right eye first
    xy = np.array([[190.0, 210.0], [310.0, 190.0], [249.0, 0.0]])
    assert eye_side(xy, points).tolist() == [0, 1, 0]


def test_fill_eyes_reads_nothing_inside_the_zone():
    rng = np.random.default_rng(0)
    img = rng.random((40, 40, 3)).astype(np.float32)
    zone = np.zeros((40, 40), bool)
    zone[10:20, 10:20] = True
    other = img.copy()
    other[zone] = 0.0
    a, b = fill_eyes(img, zone, ~zone), fill_eyes(other, zone, ~zone)
    assert np.array_equal(a, b) and np.array_equal(a[~zone], img[~zone])


def test_prepared_cat_has_the_painters_inputs(synthetic):
    _, cat = synthetic
    assert cat["target"].shape == (SIDE, SIDE, 3) and cat["target"].dtype == torch.float32
    assert cat["photo"].shape == (SIDE, SIDE, 3) and cat["tri"].shape == (SIDE, SIDE)
    assert cat["field"].shape == (SIDE, SIDE, 2)
    norm = torch.linalg.norm(cat["field"], dim=-1)
    assert float(norm.median()) == pytest.approx(1.0, abs=0.05)
    head = cat["tri"] == 1
    assert 0.2 < head.mean() < 0.7
    assert torch.all(cat["target"][torch.from_numpy(~head & (cat["tri"] == 0))] == 1.0)
    for part in ("eye", "eye_rim", "pupil", "nose", "mouth", "whisker_pad", "dot_pad", "iris", "ear_inner"):
        assert cat[part].shape == (SIDE, SIDE) and cat[part].any(), part
    assert not (cat["pupil"] & ~cat["eye"]).any()
    assert len(cat["eye_outlines"]) == 2 and len(cat["pupils"]) == 2
    assert np.allclose(cat["points"], synthetic[0].points * 2)


def test_one_eye_keeps_only_that_eyes_parts(synthetic):
    _, cat = synthetic
    left = one_eye(cat, 0)
    ys, xs = np.nonzero(left["eye"])
    assert xs.max() < SIDE / 2 and len(left["eye_outlines"]) == 1 and len(left["pupils"]) == 1


def test_changing_an_eye_changes_nothing_the_other_stages_see(synthetic):
    inputs, cat = synthetic
    blue_eye = (0.3, 0.45, 0.85)
    inside = cat["eye"]                                           # an edit of the eyes' colour, as recolouring does
    edited = {**cat, "target": torch.where(torch.from_numpy(inside)[..., None], torch.tensor(blue_eye), cat["target"]),
              "photo": np.where(inside[..., None], np.float32(blue_eye), cat["photo"]).astype(np.float32)}
    small = np.asarray(Image.fromarray(inside.astype(np.uint8) * 255).resize((256, 256), Image.BOX)) == 255
    edited_inputs = dataclasses.replace(inputs, target=np.where(small[..., None], np.uint8(200), inputs.target))
    seen, seen_edited = without_eyes(cat, inputs), without_eyes(edited, edited_inputs)
    for key in ("target", "photo", "field"):
        assert same(seen[key], seen_edited[key]), key


# ------------------------------------------------------------------ equivalence with the reference
def real_inputs(name: str):
    PS, J, H = reference("paint_stages"), reference("fur_judge"), reference("head_line")
    try:
        cutout = J.repaint_file(name)
    except FileNotFoundError:
        pytest.skip(f"no cut-out for {name}")
    photo = H.photo_path(json.loads(str(np.load(cutout, allow_pickle=True)["meta"])))
    return PS, load_inputs(cutout, photo, next(PS.PLAN.glob(f"*/shard_*/{name}.npz")))


@pytest.mark.parametrize("name", REAL_CATS)
def test_prepared_cat_matches_the_reference(name):
    PS, inputs = real_inputs(name)
    ref, mine = PS.load_cat(name), prepare_cat(inputs)
    assert set(ref) - set(mine) == {"texture", "pads", "ear_roots"}    # used only by layers the painter dropped
    for key in mine:
        assert same(ref[key], mine[key]), key
    ref_seen, mine_seen = PS.without_eyes(name, ref), without_eyes(mine, inputs)
    for key in ("target", "photo", "field"):
        assert same(ref_seen[key], mine_seen[key]), key
    for side in (0, 1):
        r, m = PS.one_eye(ref, side), one_eye(mine, side)
        for key in ("eye", "eye_rim", "pupil", "iris", "eye_outlines", "pupils"):
            assert same(r[key], m[key]), (side, key)
