import numpy as np
import pytest
from reference_data import REAL_CATS, cat_source, reference
from scipy import ndimage
from synthetic import synthetic_inputs

from brushgpt.attributes import (
    EYE_COLOURS,
    WRITE_GROW,
    coat_names,
    eye_masks,
    eye_name,
    eye_names,
    eye_targets,
    eye_view,
    painting_attributes,
    read_right,
    real_eye_colours,
    recolour_eyes,
    target_at,
    vivid_half,
)
from brushgpt.cat import load_inputs


@pytest.fixture(scope="module")
def view():
    return eye_view(synthetic_inputs())


def fake_targets() -> dict:
    """Real eyes of each colour (Lab), spread over lightness."""
    rng = np.random.default_rng(0)
    centre = {"blue": (-5, -25), "green": (-20, 15), "yellow": (2, 45), "orange": (20, 50), "copper": (25, 40)}
    return {
        n: np.c_[rng.uniform(20, 80, 60), a + rng.normal(0, 3, 60), b + rng.normal(0, 3, 60)]
        for n, (a, b) in centre.items()
    }


# ------------------------------------------------------------------ unit tests
@pytest.mark.parametrize(
    "lab, name",
    [
        ((50, 0, 0), "unclear"),
        ((15, 20, 30), "unclear"),
        ((50, -20, -20), "blue"),
        ((50, -20, 10), "green"),
        ((60, 0, 30), "yellow"),
        ((60, 20, 30), "orange"),
        ((40, 20, 30), "copper"),
    ],
)
def test_eye_name(lab, name):
    assert eye_name(np.array(lab, float)) == name and name in EYE_COLOURS


def test_eye_view_has_the_masks_the_readers_need(view):
    assert view["photo"].shape == (512, 512, 3) and view["points"].shape == (9, 2)
    left, right = eye_masks(view)
    assert left.any() and right.any() and not (left & right).any() and not (view["iris"] & ~view["eye"]).any()


def test_eye_names_read_each_eye_on_its_side(view):
    img = view["photo"].copy()
    left, right = eye_masks(view)
    img[left] = (0.25, 0.45, 0.85)
    img[right] = (0.45, 0.60, 0.20)
    assert eye_names(view, img) == ["blue", "green"]


def test_recolouring_changes_only_the_asked_eye(view):
    targets = fake_targets()
    left = (np.mgrid[:512, :512][1] + 0.5) < 0.5 * (view["points"][0, 0] + view["points"][1, 0])
    out = recolour_eyes(view["photo"], view, targets, ("blue", None))
    changed = np.abs(out - view["photo"]).max(-1) > 0
    assert changed.any()
    assert not (changed & ~(ndimage.binary_dilation(view["eye"], iterations=WRITE_GROW) & left)).any()
    assert eye_names(view, out)[0] == "blue"


def test_real_eye_targets():
    eyes = fake_targets()["yellow"]
    half = vivid_half(eyes)
    assert 0 < len(half) < len(eyes)
    ab, n = target_at(eyes, 50.0)
    assert n >= 30 and ab.shape == (2,)


def test_coat_names_and_reading_right():
    row = {"base": {"label": "grey", "confident": True}, "pattern": {"label": "tabby", "confident": False}}
    assert coat_names(row) == ["grey", "unclear"]
    assert read_right(["blue", "green", "grey", "tabby"], ["blue", "copper", "grey", "unclear"]) == [
        True,
        False,
        True,
        None,
    ]


def test_painting_attributes_reads_eyes_and_coat(view):
    img = view["photo"].copy()
    left, right = eye_masks(view)
    img[left], img[right] = (0.25, 0.45, 0.85), (0.25, 0.45, 0.85)
    coat = {"base": {"label": "white", "confident": True}, "pattern": {"label": "solid", "confident": True}}
    assert painting_attributes(view, [img], lambda imgs: [coat] * len(imgs)) == [["blue", "blue", "white", "solid"]]


# ------------------------------------------------------------------ equivalence with the reference
@pytest.fixture(scope="module")
def real():
    PS, EA, E, R = (reference(m) for m in ("paint_stages", "edit_arch_data", "eye_edit_delta", "eye_recolour_check"))
    if not R.COAT.exists():
        pytest.skip("no labelled real eyes")
    cats = []
    for name in REAL_CATS[:2]:
        s = cat_source(name)
        cats.append((PS.load_cat(name), eye_view(load_inputs(s.cutout, s.photo, s.plan))))
    return EA, E, R, cats


def test_eye_view_matches_the_reference_cat(real):
    _, _, _, cats = real
    for ref, mine in cats:
        for key in ("photo", "points", "eye", "iris"):
            assert np.array_equal(np.asarray(ref[key]), np.asarray(mine[key])), key


def test_real_eye_targets_match_the_reference(real):
    _, _, R, _ = real
    ref, mine = R.real_eyes(), real_eye_colours(R.COAT)
    assert list(ref) == list(mine) and all(np.array_equal(ref[n], mine[n]) for n in ref)
    targets = eye_targets(R.COAT)
    for n in ref:
        assert np.array_equal(targets[n], R.vivid_half(ref[n]) if n in R.VIVID else ref[n])


def test_eye_conditions_match_the_reference(real):
    EA, E, R, cats = real
    targets = eye_targets(R.COAT)
    for ref, mine in cats:
        assert eye_names(mine, mine["photo"]) == EA.eye_names(ref, ref["photo"])
        for to in ((None, "green"), ("blue", "copper"), ("yellow", None)):
            a = E.recoloured_cat(ref, targets, to)["photo"]
            b = recolour_eyes(mine["photo"], mine, targets, to)
            assert np.array_equal(a, b)
            assert eye_names(mine, b) == EA.eye_names(ref, a)


def test_coat_reader_matches_the_reference(real):
    pytest.importorskip("transformers")
    pytest.importorskip("sklearn")
    from brushgpt.attributes import CoatReader

    CL = reference("coat_label")
    _, _, _, cats = real
    images = [np.asarray(ref["target"], np.float32) for ref, _ in cats] + [cats[0][1]["photo"]]
    try:
        ref_read = CL.reader()
        mine = CoatReader(CL.DIR / "clip.npz", CL.DIR / "colour.npz", CL.DIR / "labels.jsonl")
    except OSError:
        pytest.skip("the CLIP weights are not in the local cache")
    assert mine(images) == ref_read(images)
