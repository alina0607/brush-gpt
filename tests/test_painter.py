import dataclasses
import importlib
import json
import os
import sys

import numpy as np
import pytest
import torch
from scipy import ndimage
from synthetic import synthetic_inputs

from brushgpt import painter
from brushgpt.brushes import KINDS, StrokeParams
from brushgpt.cat import load_inputs, one_eye, prepare_cat, without_eyes
from brushgpt.painter import (
    EYE_STAGE,
    FIXED_PLACES,
    STAGES,
    StageResult,
    fit_stage,
    paint_apart,
    paint_cat,
    paint_eye_versions,
    painting_layers,
    replay,
    stage_arrays,
    stage_layers,
)

REAL_CATS = ("afhq__flickr_cat_000014", "cat__CAT_DATASET_01__CAT_00__00000001_017")
FIT_LAYERS = [
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 0),
    (4, 0),
    (5, 0),
    (6, 0),
    (6, 1),
    (7, 0),
    (7, 1),
    (7, 2),
    (7, 3),
    (7, 4),
    (7, 5),
    (7, 6),
    (7, 7),
]


def reference(module: str):
    """A module of the research code this package was extracted from (path in $SELL_ANYTHING_ML)."""
    root = os.environ.get("SELL_ANYTHING_ML")
    if not root:
        pytest.skip("SELL_ANYTHING_ML is not set")
    for path in (root, os.path.join(root, "tools")):
        if path not in sys.path:
            sys.path.append(path)
    return importlib.import_module(module)


def strokes(n: int, x: float, seed: int = 0) -> StrokeParams:
    g = torch.Generator().manual_seed(seed)
    xy = torch.stack([torch.full((n,), x), 100 + 300 * torch.rand(n, generator=g)], -1)
    return StrokeParams(
        xy, torch.full((n, 2), 10.0), torch.rand(n, generator=g), torch.rand(n, 3, generator=g), torch.full((n,), 0.9)
    )


@pytest.fixture(scope="module")
def synthetic():
    inputs = synthetic_inputs()
    return inputs, prepare_cat(inputs)


# ------------------------------------------------------------------ unit tests
def test_the_stage_table_has_seven_stages_of_known_brushes():
    assert sorted(STAGES) == list(range(1, 8)) and sum(len(v) for v in STAGES.values()) == 21
    for layers in STAGES.values():
        assert len({st.name for st in layers}) == len(layers)
        for st in layers:
            assert st.kind in KINDS
            assert st.n > 0 or st.place in FIXED_PLACES


def test_stage_two_is_saved_by_layer_and_replayed_one_eye_at_a_time():
    points = np.array([[150.0, 200.0], [350.0, 200.0]])
    iris, lids = STAGES[EYE_STAGE][1], STAGES[EYE_STAGE][4]
    layers = [
        (EYE_STAGE, 1, strokes(3, 150), iris),
        (EYE_STAGE, 4, strokes(2, 140, 1), lids),
        (EYE_STAGE, 1, strokes(4, 360, 2), iris),
    ]
    canvas = torch.rand(512, 512, 3)
    arrays = stage_arrays(StageResult(EYE_STAGE, canvas, canvas, layers, 0.0), points)
    assert arrays["canvas"].dtype == np.float32 and arrays["1_xy"].shape == (7, 2) and arrays["4_xy"].shape == (2, 2)
    assert arrays["1_side"].tolist() == [0, 0, 0, 1, 1, 1, 1] and arrays["1_side"].dtype == np.int8
    assert str(arrays["1_kind"]) == "wet" and str(arrays["4_kind"]) == "liner" and "0_xy" not in arrays
    order = [(j, len(p.xy)) for _, j, p in stage_layers(arrays, EYE_STAGE)]
    assert order == [(1, 3), (4, 2), (1, 4)]  # the left eye's layers, then the right's


def test_fitting_a_layer_brings_the_canvas_closer_to_the_photo(synthetic):
    _, cat = synthetic
    st = STAGES[6][0]  # the nose
    white = torch.ones(512, 512, 3)
    p, out, st2 = fit_stage(st, cat, white, np.random.default_rng(0), iters=30)
    nose = torch.from_numpy(cat["nose"])
    before = (white - cat["target"]).abs()[nose].mean()
    assert (out - cat["target"]).abs()[nose].mean() < 0.7 * before
    assert len(p.xy) == st2.n == st.n and all(torch.isfinite(t).all() for t in p)
    far = torch.from_numpy(~ndimage.binary_dilation(cat["nose"], iterations=10))
    assert float((out - white).abs()[far].mean()) < 0.01  # outside its part a layer changes little


def test_a_painting_is_saved_in_seven_stages_and_replays_to_its_canvas(synthetic, tmp_path, monkeypatch):
    inputs, _ = synthetic
    monkeypatch.setattr(painter, "ITERS", 2)
    canvas = paint_cat(inputs, tmp_path)
    for k in range(1, 8):
        z = np.load(tmp_path / f"stage{k}.npz")
        assert z["canvas"].shape == (512, 512, 3) and z["canvas"].dtype == np.float32
        for key in z.files:
            assert key == "canvas" or key.split("_", 1)[1] in (*StrokeParams._fields, "kind", "side"), key
    assert np.array_equal(np.load(tmp_path / "stage7.npz")["canvas"], canvas.numpy())
    rows = [json.loads(r) for r in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in rows] == list(range(1, 8)) and rows[-1]["l1_cat_after"] < rows[0]["l1_cat_before"]
    assert rows[EYE_STAGE - 1]["eye_order_max_diff"] < 1e-5
    assert float((replay(painting_layers(tmp_path)) - canvas).abs().max()) < 1e-5


def test_eye_versions_repaint_only_the_eyes(synthetic, monkeypatch):
    inputs, cat = synthetic
    monkeypatch.setattr(painter, "ITERS", 1)
    blue = {
        **cat,
        "target": torch.where(torch.from_numpy(cat["eye"])[..., None], torch.tensor([0.3, 0.45, 0.85]), cat["target"]),
    }
    whole, layers = paint_apart(cat, inputs)
    (first, first_layers), (other, other_layers) = paint_eye_versions([cat, blue], inputs)
    assert torch.equal(first, whole)
    assert [(k, j) for k, j, _, _ in first_layers] == [(k, j) for r in layers for k, j, _, _ in r.layers]
    for (k, j, p, _), (k2, j2, q, _) in zip(first_layers, other_layers, strict=True):
        assert (k, j) == (k2, j2)
        if k != EYE_STAGE:
            assert all(torch.equal(a, b) for a, b in zip(p, q, strict=True))
    assert not torch.equal(first, other)


# ------------------------------------------------------------------ equivalence with the reference
def real_cat(name: str):
    PS, J, H = reference("paint_stages"), reference("fur_judge"), reference("head_line")
    try:
        cutout = J.repaint_file(name)
    except FileNotFoundError:
        pytest.skip(f"no cut-out for {name}")
    if not (PS.OUT / name / "stage7.npz").exists():
        pytest.skip(f"{name} has no reference painting")
    photo = H.photo_path(json.loads(str(np.load(cutout, allow_pickle=True)["meta"])))
    inputs = load_inputs(cutout, photo, next(PS.PLAN.glob(f"*/shard_*/{name}.npz")))
    return PS, inputs


@pytest.fixture(scope="module", params=REAL_CATS)
def real(request):
    PS, inputs = real_cat(request.param)
    ref, mine = PS.load_cat(request.param), prepare_cat(inputs)
    return request.param, PS, inputs, ref, mine, PS.without_eyes(request.param, ref), without_eyes(mine, inputs)


def saved_canvas(PS, name: str, stage: int) -> torch.Tensor:
    if stage == 1:
        return torch.ones(512, 512, 3)
    return torch.from_numpy(np.load(PS.OUT / name / f"stage{stage - 1}.npz")["canvas"])


def assert_same_strokes(a, b) -> None:
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("stage,j", FIT_LAYERS)
def test_fitting_a_layer_matches_the_reference(real, stage, j, monkeypatch):
    name, PS, _, ref, mine, ref_seen, mine_seen = real
    monkeypatch.setattr(PS, "ITERS", 3)
    monkeypatch.setattr(painter, "ITERS", 3)
    if stage == EYE_STAGE:
        ref_cat, my_cat = PS.one_eye(ref, 1), one_eye(mine, 1)
    else:
        ref_cat, my_cat = ref_seen, mine_seen
    start = saved_canvas(PS, name, stage)
    r = PS.fit_stage(PS.STAGES[stage][j], ref_cat, start, np.random.default_rng(stage))
    m = fit_stage(STAGES[stage][j], my_cat, start, np.random.default_rng(stage))
    assert (r is None) == (m is None)
    if r is not None:
        assert_same_strokes(r[0], m[0])
        assert torch.equal(r[1], m[1]) and r[2].n == m[2].n


@pytest.mark.parametrize("stage", [EYE_STAGE, 7])
def test_a_whole_stage_and_its_saved_arrays_match_the_reference(real, stage, monkeypatch):
    name, PS, _, ref, mine, ref_seen, mine_seen = real
    monkeypatch.setattr(PS, "ITERS", 2)
    monkeypatch.setattr(painter, "ITERS", 2)
    start = saved_canvas(PS, name, stage)
    plain = start * 0.97
    torch.manual_seed(stage)
    r_canvas, r_plain, _, r_saved = PS.fit_apart(stage, ref, ref_seen, start, plain)
    layers: list = []
    torch.manual_seed(stage)
    if stage == EYE_STAGE:
        m_canvas = painter.paint_eyes_apart(mine, start, layers)
        m_plain = plain
    else:
        m_canvas, m_plain = painter.paint_without_eyes(stage, mine_seen, start, plain, layers)
    assert torch.equal(r_canvas, m_canvas) and torch.equal(r_plain, m_plain)
    m_saved = stage_arrays(StageResult(stage, start, m_canvas, layers, 0.0), mine["points"])
    assert set(m_saved) == {"canvas", *r_saved}
    for key, v in r_saved.items():
        assert v.dtype == m_saved[key].dtype and np.array_equal(v, m_saved[key]), key


def test_replaying_a_saved_painting_matches_the_reference(real):
    name, PS = real[0], real[1]
    EA = reference("edit_arch_data")
    ref = EA.render([(st, j, p) for st, j, p, _ in EA.version_layers(name, "A", None)])
    mine = replay(painting_layers(PS.OUT / name))
    assert torch.equal(ref, mine)


def test_seven_stages_match_the_reference_at_few_iterations(real, monkeypatch):
    """The whole --apart painting, as the reference's command line runs it, with 1 step a layer."""
    name, PS, inputs, ref, mine, ref_seen, _ = real
    monkeypatch.setattr(PS, "ITERS", 1)
    monkeypatch.setattr(painter, "ITERS", 1)
    canvas = plain = torch.ones(512, 512, 3)
    ref_saved = []
    for stage in sorted(PS.STAGES):
        torch.manual_seed(stage)
        canvas, plain, _, saved = PS.fit_apart(stage, ref, ref_seen, canvas, plain)
        ref_saved.append({"canvas": canvas.numpy(), **saved})
    _, results = paint_apart(mine, inputs)
    for r, res in zip(ref_saved, results, strict=True):
        m = stage_arrays(res, mine["points"])
        assert set(r) == set(m)
        for key in r:
            assert np.array_equal(r[key], m[key]), (res.stage, key)


def test_stage_dataclass_round_trips():
    st = dataclasses.replace(STAGES[7][6], n=3)
    assert st.n == 3 and st.brush.kind == "whisker" and st.brush.length == 140
