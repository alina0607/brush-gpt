import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from painted import MORE_TRAIN_CAT, synthetic_cat
from reference_data import REAL_CATS, cat_source, ml_root, reference, write_cat_table

from brushgpt.brushes import StrokeParams
from brushgpt.dataset import read_duplicate_pairs, read_split
from brushgpt.eye_data import (
    NXY,
    OFFSET,
    PREFIX,
    SP,
    SPAN,
    SPECIALS,
    VOCAB,
    EyeGeometry,
    EyeRow,
    Frame,
    body_tokens,
    build,
    cat_eye_rows,
    duplicates_to_train,
    load_eye_ranges,
    on_plan_grid,
    plan_eyes,
    prefix_tokens,
    reviewed_for_eyes,
    sequences,
    strokes_from_tokens,
    with_eyes,
)
from brushgpt.eye_model import EyeData

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_eye_data.py"
RANGES = {j: np.array([[0.05, 0.05, 0.2], [1.5, 1.0, 1.0]]) for j in range(6)}


def eye_strokes(rng: np.random.Generator, n: int, centre: np.ndarray, r: float) -> StrokeParams:
    return StrokeParams(
        torch.tensor(centre + rng.uniform(-r, r, (n, 2)), dtype=torch.float32),
        torch.tensor(rng.uniform(0.1, 0.9, (n, 2)) * r, dtype=torch.float32),
        torch.tensor(rng.uniform(-3, 3, n), dtype=torch.float32),
        torch.tensor(rng.uniform(0, 1, (n, 3)), dtype=torch.float32),
        torch.tensor(rng.uniform(0.3, 1, n), dtype=torch.float32),
    )


def a_geometry(flip: bool = False) -> EyeGeometry:
    t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    outline = np.c_[200 + 30 * np.cos(t), 240 + 18 * np.sin(t)]
    frame = Frame(outline.mean(0), 30.0, flip)
    return EyeGeometry(frame, outline, np.c_[200 + 4 * np.cos(t[::3][:6]), 240 + 10 * np.sin(t[::3][:6])])


# ------------------------------------------------------------------ unit tests
def test_vocabulary_layout():
    assert PREFIX == 52 and len(SPECIALS) == 38 and VOCAB == 15942
    assert OFFSET["x"] == len(SPECIALS) and OFFSET["ring"] + 12**3 == VOCAB


def test_frame_round_trip_and_mirror():
    f = Frame(np.array([100.0, 50.0]), 20.0)
    xy = np.array([[110.0, 40.0], [90.0, 70.0]])
    assert np.allclose(f.local(xy), [[0.5, -0.5], [-0.5, 1.0]]) and np.allclose(f.canvas(f.local(xy)), xy)
    m = f.mirrored()
    assert np.allclose(m.local(xy), [[-0.5, -0.5], [0.5, 1.0]]) and np.allclose(m.canvas(m.local(xy)), xy)


def test_plan_points_go_on_the_two_pixel_grid():
    assert np.allclose(on_plan_grid(np.array([[10.2, 255.9], [-3.0, 300.0]])), [[21.0, 511.0], [1.0, 511.0]])
    plan = {
        "eyes": np.stack([np.c_[np.full(16, 160.0), np.arange(16.0)], np.c_[np.full(16, 40.0), np.arange(16.0)]]),
        "pupils": np.array([[40.0, 8.0, 2.0, 5.0, 0.0], [np.nan, 0, 0, 0, 0]]),
    }
    left, right = plan_eyes(plan)
    assert left.outline[:, 0].mean() < right.outline[:, 0].mean()
    assert left.pupil is not None and left.pupil.shape == (6, 2) and right.pupil is None


def test_prefix_is_fixed_length_with_or_without_a_pupil():
    g = a_geometry()
    with_pupil = prefix_tokens(0, "green", g.frame, g.outline, g.pupil, 5)
    without = prefix_tokens(1, None, g.frame, g.outline, None, 5)
    assert len(with_pupil) == len(without) == PREFIX
    assert with_pupil[:3] == [SP["BOS"], SP["SIDE_L"], SP["EYE_green"]] and without[2] == SP["NULL"]
    assert with_pupil[38] == SP["PUPIL"] and without[38] == SP["NO_PUPIL"] and with_pupil[-1] == SP["START"]
    assert all(OFFSET["x"] <= t < OFFSET["x"] + NXY for t in with_pupil[6:38:2])


@pytest.mark.parametrize("flip", [False, True])
def test_strokes_survive_the_tokens_up_to_the_bins(flip):
    rng = np.random.default_rng(0)
    g = a_geometry(flip)
    layers = [(0, eye_strokes(rng, 6, g.frame.centre, 25)), (3, eye_strokes(rng, 4, g.frame.centre, 10))]
    body, dropped = body_tokens(layers, g.frame, RANGES)
    assert dropped == 0 and body[0] == SP["LAYER_0"] and body[-1] == SP["EOS"]
    back = strokes_from_tokens(np.array(body), g.frame, RANGES)
    assert [j for j, _ in back] == [0, 3]
    for (_, p), (_, q) in zip(layers, back, strict=True):
        a, b = np.sort(p.xy.numpy(), 0), np.sort(q.xy.numpy(), 0)
        assert np.abs(a - b).max() <= 2 * SPAN * g.frame.r / NXY  # one x / y bin
        assert np.allclose(np.sort(p.alpha.numpy()), np.sort(q.alpha.numpy()), atol=0.1)


def test_strokes_outside_the_frame_are_dropped_and_malformed_tokens_skipped():
    rng = np.random.default_rng(1)
    g = a_geometry()
    p = eye_strokes(rng, 3, g.frame.centre, 10)
    p.xy[0] = torch.tensor([400.0, 400.0])
    body, dropped = body_tokens([(2, p)], g.frame, RANGES)
    assert dropped == 1 and (len(body) - 2) == 2 * 5
    noisy = np.array([SP["START"], OFFSET["x"]] + body[:6] + [SP["PAD"]] + body[6:])
    assert len(strokes_from_tokens(noisy, g.frame, RANGES)[0][1].xy) == 2


def test_mirrored_rows_swap_sides_and_drop_repeats():
    rng = np.random.default_rng(2)
    g = a_geometry()
    layers = [(0, eye_strokes(rng, 5, g.frame.centre, 20))]
    rows = [EyeRow("A", 0, "green", g, 3, layers), EyeRow("eye:right_blue", 0, "green", g, 3, layers)]
    seqs = sequences([rows], RANGES)
    assert [m[1:3] for m in seqs.meta] == [("A", 0), ("A~m", 1)] and seqs.counts["repeated_sequences_dropped"] == 2
    a, b = seqs.tokens
    assert a[1] == SP["SIDE_L"] and b[1] == SP["SIDE_R"] and seqs.meta[1][-1] is True


def test_duplicates_move_held_cats_to_train():
    split = {"a": "held", "b": "train", "c": "held", "d": "held"}
    out, moved = duplicates_to_train(split, [("a", "b"), ("c", "d"), ("x", "a")])
    assert out == {"a": "train", "b": "train", "c": "held", "d": "held"} and moved == 1


def test_with_eyes_replaces_the_eye_stage_in_place():
    p = eye_strokes(np.random.default_rng(3), 2, np.zeros(2), 1)
    painted = [(1, 0, p), (2, 0, p), (2, 1, p), (3, 0, p)]
    assert [x[:2] for x in with_eyes(painted, [(2, 5, p)])] == [(1, 0), (2, 5), (3, 0)]


def test_build_writes_the_format_the_trainer_reads(tmp_path):
    cats = [synthetic_cat(tmp_path, f"cat{i}", i) for i in range(4)]
    stats = build(cats, tmp_path / "eyes", split={"cat3": "held"}, threads=2)
    assert stats["cats"] == 4 and stats["held_cats"] == 1 and stats["left_out"] == {}
    z = np.load(tmp_path / "eyes" / "data.npz")
    assert z["tokens"].dtype == np.int16 and int(z["vocab"]) == VOCAB and int(z["prefix"]) == PREFIX
    assert z["version"].tolist()[:2] == ["A", "A~m"] and z["ranges"].shape == (6, 2, 3)
    assert (z["tokens"] < VOCAB).all() and (z["tokens"][:, 0] == SP["BOS"]).all()
    data = EyeData(tmp_path / "eyes" / "data.npz")
    assert len(data.held) == 2 and len(data.train) + len(data.val) + len(data.held) <= len(z["tokens"])
    assert set(load_eye_ranges(tmp_path / "eyes" / "data.npz")) == set(range(6))


def test_the_script_builds_from_a_cat_table(tmp_path, monkeypatch):
    cats = [synthetic_cat(tmp_path, f"cat{i}", i) for i in range(3)]
    write_cat_table(cats, tmp_path / "cats.csv")
    (tmp_path / "review.csv").write_text("source,id,eyes\nsyn,x,used\n")
    (tmp_path / "split.json").write_text(json.dumps({"split": {"cat1": "held"}}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_eye_data.py",
            "--cats",
            str(tmp_path / "cats.csv"),
            "--paintings",
            str(tmp_path / "paintings"),
            "--split",
            str(tmp_path / "split.json"),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    runpy.run_path(str(SCRIPT), run_name="__main__")
    z = np.load(tmp_path / "out" / "data.npz")
    assert z["cats"].tolist() == ["cat0", "cat1", "cat2"] and z["split"].tolist() == ["train", "held", "train"]
    assert reviewed_for_eyes(tmp_path / "review.csv") == {"syn__x"}


# ------------------------------------------------------------------ equivalence with the reference
@pytest.fixture(scope="module")
def ref():
    Y, R, ED = (reference(m) for m in ("eye_data", "eye_recolour_check", "edit_data"))
    data = ml_root() / "data" / "processed" / "eye_1110" / "data.npz"
    if not data.exists():
        pytest.skip("no reference eye data")
    from brushgpt.attributes import eye_targets

    return Y, ED.EA.EV, eye_targets(R.COAT), np.load(data)


def test_eye_rows_are_identical_to_the_reference_and_its_build(ref):
    Y, ev, targets, z = ref
    Y.init_worker()
    ranges = load_eye_ranges(z)
    want_tokens, want_meta, want_counts = Y.sequences([Y.raw_rows(n) for n in REAL_CATS], ranges)
    got = sequences([cat_eye_rows(cat_source(n), ev, targets) for n in REAL_CATS], ranges)
    assert len(got.tokens) == len(want_tokens) == 40 and got.counts == want_counts
    assert all(np.array_equal(a, b) for a, b in zip(got.tokens, want_tokens, strict=True))
    assert got.meta == want_meta
    assert any(m[-1] for m in got.meta)  # mirrored eyes are among them
    cats = [str(c) for c in z["cats"]]
    for ci, name in enumerate(REAL_CATS):  # the same rows as the final build
        rows = np.flatnonzero(z["cat_id"] == cats.index(name))
        mine = [t for t, m in zip(got.tokens, got.meta, strict=True) if m[0] == ci]
        assert len(rows) == len(mine)
        assert all(np.array_equal(z["tokens"][i][: z["lengths"][i]], t) for i, t in zip(rows, mine, strict=True))


def test_a_small_build_is_identical_to_the_reference(ref, tmp_path):
    Y, ev, targets, _ = ref
    CV = reference("coat_versions")
    names = [*REAL_CATS, MORE_TRAIN_CAT]  # bins from both training cats: every layer
    kept = Y.eyes_kept
    Y.eyes_kept = lambda: set(names)
    try:
        want = Y.build(0, tmp_path / "ref", 1)
    finally:
        Y.eyes_kept = kept
    pairs = ml_root() / "runs" / "afhq_duplicates" / "pairs.json"
    got = build(
        [cat_source(n) for n in names],
        tmp_path / "mine",
        split=read_split(CV.OUT / "plan.json"),
        duplicates=read_duplicate_pairs(pairs),
        eye_versions=ev,
        eye_targets=targets,
    )
    assert got == json.loads(json.dumps(want))
    a, b = np.load(tmp_path / "ref" / "data.npz"), np.load(tmp_path / "mine" / "data.npz")
    assert sorted(a.files) == sorted(b.files)
    for k in a.files:
        assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), k


def test_review_and_duplicate_moves_match_the_reference(ref):
    Y = ref[0]
    if not Y.ASSIGN.exists():
        pytest.skip("no review file")
    assert reviewed_for_eyes(Y.ASSIGN) == Y.eyes_kept()
    split = json.loads((reference("coat_versions").OUT / "plan.json").read_text())["split"]
    pairs = read_duplicate_pairs(ml_root() / "runs" / "afhq_duplicates" / "pairs.json")
    assert duplicates_to_train(split, pairs) == Y.joined_duplicates(split)
