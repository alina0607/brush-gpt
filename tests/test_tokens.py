import numpy as np
import pytest
import torch
from reference_data import REAL_CATS, cat_source, earlier_ranges, reference

from brushgpt.brushes import StrokeParams
from brushgpt.dataset import face_points, read_plan, version_layers
from brushgpt.painter import EYE_STAGE
from brushgpt.tokens import (
    LAYERS,
    NALPHA,
    NANGLE,
    NRGB,
    NSIZE,
    NXY,
    PALETTE_K,
    PARTS,
    Vocabulary,
    around,
    decode_strokes,
    layers_from_tokens,
    painting_tokens,
    palette_colours,
    part_map,
    plan_tokens,
    ranges_from_array,
    ranges_to_array,
    render,
    serpentine,
)
from brushgpt.train import layout_from_specials

FORMATS = [(False, False, False), (True, False, False), (False, True, False), (True, True, False), (True, True, True)]
POINTS = np.array([[95, 120], [161, 120], [128, 182], [50, 95], [62, 28], [100, 72],
                   [156, 72], [194, 28], [206, 95]], np.float64)


def synthetic_ranges() -> dict:
    return {lj: (np.array([4.0, 2.0, 0.3]), np.array([60.0, 20.0, 1.0])) for lj in LAYERS}


def synthetic_plan() -> dict:
    t = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    eye = lambda cx: np.c_[cx + 15 * np.cos(t), 120 + 9 * np.sin(t)]  # noqa: E731
    head = np.c_[np.linspace(40, 220, 12), np.full(12, 60.0), (np.arange(12) % 6 < 2).astype(float)]
    return {"yaw": np.float64(20.0), "pitch": np.float64(-5.0), "roll": np.float64(3.0), "head": [head],
            "eyes": np.stack([eye(161), eye(95)]),
            "pupils": np.array([[95, 120, 3, 8, 0.1, 0], [161, 120, 3, 8, 0, 0]]),
            "nose": np.c_[128 + 8 * np.cos(t), 160 + 5 * np.sin(t)],
            "mouth": np.c_[np.linspace(110, 146, 9), np.full(9, 186.0)]}


def synthetic_layers(seed: int = 0) -> list:
    """A painting whose strokes sit on bin centres (so encoding and decoding gives them back exactly): every layer,
    stage 2 once per eye, fur strokes in five colours."""
    rng = np.random.default_rng(seed)
    ranges = synthetic_ranges()
    fur = rng.integers(0, NRGB ** 3, 5)
    out = []
    order = [(k, j, None) for k, j in LAYERS if k < EYE_STAGE]
    order += [(k, j, s) for s in (0, 1) for k, j in LAYERS if k == EYE_STAGE]
    order += [(k, j, None) for k, j in LAYERS if k > EYE_STAGE]
    for k, j, side in order:
        n = 6
        x = rng.integers(40, 90, n) + (64 if side == 1 else 0) if side is not None else rng.integers(0, NXY, n)
        colour = rng.integers(0, NRGB ** 3, n) if k == EYE_STAGE else rng.choice(fur, n)
        raw = np.c_[x, rng.integers(0, NXY, n), rng.integers(0, NANGLE, n), rng.integers(0, NSIZE * NSIZE, n),
                    colour * NALPHA + rng.integers(0, NALPHA, n)]
        out.append((k, j, decode_strokes(raw, (k, j), ranges), side))
    return out


def rows_of(p: StrokeParams) -> np.ndarray:
    a = np.concatenate([t.reshape(len(t), -1).numpy() for t in p], 1)
    return a[np.lexsort(a.T[::-1])]


# ------------------------------------------------------------------ unit tests
def test_vocabulary_appends_optional_tokens_after_the_shared_ones():
    base, full = Vocabulary.build(), Vocabulary.build(True, True, True)
    assert len(base.specials) == 49 and len(full.specials) == 84
    assert full.specials[:49] == base.specials
    assert base.field_names == ["x", "y", "angle", "size", "paint"] and base.size == 49 + 256 + 256 + 32 + 36 + 13824
    assert full.field_names[-1] == "colour" and full.field_sizes["paint"] == (PALETTE_K + NRGB ** 3) * NALPHA
    assert full.size == 16344 and full.palette_k.tolist() == [16, 0] and base.palette_k.tolist() == [0, 0]
    assert full.condition(["blue", "copper", "white", "tabby"]) == base.condition(["blue", "copper", "white", "tabby"])
    assert full.condition_names(full.condition(["blue", "unclear", "white", "tabby"])) == ["blue", "unclear", "white",
                                                                                         "tabby"]


def test_layout_from_the_vocabulary_matches_the_token_format():
    v = Vocabulary.build(True, True, True)
    layout = layout_from_specials(list(v.specials), v.field_names, [v.field_sizes[f] for f in v.field_names],
                                  tuple(v.palette_k))
    names = [v.specials[i] for i in layout.slots]
    assert names[-1] == "EOS" and "PAL_ALL" in names and "EYE_L" in names and "PART_EAR_L" in names
    assert all(not n.startswith(("COND_", "TASK_")) for n in names)
    assert [v.specials[i] for i in layout.plan_markers] == [f"PLAN_{r}" for r in
                                                            ("HEAD", "EAR", "EYE", "PUPIL", "NOSE", "MOUTH")]
    assert layout.palette == ((v.ids["PAL_ALL"], PALETTE_K),)
    assert layout.field_offsets == tuple(v.offsets.values())
    assert layout.field_offsets[-1] + layout.field_sizes[-1] == v.size


@pytest.mark.parametrize("fmt", FORMATS)
def test_encoding_and_decoding_gives_every_stroke_back(fmt):
    part_order, plan, palette = fmt
    vocab, ranges, layers = Vocabulary.build(*fmt), synthetic_ranges(), synthetic_layers()
    t, f = painting_tokens(layers, ranges, vocab, part_map(POINTS) if part_order else None,
                           synthetic_plan() if plan else None)
    assert len(t) == len(f) and t.dtype == np.int32 and f.dtype == np.int8
    assert (f == 5).sum() == sum(len(p.xy) for *_, p, _ in layers) and (f == 6).sum() == PALETTE_K * palette
    back = layers_from_tokens(t, ranges, vocab)
    assert [(k, j) for k, j, _ in back] == [(k, j) for k, j, _, _ in layers]
    for (_, _, p, _), (_, _, q) in zip(layers, back, strict=True):
        if part_order:
            assert np.allclose(rows_of(p), rows_of(q), atol=1e-5)
        else:
            assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(p, q, strict=True))


def test_part_order_writes_parts_in_order_and_the_eyes_in_their_parts():
    vocab = Vocabulary.build(part_order=True)
    t, f = painting_tokens(synthetic_layers(), synthetic_ranges(), vocab, part_map(POINTS))
    parts = [vocab.specials[x] for x in t if x < len(vocab.specials) and vocab.specials[x].startswith("PART_")]
    order = [f"PART_{p.replace(' ', '_').upper()}" for p in PARTS]
    assert parts == sorted(parts, key=order.index) and len(set(parts)) == len(parts)
    eye_layers = {vocab.ids[f"LAYER{i}"] for i, (k, _) in enumerate(LAYERS) if k == EYE_STAGE}
    part = None
    for x in t:
        name = vocab.specials[x] if x < len(vocab.specials) else ""
        part = name if name.startswith("PART_") else part
        if x in eye_layers:
            assert part in ("PART_EYE_L", "PART_EYE_R")


def test_a_broken_sequence_decodes_what_it_can():
    vocab, ranges = Vocabulary.build(True, True, True), synthetic_ranges()
    t, _ = painting_tokens(synthetic_layers(), ranges, vocab, part_map(POINTS), synthetic_plan())
    whole = sum(len(p.xy) for *_, p in layers_from_tokens(t, ranges, vocab))
    cut = layers_from_tokens(t[:-3], ranges, vocab)                       # the last stroke cut short
    assert sum(len(p.xy) for *_, p in cut) == whole - 1
    eos = np.r_[t[:len(t) // 2], vocab.ids["EOS"], t[len(t) // 2:]]      # everything after EOS is ignored
    assert sum(len(p.xy) for *_, p in layers_from_tokens(eos, ranges, vocab)) < whole
    stray = np.array([vocab.offsets[k] for k in ("x", "y", "angle", "size", "paint")])
    assert layers_from_tokens(stray, ranges, vocab) == []                   # a stroke outside any layer


def test_plan_tokens_of_a_mirrored_painting_are_flipped():
    vocab = Vocabulary.build(plan=True)
    t, f = plan_tokens(synthetic_plan(), False, vocab)
    tm, fm = plan_tokens(synthetic_plan(), True, vocab)
    assert f.tolist() == fm.tolist() and t[0] == vocab.ids["PLAN_POSE"] and t[-1] == vocab.ids["PLAN_END"]
    pose = lambda toks: [vocab.specials[x] for x in toks[1:4]]  # noqa: E731
    assert pose(t) == ["POSE_9", "POSE_7", "POSE_8"] and pose(tm) == ["POSE_6", "POSE_7", "POSE_7"]
    assert sorted(set(vocab.specials[x] for x in t if x < len(vocab.specials))) == sorted(
        ["PLAN_POSE", "POSE_9", "POSE_7", "POSE_8", "PLAN_HEAD", "PLAN_EAR", "PLAN_EYE", "PLAN_PUPIL", "PLAN_NOSE",
         "PLAN_MOUTH", "PLAN_END"])
    first_eye = list(t).index(vocab.ids["PLAN_EYE"])
    assert t[first_eye + 1] - vocab.offsets["x"] < 128       # the eyes go left to right on the image


def test_serpentine_and_around_orders():
    xy = np.array([[10.0, 5.0], [50.0, 5.0], [10.0, 40.0], [50.0, 40.0]])
    assert serpentine(xy).tolist() == [0, 1, 3, 2]
    ring = np.array([[0.0, 10.0], [10.0, 0.0], [0.0, -10.0], [-10.0, 0.0], [0.0, -5.0]])
    assert around(ring, np.zeros(2)).tolist() == [4, 2, 1, 0, 3]


def test_palette_colours():
    grey = palette_colours(np.zeros((0, 3)), 4)
    assert len(set(grey.tolist())) == 1
    two = palette_colours(np.array([[0.9, 0.9, 0.9], [0.05, 0.05, 0.05], [0.9, 0.9, 0.9]]), 4)
    assert len(two) == 4 and two[0] == two[2] and two[1] == two[3] and two[0] < two[1]


def test_ranges_survive_the_array_form():
    r = synthetic_ranges()
    del r[LAYERS[3]]
    back = ranges_from_array(ranges_to_array(r))
    assert set(back) == set(r) and all(np.allclose(back[k][0], r[k][0]) for k in r)


# ------------------------------------------------------------------ equivalence with the reference
def test_vocabularies_match_the_reference():
    ED = reference("edit_data")
    for part_order in (False, True):
        for plan in (False, True):
            for palette in (False, True):
                ref, v = ED.specials_for(part_order, plan, palette), Vocabulary.build(part_order, plan, palette)
                assert list(v.specials) == ref
                fields, sizes = ED.fields_of(ref)
                assert v.field_names == fields and v.field_sizes == sizes and v.offsets == ED.offsets_of(ref)
                assert v.size == ED.vocab_of(ref) and v.palette_k.tolist() == ED.palette_k_of(ref).tolist()


@pytest.fixture(scope="module")
def real():
    ED, EA = reference("edit_data"), reference("edit_arch_data")
    ranges_file = earlier_ranges()
    out = []
    for name in REAL_CATS[:2]:
        source = cat_source(name)
        layers, _ = ED.finite(EA.version_layers(name, "A", None))
        out.append((name, source, layers))
    return ED, EA, np.load(ranges_file)["ranges"], out


@pytest.mark.parametrize("fmt", FORMATS)
def test_real_paintings_encode_and_decode_as_the_reference(real, fmt):
    ED, EA, ranges_arr, cats = real
    part_order, plan, palette = fmt
    vocab = Vocabulary.build(*fmt)
    sp = list(vocab.specials)
    fmt_ref = ED.TokenFormat(EA.ranges_from(ranges_arr), {n: i for i, n in enumerate(sp)}, ED.offsets_of(sp), plan,
                             palette)
    ranges = ranges_from_array(ranges_arr)
    for name, source, layers in cats:
        parts_ref = {f: ED.part_map(ED.face_points(name, f)) for f in (False, True)} if part_order else None
        mine_parts = {f: part_map(face_points(source, f)) for f in (False, True)} if part_order else None
        for version, lay in (("A", layers), ("A~m", ED.mirrored([("A", layers, [0, 0, 0, 0])])[0][1])):
            t_ref, f_ref = ED.version_tokens(name, version, lay, fmt_ref, parts_ref)
            flip = version.endswith("~m")
            t, f = painting_tokens(lay, ranges, vocab, mine_parts[flip] if part_order else None,
                                   read_plan(source.plan) if plan else None, flip)
            assert np.array_equal(t, t_ref) and np.array_equal(f, f_ref) and t.dtype == t_ref.dtype
            back_ref = ED.layers_from_tokens(t_ref, EA.ranges_from(ranges_arr), sp, ED.offsets_of(sp))
            back = layers_from_tokens(t, ranges, vocab)
            assert [(k, j) for k, j, _ in back] == [(k, j) for k, j, _ in back_ref]
            for (_, _, p), (_, _, q) in zip(back, back_ref, strict=True):
                assert all(torch.equal(a, b) for a, b in zip(p, q, strict=True))


def test_decoded_paintings_render_as_the_reference(real):
    ED, EA, ranges_arr, cats = real
    name, source, layers = cats[0]
    vocab = Vocabulary.build(True, True, True)
    ranges = ranges_from_array(ranges_arr)
    t, _ = painting_tokens(layers, ranges, vocab, part_map(face_points(source, False)), read_plan(source.plan))
    back = layers_from_tokens(t, ranges, vocab)
    torch.set_num_threads(3)
    assert torch.equal(render(back), EA.render(back))


def test_version_layers_load_as_the_reference(real):
    _, EA, _, cats = real
    for name, source, _ in cats:
        ref, mine = EA.version_layers(name, "A", None), version_layers(source.painting)
        assert [(k, j, s) for k, j, _, s in mine] == [(k, j, s) for k, j, _, s in ref]
        for (_, _, p, _), (_, _, q, _) in zip(mine, ref, strict=True):
            assert all(torch.equal(a, b) for a, b in zip(p, q, strict=True))
