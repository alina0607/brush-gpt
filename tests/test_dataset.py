import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from painted import COAT, MORE_TRAIN_CAT, strokes, synthetic_cat, write_painting
from reference_data import REAL_CATS, cat_source, earlier_ranges, reference, write_cat_table

from brushgpt.dataset import (
    build,
    finite,
    mirrored,
    read_duplicate_pairs,
    read_split,
    split_cats,
    version_layers,
)
from brushgpt.painter import EYE_STAGE
from brushgpt.tokens import NXY, Vocabulary
from brushgpt.train import PaintingData

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_data.py"



# ------------------------------------------------------------------ unit tests
def test_split_keeps_duplicates_together_and_holds_a_share():
    names = [f"cat{i:02d}" for i in range(40)]
    pairs = [(f"cat{i:02d}", f"cat{i + 1:02d}") for i in range(0, 20, 2)]
    split = split_cats(names, pairs)
    assert all(split[a] == split[b] for a, b in pairs)
    assert 6 <= sum(s == "held" for s in split.values()) <= 7
    assert split == split_cats(list(reversed(names)), pairs)


def test_mirroring_flips_x_only_and_swaps_the_eyes():
    rng = np.random.default_rng(0)
    layers = [(1, 0, strokes(rng, 3), None), (2, 0, strokes(rng, 2, (100, 200)), 0),
              (2, 1, strokes(rng, 2, (100, 200)), 0), (2, 0, strokes(rng, 2, (300, 400)), 1),
              (3, 0, strokes(rng, 3), None)]
    (name, flipped, cond), = mirrored([("A", layers, ["blue", "green", "grey", "tabby"])])
    assert name == "A~m" and cond == ["green", "blue", "grey", "tabby"]
    assert [(k, j, s) for k, j, _, s in flipped] == [(1, 0, None), (2, 0, 0), (2, 0, 1), (2, 1, 1), (3, 0, None)]
    p, q = layers[3][2], flipped[1][2]                     # the right eye became the left one
    assert torch.allclose(q.xy[:, 0], 512 - p.xy[:, 0]) and torch.equal(q.xy[:, 1], p.xy[:, 1])
    assert torch.allclose(q.angle, np.pi - p.angle)
    assert torch.equal(q.lw, p.lw) and torch.equal(q.colour, p.colour) and torch.equal(q.alpha, p.alpha)


def test_finite_drops_broken_strokes_and_empty_layers():
    rng = np.random.default_rng(0)
    a, b = strokes(rng, 3), strokes(rng, 1)
    a.xy[1, 0] = float("nan")
    b.colour[0, 2] = float("inf")
    kept, dropped = finite([(1, 0, a, None), (1, 1, b, None)])
    assert dropped == 2 and len(kept) == 1 and len(kept[0][2].xy) == 2


def test_version_layers_split_stage_two_by_eye(tmp_path):
    write_painting(tmp_path / "cat")
    layers = version_layers(tmp_path / "cat")
    eyes = [(j, s, len(p.xy)) for k, j, p, s in layers if k == EYE_STAGE]
    assert eyes == [(j, 0, 3) for j in range(6)] + [(j, 1, 2) for j in range(6)]
    assert [k for k, *_ in layers] == sorted(k for k, *_ in layers) and len(layers) == 21 + 6


def test_build_writes_the_format_the_trainer_reads(tmp_path):
    cats = [synthetic_cat(tmp_path, f"cat{i}", i) for i in range(4)]
    vocab = Vocabulary.build(True, True, True)
    split = {"cat0": "held"}
    summary = build(cats, dict.fromkeys(["cat0", "cat1", "cat2", "cat3"], COAT), tmp_path / "data", vocab,
                    split=split, mirror=True)
    assert summary["cats"] == 4 and summary["held"] == 1 and summary["paintings"] == 8
    z = np.load(tmp_path / "data" / "data.npz")
    assert z["versions"].tolist() == ["A", "A~m"] * 4 and z["cat_id"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert z["tokens"].dtype == np.int32 and z["fields"].dtype == np.int8 and z["cond"].dtype == np.int32
    assert z["split"].tolist() == ["held", "train", "train", "train"] and int(z["vocab"]) == vocab.size
    assert [vocab.specials[c] for c in z["cond"][0]] == ["COND_green", "COND_green", "COND_BASE_brown",
                                                         "COND_PAT_unclear"]
    data = PaintingData(tmp_path / "data" / "data.npz")
    assert data.held == [0] and len(data.train) == 6 and data.layout.palette == ((vocab.ids["PAL_ALL"], 16),)
    x = z["tokens"][0][z["fields"][0] == 1] - vocab.offsets["x"]
    xm = z["tokens"][1][z["fields"][1] == 1] - vocab.offsets["x"]
    assert x.min() >= 0 and x.max() < NXY and len(x) == len(xm)


def test_the_script_builds_from_a_cat_table(tmp_path, monkeypatch):
    cats = [synthetic_cat(tmp_path, f"cat{i}", i) for i in range(3)]
    write_cat_table(cats, tmp_path / "cats.csv")
    (tmp_path / "coat.jsonl").write_text("".join(json.dumps({"name": c.name, **COAT}) + "\n" for c in cats))
    (tmp_path / "split.json").write_text(json.dumps({"split": {"cat1": "held"}}))
    assert read_split(tmp_path / "split.json") == {"cat1": "held"}
    monkeypatch.setattr(sys, "argv", ["build_data.py", "--cats", str(tmp_path / "cats.csv"), "--paintings",
                                      str(tmp_path / "paintings"), "--coat-labels", str(tmp_path / "coat.jsonl"),
                                      "--split", str(tmp_path / "split.json"), "--part-order", "--plan",
                                      "--out", str(tmp_path / "out")])
    runpy.run_path(str(SCRIPT), run_name="__main__")
    z = np.load(tmp_path / "out" / "data.npz")
    assert z["cats"].tolist() == ["cat0", "cat1", "cat2"] and z["split"].tolist() == ["train", "held", "train"]


# ------------------------------------------------------------------ equivalence with the reference
def reference_build(out: Path, names: list[str], ranges: Path | None, eye_versions: bool, mirror: bool,
                    part_order: bool, plan: bool, palette: bool) -> None:
    ED = reference("edit_data")
    (out.parent / "whole.txt").write_text("\n".join(names))
    keep = list(ED.WHOLE_LISTS)
    ED.WHOLE_LISTS[:] = [out.parent / "whole.txt"]
    try:
        ED.build(0, (), out, ("gen",), False, ranges, eye_versions, mirror, part_order, plan, palette)
    finally:
        ED.WHOLE_LISTS[:] = keep


@pytest.mark.parametrize("fmt", [
    (True, True, True, True, True),       # the final builds: plan, part order, palette or not, mirror, eye versions
    (True, True, False, True, True),
    (False, False, False, False, False),  # layer order, bins from these cats, no versions
])
def test_a_small_build_is_identical_to_the_reference(tmp_path, fmt):
    part_order, plan, palette, mirror, eye_versions = fmt
    ED, CL, R, CV = (reference(m) for m in ("edit_data", "coat_label", "eye_recolour_check", "coat_versions"))
    names = list(REAL_CATS) + ([] if eye_versions else [MORE_TRAIN_CAT])  # bins from these cats: train cats cover
    sources = [cat_source(n) for n in names]                                # every layer
    ranges = earlier_ranges() if eye_versions else None
    reference_build(tmp_path / "ref", names, ranges, eye_versions, mirror, part_order, plan, palette)
    from brushgpt.attributes import eye_targets, read_coat_labels
    from brushgpt.dataset import load_ranges
    build(sources, read_coat_labels(CL.DIR / "learned.jsonl"), tmp_path / "mine",
          Vocabulary.build(part_order, plan, palette), split=read_split(CV.OUT / "plan.json"),
          ranges=load_ranges(ranges) if ranges else None,
          eye_versions=ED.EA.EV if eye_versions else None, eye_targets=eye_targets(R.COAT) if eye_versions else None,
          mirror=mirror)
    a, b = np.load(tmp_path / "ref" / "data.npz"), np.load(tmp_path / "mine" / "data.npz")
    assert sorted(a.files) == sorted(b.files)
    for k in a.files:
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape, k
        assert np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f"), k


def test_split_matches_the_reference():
    EA, PS = reference("edit_arch_data"), reference("paint_stages")
    pairs_file = PS.ML / "runs" / "afhq_duplicates" / "pairs.json"
    if not pairs_file.exists():
        pytest.skip("no duplicate pairs")
    pairs = read_duplicate_pairs(pairs_file)
    names = sorted({n for p in pairs[:40] for n in p} | {f"afhq__flickr_cat_{i:06d}" for i in range(60)})
    assert split_cats(names, pairs) == EA.split_cats(names)
