import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from painted import COAT, synthetic_cat
from PIL import Image
from reference_data import cat_source, ml_root, reference, write_cat_table

from brushgpt.dataset import build as build_whole
from brushgpt.dataset import read_plan
from brushgpt.evaluate import (
    CROP_SIZE,
    CatViews,
    WholeFaceData,
    contact_sheet,
    crop,
    eye_measures,
    eye_summary,
    measure_eyes_alone,
    measure_system,
    measure_whole_face,
    plan_from_tokens,
)
from brushgpt.eye_data import PREFIX, build, load_eye_ranges, on_plan_grid
from brushgpt.eye_model import EyeData, EyeSamples, model_config
from brushgpt.layers import GPT
from brushgpt.tokens import Vocabulary, plan_tokens

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
NAMES = ["cat0", "cat1", "cat2"]


def every_coat(images):
    return [{"base": {"label": "brown", "confident": True}, "pattern": {"label": "tabby", "confident": True}}
            for _ in images]


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """Three synthetic cats (the last one held): their sources, a whole-face data file and an eye data file."""
    root = tmp_path_factory.mktemp("judge")
    cats = [synthetic_cat(root, n, i) for i, n in enumerate(NAMES)]
    split = {"cat2": "held"}
    build_whole(cats, dict.fromkeys(NAMES, COAT), root / "whole", Vocabulary.build(True, True), split=split)
    build(cats, root / "eyes", split=split)
    return root, {c.name: c for c in cats}


def own_samples(eye_data: Path) -> EyeSamples:
    """The held eyes' own bodies as if a model had sampled them."""
    z = np.load(eye_data)
    rows = np.flatnonzero((z["split"][z["cat_id"]] == "held") & (z["version"] == "A"))
    bodies = [z["tokens"][r, PREFIX:z["lengths"][r]] for r in rows]
    width = max(map(len, bodies))
    return EyeSamples(rows, z["tokens"][rows, :PREFIX].astype(np.int64),
                      np.stack([np.pad(b, (0, width - len(b)), constant_values=3) for b in bodies]).astype(np.int64),
                      1.0, 1.0, 0)


# ------------------------------------------------------------------ unit tests
def test_eye_measures_on_a_drawn_eye():
    t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    outline = np.c_[200 + 40 * np.cos(t), 240 + 20 * np.sin(t)]
    pupil = np.c_[200 + 6 * np.cos(t[::3][:6]), 240 + 15 * np.sin(t[::3][:6])]
    canvas = np.ones((512, 512, 3))
    canvas[226:254, 195:205] = 0.0
    xy = torch.tensor([[200.0, 240.0], [210.0, 245.0], [300.0, 300.0]])
    from brushgpt.brushes import StrokeParams
    p = StrokeParams(xy, torch.ones(3, 2), torch.zeros(3), torch.zeros(3, 3), torch.ones(3))
    m = eye_measures(canvas, [(0, p)], outline, pupil)
    assert m["strokes"] == 3 and m["inside"] == pytest.approx(2 / 3) and m["pupil_contrast"] > 0.8
    assert np.isnan(eye_measures(canvas, [], outline, None)["pupil_contrast"])


def test_eye_summary_counts_only_asked_colours():
    rows = [{"asked": ["green", "unclear"], "names": {"m": ["green", "blue"]},
             "eyes": {"m": [{"strokes": 10, "inside": 0.5, "pupil_contrast": 0.1}]}},
            {"asked": ["blue", "copper"], "names": {"m": ["unclear", "copper"]}, "eyes": {}}]
    assert eye_summary(rows, ["m"]) == {"m": {"read_right": "2/3", "read_unclear": 1, "strokes": 10.0,
                                              "inside": 0.5, "pupil_contrast": 0.1}}


def test_plan_is_read_back_from_the_whole_face_tokens(synthetic):
    _, sources = synthetic
    plan = read_plan(sources["cat0"].plan)
    vocab = Vocabulary.build(True, True)
    toks, _ = plan_tokens(plan, False, vocab)
    outlines, pupils = plan_from_tokens(toks, vocab)
    assert len(outlines) == 2 and all(len(o) == 16 for o in outlines) and all(len(p) == 6 for p in pupils)
    assert np.allclose(sorted(o[:, 0].mean() for o in outlines),
                       sorted(on_plan_grid(np.asarray(e, float))[:, 0].mean() for e in plan["eyes"]))


def test_the_whole_face_measure_on_the_painters_own_tokens(synthetic):
    root, sources = synthetic
    data = WholeFaceData(root / "whole" / "data.npz")
    rows = np.array([0, 1, 2])
    tokens = np.load(root / "whole" / "data.npz")["tokens"][rows]
    sampled = np.c_[tokens, np.full(3, -1)]                     # a sampled painting ends with EOS
    sampled[np.arange(3), (tokens >= 0).sum(1)] = data.vocab.ids["EOS"]
    res = measure_whole_face(sampled, rows, data, CatViews(sources), every_coat)
    assert res["n"] == 3 and res["gen_ends_with_eos"] == "3/3"
    assert res["model"] == res["painter"] and all(r["model"]["names"] == r["painter"]["names"] for r in res["rows"])
    assert res["painter"]["gen_read_right"]["base"] == "3/3"


def test_the_eye_alone_measure_on_the_painters_own_eyes(synthetic):
    root, sources = synthetic
    s = own_samples(root / "eyes" / "data.npz")
    res, pics = measure_eyes_alone(s.__dict__, np.load(root / "eyes" / "data.npz"), sources, CatViews(sources))
    (row,) = res["rows"]
    assert row["cat"] == "cat2" and row["names"]["eye model"] == row["names"]["painter"]
    assert len(pics) == 1 and pics[0][0][1].size == CROP_SIZE


def test_the_system_measure_repaints_both_eyes(synthetic):
    root, sources = synthetic
    eye_data = EyeData(root / "eyes" / "data.npz")
    torch.manual_seed(0)
    model = GPT(model_config(eye_data, 32, 1, 4, 0.0)).eval()
    whole = np.load(root / "whole" / "data.npz")
    res, pics = measure_system(whole["tokens"][[2]], np.array([2]), WholeFaceData(root / "whole" / "data.npz"),
                               model, eye_data, load_eye_ranges(root / "eyes" / "data.npz"), CatViews(sources),
                               guidance=2.0, max_new=60)
    (row,) = res["rows"]
    assert row["cat"] == "cat2" and set(row["names"]) == {"whole-face model", "whole + eye model"}
    assert res["summary"]["plan_not_two_outlines"] == 0 and len(row["eyes"]["whole + eye model"]) == 2


def test_contact_sheet(tmp_path):
    img = crop(np.ones((512, 512, 3)), [np.array([[180.0, 230.0], [220.0, 250.0]])])
    contact_sheet([[("a", img), ("b", img)]], tmp_path / "s.png")
    assert Image.open(tmp_path / "s.png").size == (2 * CROP_SIZE[0], CROP_SIZE[1] + 4)


def test_the_script_measures_eyes_alone(synthetic, tmp_path, monkeypatch):
    root, sources = synthetic
    write_cat_table(list(sources.values()), tmp_path / "cats.csv")
    own_samples(root / "eyes" / "data.npz").save(tmp_path / "s.npz")
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "alone", str(tmp_path / "s.npz"), "--eye-data",
                                      str(root / "eyes" / "data.npz"), "--cats", str(tmp_path / "cats.csv"),
                                      "--paintings", str(root / "paintings")])
    runpy.run_path(str(SCRIPT), run_name="__main__")
    res = json.loads((tmp_path / "s.alone.json").read_text())
    assert res["summary"]["eye model"]["strokes"] <= res["summary"]["painter"]["strokes"]   # strokes off the frame
    assert [r["cat"] for r in res["rows"]] == ["cat2"]                                          # are not written
    assert (tmp_path / "s.alone.png").exists()


# ------------------------------------------------------------------ equivalence with the reference
class RealSources(dict):
    """The research code's painted cats, found when first asked for."""

    def __missing__(self, name):
        self[name] = cat_source(name)
        return self[name]


@pytest.fixture(scope="module")
def real():
    reference("paint_stages")
    root = ml_root()
    e1, part = root / "runs/eyegpt/E1", root / "data/kaggle/gen_cfg/out/out/part"
    files = [e1 / "eyes_g2.npz", e1 / "best.pt", part / "gen_g2.0.npz", root / "data/processed/eye_1110/data.npz",
             root / "data/processed/edit_1110_part/data.npz"]
    if not all(f.exists() for f in files):
        pytest.skip("the final runs' samples or data are not there")
    torch.set_num_threads(3)
    sources = RealSources()
    return root, e1, part, sources, CatViews(sources)


def same_rows(got: list[dict], want: list[dict]) -> bool:
    """Rows equal as JSON (a measure that is NaN, an eye without strokes, compares equal to NaN)."""
    return json.dumps(got) == json.dumps(want)


def whole_samples(part: Path, g: int) -> tuple[np.ndarray, np.ndarray]:
    """The final whole-face samples, written by the research code with (source, target) pairs; the target is the
    data row this package's samples call `rows`."""
    z = np.load(part / f"gen_g{g}.0.npz")
    return z["tokens"], z["pairs"][:, 1]


@pytest.mark.parametrize("samples, n", [("eyes_g2", 2), ("eyes_g2_blue", 1)])
def test_eye_alone_reproduces_the_final_measures(real, samples, n):
    root, e1, _, sources, views = real
    res, _ = measure_eyes_alone(np.load(e1 / f"{samples}.npz"), np.load(root / "data/processed/eye_1110/data.npz"),
                                sources, views, n)
    want = json.loads((e1 / f"{samples}.alone.json").read_text())
    assert same_rows(res["rows"], want["rows"][:n])


def test_system_reproduces_the_final_measures(real):
    root, e1, part, _, views = real
    tokens, rows = whole_samples(part, 2)
    from brushgpt.eye_model import load_checkpoint
    eye_path = root / "data/processed/eye_1110/data.npz"
    res, _ = measure_system(tokens, rows, WholeFaceData(root / "data/processed/edit_1110_part/data.npz"),
                            load_checkpoint(e1 / "best.pt"), EyeData(eye_path), load_eye_ranges(eye_path), views,
                            guidance=2.0, n_cats=2)
    want = json.loads((e1 / "system_whole_g2_eye_g2.json").read_text())
    assert same_rows(res["rows"], want["rows"][:2])


def test_whole_face_measure_reproduces_the_final_measures(real):
    pytest.importorskip("transformers")
    pytest.importorskip("sklearn")
    from brushgpt.attributes import CoatReader
    root, _, part, _, views = real
    CL = reference("coat_label")
    try:
        reader = CoatReader(CL.DIR / "clip.npz", CL.DIR / "colour.npz", CL.DIR / "labels.jsonl")
    except OSError:
        pytest.skip("the CLIP weights are not in the local cache")
    tokens, rows = whole_samples(part, 2)
    res = measure_whole_face(tokens[:2], rows[:2], WholeFaceData(root / "data/processed/edit_1110_part/data.npz"),
                             views, reader)
    want = json.loads((part / "gen_g2.0.measure.json").read_text())
    assert same_rows(res["rows"], want["rows"][:2])
