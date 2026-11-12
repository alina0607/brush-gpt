import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from brushgpt.model import BrushGPTConfig, convert_reference_state_dict, drop_attributes, load_checkpoint
from brushgpt.train import PaintingData, TrainConfig, layout_from_specials, lr_at, sample_held, train

SPECIALS = ["BOS", "EOS", "SEP", "EYE_L", "EYE_R", *(f"LAYER{k}" for k in range(8)), "COND_unclear", "COND_blue",
            "COND_green", "COND_yellow", "COND_BASE_white", "COND_BASE_black", "COND_PAT_solid", "COND_PAT_tabby",
            "TASK_GEN", "TASK_EYES", "PLAN_HEAD", "PLAN_EYE", "PLAN_END", "PAL_ALL"]
FIELD_NAMES = ["x", "y", "angle", "size", "paint", "colour"]
FIELD_SIZES = [16, 16, 4, 3, 10, 6]


def reference(module: str):
    """A module of the research code this package was extracted from (path in $SELL_ANYTHING_ML)."""
    root = os.environ.get("SELL_ANYTHING_ML")
    if not root:
        pytest.skip("SELL_ANYTHING_ML is not set")
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(module)


def make_data(path: Path, n_cats: int = 6, seed: int = 0) -> Path:
    """A small data file in the built format: a plan outline, a palette, then strokes in four layers (two in the eye
    stage); every cat has its painting A and a mirror, some also an eye-colour version; the last two cats are held."""
    rng = np.random.default_rng(seed)
    sp = {n: i for i, n in enumerate(SPECIALS)}
    offsets = np.cumsum([len(SPECIALS), *FIELD_SIZES[:-1]])

    def tok(f: int) -> int:
        return int(offsets[f] + rng.integers(FIELD_SIZES[f]))

    rows, fields, conds, cat_id, versions = [], [], [], [], []
    for c in range(n_cats):
        for version in ["A", "A~m", "eye:both_green"][:2 + c % 2]:
            t, f = [sp["PLAN_HEAD"]], [0]
            for _ in range(rng.integers(2, 5)):
                t += [tok(0), tok(1)]
                f += [1, 2]
            t += [sp["PLAN_END"], sp["PAL_ALL"], *(tok(5) for _ in range(3))]
            f += [0, 0, 6, 6, 6]
            for layer in ("LAYER0", "LAYER2", "LAYER3", "LAYER5"):
                t.append(sp[layer])
                f.append(0)
                for _ in range(rng.integers(1, 4)):
                    t += [tok(k) for k in range(5)]
                    f += [1, 2, 3, 4, 5]
            rows.append(t)
            fields.append(f)
            eyes = rng.choice([sp["COND_blue"], sp["COND_green"], sp["COND_yellow"], sp["COND_unclear"]], 2)
            conds.append([*eyes, sp["COND_BASE_white"] + c % 2, sp["COND_PAT_solid"] + c % 2])
            cat_id.append(c)
            versions.append(version)
    width = max(map(len, rows)) + 3
    tokens = np.full((len(rows), width), -1, np.int32)
    flds = np.full((len(rows), width), -1, np.int8)
    for i, (t, f) in enumerate(zip(rows, fields, strict=True)):
        tokens[i, :len(t)], flds[i, :len(f)] = t, f
    np.savez(path, tokens=tokens, fields=flds, cond=np.array(conds, np.int32), cat_id=np.array(cat_id),
             group=np.array(cat_id), kind=np.array(["gen"] * len(rows)), versions=np.array(versions),
             cats=np.array([f"cat{c}" for c in range(n_cats)]),
             split=np.array(["train"] * (n_cats - 2) + ["held"] * 2), specials=np.array(SPECIALS),
             vocab=np.array(len(SPECIALS) + sum(FIELD_SIZES)), field_names=np.array(FIELD_NAMES),
             field_sizes=np.array(FIELD_SIZES), palette_k=np.array([3, 0]))
    return path


@pytest.fixture
def data_file(tmp_path: Path) -> Path:
    return make_data(tmp_path / "data.npz")


# ------------------------------------------------------------------------------------------------- unit tests
def test_learning_rate_warms_up_then_decays_to_a_tenth():
    cfg = TrainConfig(steps=1000, warmup=100, lr=1e-3)
    assert lr_at(0, cfg) == pytest.approx(1e-5)
    assert lr_at(99, cfg) == pytest.approx(1e-3)
    assert lr_at(100, cfg) == pytest.approx(1e-3)
    assert lr_at(1000, cfg) == pytest.approx(1e-4)
    assert all(lr_at(s, cfg) >= lr_at(s + 1, cfg) for s in range(100, 999))


def test_layout_from_specials():
    layout = layout_from_specials(SPECIALS, FIELD_NAMES, FIELD_SIZES, (3, 0))
    assert layout.eos == SPECIALS.index("EOS") and layout.bos == 0
    assert layout.field_offsets[0] == len(SPECIALS)
    assert layout.plan_markers == (SPECIALS.index("PLAN_HEAD"), SPECIALS.index("PLAN_EYE"))
    assert layout.palette == ((SPECIALS.index("PAL_ALL"), 3),)
    assert SPECIALS.index("PLAN_END") in layout.slots and SPECIALS.index("TASK_GEN") not in layout.slots


def test_painting_data_splits_by_cat_and_scores_painting_a_only(data_file: Path):
    data = PaintingData(data_file)
    assert {int(data.cat_id[i]) for i in data.train} == {0, 1, 2, 3}
    assert {int(data.cat_id[i]) for i in data.held} == {4, 5}
    assert all(str(data.versions[i]) == "A" for i in data.held)
    cond, tokens, fields = data.example(data.train[0])
    assert int(cond[0]) == SPECIALS.index("TASK_GEN") and len(cond) == 5
    assert tokens.shape == fields.shape and int(tokens.min()) >= 0


def test_eye_paint_marks_paint_tokens_of_eye_layers_only(data_file: Path):
    data = PaintingData(data_file)
    _, tokens, fields = data.example(0)
    marked = data.eye_paint(tokens, fields)
    layer = None
    for t, f, m in zip(tokens.tolist(), fields.tolist(), marked.tolist(), strict=True):
        if f == 0 and t < len(SPECIALS) and SPECIALS[t].startswith(("LAYER", "PAL_")):
            layer = SPECIALS[t]
        assert m == (f == 5 and layer in ("LAYER2", "LAYER3", "LAYER5"))


def test_training_writes_checkpoints_and_sampling_gives_one_painting_per_cat(data_file: Path, tmp_path: Path):
    data = PaintingData(data_file)
    model_cfg = BrushGPTConfig(vocab_size=data.vocab + 1, d_model=32, n_heads=4, enc_layers=1, dec_layers=1,
                               max_len=512, null_token=True)
    out = tmp_path / "run"
    model = train(data, model_cfg, TrainConfig(steps=3, accum=2, warmup=1, eval_every=2, cond_drop=0.5), out)
    assert {"info.json", "log.jsonl", "last.pt", "best.pt"} <= {p.name for p in out.iterdir()}
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [2, 3] and all("gen_nll" in r for r in rows)
    again = load_checkpoint(torch.load(out / "last.pt"), data.layout)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), again.state_dict().values(),
                                                  strict=True))
    tokens, items = sample_held(again, data, n_cats=2, batch=2, guidance=2.0)
    assert tokens.shape[0] == 2 and list(items) == data.held[:2]


# ----------------------------------------------------------------------------------- equivalence with the reference
def test_data_matches_the_reference(data_file: Path):
    ea = reference("sellanything_ml.sketch.edit_arch")
    ref, mine = ea.Pairs(data_file, "cpu"), PaintingData(data_file)
    lay = mine.layout
    assert list(lay.slots) == ref.slots and list(lay.field_offsets) == ref.offsets
    assert list(lay.field_sizes) == ref.sizes and list(lay.plan_markers) == ref.plan_runs
    assert dict(lay.palette) == ref.palette_runs
    assert mine.train == ref.gens["train"] and mine.held == ref.gens["held"]
    for i in range(len(mine.tokens)):
        cond, tokens, fields = mine.example(i)
        r_cond, _, r_new, r_field = ref.pair(-1, i)
        assert torch.equal(cond, r_cond) and torch.equal(tokens, r_new) and torch.equal(fields, r_field)
        assert torch.equal(mine.eye_paint(tokens, fields), ref.eye_paint(tokens, fields))
        assert torch.equal(mine.eyes_swapped(cond), ref.eyes_swapped(cond))
        assert torch.equal(drop_attributes(cond, 99, mine.n_attributes_kept), ref.without_attributes(cond, 99))


def _reference_args(data: Path, out: Path, **kw) -> argparse.Namespace:
    args = dict(arch="encdec", layout="free", data=str(data), out=str(out), steps=6, accum=3, lr=3e-4, d_model=32,
                n_layers=8, enc_layers=1, dec_layers=2, eval_every=3, eval_pairs=60, stop_after_min=0, gen_frac=1.0,
                init=None, eval_gen=3, dropout=0.1, weight_decay=0.1, cond_drop=0.3, balance_cats=True)
    return argparse.Namespace(**{**args, **kw})


@pytest.mark.parametrize("balance_cats", [True, False])
def test_training_and_sampling_match_the_reference(tmp_path: Path, balance_cats: bool):
    ea = reference("sellanything_ml.sketch.edit_arch")
    data_file = make_data(tmp_path / "data.npz", n_cats=8)
    ref_out, my_out = tmp_path / "ref", tmp_path / "mine"
    a = _reference_args(data_file, ref_out, balance_cats=balance_cats)
    ea.train(a)
    data = PaintingData(data_file)
    model_cfg = BrushGPTConfig(vocab_size=data.vocab + 1, d_model=32, enc_layers=1, dec_layers=2, dropout=0.1,
                               null_token=True)
    train(data, model_cfg, TrainConfig(steps=6, accum=3, eval_every=3, eval_paintings=3, cond_drop=0.3,
                                       balance_cats=balance_cats), my_out)

    def logged(out: Path) -> list[dict]:
        keys = ("step", "loss", "lr", "gen_acc", "gen_nll", "gen_tokens", "eye_paint_nll", "eye_swap_gap")
        return [{k: r.get(k) for k in keys} for r in map(json.loads, (out / "log.jsonl").read_text().splitlines())]

    assert logged(ref_out) == logged(my_out)
    ref_ck, my_ck = torch.load(ref_out / "last.pt"), torch.load(my_out / "last.pt")
    ref_state = convert_reference_state_dict(ref_ck["model"])
    assert all(torch.equal(ref_state[k], v) for k, v in my_ck["model"].items())

    for guidance in (1.0, 2.0):
        s = argparse.Namespace(ckpt=str(ref_out / "best.pt"), data=str(data_file), out=str(tmp_path / "ref.npz"),
                               cats=2, temperature=1.0, seed=0, task="gen", batch=2, guidance=guidance)
        ea.sample_cmd(s)
        ref_tokens = np.load(tmp_path / "ref.npz")["tokens"]
        model = load_checkpoint(torch.load(my_out / "best.pt"), data.layout)
        tokens, items = sample_held(model, data, n_cats=2, batch=2, guidance=guidance, seed=0)
        assert np.array_equal(ref_tokens, tokens)


def test_real_data_file_reads_as_the_reference_reads_it():
    ea = reference("sellanything_ml.sketch.edit_arch")
    path = Path(os.environ["SELL_ANYTHING_ML"]) / "data/processed/edit_1110_gen_furpal/data.npz"
    if not path.exists():
        pytest.skip("the real data file is not there")
    ref, mine = ea.Pairs(path, "cpu"), PaintingData(path)
    lay = mine.layout
    assert (list(lay.slots), list(lay.field_offsets), list(lay.field_sizes), list(lay.plan_markers)) == \
        (ref.slots, ref.offsets, ref.sizes, ref.plan_runs)
    assert dict(lay.palette) == ref.palette_runs
    assert mine.train == ref.gens["train"] and mine.held == ref.gens["held"]
    for i in (mine.train[0], mine.train[-1], mine.held[0]):
        got, want = mine.example(i), ref.pair(-1, i)
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[2]) and torch.equal(got[2], want[3])
        assert torch.equal(mine.eye_paint(got[1], got[2]), ref.eye_paint(got[1], got[2]))
        assert torch.equal(mine.eyes_swapped(got[0]), ref.eyes_swapped(got[0]))
