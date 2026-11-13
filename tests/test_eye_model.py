import argparse
import json
import runpy
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from painted import synthetic_cat
from reference_data import ml_root, reference

from brushgpt.eye_data import PREFIX, SP, Frame, build, load_eye_ranges, strokes_from_tokens
from brushgpt.eye_model import (
    VAL_EVERY,
    EyeData,
    EyeGrammar,
    EyeSamples,
    EyeTrainConfig,
    generate,
    geometry,
    inside,
    load_checkpoint,
    lr_at,
    mean_body_loss,
    model_config,
    sample_held,
    train,
    without_colour,
)
from brushgpt.layers import GPT

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
TINY: dict[str, Any] = {"d_model": 32, "n_layers": 2, "n_heads": 4, "dropout": 0.0}


@pytest.fixture(scope="module")
def synthetic_data(tmp_path_factory) -> Path:
    """Eye data built from five synthetic cats: one held, one a validation cat, three training cats."""
    root = tmp_path_factory.mktemp("eyes")
    val_name = next(f"cat{i}" for i in range(1000) if zlib.crc32(f"cat{i}".encode()) % VAL_EVERY == 0)
    names = ["cat0", "cat1", "cat2", "cat3", val_name]
    build([synthetic_cat(root, n, i) for i, n in enumerate(names)], root / "data", split={"cat3": "held"})
    return root / "data" / "data.npz"


def tiny_model(data: EyeData, seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(model_config(data, **TINY)).eval()


def assert_well_formed(body: np.ndarray, data: EyeData) -> None:
    """A body follows the grammar: layers in increasing order, each followed by whole strokes, then EOS."""
    fields = ["x", "y", "angle", "size", "paint"]
    layer, k = -1, 0
    for t in body:
        if t == SP["EOS"]:
            assert k == 0 and layer >= 0
            return
        name = data.specials[t] if t < len(data.specials) else ""
        if name.startswith("LAYER_"):
            assert k == 0 and int(name[6:]) > layer
            layer = int(name[6:])
            continue
        a, b = data.range[fields[k]]
        assert layer >= 0 and a <= t < b
        k = (k + 1) % 5


# ------------------------------------------------------------------ unit tests
def test_data_splits_and_batches(synthetic_data):
    data = EyeData(synthetic_data)
    assert len(data.val) and len(data.held) == 2 and not set(data.val) & set(data.train)
    x, y, m = data.batch(data.train[:3], torch.device("cpu"))
    assert x.shape == y.shape == m.shape and torch.equal(x[:, 1:], y[:, :-1])
    assert not m[:, : PREFIX - 1].any() and m[:, PREFIX - 1].all()
    assert int(m.sum()) == int((data.lengths[data.train[:3]] - PREFIX).sum())
    x, _, _ = data.batch(data.train[:3], torch.device("cpu"), 1.0, np.random.default_rng(0))
    assert (x[:, 2] == data.sp["NULL"]).all()
    assert (data.prefixes(data.held, "blue")[:, 2] == data.sp["EYE_blue"]).all()


def test_learning_rate_warms_up_then_decays():
    assert lr_at(0, 100, 1.0, 10) == pytest.approx(0.1) and lr_at(10, 100, 1.0, 10) == pytest.approx(1.0)
    assert lr_at(100, 100, 1.0, 10) == pytest.approx(0.1)


@pytest.mark.parametrize("guidance", [1.0, 2.0])
def test_sampling_follows_the_grammar(synthetic_data, guidance):
    data = EyeData(synthetic_data)
    model = tiny_model(data)
    pre = data.prefixes(data.held)
    got = generate(
        model,
        pre,
        without_colour(pre, data.sp["NULL"]),
        EyeGrammar(data, "cpu"),
        300,
        1.0,
        guidance,
        torch.Generator().manual_seed(0),
    ).numpy()
    frame, ranges = Frame(np.array([200.0, 240.0]), 30.0), load_eye_ranges(synthetic_data)
    for body in got:
        assert_well_formed(body, data)
        n_strokes = sum(t >= data.range["x"][0] for t in body if t != SP["EOS"]) // 5
        assert sum(len(p.xy) for _, p in strokes_from_tokens(body, frame, ranges)) == n_strokes


def test_guidance_one_ignores_the_unconditional_half(synthetic_data):
    data = EyeData(synthetic_data)
    model = tiny_model(data)
    pre, grammar = data.prefixes(data.held), EyeGrammar(data, "cpu")
    a = generate(model, pre, without_colour(pre, 4), grammar, 50, 1.0, 1.0, torch.Generator().manual_seed(3))
    b = generate(model, pre, None, grammar, 50, 1.0, 1.0, torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_geometry_of_an_eye_against_itself(synthetic_data):
    data = EyeData(synthetic_data)
    r = int(data.held[0])
    t = data.tokens[r, : data.lengths[r]]
    g = geometry(t[:PREFIX], t[PREFIX:], t[PREFIX:], data)
    assert g["strokes_ratio"] == 1.0 and 0 <= g["inside"] <= 1
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert inside(np.array([[5.0, 5.0], [15.0, 5.0]]), square).tolist() == [True, False]


def test_training_writes_a_checkpoint_that_samples(synthetic_data, tmp_path):
    data = EyeData(synthetic_data)
    cfg = EyeTrainConfig(steps=3, batch=4, warmup=1, eval_every=2, log_every=1, sample_every=2, sample_n=1)
    train(data, model_config(data, **TINY), cfg, tmp_path, "cpu")
    log = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    evals = [r for r in log if "val" in r]
    assert [r["step"] for r in evals] == [0, 2, 3] and "sample_inside" in evals[1]
    model = load_checkpoint(tmp_path / "best.pt", "cpu", PREFIX + 100)
    samples = sample_held(model, data, n_cats=1, guidance=2.0, max_new=100)
    assert samples.bodies.shape[0] == 2 and samples.rows.tolist() == data.held[:2].tolist()
    samples.save(tmp_path / "s.npz")
    back = EyeSamples.load(tmp_path / "s.npz")
    assert np.array_equal(back.bodies, samples.bodies) and back.guidance == 2.0 and back.eye_as == ""


def test_the_scripts_train_and_sample(synthetic_data, tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_eyes.py",
            "--data",
            str(synthetic_data),
            "--out",
            str(tmp_path),
            "--d-model",
            "32",
            "--layers",
            "1",
            "--heads",
            "4",
            "--steps",
            "2",
            "--batch",
            "2",
            "--eval-every",
            "2",
            "--sample-every",
            "0",
            "--device",
            "cpu",
        ],
    )
    runpy.run_path(str(SCRIPTS / "train_eyes.py"), run_name="__main__")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sample_eyes.py",
            "--ckpt",
            str(tmp_path / "best.pt"),
            "--data",
            str(synthetic_data),
            "--out",
            str(tmp_path / "s.npz"),
            "--cats",
            "1",
            "--max-new",
            "40",
            "--eye-as",
            "blue",
            "--device",
            "cpu",
        ],
    )
    runpy.run_path(str(SCRIPTS / "sample_eyes.py"), run_name="__main__")
    s = EyeSamples.load(tmp_path / "s.npz")
    assert s.eye_as == "blue" and (s.prefix[:, 2] == SP["EYE_blue"]).all()


# ------------------------------------------------------------------ equivalence with the reference
@pytest.fixture(scope="module")
def real(tmp_path_factory):
    """The eye model's real data cut down to 2 held cats, 40 training cats and 3 validation cats, and
    the final run's checkpoint."""
    G = reference("sellanything_ml.sketch.eye_gpt")
    root = ml_root()
    ckpt, path = root / "runs/eyegpt/E1/best.pt", root / "data/processed/eye_1110/data.npz"
    if not (ckpt.exists() and path.exists()):
        pytest.skip("the eye model's checkpoint or data is not there")
    z = np.load(path)
    split = z["split"]
    held = [c for c in range(len(split)) if split[c] == "held"][:2]
    train_cats = [c for c in range(len(split)) if split[c] == "train"]
    val_cats = [c for c in train_cats if zlib.crc32(str(z["cats"][c]).encode()) % VAL_EVERY == 0][:3]
    keep = sorted(set(held + train_cats[:40] + val_cats))
    rows = np.flatnonzero(np.isin(z["cat_id"], keep))
    new_id = {c: i for i, c in enumerate(keep)}
    small = {k: z[k] for k in z.files}
    for k in ("tokens", "lengths", "version", "side", "eye", "centre", "r", "flip"):
        small[k] = z[k][rows]
    small["cat_id"] = np.array([new_id[c] for c in z["cat_id"][rows]], np.int32)
    small["cats"], small["split"] = z["cats"][keep], split[keep]
    out = tmp_path_factory.mktemp("real") / "data.npz"
    np.savez(out, **small)
    return G, ckpt, out


def reference_model(G, ckpt: Path):
    ck = torch.load(ckpt, map_location="cpu")
    ref = G.GPT(G.GPTConfig(**ck["config"])).eval()
    ref.load_state_dict(ck["model"])
    return ref


def test_real_checkpoint_gives_the_reference_logits_and_loss(real):
    G, ckpt, path = real
    data, ref = EyeData(path), reference_model(G, ckpt)
    mine = load_checkpoint(ckpt)
    rows = data.held[:4]
    x, _, _ = data.batch(rows, torch.device("cpu"))
    with torch.no_grad():
        assert torch.equal(ref(x)[0], mine(x)[0])
    want = G.held_loss(ref, G.EyeData(path), torch.device("cpu"), 2, ix=rows)
    assert mean_body_loss(mine, data, rows, torch.device("cpu"), 2) == want


@pytest.mark.parametrize("guidance, eye_as", [(1.0, ""), (2.0, ""), (2.0, "blue")])
def test_sampling_is_identical_to_the_reference(real, tmp_path, guidance, eye_as):
    G, ckpt, path = real
    G.sample(
        argparse.Namespace(
            ckpt=str(ckpt),
            data=str(path),
            out=str(tmp_path / "ref.npz"),
            cats=1,
            guidance=guidance,
            temperature=1.0,
            eye_as=eye_as,
            max_new=1200,
            batch=32,
            seed=0,
            device="cpu",
        )
    )
    data = EyeData(path)
    model = load_checkpoint(ckpt, "cpu", data.prefix + 1200)
    sample_held(model, data, 1, guidance, 1.0, eye_as, 1200, 32, 0).save(tmp_path / "mine.npz")
    a, b = np.load(tmp_path / "ref.npz"), np.load(tmp_path / "mine.npz")
    assert sorted(a.files) == sorted(b.files)
    for k in a.files:
        assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), k


@pytest.mark.parametrize("balance_cats", [False, True])
def test_training_steps_are_identical_to_the_reference(real, tmp_path, balance_cats):
    G, _, path = real
    args = dict(
        d_model=32,
        layers=2,
        heads=4,
        dropout=0.1,
        weight_decay=0.1,
        lr=1e-3,
        warmup=2,
        steps=4,
        batch=4,
        cond_drop=0.5,
        balance_cats=balance_cats,
        eval_every=2,
        eval_limit=4,
        log_every=1,
        sample_every=2,
        sample_n=2,
        stop_after_min=0,
        seed=0,
    )
    G.train(argparse.Namespace(data=str(path), out=str(tmp_path / "ref"), device="cpu", **args))
    data = EyeData(path)
    cfg = EyeTrainConfig(**{k: v for k, v in args.items() if k not in ("d_model", "layers", "heads", "dropout")})
    train(data, model_config(data, 32, 2, 4, 0.1), cfg, tmp_path / "mine", "cpu")
    a, b = torch.load(tmp_path / "ref" / "best.pt"), torch.load(tmp_path / "mine" / "best.pt")
    assert a["step"] == b["step"] and a["val"] == b["val"] and a["held"] == b["held"]
    assert a["config"] == {**b["config"], "abs_pos": False}
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])

    def records(run: str) -> list[dict]:
        rows = [json.loads(x) for x in (tmp_path / run / "log.jsonl").read_text().splitlines()]
        return [{k: v for k, v in r.items() if k not in ("seconds", "tokens_per_s")} for r in rows]

    assert records("ref") == records("mine")
