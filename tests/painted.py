"""Painted test cats: the synthetic cat's inputs (synthetic.py) written to files, with a painting of random strokes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from synthetic import synthetic_inputs

from brushgpt.brushes import StrokeParams
from brushgpt.dataset import CatSource
from brushgpt.painter import EYE_STAGE, STAGES

COAT = {"base": {"label": "brown", "confident": True}, "pattern": {"label": "tabby", "confident": False}}
MORE_TRAIN_CAT = "afhq__flickr_cat_000014"


def strokes(rng: np.random.Generator, n: int, x: tuple[float, float] = (40, 470)) -> StrokeParams:
    return StrokeParams(
        torch.tensor(np.c_[rng.uniform(*x, n), rng.uniform(40, 470, n)], dtype=torch.float32),
        torch.tensor(rng.uniform(3, 50, (n, 2)), dtype=torch.float32),
        torch.tensor(rng.uniform(-3, 3, n), dtype=torch.float32),
        torch.tensor(rng.uniform(0, 1, (n, 3)), dtype=torch.float32),
        torch.tensor(rng.uniform(0.3, 1, n), dtype=torch.float32),
    )


def write_painting(folder: Path, seed: int = 0) -> None:
    """stage1.npz .. stage7.npz with a few strokes in every layer (stage 2: both eyes, left first)."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    for k, layers in STAGES.items():
        arrays = {"canvas": np.ones((512, 512, 3), np.float32)}
        for j in range(len(layers)):
            if k == EYE_STAGE:
                sides = [strokes(rng, 3, (150, 230)), strokes(rng, 2, (290, 370))]
                p = StrokeParams(*(torch.cat(f) for f in zip(*sides, strict=True)))
                arrays[f"{j}_side"] = np.array([0, 0, 0, 1, 1], np.int8)
            else:
                p = strokes(rng, 4)
            for f, v in zip(p._fields, p, strict=True):
                arrays[f"{j}_{f}"] = v.numpy()
        np.savez(folder / f"stage{k}.npz", **arrays)  # type: ignore[arg-type]  # keys are array names, not allow_pickle


def synthetic_cat(root: Path, name: str, seed: int) -> CatSource:
    """A cat whose inputs are the synthetic cut-out, photo and plan, and whose painting has random strokes."""
    inputs = synthetic_inputs()
    root.mkdir(parents=True, exist_ok=True)
    np.savez(
        root / f"{name}_cutout.npz",
        target=inputs.target,
        tri=inputs.tri,
        points=inputs.points,
        box=np.array(inputs.box),
    )
    inputs.photo.save(root / f"{name}.png")
    head = np.c_[np.linspace(40, 220, 12), np.full(12, 60.0), (np.arange(12) < 3).astype(float)]
    plan = {**inputs.plan, "yaw": 10.0, "pitch": 0.0, "roll": -4.0, "head": np.array([head, head[:5]], object)}
    np.savez(root / f"{name}_plan.npz", **plan)  # type: ignore[arg-type]  # keys are array names, not allow_pickle
    write_painting(root / "paintings" / name, seed)
    return CatSource(
        name, root / f"{name}_cutout.npz", root / f"{name}.png", root / f"{name}_plan.npz", root / "paintings" / name
    )
