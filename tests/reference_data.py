"""Shared helpers for the equivalence tests of the token format, the attribute readers and the data builder: the
research code this package was extracted from ($SELL_ANYTHING_ML) and real painted cats from its folders."""

from __future__ import annotations

import csv
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from brushgpt.dataset import CatSource

# a training cat and a held cat with eye-colour versions, and a CAT-dataset cat without
REAL_CATS = ("afhq__flickr_cat_000002", "afhq__flickr_cat_000009", "cat__CAT_DATASET_02__CAT_03__00000801_006")


def reference(module: str):
    """A module of the research code (path in $SELL_ANYTHING_ML); the test is skipped when it is not set."""
    root = os.environ.get("SELL_ANYTHING_ML")
    if not root:
        pytest.skip("SELL_ANYTHING_ML is not set")
    for path in (root, os.path.join(root, "tools")):
        if path not in sys.path:
            sys.path.append(path)
    return importlib.import_module(module)


def ml_root() -> Path:
    return Path(reference("paint_stages").ML)


def cat_source(name: str) -> CatSource:
    """A painted cat of the research code's folders; skipped when its files are missing."""
    PS, J, H = reference("paint_stages"), reference("fur_judge"), reference("head_line")
    try:
        cutout = J.repaint_file(name)
    except FileNotFoundError:
        pytest.skip(f"no cut-out for {name}")
    painting = PS.OUT / name
    if not (painting / "stage7.npz").exists():
        pytest.skip(f"{name} is not painted")
    photo = H.photo_path(json.loads(str(np.load(cutout, allow_pickle=True)["meta"])))
    return CatSource(name, Path(cutout), Path(photo), next(PS.PLAN.glob(f"*/shard_*/{name}.npz")), painting)


def write_cat_table(sources: list[CatSource], path: Path) -> Path:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "cutout", "photo", "plan"])
        for s in sources:
            w.writerow([s.name, s.cutout, s.photo, s.plan])
    return path


def earlier_ranges() -> Path:
    """The earlier build whose bins the final builds use."""
    path = ml_root() / "data" / "processed" / "edit_1108" / "data.npz"
    if not path.exists():
        pytest.skip("no earlier build to take the bins from")
    return path
