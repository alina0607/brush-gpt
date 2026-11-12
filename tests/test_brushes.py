import importlib
import math
import os
import sys

import numpy as np
import pytest
import torch

from brushgpt.brushes import (
    KINDS,
    Brush,
    StrokeParams,
    coverage,
    hash01,
    lay,
    lay_layer,
    render,
    seed_of,
    stroke_seeds,
)


def reference(module: str):
    """A module of the research code this package was extracted from (path in $SELL_ANYTHING_ML)."""
    root = os.environ.get("SELL_ANYTHING_ML")
    if not root:
        pytest.skip("SELL_ANYTHING_ML is not set")
    for path in (root, os.path.join(root, "tools")):
        if path not in sys.path:
            sys.path.append(path)
    return importlib.import_module(module)


def random_strokes(n: int, side: int, size: tuple[float, float], seed: int) -> StrokeParams:
    g = torch.Generator().manual_seed(seed)
    u = lambda *shape: torch.rand(*shape, generator=g)  # noqa: E731
    return StrokeParams(xy=u(n, 2) * side, lw=torch.tensor(size) * (0.5 + u(n, 2)), angle=(u(n) - 0.5) * 2 * math.pi,
                        colour=u(n, 3), alpha=0.3 + 0.7 * u(n))


SIZES = {"wet": (40, 14), "drag": (40, 14), "stick": (60, 30), "pencil": (30, 2), "whisker": (60, 1.6),
         "rake": (20, 4), "liner": (12, 3), "dot": (8, 6)}


# ------------------------------------------------------------------ unit tests
def test_hash01_is_deterministic_and_in_the_unit_interval():
    k = torch.arange(1000.0)
    a, b = hash01(k, k * 3), hash01(k, k * 3)
    assert torch.equal(a, b)
    assert float(a.min()) >= 0 and float(a.max()) < 1
    assert len(torch.unique(a)) > 990


def test_strokes_in_the_same_token_bin_share_a_seed():
    a = seed_of(np.array([[100.1, 50.2]]), np.array([0.30]))
    b = seed_of(np.array([[100.9, 50.9]]), np.array([0.31]))     # same 2 px cell, same angle bin
    c = seed_of(np.array([[103.0, 50.2]]), np.array([0.30]))
    assert torch.equal(a, b) and not torch.equal(a, c)


@pytest.mark.parametrize("kind", KINDS)
def test_every_brush_covers_0_to_1_and_has_a_gradient_for_each_value(kind):
    p = random_strokes(3, 128, SIZES[kind], 0)
    params = [t.clone().requires_grad_(True) for t in p]
    q = StrokeParams(*params)
    start = torch.linspace(0.2, 0.9, 128)[None, :, None].expand(128, 128, 3).contiguous()
    out = lay(start, start, q, stroke_seeds(q, 128), Brush(kind, *SIZES[kind]))
    out.sum().backward()
    cov, _ = coverage(kind, p, stroke_seeds(p, 128), torch.rand(3, 9, 9, 2) * 128)
    assert float(cov.min()) >= 0 and float(cov.max()) <= 1
    for name, t in zip(StrokeParams._fields, params, strict=True):
        if kind == "drag" and name == "colour":           # drag carries no paint of its own
            assert t.grad is None or float(t.grad.abs().sum()) == 0
            continue
        assert t.grad is not None and torch.isfinite(t.grad).all() and float(t.grad.abs().sum()) > 0, name


def test_clear_strokes_leave_the_canvas_unchanged():
    p = random_strokes(4, 64, (20, 8), 1)._replace(alpha=torch.zeros(4))
    canvas = torch.rand(64, 64, 3)
    assert torch.equal(lay_layer(canvas, p, Brush("wet", 20, 8)), canvas)


def test_an_opaque_liner_paints_its_colour_at_its_centre():
    p = StrokeParams(torch.tensor([[32.0, 32.0]]), torch.tensor([[20.0, 6.0]]), torch.zeros(1),
                     torch.tensor([[0.2, 0.4, 0.6]]), torch.ones(1))
    out = render([(Brush("liner", 20, 6), p)], side=64)
    assert torch.allclose(out[32, 32], torch.tensor([0.2, 0.4, 0.6]), atol=1e-3)
    assert torch.equal(out[0, 0], torch.ones(3))


def test_an_unknown_brush_is_an_error():
    p = random_strokes(1, 32, (10, 4), 2)
    with pytest.raises(ValueError):
        coverage("crayon", p, stroke_seeds(p, 32), torch.zeros(1, 3, 3, 2))


# ------------------------------------------------------------------ equivalence with the reference
def test_seeds_match_the_reference():
    BL = reference("brush_lab")
    xy, angle = np.random.default_rng(0).uniform(0, 512, (500, 2)), np.random.default_rng(1).uniform(-7, 7, 500)
    assert torch.equal(BL.seed_of(xy, angle, 512), seed_of(xy, angle, 512))


@pytest.mark.parametrize("kind", KINDS)
def test_laying_random_strokes_matches_the_reference(kind):
    PS = reference("paint_stages")
    B = reference("sellanything_ml.sketch.catbrush")
    length, width = SIZES[kind]
    p = random_strokes(12, 256, (length, width), 3)
    start = torch.rand(256, 256, 3, generator=torch.Generator().manual_seed(4))
    canvas = torch.rand(256, 256, 3, generator=torch.Generator().manual_seed(5))
    seed = stroke_seeds(p, 256)
    ref_stage = PS.Stage("test", kind, len(p.xy), length, width, (0.3, 1.0), None, "error")
    ref_p = B.StrokeParams(*[t.clone().requires_grad_(True) for t in p])
    ref = PS.layer(canvas, start, ref_p, seed, ref_stage)
    mine_p = StrokeParams(*[t.clone().requires_grad_(True) for t in p])
    mine = lay(canvas, start, mine_p, seed, Brush(kind, length, width))
    assert torch.equal(ref, mine)
    ref.square().sum().backward()
    mine.square().sum().backward()
    for a, b in zip(ref_p, mine_p, strict=True):
        assert (a.grad is None and b.grad is None) or torch.equal(a.grad, b.grad)
