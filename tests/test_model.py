import importlib
import os
import sys
from pathlib import Path
from typing import cast

import pytest
import torch

from brushgpt.layers import DecoderLayerCache
from brushgpt.model import (
    COLOUR_CHOICE,
    BrushGPT,
    BrushGPTConfig,
    GrammarState,
    StrokeGrammar,
    TokenLayout,
    convert_reference_state_dict,
    drop_attributes,
    load_checkpoint,
)

SPECIALS = [
    "BOS",
    "EOS",
    "SEP",
    "EYE_L",
    "EYE_R",
    "LAYER0",
    "LAYER1",
    "LAYER2",
    "COND_blue",
    "COND_green",
    "COND_unclear",
    "COND_BASE_white",
    "COND_PAT_solid",
    "TASK_GEN",
    "PLAN_HEAD",
    "PLAN_END",
    "PAL_ALL",
]
SP = {n: i for i, n in enumerate(SPECIALS)}
SIZES = (8, 8, 4, 3, 6, 5)  # x, y, angle, size, paint, colour
OFFSETS = tuple(len(SPECIALS) + sum(SIZES[:i]) for i in range(len(SIZES)))
VOCAB = OFFSETS[-1] + SIZES[-1]
SLOTS = tuple(SP[n] for n in ("EYE_L", "EYE_R", "LAYER0", "LAYER1", "LAYER2", "PLAN_HEAD", "PLAN_END", "EOS"))
LAYOUT = TokenLayout(
    bos=SP["BOS"],
    slots=SLOTS,
    field_offsets=OFFSETS,
    field_sizes=SIZES,
    plan_markers=(SP["PLAN_HEAD"],),
    palette=((SP["PAL_ALL"], 3),),
)
# The same vocabulary with the palette marker among the slots, so sampling can open a palette.
PAL_LAYOUT = TokenLayout(
    bos=SP["BOS"],
    slots=(*SLOTS[:-1], SP["PAL_ALL"], SP["EOS"]),
    field_offsets=OFFSETS,
    field_sizes=SIZES,
    plan_markers=(SP["PLAN_HEAD"],),
    palette=((SP["PAL_ALL"], 3),),
)


def reference(module: str):
    """A module of the research code this package was extracted from (path in $SELL_ANYTHING_ML)."""
    root = os.environ.get("SELL_ANYTHING_ML")
    if not root:
        pytest.skip("SELL_ANYTHING_ML is not set")
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(module)


def painting(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """A short painting in the token format: a plan outline, a palette, two layers of strokes."""
    g = torch.Generator().manual_seed(seed)

    def field(f: int) -> int:
        return OFFSETS[f] + int(torch.randint(SIZES[f], (1,), generator=g))

    toks, flds = [SP["PLAN_HEAD"]], [0]
    for _ in range(3):
        toks += [field(0), field(1)]
        flds += [1, 2]
    toks += [SP["PLAN_END"], SP["PAL_ALL"]] + [field(5) for _ in range(3)]
    flds += [0, 0, 6, 6, 6]
    for layer, n in (("LAYER0", 3), ("LAYER2", 2)):
        toks.append(SP[layer])
        flds.append(0)
        for _ in range(n):
            toks += [field(f) for f in range(5)]
            flds += [1, 2, 3, 4, 5]
    return torch.tensor(toks), torch.tensor(flds)


COND = torch.tensor([SP["TASK_GEN"], SP["COND_blue"], SP["COND_green"], SP["COND_BASE_white"], SP["COND_PAT_solid"]])


def tiny(layout: TokenLayout = LAYOUT, dropout: float = 0.0, null: bool = True, seed: int = 0) -> BrushGPT:
    torch.manual_seed(seed)
    cfg = BrushGPTConfig(
        vocab_size=VOCAB + int(null),
        d_model=32,
        n_heads=4,
        enc_layers=2,
        dec_layers=2,
        max_len=256,
        dropout=dropout,
        null_token=null,
    )
    return BrushGPT(cfg, layout)


# ------------------------------------------------------------------------------------------------- unit tests
def test_target_choices_index_each_token_within_its_choice_set():
    toks, flds = painting()
    choice, index = LAYOUT.target_choices(toks, flds)
    start_ids = LAYOUT.stroke_start_ids("cpu")
    for t, f, c, i in zip(toks.tolist(), flds.tolist(), choice.tolist(), index.tolist(), strict=True):
        if f <= 1 and t != SP["PAL_ALL"]:
            assert c == 0 and int(start_ids[i]) == t
        elif f >= 2:
            assert c == f - 1 and OFFSETS[c] + i == t


def test_a_marker_outside_the_slots_maps_to_the_first_slot():
    choice, index = LAYOUT.target_choices(torch.tensor([SP["PAL_ALL"]]), torch.tensor([0]))
    assert choice.tolist() == [0] and index.tolist() == [0]


def test_grammar_follows_a_painting_written_in_the_format():
    toks, flds = painting()
    toks = torch.cat([toks, torch.tensor([SP["EOS"]])])
    flds = torch.cat([flds, torch.tensor([0])])
    grammar = StrokeGrammar(PAL_LAYOUT)
    state = grammar.start(1, "cpu")
    for t, f in zip(toks.tolist(), flds.tolist(), strict=True):
        expected = 0 if f <= 1 else f - 1
        assert int(grammar.choice(state)[0]) == expected
        state = grammar.advance(state, torch.tensor([t]))


def test_grammar_counts_palette_colours():
    grammar = StrokeGrammar(PAL_LAYOUT)
    state = grammar.start(1, "cpu")
    seen = []
    for t in [SP["PAL_ALL"], OFFSETS[5], OFFSETS[5] + 1, OFFSETS[5] + 2, SP["LAYER0"]]:
        state = grammar.advance(state, torch.tensor([t]))
        seen.append(int(state.choice[0]))
    assert seen == [COLOUR_CHOICE, COLOUR_CHOICE, COLOUR_CHOICE, 0, 0]


def test_loss_is_a_mean_over_every_target_and_its_eos():
    model = tiny()
    toks, flds = painting()
    loss, nll, right = model.loss(COND, toks, flds)
    assert nll.shape == right.shape == (len(toks) + 1,)
    assert torch.allclose(loss, nll.mean(), atol=1e-6)
    loss.backward()
    assert model.embed.weight.grad is not None


def test_cached_decoding_matches_the_full_decoder():
    model = tiny().eval()
    toks, _ = painting()
    inp = torch.cat([torch.tensor([LAYOUT.bos]), toks])[None]
    memory = model.encode(COND[None])
    full = model.decode(inp, memory)
    cache = [DecoderLayerCache.empty(64) for _ in model.decoder]
    steps = [model.decode(inp[:, :1], memory, 0, cache)]
    steps += [model.decode(inp[:, t : t + 1], memory, t, cache) for t in range(1, inp.size(1))]
    assert torch.allclose(torch.cat(steps, 1), full, atol=1e-5)


def test_samples_follow_the_grammar_and_end_with_eos():
    model = tiny(PAL_LAYOUT).eval()
    cond = COND.repeat(4, 1)
    out = model.sample(cond, 60, generator=torch.Generator().manual_seed(0))
    assert out.shape == (4, 60)
    grammar = StrokeGrammar(PAL_LAYOUT)
    state = grammar.start(4, "cpu")
    allowed = [PAL_LAYOUT.stroke_start_ids("cpu")] + [
        torch.arange(OFFSETS[c], OFFSETS[c] + SIZES[c]) for c in range(1, len(SIZES))
    ]
    for i in range(60):
        tok = out[:, i]
        live = tok >= 0
        for r in torch.nonzero(live).flatten().tolist():
            assert int(tok[r]) in allowed[int(state.choice[r])]
        state = grammar.advance(state, torch.where(live, tok, torch.full_like(tok, PAL_LAYOUT.eos)))
    for row in out:
        ended = (row == PAL_LAYOUT.eos).nonzero()
        if len(ended):
            assert torch.all(row[int(ended[0]) + 1 :] == -1)


def test_guidance_one_is_the_conditional_model():
    model = tiny().eval()
    cond = COND.repeat(2, 1)
    uncond = drop_attributes(cond, model.cfg.null_id)
    a = model.sample(cond, 30, generator=torch.Generator().manual_seed(5))
    b = model.sample(cond, 30, generator=torch.Generator().manual_seed(5), uncond=uncond, guidance=1.0)
    assert torch.equal(a, b)


def test_drop_attributes_keeps_the_task_token():
    out = drop_attributes(COND, 99)
    assert out[0] == COND[0] and torch.all(out[1:] == 99) and torch.all(COND[1:] != 99)


def test_checkpoint_round_trip(tmp_path: Path):
    model = tiny()
    torch.save({"config": model.cfg.to_dict(), "model": model.state_dict()}, tmp_path / "m.pt")
    again = load_checkpoint(torch.load(tmp_path / "m.pt"), LAYOUT)
    toks, flds = painting()
    assert torch.equal(model.eval().loss(COND, toks, flds)[0], again.eval().loss(COND, toks, flds)[0])


# ----------------------------------------------------------------------------------- equivalence with the reference
def reference_model(layout: TokenLayout = LAYOUT, dropout: float = 0.0, null: bool = True, seed: int = 0):
    ea = reference("sellanything_ml.sketch.edit_arch")
    cfg = ea.Config(
        arch="encdec",
        layout="free",
        d_model=32,
        n_heads=4,
        enc_layers=2,
        dec_layers=2,
        max_len=256,
        vocab=VOCAB + int(null),
        dropout=dropout,
        cond_drop=0.15 if null else 0.0,
    )
    torch.manual_seed(seed)
    ref = ea.EditModel(
        cfg, list(layout.field_offsets), list(layout.field_sizes), list(layout.slots), list(layout.plan_markers)
    )
    ref.palette_runs = dict(layout.palette)
    return ref


def test_same_seed_gives_the_reference_weights_under_the_documented_key_mapping():
    ref, mine = reference_model(), tiny()
    a, b = convert_reference_state_dict(ref.state_dict()), mine.state_dict()
    assert list(a) == list(b)
    assert all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_loss_and_gradients_match_the_reference(dropout: float):
    ref, mine = reference_model(dropout=dropout), tiny(dropout=dropout)
    toks, flds = painting(3)
    torch.manual_seed(7)
    r_loss, r = ref.loss(COND, toks.new_zeros(0), toks, flds, SP["SEP"], SP["BOS"])
    torch.manual_seed(7)
    m_loss, m_nll, m_right = mine.loss(COND, toks, flds)
    assert torch.equal(r_loss, m_loss) and torch.equal(r["nll"], m_nll) and torch.equal(r["right"], m_right)
    r_loss.backward()
    m_loss.backward()
    ga = convert_reference_state_dict({n: p.grad for n, p in ref.named_parameters()})
    gb = cast(dict[str, torch.Tensor], {n: p.grad for n, p in mine.named_parameters()})  # set by backward()
    assert all(torch.equal(ga[k], gb[k]) for k in ga)


def test_reference_state_dict_loads_and_gives_the_same_outputs():
    ref = reference_model(seed=11).eval()
    ck = {"config": {**vars(ref.cfg)}, "model": ref.state_dict()}
    mine = load_checkpoint(ck, LAYOUT).eval()
    toks, flds = painting(4)
    assert torch.equal(ref.loss(COND, toks.new_zeros(0), toks, flds, 2, 0)[0], mine.loss(COND, toks, flds)[0])


@pytest.mark.parametrize("layout", [LAYOUT, PAL_LAYOUT])
@pytest.mark.parametrize("guidance,temperature", [(1.0, 1.0), (2.5, 1.0), (3.0, 0.7)])
def test_sampling_matches_the_reference(layout: TokenLayout, guidance: float, temperature: float):
    ref, mine = reference_model(layout).eval(), tiny(layout).eval()
    cond = torch.stack([COND, COND, COND])
    cond[1, 1] = SP["COND_unclear"]
    assert mine.cfg.null_id is not None  # tiny() has a null token
    uncond = drop_attributes(cond, mine.cfg.null_id)
    a = ref.sample_free(
        cond,
        cond.new_zeros(0),
        SP["SEP"],
        SP["BOS"],
        temperature,
        torch.Generator().manual_seed(4),
        n_max=80,
        uncond=uncond,
        guidance=guidance,
    )
    b = mine.sample(cond, 80, temperature, torch.Generator().manual_seed(4), uncond, guidance)
    assert torch.equal(a, b)


def test_grammar_matches_the_reference_on_random_streams():
    ref = reference_model(PAL_LAYOUT)
    grammar = StrokeGrammar(PAL_LAYOUT)
    marks = (
        torch.tensor(ref.plan_runs),
        torch.tensor(list(ref.palette_runs)),
        torch.tensor(list(ref.palette_runs.values())),
    )
    g = torch.Generator().manual_seed(0)
    b = 16
    step, inplan, left = (
        torch.zeros(b, dtype=torch.long),
        torch.zeros(b, dtype=torch.bool),
        torch.zeros(b, dtype=torch.long),
    )
    state = grammar.start(b, "cpu")
    start_ids = PAL_LAYOUT.stroke_start_ids("cpu")
    for _ in range(200):
        tok = torch.empty(b, dtype=torch.long)
        for r in range(b):
            c = int(step[r])
            pool = start_ids if c == 0 else torch.arange(OFFSETS[c], OFFSETS[c] + SIZES[c])
            tok[r] = pool[torch.randint(len(pool), (1,), generator=g)]
        step, (inplan, left) = ref.advance(step, tok, (inplan, left), marks)
        state = grammar.advance(state, tok)
        assert isinstance(state, GrammarState)
        assert torch.equal(step, state.choice) and torch.equal(inplan, state.in_plan)
        assert torch.equal(left, state.palette_left)


def test_real_checkpoint_matches_the_reference_on_a_real_painting():
    ea = reference("sellanything_ml.sketch.edit_arch")
    root = Path(os.environ["SELL_ANYTHING_ML"])
    ckpt, data = root / "data/kaggle/gen_more/out/out/more/best.pt", root / "data/processed/edit_1109/data.npz"
    if not (ckpt.exists() and data.exists()):
        pytest.skip("the real checkpoint or its data is not there")
    pairs = ea.Pairs(data, "cpu")
    ck = torch.load(ckpt, map_location="cpu")
    ref = ea.EditModel(ea.Config(**ck["config"]), pairs.offsets, pairs.sizes, pairs.slots, pairs.plan_runs).eval()
    ref.palette_runs = pairs.palette_runs
    ref.load_state_dict(ck["model"])
    layout = TokenLayout(
        bos=pairs.specials.index("BOS"),
        slots=tuple(pairs.slots),
        field_offsets=tuple(pairs.offsets),
        field_sizes=tuple(pairs.sizes),
        plan_markers=tuple(pairs.plan_runs),
        palette=tuple(pairs.palette_runs.items()),
    )
    mine = load_checkpoint(ck, layout).eval()
    sep, bos = pairs.specials.index("SEP"), pairs.specials.index("BOS")
    item = pairs.gens["held"][0]
    cond, old, new, field = pairs.pair(-1, item)
    with torch.no_grad():
        r_loss, r = ref.loss(cond, old, new, field, sep, bos)
        m_loss, m_nll, m_right = mine.loss(cond, new, field)
    assert torch.equal(r_loss, m_loss) and torch.equal(r["nll"], m_nll) and torch.equal(r["right"], m_right)
    conds = torch.stack([pairs.pair(-1, i)[0] for i in pairs.gens["held"][:2]])
    a = ref.sample_free(conds, conds.new_zeros(0), sep, bos, 1.0, torch.Generator().manual_seed(0), n_max=40)
    b = mine.sample(conds, 40, 1.0, torch.Generator().manual_seed(0))
    assert torch.equal(a, b)
