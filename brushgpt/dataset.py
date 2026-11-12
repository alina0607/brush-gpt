"""Building the training data: every painted cat's paintings as token sequences with their conditions.

Each cat gives one or more paintings (versions):

  A        the data painter's painting of the photo
  eye:<v>  (with eye versions) the same painting with only the eyes repainted for a photo whose eyes were recoloured
           (`painter.paint_eye_versions`); whole paintings that differ only in the eyes
  <v>~m    (with mirroring) each of the above flipped left to right: x -> 512 - x, angle -> pi - angle, the two eyes'
           layers swapped (the image's left eye is still written first), the eye conditions swapped and the plan and
           part map flipped. Paintings are never flipped upside down.

A version's condition is four attribute names (`brushgpt.attributes`): the left and right eye's colour read on the
photo it was painted from (the recoloured photo for an eye version) and the coat's base colour and pattern from the
coat reader's reading of the cat's photo. Strokes with a non-finite value are dropped (a painting fitted on a GPU
can have a few), and cats whose finished canvas is not finite are left out.

The split is by cat: a given split (cats it does not name train), or HOLD_SHARE of the cats held out (seed 0), with
cats that are the same photo kept on the same side. The bins of each layer's size and opacity span the training
cats' A paintings, or are taken from an earlier build (so that a token means the same stroke in both).

Output (data.npz): tokens and fields per painting (padded with -1), cond (4 condition token ids), cat_id, group,
kind ("gen"), versions, cats, split per cat, ranges (21 x 6), vocab, specials, field_names, field_sizes and
palette_k; data.json: counts and token lengths.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .attributes import coat_names, eye_names, eye_view, recolour_eyes
from .brushes import StrokeParams
from .cat import INPUT_SIDE, load_inputs
from .painter import EYE_STAGE, SIDE, STAGES
from .tokens import PaintedLayer, Ranges, Vocabulary, painting_tokens, part_map, ranges_from_array, ranges_to_array

HOLD_SHARE, SPLIT_SEED = 0.15, 0
KIND = "gen"

Version = tuple[str, list[PaintedLayer], list[str]]    # (name, layers in painting order, condition names)


@dataclass(frozen=True)
class CatSource:
    """Where one cat's inputs are: its cut-out, its photo and its sketch plan (as `scripts/paint.py` takes them),
    and the folder with its painting (stage1.npz .. stage7.npz)."""

    name: str
    cutout: Path
    photo: Path
    plan: Path
    painting: Path


def read_cat_table(path: Path | str, paintings: Path | str) -> list[CatSource]:
    """A CSV with the columns name, cutout, photo, plan; each cat's painting is paintings/<name>."""
    with open(path, newline="") as f:
        return [CatSource(r["name"], Path(r["cutout"]), Path(r["photo"]), Path(r["plan"]), Path(paintings) / r["name"])
                for r in csv.DictReader(f)]


# ------------------------------------------------------------------ paintings
def stored_layers(arrays: Mapping[str, np.ndarray], stage: int, prefix: str = ""
                  ) -> list[tuple[int, int, StrokeParams, np.ndarray | None]]:
    """A stage's stored layers (stage, j, strokes, each stroke's eye side or None); `prefix`: a version's stage 2
    in an eye-versions file."""
    files = set(arrays.keys())
    out = []
    for j in range(len(STAGES[stage])):
        if f"{prefix}{j}_xy" in files:
            p = StrokeParams(*(torch.from_numpy(np.asarray(arrays[f"{prefix}{j}_{f}"])) for f in StrokeParams._fields))
            side = np.asarray(arrays[f"{prefix}{j}_side"]) if f"{prefix}{j}_side" in files else None
            out.append((stage, j, p, side))
    return out


def version_layers(painting: Path, eyes: Mapping[str, np.ndarray] | None = None, version: str = ""
                   ) -> list[PaintedLayer]:
    """Every layer of a painting in painting order, stage 2 split by eye (all of the left eye's layers, then the
    right's). eyes / version: take stage 2 from that version of an eye-versions file instead."""
    out: list[PaintedLayer] = []
    for k in sorted(STAGES):
        with np.load(painting / f"stage{k}.npz") as z:
            if k != EYE_STAGE:
                out += [(st, j, p, None) for st, j, p, _ in stored_layers(z, k)]
                continue
            two = stored_layers(z, k) if eyes is None else stored_layers(eyes, k, f"{version}/")
        for side in (0, 1):
            for st, j, p, s in two:
                keep = torch.from_numpy(s == side)
                if keep.any():
                    out.append((st, j, StrokeParams(*(t[keep] for t in p)), side))
    return out


def finite(layers: list[PaintedLayer]) -> tuple[list[PaintedLayer], int]:
    """The layers without their strokes that have a non-finite value (a layer left empty is dropped), and how many
    strokes were dropped."""
    out, dropped = [], 0
    for st, j, p, side in layers:
        k = torch.ones(len(p.xy), dtype=torch.bool)
        for t in p:
            k &= torch.isfinite(t.reshape(len(t), -1)).all(1)
        dropped += int((~k).sum())
        if k.any():
            out.append((st, j, StrokeParams(*(t[k] for t in p)), side))
    return out, dropped


def has_finite_canvas(painting: Path) -> bool:
    with np.load(painting / "stage7.npz") as z:
        return bool(np.isfinite(z["canvas"]).all())


def layer_ranges(paintings: Iterable[Path]) -> Ranges:
    """Each layer's lowest and highest stroke length, width and opacity over these paintings (finite strokes)."""
    r: Ranges = {}
    for painting in paintings:
        layers, _ = finite(version_layers(painting))
        for st, j, p, _ in layers:
            v = torch.cat([p.lw, p.alpha[:, None]], 1).numpy()
            lo, hi = r.get((st, j), (np.full(3, np.inf), np.full(3, -np.inf)))
            r[(st, j)] = (np.minimum(lo, v.min(0)), np.maximum(hi, v.max(0)))
    return r


def mirrored(versions: list[Version]) -> list[Version]:
    """Each painting flipped left to right, named "<version>~m"."""
    out = []
    for v, layers, cond in versions:
        flip = [(st, j, StrokeParams(torch.stack([SIDE - p.xy[:, 0], p.xy[:, 1]], 1), p.lw, np.pi - p.angle,
                                     p.colour, p.alpha), None if side is None else 1 - side)
                for st, j, p, side in layers]
        at = next((i for i, x in enumerate(flip) if x[3] is not None), len(flip))
        rest = [x for x in flip if x[3] is None]
        eyes = [x for x in flip if x[3] == 0] + [x for x in flip if x[3] == 1]
        out.append((f"{v}~m", rest[:at] + eyes + rest[at:], [cond[1], cond[0], *cond[2:]]))
    return out


# ------------------------------------------------------------------ split
def read_duplicate_pairs(path: Path | str, prefix: str = "afhq__") -> list[tuple[str, str]]:
    """Pairs of cats that are the same photo ({"pairs": [{"a", "b"}]}, AFHQ ids)."""
    return [(f"{prefix}{p['a']}", f"{prefix}{p['b']}") for p in json.loads(Path(path).read_text())["pairs"]]


def split_cats(names: list[str], duplicates: Iterable[tuple[str, str]] = (), share: float = HOLD_SHARE,
               seed: int = SPLIT_SEED) -> dict[str, str]:
    """"held" or "train" per cat: groups of the same photo (joined through the duplicate pairs) are shuffled and
    held out until `share` of the cats are."""
    parent = {n: n for n in names}

    def find(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    for a, b in duplicates:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    out, held = {}, 0
    for k in keys:
        side = "held" if held < share * len(names) else "train"
        held += len(groups[k]) if side == "held" else 0
        out.update({n: side for n in groups[k]})
    return out


def read_split(path: Path | str) -> dict[str, str]:
    """A split file: JSON mapping cat -> "held" / "train", or an object with such a mapping under "split"."""
    d = json.loads(Path(path).read_text())
    return d["split"] if "split" in d else d


# ------------------------------------------------------------------ one cat
def cat_versions(source: CatSource, coat: Mapping, eye_versions: Path | None = None,
                 eye_targets: Mapping[str, np.ndarray] | None = None, mirror: bool = False) -> list[Version]:
    """One cat's paintings with their condition names: A, its eye versions (eye_versions/<name>.json and .npz;
    eye_targets: the real eyes each colour is recoloured towards), and with `mirror` all of them flipped."""
    view = eye_view(load_inputs(source.cutout, source.photo, source.plan))
    own = coat_names(coat)
    versions: list[Version] = [("A", version_layers(source.painting), eye_names(view, view["photo"]) + own)]
    info_file = eye_versions / f"{source.name}.json" if eye_versions is not None else None
    if info_file is not None and info_file.exists():
        info = json.loads(info_file.read_text())["versions"]
        if "A_cpu" in info:        # the original painted with the versions; its eye versions are comparable to it
            if eye_targets is None:
                raise ValueError("eye versions need the eye colour targets")
            with np.load(eye_versions / f"{source.name}.npz") as arrays:
                for v, row in info.items():
                    if v == "A_cpu":
                        continue
                    to = tuple(row["to"])
                    photo = recolour_eyes(view["photo"], view, eye_targets, to) if any(to) else view["photo"]
                    versions.append((f"eye:{v}", version_layers(source.painting, arrays, v),
                                     eye_names(view, photo) + own))
    return versions + (mirrored(versions) if mirror else [])


def face_points(source: CatSource, flip: bool) -> np.ndarray:
    """The cat's nine face points on the 256 grid, x flipped for a mirrored painting."""
    with np.load(source.cutout, allow_pickle=True) as z:
        p = z["points"].astype(np.float64)
    return np.c_[INPUT_SIDE - p[:, 0], p[:, 1]] if flip else p


def read_plan(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in ("yaw", "pitch", "roll", "head", "eyes", "pupils", "nose", "mouth")}


# ------------------------------------------------------------------ the build
def pad_rows(rows: list[np.ndarray], width: int, dtype: type) -> np.ndarray:
    return np.stack([np.pad(r, (0, width - len(r)), constant_values=-1).astype(dtype) for r in rows])


def build(cats: list[CatSource], coat_labels: Mapping[str, Mapping], out: Path | str, vocab: Vocabulary,
          split: Mapping[str, str] | None = None, duplicates: Iterable[tuple[str, str]] = (),
          ranges: Ranges | None = None, eye_versions: Path | None = None,
          eye_targets: Mapping[str, np.ndarray] | None = None, mirror: bool = False, limit: int = 0,
          log: Callable[[dict], None] | None = None) -> dict:
    """Build out/data.npz and out/data.json from the painted cats (in name order; cats without a finished, finite
    painting are left out; `limit`: only the first ones). The vocabulary sets the format (`Vocabulary.build`):
    with part markers the strokes go in part order, with plan tokens the sketch plan comes first."""
    out = Path(out)
    painted = sorted((c for c in cats if (c.painting / "stage7.npz").exists()), key=lambda c: c.name)
    nan_cats = [c.name for c in painted if not has_finite_canvas(c.painting)]
    painted = [c for c in painted if c.name not in nan_cats]
    painted = painted[:limit] if limit else painted
    names = [c.name for c in painted]
    split = split_cats(names, duplicates) if split is None else split
    split = {n: split.get(n, "train") for n in names}
    if ranges is None:
        ranges = layer_ranges(c.painting for c in painted if split[c.name] == "train")
    toks, fields, cond, cat_id, group, vnames = [], [], [], [], [], []
    dropped = 0
    for ci, source in enumerate(painted):
        parts = ({flip: part_map(face_points(source, flip)) for flip in (False, True)} if vocab.has_parts else None)
        plan = read_plan(source.plan) if vocab.has_plan else None
        g = len(set(group))
        for v, layers, names_of in cat_versions(source, coat_labels[source.name], eye_versions, eye_targets, mirror):
            layers, n = finite(layers)
            dropped += n
            flip = v.endswith("~m")
            t, f = painting_tokens(layers, ranges, vocab, parts[flip] if parts else None, plan, flip)
            toks.append(t)
            fields.append(f)
            cond.append(vocab.condition(names_of))
            cat_id.append(ci)
            group.append(g)
            vnames.append(v)
        if log is not None:
            log({"cat": source.name, "versions": len(toks)})
    width = max(len(t) for t in toks)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "data.npz", tokens=pad_rows(toks, width, np.int32),
                        fields=pad_rows(fields, width, np.int8), cond=np.array(cond, np.int32),
                        cat_id=np.array(cat_id), group=np.array(group),
                        kind=np.array([KIND] * len(toks)), versions=np.array(vnames), cats=np.array(names),
                        split=np.array([split[c] for c in names]), ranges=ranges_to_array(ranges),
                        vocab=np.array(vocab.size), specials=np.array(vocab.specials),
                        field_names=np.array(vocab.field_names),
                        field_sizes=np.array([vocab.field_sizes[f] for f in vocab.field_names]),
                        palette_k=vocab.palette_k)
    lens = np.array([len(t) for t in toks])
    summary = {"cats": len(names), "held": sum(split[c] == "held" for c in names), "paintings": len(toks),
               "held_paintings": sum(split[names[c]] == "held" for c in cat_id),
               "part_order": vocab.has_parts, "plan": vocab.has_plan, "palette": vocab.has_palette,
               "eye_versions": eye_versions is not None, "mirror": mirror, "strokes_dropped": dropped,
               "nan_canvas_cats_left_out": len(nan_cats),
               "tokens_min_median_max": [int(lens.min()), int(np.median(lens)), int(lens.max())], "vocab": vocab.size}
    (out / "data.json").write_text(json.dumps(summary, indent=1))
    return summary


def load_ranges(data: Path | str) -> Ranges:
    """The bins' ranges of an earlier build's data.npz."""
    with np.load(data) as z:
        return ranges_from_array(z["ranges"])
