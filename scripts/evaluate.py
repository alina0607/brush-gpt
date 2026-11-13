"""Judge sampled paintings on held-out cats (see brushgpt.evaluate). Three modes:

    # the whole-face model's samples (scripts/sample.py) against the painter's paintings, every attribute
    python scripts/evaluate.py whole runs/furpal/gen_g2.npz --whole-data data/gen_furpal/data.npz --cats cats.csv \
        --coat-features coat/clip.npz --coat-colours coat/colour.npz --coat-labels coat/labels.jsonl

    # the eye model's eyes (scripts/sample_eyes.py) in the painter's paintings
    python scripts/evaluate.py alone runs/eyes/eyes_g2.npz --eye-data data/eyes/data.npz --cats cats.csv \
        --paintings runs/paint

    # the whole-face model's samples with their eyes repainted by the eye model
    python scripts/evaluate.py system runs/furpal/gen_g2.npz --whole-data data/gen_furpal/data.npz \
        --eye-data data/eyes/data.npz --ckpt runs/eyes/best.pt --guidance 2 --cats cats.csv

cats.csv lists the cats (name, cutout, photo, plan); the readers need each cat's inputs, `alone` also its painting
(PAINTINGS/<name>). Writes the measures as JSON next to the samples (SAMPLES.measure.json, .alone.json or
.system_g<g>.json; or --out) and, for the eye modes, a contact sheet of eye crops with the same name (.png).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from brushgpt.attributes import CoatReader
from brushgpt.dataset import read_cat_table
from brushgpt.evaluate import (
    CatViews,
    WholeFaceData,
    contact_sheet,
    measure_eyes_alone,
    measure_system,
    measure_whole_face,
    read_whole_samples,
)
from brushgpt.eye_data import load_eye_ranges
from brushgpt.eye_model import EyeData, load_checkpoint, resolve_device

SHEET_ROWS = 16


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["whole", "alone", "system"])
    ap.add_argument("samples", help="the samples .npz")
    ap.add_argument("--cats", required=True, help="CSV of cats: name, cutout, photo, plan")
    ap.add_argument("--paintings", default=".", help="alone: folder with one painting folder per cat")
    ap.add_argument("--whole-data", help="whole, system: the whole-face model's data file")
    ap.add_argument("--eye-data", help="alone, system: the eye model's data file")
    ap.add_argument("--ckpt", help="system: the eye model")
    ap.add_argument("--coat-features", help="whole: the coat reader's CLIP features (.npz)")
    ap.add_argument("--coat-colours", help="whole: the coat reader's colour statistics (.npz)")
    ap.add_argument("--coat-labels", help="whole: the coat reader's labels (JSON lines)")
    ap.add_argument("--guidance", type=float, default=2.0, help="system: the eye model's guidance weight")
    ap.add_argument("--temperature", type=float, default=1.0, help="system: the eye model's temperature")
    ap.add_argument("--max-new", type=int, default=1200, help="system: most tokens per eye")
    ap.add_argument("--n-cats", type=int, default=32, help="alone, system: the first N cats of the samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="system: where the eye model samples")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", help="the JSON (default: next to the samples)")
    a = ap.parse_args()
    need = {
        "whole": ("whole_data", "coat_features", "coat_colours", "coat_labels"),
        "alone": ("eye_data",),
        "system": ("whole_data", "eye_data", "ckpt"),
    }[a.mode]
    missing = [n for n in need if getattr(a, n) is None]
    if missing:
        ap.error(f"{a.mode} needs " + ", ".join("--" + n.replace("_", "-") for n in missing))
    torch.set_num_threads(a.threads)
    sources = {c.name: c for c in read_cat_table(a.cats, a.paintings)}
    views = CatViews(sources)
    suffix = {"whole": ".measure.json", "alone": ".alone.json", "system": f".system_g{a.guidance:g}.json"}[a.mode]
    out = Path(a.out or Path(a.samples).with_suffix(suffix))
    log = lambda row: print(json.dumps(row), flush=True)  # noqa: E731
    pics = None
    if a.mode == "whole":
        tokens, rows = read_whole_samples(a.samples)
        reader = CoatReader(a.coat_features, a.coat_colours, a.coat_labels)
        res = {
            "samples": a.samples,
            **measure_whole_face(tokens, rows, WholeFaceData(a.whole_data), views, reader, log),
        }
        summary = {k: v for k, v in res.items() if k != "rows"}
    elif a.mode == "alone":
        res, pics = measure_eyes_alone(np.load(a.samples), np.load(a.eye_data), sources, views, a.n_cats)
        summary = res["summary"]
    else:
        tokens, rows = read_whole_samples(a.samples)
        model = load_checkpoint(a.ckpt, resolve_device(a.device))
        res, pics = measure_system(
            tokens,
            rows,
            WholeFaceData(a.whole_data),
            model,
            EyeData(a.eye_data),
            load_eye_ranges(a.eye_data),
            views,
            a.guidance,
            a.temperature,
            a.max_new,
            a.n_cats,
            a.seed,
        )
        summary = res["summary"]
    out.write_text(json.dumps(res, indent=1))
    if pics:
        contact_sheet(pics[:SHEET_ROWS], out.with_suffix(".png"))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
