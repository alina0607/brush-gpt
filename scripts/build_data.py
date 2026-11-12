"""Build Brush GPT's training data from painted cats.

    python scripts/build_data.py --cats cats.csv --paintings runs/paint --coat-labels coat_labels.jsonl \
        --eye-versions runs/eye_versions --real-eyes real_eyes.jsonl --split split.json \
        --ranges-from data/gen_first/data.npz --part-order --plan --palette --mirror --out data/gen_furpal

cats.csv lists the cats (name, cutout, photo, plan; as scripts/paint.py takes them); each cat's painting is
PAINTINGS/<name>/stage1.npz .. stage7.npz. The final format is --part-order --plan, with --palette for the fur palette;
--mirror and --eye-versions add each cat's mirrored and eye-recoloured paintings. Writes OUT/data.npz and
OUT/data.json (see brushgpt.dataset).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from brushgpt.attributes import eye_targets, read_coat_labels
from brushgpt.dataset import build, load_ranges, read_cat_table, read_duplicate_pairs, read_split
from brushgpt.tokens import Vocabulary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cats", required=True, help="CSV of cats: name, cutout, photo, plan")
    ap.add_argument("--paintings", required=True, help="folder with one painting folder per cat")
    ap.add_argument("--coat-labels", required=True, help="JSON lines: the coat reader's reading of each cat")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--eye-versions", help="folder with <cat>.json and <cat>.npz eye-colour versions")
    ap.add_argument("--real-eyes", help="JSON lines of labelled real eyes (needed with --eye-versions)")
    ap.add_argument("--split", help="JSON split per cat (cats it does not name train)")
    ap.add_argument("--duplicates", help="without --split: JSON pairs of cats that are the same photo")
    ap.add_argument("--ranges-from", help="an earlier build's data.npz whose bins this build uses")
    ap.add_argument("--part-order", action="store_true", help="write each painting part by part")
    ap.add_argument("--plan", action="store_true", help="write each painting's sketch plan before its strokes")
    ap.add_argument("--palette", action="store_true", help="write the fur strokes on a 16-colour palette")
    ap.add_argument("--mirror", action="store_true", help="also every painting flipped left to right")
    ap.add_argument("--limit", type=int, default=0, help="only the first N cats (a quick build)")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    if a.palette and not a.part_order:
        ap.error("--palette needs --part-order")
    if a.eye_versions and not a.real_eyes:
        ap.error("--eye-versions needs --real-eyes")
    torch.set_num_threads(a.threads)
    summary = build(
        read_cat_table(a.cats, a.paintings), read_coat_labels(a.coat_labels), Path(a.out),
        Vocabulary.build(a.part_order, a.plan, a.palette),
        split=read_split(a.split) if a.split else None,
        duplicates=read_duplicate_pairs(a.duplicates) if a.duplicates else (),
        ranges=load_ranges(a.ranges_from) if a.ranges_from else None,
        eye_versions=Path(a.eye_versions) if a.eye_versions else None,
        eye_targets=eye_targets(a.real_eyes) if a.real_eyes else None,
        mirror=a.mirror, limit=a.limit, log=lambda row: print(json.dumps(row), flush=True))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
