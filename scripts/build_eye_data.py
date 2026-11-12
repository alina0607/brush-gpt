"""Build the eye part model's training data from painted cats.

    python scripts/build_eye_data.py --cats cats.csv --paintings runs/paint --eye-versions runs/eye_versions \
        --real-eyes real_eyes.jsonl --review review.csv --split split.json --duplicates duplicates.json --out data/eyes

cats.csv lists the cats (name, cutout, photo, plan; as scripts/paint.py takes them); each cat's painting is
PAINTINGS/<name>/stage1.npz .. stage7.npz. --review keeps only the cats a review kept for the eyes (a CSV with the
columns source, id, eyes; eyes == "used"). --split takes the whole-face data's split so both models are scored on the
same held cats; --duplicates moves a held cat whose photo is also in training to train. Writes OUT/data.npz and
OUT/stats.json (see brushgpt.eye_data).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from brushgpt.attributes import eye_targets
from brushgpt.dataset import read_cat_table, read_duplicate_pairs, read_split
from brushgpt.eye_data import build, reviewed_for_eyes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cats", required=True, help="CSV of cats: name, cutout, photo, plan")
    ap.add_argument("--paintings", required=True, help="folder with one painting folder per cat")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--eye-versions", help="folder with <cat>.json and <cat>.npz eye-colour versions")
    ap.add_argument("--real-eyes", help="JSON lines of labelled real eyes (needed with --eye-versions)")
    ap.add_argument("--review", help="CSV of reviewed cats (source, id, eyes): only those with eyes == used")
    ap.add_argument("--split", help="JSON split per cat (cats it does not name train)")
    ap.add_argument("--duplicates", help="JSON pairs of cats that are the same photo")
    ap.add_argument("--limit", type=int, default=0, help="only the first N cats (a quick build)")
    ap.add_argument("--threads", type=int, default=3, help="worker processes")
    a = ap.parse_args()
    if a.eye_versions and not a.real_eyes:
        ap.error("--eye-versions needs --real-eyes")
    torch.set_num_threads(1 if a.threads > 1 else 3)
    cats = read_cat_table(a.cats, a.paintings)
    if a.review:
        kept = reviewed_for_eyes(a.review)
        cats = [c for c in cats if c.name in kept]
    stats = build(cats, Path(a.out), split=read_split(a.split) if a.split else None,
                  duplicates=read_duplicate_pairs(a.duplicates) if a.duplicates else (),
                  eye_versions=Path(a.eye_versions) if a.eye_versions else None,
                  eye_targets=eye_targets(a.real_eyes) if a.real_eyes else None, limit=a.limit, threads=a.threads,
                  log=lambda row: print(json.dumps(row), flush=True))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
