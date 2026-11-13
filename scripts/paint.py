"""Paint cats with the data painter: seven stages of fitted brush strokes, eyes apart.

One cat:

    python scripts/paint.py --cutout CUTOUT.npz --photo PHOTO.png --plan PLAN.npz --out runs/paint/CAT

A list of cats (a CSV with the columns name, cutout, photo, plan), each into OUT/<name>; a cat whose stage7.npz
already exists is skipped, so an interrupted run can be started again:

    python scripts/paint.py --list cats.csv --out runs/paint --device mps

Each cat's folder gets stage1.npz .. stage7.npz (see brushgpt.painter) and log.jsonl (one row a stage).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

from brushgpt.cat import load_inputs
from brushgpt.painter import paint_cat


def paint_one(name: str, cutout: str, photo: str, plan: str, out: Path, device: str) -> None:
    t0 = time.time()
    paint_cat(
        load_inputs(cutout, photo, plan),
        out,
        device,
        on_row=lambda row: print(json.dumps({"cat": name, **row}), flush=True),
    )
    print(json.dumps({"cat": name, "done": True, "seconds": round(time.time() - t0, 1)}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cutout", help="the cat's cut-out .npz (target, tri, points, box)")
    ap.add_argument("--photo", help="the original photo")
    ap.add_argument("--plan", help="the cat's sketch plan .npz")
    ap.add_argument("--list", help="CSV of cats to paint: name, cutout, photo, plan")
    ap.add_argument("--out", required=True, help="the cat's folder (one cat) or the parent folder (--list)")
    ap.add_argument("--device", default="cpu", help="where strokes are fitted: cpu, mps or cuda")
    ap.add_argument("--threads", type=int, default=6, help="CPU threads")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    if a.list:
        with open(a.list, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if (out / r["name"] / "stage7.npz").exists():
                continue
            try:
                paint_one(r["name"], r["cutout"], r["photo"], r["plan"], out / r["name"], a.device)
            except Exception as e:  # noqa: BLE001 - one failed cat should not stop the list
                print(json.dumps({"cat": r["name"], "error": repr(e)}), file=sys.stderr, flush=True)
        return
    if not (a.cutout and a.photo and a.plan):
        ap.error("give --cutout, --photo and --plan, or --list")
    paint_one(out.name, a.cutout, a.photo, a.plan, out, a.device)


if __name__ == "__main__":
    main()
