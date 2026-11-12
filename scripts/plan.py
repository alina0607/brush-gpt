"""Make sketch plans: the head pose and the outlines a painter draws before painting (see brushgpt.plan).

One cat (its cut-out .npz holds target, tri and points on the 256 grid):

    python scripts/plan.py --cutout CUTOUT.npz --out PLAN.npz

A list of cats (a CSV with the columns name, cutout), each into OUT/<name>.npz; a cat whose plan already exists is
skipped, so an interrupted run can be started again:

    python scripts/plan.py --list cats.csv --out runs/plan

Refit the mean 3D head from the face points of many cats (.npz files with a (n, 9, 2) `points` array or a single
cat's (9, 2) points); the package ships the template fitted to 6,000 cats:

    python scripts/plan.py --fit-template shard_000.npz shard_001.npz ... --out cat_head_template.npy

Segment Anything (facebook/sam-vit-base) is downloaded from the Hugging Face Hub on first use.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from brushgpt.plan import Sam, fit_head_template, load_head_template, load_sam, make_plan, save_plan


def plan_one(name: str, cutout: str, out: Path, template: np.ndarray, sam: Sam) -> None:
    t0 = time.time()
    z = np.load(cutout, allow_pickle=True)
    plan = make_plan(z["target"], z["tri"], z["points"], template, sam)
    save_plan(out, plan)
    print(json.dumps({"cat": name, "yaw": round(plan.pose.yaw, 1), "pitch": round(plan.pose.pitch, 1),
                      "roll": round(plan.pose.roll, 1), "tokens": len(plan.tokens), "nose": plan.nose is not None,
                      "seconds": round(time.time() - t0, 1)}), flush=True)


def fit_template(files: list[str], out: Path) -> None:
    points = np.concatenate([np.load(f)["points"].reshape(-1, 9, 2) for f in files])
    template = fit_head_template(points)
    np.save(out, template)
    print(json.dumps({"views": len(points), "template": np.round(template.astype(float), 3).tolist()}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cutout", help="the cat's cut-out .npz (target, tri, points)")
    ap.add_argument("--list", help="CSV of cats to plan: name, cutout")
    ap.add_argument("--fit-template", nargs="+", metavar="POINTS", help="refit the mean 3D head from these .npz files")
    ap.add_argument("--out", required=True, help="the plan .npz (one cat), the folder (--list) or the template .npy")
    ap.add_argument("--template", help="a mean 3D head .npy (default: the one shipped with the package)")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads")
    a = ap.parse_args()
    out = Path(a.out)
    if a.fit_template:
        fit_template(a.fit_template, out)
        return
    if not (a.cutout or a.list):
        ap.error("give --cutout, --list or --fit-template")
    torch.set_num_threads(a.threads)
    template = load_head_template(a.template) if a.template else load_head_template()
    sam = load_sam()
    if a.list:
        out.mkdir(parents=True, exist_ok=True)
        with open(a.list, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            path = out / f"{r['name']}.npz"
            if path.exists():
                continue
            try:
                plan_one(r["name"], r["cutout"], path, template, sam)
            except Exception as e:  # noqa: BLE001 - one failed cat should not stop the list
                print(json.dumps({"cat": r["name"], "error": repr(e)}), file=sys.stderr, flush=True)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    plan_one(out.stem, a.cutout, out, template, sam)


if __name__ == "__main__":
    main()
