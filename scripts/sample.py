"""Paint held-out cats with a trained Brush GPT, one painting per cat, nothing picked among samples.

    python scripts/sample.py --ckpt runs/furpal/best.pt --data data/gen_furpal/data.npz \
        --out runs/furpal/gen_g2.npz --cats 32 --guidance 2

The output .npz holds `tokens` (one row per cat, padded with -1), `rows` (each painting's row in the data file) and
the sampling settings.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from brushgpt.model import load_checkpoint
from brushgpt.train import PaintingData, sample_held


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="the data file the model was trained on")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--cats", type=int, default=8, help="the first N held cats")
    ap.add_argument("--batch", type=int, default=8, help="paintings sampled together")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--guidance", type=float, default=1.0, help="classifier-free guidance weight (1: none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="cuda, cpu, or auto (cuda when available)")
    a = ap.parse_args()

    device = a.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    data = PaintingData(a.data, device)
    model = load_checkpoint(torch.load(a.ckpt, map_location=device), data.layout, device)
    t0 = time.time()
    tokens, rows = sample_held(model, data, a.cats, a.batch, a.temperature, a.guidance, a.seed)
    np.savez_compressed(a.out, tokens=tokens, rows=rows, temperature=a.temperature, guidance=a.guidance, seed=a.seed)
    print(json.dumps({"paintings": len(rows), "seconds": round(time.time() - t0, 1), "out": a.out}), flush=True)


if __name__ == "__main__":
    main()
