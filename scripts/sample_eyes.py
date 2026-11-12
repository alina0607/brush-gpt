"""Paint held-out cats' eyes with a trained eye model, from the painter's own sketch plan, one sample per eye,
nothing picked.

    python scripts/sample_eyes.py --ckpt runs/eyes/best.pt --data data/eyes/data.npz --out runs/eyes/eyes_g2.npz \
        --cats 32 --guidance 2
    python scripts/sample_eyes.py ... --out runs/eyes/eyes_g2_blue.npz --guidance 2 --eye-as blue   # the swap test

The output .npz holds `rows` (each eye's row in the data file), `prefix`, `bodies` (padded with EOS) and the
sampling settings; scripts/evaluate.py alone paints and measures them.
"""

from __future__ import annotations

import argparse
import json
import time

from brushgpt.eye_model import EyeData, load_checkpoint, resolve_device, sample_held


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="the eye data file the model was trained on")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--cats", type=int, default=32, help="the first N held cats' two eyes (0: every held eye)")
    ap.add_argument("--guidance", type=float, default=1.0, help="classifier-free guidance weight (1: none)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--eye-as", default="", help="ask every eye in this colour (the swap test)")
    ap.add_argument("--max-new", type=int, default=1200, help="most tokens per eye")
    ap.add_argument("--batch", type=int, default=32, help="eyes sampled together")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="cuda, mps, cpu, or auto")
    a = ap.parse_args()

    data = EyeData(a.data)
    model = load_checkpoint(a.ckpt, resolve_device(a.device), data.prefix + a.max_new)
    t0 = time.time()
    samples = sample_held(model, data, a.cats, a.guidance, a.temperature, a.eye_as, a.max_new, a.batch, a.seed)
    samples.save(a.out)
    print(json.dumps({"eyes": len(samples.rows), "seconds": round(time.time() - t0, 1), "out": a.out}), flush=True)


if __name__ == "__main__":
    main()
