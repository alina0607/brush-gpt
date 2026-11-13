"""Train the eye part model on a built eye data file.

    python scripts/train_eyes.py --data data/eyes/data.npz --out runs/eyes --steps 16000

The defaults are the final run's settings (d = 384, 8 layers, 6 heads, dropout 0.1, weight decay 0.1, peak learning
rate 6e-4 after 300 warm-up steps, batches of 32 eyes, the colour dropped for 15% of the eyes). Writes info.json,
log.jsonl and best.pt (the lowest validation loss) to OUT.
"""

from __future__ import annotations

import argparse

from brushgpt.eye_model import EyeData, EyeTrainConfig, model_config, resolve_device, train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="the built eye data file (.npz)")
    ap.add_argument("--out", required=True, help="directory for info.json, log.jsonl and best.pt")
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1, help="on the attention weights, training only")
    d = EyeTrainConfig()
    ap.add_argument("--steps", type=int, default=d.steps)
    ap.add_argument("--batch", type=int, default=d.batch, help="eyes per step")
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--warmup", type=int, default=d.warmup)
    ap.add_argument("--weight-decay", type=float, default=d.weight_decay)
    ap.add_argument("--cond-drop", type=float, default=d.cond_drop, help="share of eyes trained without their colour")
    ap.add_argument("--balance-cats", action="store_true", help="draw a cat first, then one of its eyes")
    ap.add_argument("--eval-every", type=int, default=d.eval_every)
    ap.add_argument("--eval-limit", type=int, default=d.eval_limit, help="score only the first N eyes (0: all)")
    ap.add_argument("--log-every", type=int, default=d.log_every)
    ap.add_argument(
        "--sample-every",
        type=int,
        default=d.sample_every,
        help="also sample validation eyes and check their geometry (0: never)",
    )
    ap.add_argument("--sample-n", type=int, default=d.sample_n)
    ap.add_argument("--stop-after-min", type=float, default=0.0, help="stop after this many minutes (0: never)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="cuda, mps, cpu, or auto")
    a = ap.parse_args()

    data = EyeData(a.data)
    cfg = EyeTrainConfig(
        steps=a.steps,
        batch=a.batch,
        lr=a.lr,
        warmup=a.warmup,
        weight_decay=a.weight_decay,
        cond_drop=a.cond_drop,
        balance_cats=a.balance_cats,
        eval_every=a.eval_every,
        eval_limit=a.eval_limit,
        log_every=a.log_every,
        sample_every=a.sample_every,
        sample_n=a.sample_n,
        stop_after_min=a.stop_after_min,
        seed=a.seed,
    )
    train(data, model_config(data, a.d_model, a.layers, a.heads, a.dropout), cfg, a.out, resolve_device(a.device))


if __name__ == "__main__":
    main()
