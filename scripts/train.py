"""Train Brush GPT on a built data file.

    python scripts/train.py --data data/gen_furpal/data.npz --out runs/furpal --steps 12000 \
        --dropout 0.1 --cond-drop 0.15 --balance-cats --eval-every 500 --eval-paintings 100
"""

from __future__ import annotations

import argparse

import torch

from brushgpt.model import BrushGPTConfig
from brushgpt.train import PaintingData, TrainConfig, train


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="the built data file (.npz)")
    ap.add_argument("--out", required=True, help="directory for info.json, log.jsonl, last.pt and best.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--accum", type=int, default=8, help="paintings per optimiser step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--enc-layers", type=int, default=4)
    ap.add_argument("--dec-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0, help="training only: embeddings, attention, residuals")
    ap.add_argument("--cond-drop", type=float, default=0.0,
                    help="share of paintings trained without their attributes (enables guidance)")
    ap.add_argument("--balance-cats", action="store_true", help="draw a cat first, then one of its paintings")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-paintings", type=int, default=60, help="held paintings scored per evaluation")
    ap.add_argument("--stop-after-min", type=float, default=0.0, help="stop after this many minutes (0: never)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="cuda, cpu, or auto (cuda when available)")
    a = ap.parse_args()

    device = resolve_device(a.device)
    data = PaintingData(a.data, device)
    model_cfg = BrushGPTConfig(vocab_size=data.vocab + (1 if a.cond_drop > 0 else 0), d_model=a.d_model,
                               enc_layers=a.enc_layers, dec_layers=a.dec_layers, dropout=a.dropout,
                               null_token=a.cond_drop > 0)
    train_cfg = TrainConfig(steps=a.steps, accum=a.accum, lr=a.lr, weight_decay=a.weight_decay, cond_drop=a.cond_drop,
                            balance_cats=a.balance_cats, eval_every=a.eval_every, eval_paintings=a.eval_paintings,
                            stop_after_min=a.stop_after_min, seed=a.seed)
    train(data, model_cfg, train_cfg, a.out, device)


if __name__ == "__main__":
    main()
