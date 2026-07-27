"""Train OFRSNet on the hand-annotated KITTI-OFRS dataset.

Input  : 11-class OFRS semantic maps  (data/processed/semantic_ofrs/)
Target : occlusion-free road masks    (data/amodal_road/)

    python -m src.ofrs.train
    python -m src.ofrs.train --epochs 100 --batch-size 4 --lr 1e-3

Checkpoints (best val road-IoU + last) are written to checkpoints/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config  # noqa: E402
from src import common  # noqa: E402
from src.ofrs.dataset import (OFRSDataset, annotated_bases,  # noqa: E402
                              split_bases)
from src.ofrs.loss import SpatiallyWeightedCELoss  # noqa: E402
from src.ofrs.model import OFRSNet  # noqa: E402


@torch.no_grad()
def road_iou(logits, target):
    pred = logits.argmax(1)
    p, t = pred == 1, target == 1
    inter = (p & t).sum().item()
    union = (p | t).sum().item()
    return inter / union if union else 1.0


def evaluate(model, loader, device, criterion):
    model.eval()
    losses, ious = [], []
    with torch.no_grad():
        for x, y, w, _ in loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            out = model(x)
            losses.append(criterion(out, y, w).item())
            ious.append(road_iou(out, y))
    return float(np.mean(losses)), float(np.mean(ious))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=config.OFRS_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.OFRS_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.OFRS_LR)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    config.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = common.pick_device()
    print(f"[device] {device}")

    bases = annotated_bases()
    if not bases:
        raise SystemExit(
            "No annotated samples found. You need BOTH a semantic map "
            f"({config.SEMANTIC_DIR}) and an amodal mask ({config.AMODAL_DIR}) "
            "for at least one image. Run Step 5 and annotate some images first."
        )
    train_bases, val_bases = split_bases(bases)
    print(f"[data] {len(bases)} annotated | train {len(train_bases)} / val {len(val_bases)}")

    train_ds = OFRSDataset(train_bases, train=True)
    val_ds = OFRSDataset(val_bases, train=False)
    # MPS/CPU: pin_memory only helps CUDA.
    pin = device.type == "cuda"
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=pin, drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=args.workers, pin_memory=pin)

    model = OFRSNet(in_channels=config.OFRS_NUM_CLASSES, num_classes=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] OFRSNet, {n_params/1e6:.2f}M params")

    criterion = SpatiallyWeightedCELoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_losses = []
        for x, y, w, _ in train_ld:
            x, y, w = x.to(device), y.to(device), w.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y, w)
            loss.backward()
            optim.step()
            ep_losses.append(loss.item())
        sched.step()

        val_loss, val_iou = evaluate(model, val_ld, device, criterion)
        print(f"[{epoch:03d}/{args.epochs}] train_loss {np.mean(ep_losses):.4f}  "
              f"val_loss {val_loss:.4f}  val_road_IoU {val_iou:.4f}  "
              f"lr {sched.get_last_lr()[0]:.2e}")

        torch.save({"model": model.state_dict(), "epoch": epoch,
                    "val_iou": val_iou, "config_classes": config.OFRS_CLASSES},
                   config.CKPT_DIR / "ofrsnet_last.pt")
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_iou": val_iou, "config_classes": config.OFRS_CLASSES},
                       config.CKPT_DIR / "ofrsnet_best.pt")
            print(f"       * new best road IoU {best_iou:.4f} -> ofrsnet_best.pt")

    print(f"[done] best val road IoU {best_iou:.4f}")


if __name__ == "__main__":
    main()
