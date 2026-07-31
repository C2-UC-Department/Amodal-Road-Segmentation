"""Train the mobile semantic student by distilling the Mask2Former/Mapillary-
Vistas teacher (src/mobile_semantic/model.py; see that file's docstring and
the iOS migration plan's Section 1 for why this architecture).

Input : every source with a teacher label already exported --
        run `python -m src.s5_export_semantics --source <kitti|footage|realworld>`
        first for any source not yet covered (data/processed/semantic_ofrs/).
Target: the teacher's own OFRS-11 map (hard-label distillation), PLUS an
        auxiliary road-channel loss against KITTI's real hand-labelled road
        mask where available -- see src/mobile_semantic/dataset.py's
        docstring for why only KITTI contributes that term.

    python -m src.mobile_semantic.train
    python -m src.mobile_semantic.train --epochs 50 --batch-size 8

Checkpoints (best val road-IoU-vs-teacher + last) are written to checkpoints/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config  # noqa: E402
from src import common  # noqa: E402
from src.mobile_semantic.dataset import (MobileSemanticDataset,  # noqa: E402
                                         all_labeled_samples,
                                         mobile_semantic_collate, split_samples)
from src.mobile_semantic.model import MobileSemanticNet  # noqa: E402

ROAD_IDX = config.OFRS_CLASSES.index("road")


def compute_loss(logits, label, road_gt, has_road_gt, valid):
    """Teacher-distillation CE (all valid pixels) + auxiliary road BCE
    (KITTI pixels with real ground truth only). Padded/invalid pixels are
    excluded from both terms via `valid`, same rationale as
    src/ofrs/dataset.py's weight>0 filtering.
    """
    ce = F.cross_entropy(logits, label, reduction="none")           # (N, H, W)
    ce = (ce * valid).sum() / valid.sum().clamp(min=1)

    road_mask = has_road_gt.view(-1, 1, 1) & valid & (road_gt >= 0)
    if road_mask.any():
        road_logit = logits[:, ROAD_IDX]
        bce = F.binary_cross_entropy_with_logits(road_logit, road_gt, reduction="none")
        aux = (bce * road_mask).sum() / road_mask.sum().clamp(min=1)
    else:
        aux = torch.zeros((), device=logits.device)
    return ce + config.MOBILE_SEM_ROAD_AUX_WEIGHT * aux, ce.item(), aux.item()


@torch.no_grad()
def road_iou(logits, target, valid):
    pred = logits.argmax(1)
    p = (pred == ROAD_IDX) & valid
    t = (target == ROAD_IDX) & valid
    inter = (p & t).sum().item()
    union = (p | t).sum().item()
    return inter / union if union else 1.0


@torch.no_grad()
def road_iou_vs_real_gt(logits, road_gt, has_road_gt, valid):
    """Road IoU against REAL KITTI annotations only -- the trustworthy number
    (Section 7.3 of the migration plan: road-class IoU gates everything
    downstream, unlike the other 10 classes)."""
    mask = has_road_gt.view(-1, 1, 1) & valid & (road_gt >= 0)
    if not mask.any():
        return None
    pred = logits.argmax(1) == ROAD_IDX
    real = road_gt > 0.5
    inter = (pred & mask & real).sum().item()
    union = ((pred | real) & mask).sum().item()
    return inter / union if union else 1.0


def evaluate(model, loader, device):
    model.eval()
    losses, ious, real_ious = [], [], []
    with torch.no_grad():
        for rgb, label, road_gt, has_road_gt, valid, _ in loader:
            rgb, label = rgb.to(device), label.to(device)
            road_gt, has_road_gt, valid = road_gt.to(device), has_road_gt.to(device), valid.to(device)
            logits = model(rgb)
            loss, _, _ = compute_loss(logits, label, road_gt, has_road_gt, valid)
            losses.append(loss.item())
            ious.append(road_iou(logits, label, valid))
            real_iou = road_iou_vs_real_gt(logits, road_gt, has_road_gt, valid)
            if real_iou is not None:
                real_ious.append(real_iou)
    return (float(np.mean(losses)), float(np.mean(ious)),
            float(np.mean(real_ious)) if real_ious else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=config.MOBILE_SEM_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.MOBILE_SEM_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.MOBILE_SEM_LR)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sources", nargs="+", default=["kitti", "footage", "realworld"])
    args = ap.parse_args()

    config.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = common.pick_device()
    print(f"[device] {device}")

    samples = all_labeled_samples(tuple(args.sources))
    if not samples:
        raise SystemExit(
            f"No teacher-labeled samples found for sources {args.sources}. Run "
            "`python -m src.s5_export_semantics --source <source>` first.")
    train_s, val_s = split_samples(samples)
    n_kitti_val = sum(1 for s in val_s if s.gt_path is not None)
    print(f"[data] {len(samples)} labeled | train {len(train_s)} / val {len(val_s)} "
          f"({n_kitti_val} val samples have real KITTI road GT)")

    train_ds = MobileSemanticDataset(train_s)
    val_ds = MobileSemanticDataset(val_s)
    pin = device.type == "cuda"
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=pin, drop_last=False,
                          collate_fn=mobile_semantic_collate)
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=args.workers, pin_memory=pin,
                        collate_fn=mobile_semantic_collate)

    model = MobileSemanticNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] MobileSemanticNet (LR-ASPP/MobileNetV3-Large), {n_params/1e6:.2f}M params")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_losses = []
        for rgb, label, road_gt, has_road_gt, valid, _ in train_ld:
            rgb, label = rgb.to(device), label.to(device)
            road_gt, has_road_gt, valid = road_gt.to(device), has_road_gt.to(device), valid.to(device)
            optim.zero_grad()
            logits = model(rgb)
            loss, _, _ = compute_loss(logits, label, road_gt, has_road_gt, valid)
            loss.backward()
            optim.step()
            ep_losses.append(loss.item())
        sched.step()

        val_loss, val_iou, val_real_iou = evaluate(model, val_ld, device)
        real_iou_str = f"{val_real_iou:.4f}" if val_real_iou is not None else "n/a"
        print(f"[{epoch:03d}/{args.epochs}] train_loss {np.mean(ep_losses):.4f}  "
              f"val_loss {val_loss:.4f}  val_road_IoU(teacher) {val_iou:.4f}  "
              f"val_road_IoU(real_gt) {real_iou_str}  lr {sched.get_last_lr()[0]:.2e}")

        ckpt = {"model": model.state_dict(), "epoch": epoch, "val_iou": val_iou,
                "val_real_iou": val_real_iou, "ofrs_classes": config.OFRS_CLASSES}
        torch.save(ckpt, config.CKPT_DIR / "mobile_semantic_last.pt")
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(ckpt, config.CKPT_DIR / "mobile_semantic_best.pt")
            print(f"       * new best road IoU {best_iou:.4f} -> mobile_semantic_best.pt")

    print(f"[done] best val road IoU (vs teacher) {best_iou:.4f}")


if __name__ == "__main__":
    main()
