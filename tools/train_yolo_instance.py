"""Train the vehicle-instance student: YOLOv8n-seg on the distillation set
built by tools/export_yolo_instance_dataset.py.

Thinner than src/mobile_semantic/train.py's from-scratch loop -- Ultralytics
owns the training loop, augmentation, and checkpointing internally; this is a
thin CLI wrapper over `model.train(...)`, not a reimplementation.

Usage:
    python -m tools.export_yolo_instance_dataset   # build the dataset first
    python -m tools.train_yolo_instance
    python -m tools.train_yolo_instance --epochs 100 --imgsz 640
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import common  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(config.YOLO_INSTANCE_DIR / "data.yaml"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default="yolov8n-seg.pt", help="pretrained base weights")
    args = ap.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"{args.data} not found -- run "
                         "`python -m tools.export_yolo_instance_dataset` first.")

    from ultralytics import YOLO

    device = common.pick_device()
    print(f"[device] {device}")

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
               batch=args.batch, device=str(device), seed=0,
               project=str(config.CKPT_DIR), name="yolo_instance")

    best = config.CKPT_DIR / "yolo_instance" / "weights" / "best.pt"
    print(f"[done] best weights at {best}")


if __name__ == "__main__":
    main()
