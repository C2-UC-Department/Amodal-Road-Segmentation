"""Inference + visualization: image directory -> OFRSNet amodal road.

Pipeline per image:
    RGB  --Mask2Former(Mapillary)-->  11-class OFRS semantic map
         --one-hot, resize 384x1248-->  OFRSNet  -->  amodal road mask
         --resize back-->  overlay on the original image.

Writes, for each input image, a stacked visualization (original / semantic /
predicted amodal road) to the output dir, plus the raw binary mask.

Usage:
    python -m src.predict --input path/to/images --out data/predictions
    python -m src.predict --input data/raw/data_road/testing/image_2 --limit 20
    python -m src.predict --input my_dashcam/ --ckpt checkpoints/ofrsnet_best.pt --no-stack
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import common  # noqa: E402
from src import s3_detect_foreground as s3  # noqa: E402
from src import s5_export_semantics as s5  # noqa: E402
from src.ofrs.model import OFRSNet  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
COL_ROAD = (40, 220, 90)     # predicted amodal road (green)
COL_OBJECT = (255, 60, 60)   # person/vehicle from semantics (red)


def list_images(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def load_ofrsnet(ckpt_path: Path, device):
    model = OFRSNet(in_channels=config.OFRS_NUM_CLASSES, num_classes=2).to(device)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[ckpt] loaded {ckpt_path}  (epoch {ckpt.get('epoch','?')}, "
              f"val_road_IoU {ckpt.get('val_iou','?')})")
    else:
        print(f"[warn] no checkpoint at {ckpt_path}; using RANDOM weights "
              "(train OFRSNet first for meaningful output).")
    model.eval()
    return model


@torch.no_grad()
def predict_amodal(model, sem_labels: np.ndarray, out_hw: tuple[int, int],
                   device) -> np.ndarray:
    """sem_labels (H,W) int 0..10 -> predicted amodal road mask at out_hw (bool)."""
    h, w = config.OFRS_INPUT_SIZE
    sem = cv2.resize(sem_labels.astype(np.uint8), (w, h),
                     interpolation=cv2.INTER_NEAREST).astype(np.int64)
    sem = np.clip(sem, 0, config.OFRS_NUM_CLASSES - 1)
    onehot = np.eye(config.OFRS_NUM_CLASSES, dtype=np.float32)[sem].transpose(2, 0, 1)
    x = torch.from_numpy(onehot).unsqueeze(0).to(device)
    pred = model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)   # (h,w)
    return cv2.resize(pred, (out_hw[1], out_hw[0]),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def make_stack(rgb, sem_labels, amodal) -> np.ndarray:
    """Vertical stack: original / semantic / amodal overlay."""
    sem_color = s5.OFRS_PALETTE[sem_labels]
    sem_blend = (0.45 * rgb + 0.55 * sem_color).astype(np.uint8)

    obj = np.isin(sem_labels, [config.OFRS_CLASSES.index("person"),
                               config.OFRS_CLASSES.index("vehicle")])
    over = common.overlay_mask(rgb, amodal, COL_ROAD, 0.5)
    over = common.overlay_mask(over, obj, COL_OBJECT, 0.35)

    def label(img, text):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(img, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img

    return np.vstack([label(rgb, "input"),
                      label(sem_blend, "Mask2Former semantics (OFRS 11-class)"),
                      label(over, "OFRSNet predicted amodal road (green)")])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="image file or directory")
    ap.add_argument("--out", default=str(config.DATA_DIR / "predictions"))
    ap.add_argument("--ckpt", default=str(config.CKPT_DIR / "ofrsnet_best.pt"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-stack", action="store_true",
                    help="save only the amodal overlay, not the 3-panel stack")
    args = ap.parse_args()

    images = list_images(Path(args.input))
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found under {args.input}")

    out_dir = Path(args.out)
    (out_dir / "mask").mkdir(parents=True, exist_ok=True)

    device = common.pick_device()
    print(f"[device] {device}")

    # Upstream Mask2Former + Mapillary->OFRS LUT (shared with Step 5).
    processor, model_m2f, _ = s3.load_model(device)
    id2label = {int(k): v for k, v in model_m2f.config.id2label.items()}
    lut = s5.build_mapillary_to_ofrs(id2label)

    ofrsnet = load_ofrsnet(Path(args.ckpt), device)

    for path in tqdm(images, desc="predict"):
        rgb = common.read_rgb(path)
        image = Image.fromarray(rgb)
        seg = s3.segment(processor, model_m2f, device, image)      # mapillary ids
        sem = lut[np.clip(seg, 0, len(lut) - 1)]                    # OFRS 0..10
        amodal = predict_amodal(ofrsnet, sem, rgb.shape[:2], device)

        common.write_mask(out_dir / "mask" / f"{path.stem}.png", amodal)
        if args.no_stack:
            viz = common.overlay_mask(rgb, amodal, COL_ROAD, 0.5)
        else:
            viz = make_stack(rgb, sem, amodal)
        Image.fromarray(viz).save(out_dir / f"{path.stem}_viz.png")

    print(f"[ok] wrote predictions to {out_dir}")


if __name__ == "__main__":
    main()
