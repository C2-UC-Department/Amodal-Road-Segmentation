"""Step 2: Detect foreground objects with Mask2Former (Mapillary Vistas).

Runs semantic segmentation over every KITTI image and writes a binary
foreground-object mask (cars, trucks, buses, pedestrians, riders, cycles, ...)
to data/processed/foreground/<base>.png.

We use the *semantic* Mask2Former head: we only need which pixels are foreground
objects, not per-instance separation. Foreground classes are selected by
substring-matching config.FOREGROUND_KEYWORDS against the model's id2label, so
the exact Mapillary label spelling does not matter.

Usage:
    python -m src.s3_detect_foreground
    python -m src.s3_detect_foreground --limit 5      # quick test on 5 images
    python -m src.s3_detect_foreground --preview       # also write QA overlays
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


def build_foreground_class_ids(id2label: dict[int, str]) -> set[int]:
    """Return the set of class ids whose label matches a foreground keyword."""
    keywords = [k.lower() for k in config.FOREGROUND_KEYWORDS]
    excludes = [k.lower() for k in getattr(config, "FOREGROUND_EXCLUDE_KEYWORDS", [])]
    ids = set()
    for cid, label in id2label.items():
        low = label.lower()
        if any(k in low for k in keywords) and not any(x in low for x in excludes):
            ids.add(int(cid))
    return ids


def load_model(device):
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    print(f"[model] loading {config.MODEL_ID} on {device} ...")
    processor = AutoImageProcessor.from_pretrained(config.MODEL_ID)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(config.MODEL_ID)
    model.to(device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    fg_ids = build_foreground_class_ids(id2label)
    print(f"[model] {len(fg_ids)} foreground classes selected:")
    for cid in sorted(fg_ids):
        print(f"         {cid:>3}  {id2label[cid]}")
    return processor, model, fg_ids


@torch.no_grad()
def segment(processor, model, device, image: Image.Image) -> np.ndarray:
    """Return an (H, W) int array of per-pixel Mapillary class ids."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    seg = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]  # (H, W)
    )[0]
    return seg.cpu().numpy().astype(np.int32)


def foreground_from_seg(seg: np.ndarray, fg_ids: set[int], dilate_px: int) -> np.ndarray:
    mask = np.isin(seg, list(fg_ids))
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    return mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["kitti", "footage"], default="kitti",
                    help="kitti = data_road images; footage = data/footage_frames")
    ap.add_argument("--limit", type=int, default=0, help="process only N images (0 = all)")
    ap.add_argument("--preview", action="store_true", help="write QA overlays too")
    ap.add_argument("--overwrite", action="store_true", help="recompute existing masks")
    args = ap.parse_args()

    config.FOREGROUND_DIR.mkdir(parents=True, exist_ok=True)
    if args.preview:
        config.PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    device = common.pick_device()
    processor, model, fg_ids = load_model(device)

    samples = list(common.iter_source_samples(args.source))
    if args.limit:
        samples = samples[: args.limit]

    for s in tqdm(samples, desc="foreground detection"):
        out_path = config.FOREGROUND_DIR / f"{s.base}.png"
        if out_path.exists() and not args.overwrite:
            continue
        image = Image.open(s.image_path).convert("RGB")
        seg = segment(processor, model, device, image)
        fg = foreground_from_seg(seg, fg_ids, config.FOREGROUND_DILATE_PX)
        common.write_mask(out_path, fg)

        if args.preview:
            rgb = np.asarray(image)
            prev = common.overlay_mask(rgb, fg, color=(255, 60, 60), alpha=0.5)
            Image.fromarray(prev).save(config.PREVIEW_DIR / f"{s.base}_fg.png")

    print(f"[ok] foreground masks in {config.FOREGROUND_DIR}")


if __name__ == "__main__":
    main()
