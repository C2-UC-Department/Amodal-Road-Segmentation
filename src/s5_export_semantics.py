"""Step 5: Export the 11-class OFRS semantic map — the OFRSNet INPUT.

OFRSNet does not consume RGB; it consumes a semantic segmentation map unified
to 11 classes (road, sidewalk, building, wall, fence, pole, traffic_sign,
vegetation, person, vehicle, unlabeled). We reuse the Step-2 Mask2Former
(Mapillary Vistas) prediction and remap its 65 classes into those 11.

Output: data/processed/semantic_ofrs/<base>.png  — single-channel PNG whose
pixel values are the OFRS class index (0..10). This file is the network input;
data/amodal_road/<base>.png is the target.

Usage:
    python -m src.s5_export_semantics
    python -m src.s5_export_semantics --limit 5 --preview
    python -m src.s5_export_semantics --overwrite
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import common  # noqa: E402
from src import s3_detect_foreground as s3  # noqa: E402  (reuse model load + segment)

# A distinct colour per OFRS class for the QA preview (RGB).
OFRS_PALETTE = np.array([
    [128, 64, 128],   # road
    [244, 35, 232],   # sidewalk
    [70, 70, 70],     # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [220, 220, 0],    # traffic_sign
    [107, 142, 35],   # vegetation
    [220, 20, 60],    # person
    [0, 0, 142],      # vehicle
    [0, 0, 0],        # unlabeled
], dtype=np.uint8)


def build_mapillary_to_ofrs(id2label: dict[int, str]) -> np.ndarray:
    """Return a LUT array `lut[mapillary_id] = ofrs_class_index` (0..10)."""
    name_to_idx = {c: i for i, c in enumerate(config.OFRS_CLASSES)}
    max_id = max(id2label) + 1
    lut = np.full(max_id, config.OFRS_UNLABELED_ID, dtype=np.uint8)

    for cid, label in id2label.items():
        low = label.lower()
        assigned = config.OFRS_UNLABELED_ID
        for ofrs_class, keywords, excludes in config.OFRS_CLASS_KEYWORDS:
            if any(x in low for x in excludes):
                continue
            if any(k in low for k in keywords):
                assigned = name_to_idx[ofrs_class]
                break
        lut[cid] = assigned
    return lut


def report_mapping(id2label: dict[int, str], lut: np.ndarray) -> None:
    buckets: dict[str, list[str]] = {c: [] for c in config.OFRS_CLASSES}
    for cid, label in id2label.items():
        buckets[config.OFRS_CLASSES[lut[cid]]].append(label)
    print("[map] Mapillary -> OFRS class grouping:")
    for c in config.OFRS_CLASSES:
        print(f"  {c:12s}: {', '.join(sorted(buckets[c])) or '-'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["kitti", "footage"], default="kitti",
                    help="kitti = data_road images; footage = data/footage_frames")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    config.SEMANTIC_DIR.mkdir(parents=True, exist_ok=True)
    if args.preview:
        config.PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    device = common.pick_device()
    processor, model, _fg_ids = s3.load_model(device)  # prints fg classes too
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    lut = build_mapillary_to_ofrs(id2label)
    report_mapping(id2label, lut)

    samples = list(common.iter_source_samples(args.source))
    if args.limit:
        samples = samples[: args.limit]

    for s in tqdm(samples, desc="OFRS semantic export"):
        out_path = config.SEMANTIC_DIR / f"{s.base}.png"
        if out_path.exists() and not args.overwrite:
            continue
        image = Image.open(s.image_path).convert("RGB")
        seg = s3.segment(processor, model, device, image)      # (H,W) mapillary ids
        ofrs = lut[np.clip(seg, 0, len(lut) - 1)]              # (H,W) 0..10
        Image.fromarray(ofrs.astype(np.uint8), mode="L").save(out_path)

        if args.preview:
            color = OFRS_PALETTE[ofrs]
            blend = (0.5 * np.asarray(image) + 0.5 * color).astype(np.uint8)
            Image.fromarray(blend).save(config.PREVIEW_DIR / f"{s.base}_sem.png")

    print(f"[ok] OFRS semantic maps in {config.SEMANTIC_DIR}")


if __name__ == "__main__":
    main()
