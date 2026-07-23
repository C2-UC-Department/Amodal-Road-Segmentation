"""Step 3: Build the "incomplete road" + occlusion-hint masks.

For each image we combine the visible-road mask (KITTI) with the foreground
mask (Mask2Former) to identify where the road is likely occluded:

    occlusion_hint = foreground AND (dilate(visible_road) )

i.e. foreground-object pixels that sit inside or right next to the visible
road are the regions the annotator must "hallucinate" road into. This is only
a *hint* to guide annotation; the ground truth is produced by hand in Step 4.

Also seeds the amodal output: if no amodal mask exists yet for an image, we
initialise it to a copy of the visible-road mask so the annotator starts from
the correct visible geometry.

Usage:
    python -m src.s4_build_incomplete
    python -m src.s4_build_incomplete --no-seed   # don't pre-create amodal seeds
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import common  # noqa: E402


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-seed", action="store_true",
                    help="do not create amodal seed masks from the visible road")
    ap.add_argument("--preview", action="store_true", help="write QA overlays")
    args = ap.parse_args()

    config.OCCLUSION_DIR.mkdir(parents=True, exist_ok=True)
    if not args.no_seed:
        config.AMODAL_DIR.mkdir(parents=True, exist_ok=True)
    if args.preview:
        config.PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    n = 0
    for s in tqdm(list(common.iter_samples(config.SPLIT)), desc="occlusion hints"):
        vis_path = config.VISIBLE_DIR / f"{s.base}.png"
        fg_path = config.FOREGROUND_DIR / f"{s.base}.png"
        if not vis_path.exists() or not fg_path.exists():
            continue

        visible = common.read_mask(vis_path)
        foreground = common.read_mask(fg_path)

        road_neigh = dilate(visible, config.ROAD_NEIGHBOURHOOD_PX)
        occlusion = foreground & road_neigh
        common.write_mask(config.OCCLUSION_DIR / f"{s.base}.png", occlusion)

        # Seed the amodal mask from the visible road (annotator will extend it).
        if not args.no_seed:
            amodal_path = config.AMODAL_DIR / f"{s.base}.png"
            if not amodal_path.exists():
                common.write_mask(amodal_path, visible)

        if args.preview:
            rgb = common.read_rgb(s.image_path)
            prev = common.overlay_mask(rgb, visible, (60, 120, 255), 0.4)
            prev = common.overlay_mask(prev, occlusion, (255, 220, 0), 0.55)
            Image.fromarray(prev).save(config.PREVIEW_DIR / f"{s.base}_occ.png")
        n += 1

    print(f"[ok] wrote {n} occlusion masks to {config.OCCLUSION_DIR}")
    if not args.no_seed:
        print(f"[ok] seeded amodal masks in {config.AMODAL_DIR} (visible-road copies)")


if __name__ == "__main__":
    main()
