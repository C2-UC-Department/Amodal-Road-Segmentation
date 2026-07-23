"""Step 1b: Decode KITTI Road GT PNGs into clean binary visible-road masks.

Reads data_road/<split>/gt_image_2/*_road_*.png and writes one binary mask
(0/255, single channel) per image to data/processed/visible_road/<base>.png.
These masks are the *visible* road only (objects on the road are NOT labelled
road by KITTI) and become the starting point for amodal annotation.

Usage:
    python -m src.s2_prepare_visible_masks
"""
from __future__ import annotations

import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import common  # noqa: E402


def main() -> None:
    config.VISIBLE_DIR.mkdir(parents=True, exist_ok=True)
    samples = list(common.iter_samples(config.SPLIT))
    written = 0
    skipped = 0
    for s in tqdm(samples, desc="visible road masks"):
        if s.gt_path is None:
            skipped += 1
            continue
        mask = common.decode_kitti_road_gt(s.gt_path)
        common.write_mask(config.VISIBLE_DIR / f"{s.base}.png", mask)
        written += 1
    print(f"[ok] wrote {written} masks to {config.VISIBLE_DIR}")
    if skipped:
        print(f"[note] {skipped} images had no road GT (expected for the testing split).")


if __name__ == "__main__":
    main()
