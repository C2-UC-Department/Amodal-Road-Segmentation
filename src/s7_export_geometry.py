"""Step 7: Precompute the ground-manifold geometry (GroundNet depth stream).

For each image, runs monocular metric depth, isolates the road pixels using the
EXISTING OFRS semantic map from Step 5 (no second segmentation model), fits a
ground plane by RANSAC, and caches everything to
data/processed/geometry/<base>.npz.

This is a precompute step for the same reason Step 5 is: depth inference is far
too slow to repeat every training epoch and every inference call. The cache
stores raw primitives (depth map + intrinsics + plane), and the dense fields the
network consumes (ground footprint G, height-above-plane h) are derived from
them cheaply at load time -- so changing the field derivation never requires
re-running the depth model.

Usage:
    python -m src.s7_export_geometry                       # KITTI
    python -m src.s7_export_geometry --source footage
    python -m src.s7_export_geometry --limit 5 --preview
    python -m src.s7_export_geometry --overwrite
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
from src import geometry as geo  # noqa: E402

ROAD_IDX = config.OFRS_CLASSES.index("road")


def colorize_height(h, gvalid, lo=-3.0, hi=3.0):
    """Height field -> BGR-ish visualisation (blue below plane, red above)."""
    t = np.clip((h - lo) / (hi - lo), 0, 1)
    img = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_COOL)
    img[~gvalid] = 0
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["kitti", "footage"], default="kitti")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--preview", action="store_true", help="write QA overlays")
    args = ap.parse_args()

    config.GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
    if args.preview:
        config.PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    device = common.pick_device()
    print(f"[device] {device}")
    print(f"[model]  {config.DEPTH_MODEL_ID}")

    samples = list(common.iter_source_samples(args.source))
    if args.limit:
        samples = samples[: args.limit]

    n_ok = n_fail = n_skip = 0
    src_counts: dict[str, int] = {}
    n_stale = 0
    for s in tqdm(samples, desc="geometry export"):
        if geo.geometry_path(s.base).exists() and not args.overwrite:
            n_skip += 1
            # Flag caches built before/without a real calibration source, so a
            # stale heuristic-K entry is never silently reused.
            cached = geo.load_geometry(s.base)
            if cached is not None and cached.get("k_source") in ("unknown", "fov_prior"):
                n_stale += 1
            continue

        sem_path = config.SEMANTIC_DIR / f"{s.base}.png"
        if not sem_path.exists():
            n_fail += 1
            continue

        rgb = common.read_rgb(s.image_path)
        orig_h, orig_w = rgb.shape[:2]

        # Work at the same capped resolution the network sees, so the cached
        # depth lines up with the semantic map without another resize.
        sem = np.asarray(Image.open(sem_path).convert("L"))
        small, scale = common.fit_within_max_side(sem, config.OFRS_MAX_SIDE)
        h_img, w_img = small.shape
        if (h_img, w_img) != (orig_h, orig_w):
            rgb_small = cv2.resize(rgb, (w_img, h_img), interpolation=cv2.INTER_AREA)
        else:
            rgb_small = rgb

        cal = geo.calibrate_sample(s.image_path, s.base, w_img, h_img,
                                   orig_w=orig_w, orig_h=orig_h, device=device)
        K, calibrated = cal.K, cal.calibrated
        src_counts[cal.source] = src_counts.get(cal.source, 0) + 1

        depth = geo.estimate_depth_metric(rgb_small, device)
        ground = small == ROAD_IDX

        if int(ground.sum()) < config.GEOM_MIN_GROUND_PX:
            # Too little visible road to trust a plane fit -- cache an
            # explicitly invalid entry so training falls back cleanly rather
            # than silently consuming a garbage plane.
            geo.save_geometry(s.base, depth, None, K, calibrated,
                              int(ground.sum()), k_source=cal.source)
            n_fail += 1
            continue

        n_plane, n_pts = geo.estimate_ground_plane(depth, ground, K)
        geo.save_geometry(s.base, depth, n_plane, K, calibrated, n_pts,
                          k_source=cal.source)
        if n_plane is None:
            n_fail += 1
        else:
            n_ok += 1

        if args.preview and n_plane is not None:
            G, hh, gvalid = geo.derive_ground_fields(depth, n_plane, K)
            hv = colorize_height(hh, gvalid)
            blend = (0.5 * rgb_small + 0.5 * hv).astype(np.uint8)
            Image.fromarray(blend).save(config.PREVIEW_DIR / f"{s.base}_geom.png")

    print(f"[ok] plane fitted for {n_ok} images "
          f"({n_fail} failed/insufficient road, {n_skip} already cached)")
    if src_counts:
        print("[intrinsics] source used: "
              + ", ".join(f"{k}={v}" for k, v in sorted(src_counts.items())))
    if n_stale:
        print(f"[stale] {n_stale} cached entries were built with a heuristic/unknown "
              f"focal length.\n        Re-run with --overwrite to recompute them "
              f"with the current calibration chain.")
    print(f"     -> {config.GEOMETRY_DIR}")


if __name__ == "__main__":
    main()
