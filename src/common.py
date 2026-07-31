"""Shared helpers: device selection, KITTI file discovery, mask IO, compositing."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, NamedTuple

import cv2
import numpy as np
from PIL import Image

# Make the project root importable when scripts are run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
def pick_device():
    """Return the best available torch device following DEVICE_PREFERENCE."""
    import torch

    for name in config.DEVICE_PREFERENCE:
        if name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if name == "cpu":
            return torch.device("cpu")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# KITTI file discovery
# --------------------------------------------------------------------------- #
class Sample(NamedTuple):
    base: str            # shared base name, e.g. "um_000000"
    image_path: Path     # .../image_2/um_000000.png
    gt_path: Path | None # .../gt_image_2/um_road_000000.png (None for testing split)


def _split_dirs(split: str) -> tuple[Path, Path | None]:
    image_dir = config.KITTI_ROOT / split / "image_2"
    gt_dir = config.KITTI_ROOT / split / "gt_image_2"
    return image_dir, (gt_dir if gt_dir.exists() else None)


def iter_samples(split: str = config.SPLIT) -> Iterator[Sample]:
    """Yield every KITTI image in `split`, paired with its road GT if present.

    KITTI names images `um_000000.png` and the matching road GT
    `um_road_000000.png`. We ignore `*_lane_*` GTs (lane-marking labels).
    """
    image_dir, gt_dir = _split_dirs(split)
    if not image_dir.exists():
        raise FileNotFoundError(
            f"KITTI images not found at {image_dir}. Run the download step first."
        )
    for image_path in sorted(image_dir.glob("*.png")):
        base = image_path.stem                      # um_000000
        gt_path = None
        if gt_dir is not None:
            prefix, num = base.split("_", 1)        # "um", "000000"
            candidate = gt_dir / f"{prefix}_road_{num}.png"
            gt_path = candidate if candidate.exists() else None
        yield Sample(base=base, image_path=image_path, gt_path=gt_path)


def list_bases(split: str = config.SPLIT) -> list[str]:
    return [s.base for s in iter_samples(split)]


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def iter_footage_samples() -> Iterator[Sample]:
    """Yield every extracted real-world frame under config.FRAMES_DIR.

    There is no ground truth for footage frames (gt_path is always None) --
    unlike KITTI, "visible road" and the amodal seed must come from the model
    itself (Mask2Former semantics / OFRSNet), not a dataset annotation.
    """
    if not config.FRAMES_DIR.exists():
        raise FileNotFoundError(
            f"No footage frames at {config.FRAMES_DIR}. Run "
            "`python -m src.extract_frames` first."
        )
    for image_path in sorted(config.FRAMES_DIR.rglob("*")):
        if image_path.suffix.lower() in IMAGE_EXTS:
            yield Sample(base=image_path.stem, image_path=image_path, gt_path=None)


def iter_realworld_samples() -> Iterator[Sample]:
    """Yield every photo under config.REALWORLD_DIR. No ground truth, same as
    footage frames -- these are your own real-world photos, not KITTI."""
    if not config.REALWORLD_DIR.exists():
        raise FileNotFoundError(f"No real-world photos at {config.REALWORLD_DIR}.")
    for image_path in sorted(config.REALWORLD_DIR.rglob("*")):
        if image_path.suffix.lower() in IMAGE_EXTS:
            yield Sample(base=image_path.stem, image_path=image_path, gt_path=None)


def iter_source_samples(source: str) -> Iterator[Sample]:
    """Dispatch to the right sample iterator by source name."""
    if source == "kitti":
        yield from iter_samples(config.SPLIT)
    elif source == "footage":
        yield from iter_footage_samples()
    elif source == "realworld":
        yield from iter_realworld_samples()
    else:
        raise ValueError(f"unknown source {source!r} (expected 'kitti', 'footage', or 'realworld')")


# --------------------------------------------------------------------------- #
# Mask IO (single-channel 0/255 PNG)
# --------------------------------------------------------------------------- #
def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_mask(path: Path) -> np.ndarray:
    """Read a binary mask PNG -> bool array (H, W)."""
    return np.asarray(Image.open(path).convert("L")) > 127


def write_mask(path: Path, mask: np.ndarray) -> None:
    """Write a bool/uint8 mask as a 0/255 single-channel PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.asarray(mask).astype(bool).astype(np.uint8)) * 255
    Image.fromarray(arr, mode="L").save(path)


def decode_kitti_road_gt(gt_path: Path) -> np.ndarray:
    """Decode a KITTI Road GT PNG into a binary visible-road mask.

    KITTI encodes the road in the BLUE channel (road pixels are magenta,
    (255, 0, 255)); the red channel marks the valid/evaluated area. So the
    visible road is simply `blue > 0`.
    """
    gt = np.asarray(Image.open(gt_path).convert("RGB"))
    return gt[:, :, 2] > 0


# --------------------------------------------------------------------------- #
# Compositing (used by previews and the annotator)
# --------------------------------------------------------------------------- #
def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color, alpha: float) -> np.ndarray:
    """Alpha-blend a flat colour over `rgb` wherever `mask` is true.

    rgb   : (H, W, 3) uint8
    mask  : (H, W) bool
    color : (r, g, b)
    """
    out = rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    color = np.asarray(color, dtype=np.float32)
    out[m] = (1.0 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Occlusion geometry (shared by training-hint generation, the annotator, and
# inference-time compositing in predict.py)
# --------------------------------------------------------------------------- #
def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def occluder_blob_mask(occluders: np.ndarray, visible: np.ndarray,
                       near_px: int) -> np.ndarray:
    """Whole connected components of `occluders` that touch the near-road band.

    A per-pixel distance test (plain `occluders & dilate(visible, near_px)`)
    only admits the slice of an occluder within `near_px` of the visible
    road -- for a vehicle, that's roughly the tire/undercarriage band, since
    the roof can be 100+ px further away. That silently discards the rest of
    the vehicle's footprint regardless of how well a downstream model (e.g.
    OFRSNet) reconstructs the road under the whole thing.

    Here, instead, any connected blob of `occluders` that touches the near-road
    band at all is admitted IN FULL: the eligibility test is per-object, not
    per-pixel, which is what "this vehicle is occluding some of the road"
    actually means.
    """
    road_neigh = dilate(visible, near_px)
    n_labels, labels = cv2.connectedComponents(occluders.astype(np.uint8), connectivity=8)
    result = np.zeros_like(occluders, dtype=bool)
    for label in range(1, n_labels):  # 0 = background
        comp = labels == label
        if (comp & road_neigh).any():
            result |= comp
    return result


# --------------------------------------------------------------------------- #
# Resolution handling for OFRSNet (fully convolutional -- no fixed input size
# is required; see config.OFRS_MAX_SIDE / OFRS_NET_STRIDE)
# --------------------------------------------------------------------------- #
def fit_within_max_side(labels: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Uniformly downscale (never up) so the longer side is <= max_side.

    Unlike resizing to a fixed target shape, this scales BOTH axes by the same
    factor, so aspect ratio is exactly preserved -- it only exists to bound
    compute on very large images. Returns (labels, scale) so the caller can
    invert it; scale == 1.0 means no resize happened.
    """
    h, w = labels.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return labels, 1.0
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(labels.astype(np.uint8), (new_w, new_h),
                         interpolation=cv2.INTER_NEAREST)
    return resized, scale


def pad_to_multiple(arr: np.ndarray, multiple: int, pad_value) -> np.ndarray:
    """Pad a 2D (H, W[, ...]) array at the bottom/right up to a multiple of
    `multiple` in H and W. The original content always stays the top-left
    [0:h, 0:w] corner, so callers can invert this with a plain crop.
    """
    h, w = arr.shape[:2]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return arr
    pad_width = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (arr.ndim - 2)
    return np.pad(arr, pad_width, mode="constant", constant_values=pad_value)
