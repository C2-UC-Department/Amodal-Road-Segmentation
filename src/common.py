"""Shared helpers: device selection, KITTI file discovery, mask IO, compositing."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, NamedTuple

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


def iter_source_samples(source: str) -> Iterator[Sample]:
    """Dispatch to the right sample iterator by source name."""
    if source == "kitti":
        yield from iter_samples(config.SPLIT)
    elif source == "footage":
        yield from iter_footage_samples()
    else:
        raise ValueError(f"unknown source {source!r} (expected 'kitti' or 'footage')")


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
