"""Camera intrinsics estimation, with a graceful fallback chain.

Every 3D quantity in this project -- the unprojected point cloud, the fitted
ground plane, the per-pixel height `h`, the ground footprint `G` -- is computed
through the camera matrix K. Get K wrong and the whole ground manifold is
wrong: the depth map is metrically fine, but back-projecting it with the wrong
focal length shears the reconstruction.

The project is meant to run on arbitrary user photos, where the camera differs
every time, so we try progressively weaker sources and record which one was
used:

  1. `kitti_calib`  Real, per-image calibration shipped with KITTI (P2 = the
                    rectified colour camera that produces image_2). Exact.
  2. `geocalib`     GeoCalib (Veicht et al., ECCV 2024) -- a learned
                    single-image estimator of focal length AND gravity.
                    Optional dependency; see requirements.txt.
  3. `exif`         EXIF `FocalLengthIn35mmFilm` (or focal-plane resolution),
                    which most phone cameras write. Free and needs no model,
                    but is stripped by many editing/transcoding steps -- in
                    particular ffmpeg drops it when extracting video frames.
  4. `fov_prior`    Last resort: assume config.GEOM_FALLBACK_HFOV_DEG. This is
                    a blind guess and measurably poor -- on our own footage it
                    underestimates the focal length by 35-45%.

GeoCalib additionally returns the **gravity direction**, which is an estimate
of the ground-plane normal that is completely independent of the depth stream.
`src/geometry.py` uses it both as GroundNet's geometric-consistency check
(Eq. 11) and as a robust prior when the depth-based plane fit is unreliable.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

_CACHE: dict = {}


@dataclass
class Calibration:
    K: np.ndarray                  # 3x3, valid for the (w, h) it was built at
    source: str                    # kitti_calib | geocalib | exif | fov_prior
    gravity: np.ndarray | None = None      # unit 3-vector, camera frame
    focal_uncertainty: float | None = None  # px, GeoCalib only

    @property
    def calibrated(self) -> bool:
        """True when K came from real calibration or a learned estimate."""
        return self.source in ("kitti_calib", "geocalib", "exif")


# --------------------------------------------------------------------------- #
# Basic constructors
# --------------------------------------------------------------------------- #
def K_from_focal(f_px: float, w: int, h: int) -> np.ndarray:
    """Pinhole K with a principal point at the image centre."""
    return np.array([[f_px, 0.0, w / 2.0],
                     [0.0, f_px, h / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def K_from_fov(w: int, h: int, hfov_deg: float) -> np.ndarray:
    return K_from_focal((w / 2.0) / np.tan(np.radians(hfov_deg) / 2.0), w, h)


def scale_K(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64).copy()
    K[0, :] *= sx
    K[1, :] *= sy
    return K


# --------------------------------------------------------------------------- #
# 1. KITTI calibration
# --------------------------------------------------------------------------- #
def K_from_kitti(base: str) -> np.ndarray | None:
    """Parse P2 from KITTI's calib file; its 3x3 block is the intrinsic matrix."""
    calib = config.KITTI_ROOT / config.SPLIT / "calib" / f"{base}.txt"
    if not calib.exists():
        return None
    try:
        for line in calib.read_text().splitlines():
            if line.startswith("P2:"):
                vals = [float(v) for v in line.split()[1:]]
                return np.array(vals, dtype=np.float64).reshape(3, 4)[:, :3]
    except Exception:  # noqa: BLE001
        return None
    return None


# --------------------------------------------------------------------------- #
# 2. GeoCalib (optional dependency)
# --------------------------------------------------------------------------- #
def geocalib_available() -> bool:
    try:
        import geocalib  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _geocalib_model(device):
    key = f"geocalib::{config.GEOCALIB_WEIGHTS}"
    if key not in _CACHE:
        from geocalib import GeoCalib
        _CACHE[key] = GeoCalib(weights=config.GEOCALIB_WEIGHTS).to(device)
    return _CACHE[key]


def K_from_geocalib(image_path: Path, device) -> Calibration | None:
    """Run GeoCalib; returns K at the image's own resolution, plus gravity."""
    if not geocalib_available():
        return None
    try:
        import torch
        model = _geocalib_model(device)
        img = model.load_image(str(image_path)).to(device)
        with torch.no_grad():
            res = model.calibrate(img)
        cam = res["camera"]
        f = cam.f[0].detach().cpu().numpy()            # (fx, fy)
        c = cam.c[0].detach().cpu().numpy()            # (cx, cy)
        grav = res["gravity"].vec3d[0].detach().cpu().numpy().astype(np.float64)
        unc = res.get("focal_uncertainty")
        unc = float(unc[0].detach().cpu()) if unc is not None else None

        K = np.array([[float(f[0]), 0.0, float(c[0])],
                      [0.0, float(f[1]), float(c[1])],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        grav = grav / (np.linalg.norm(grav) + 1e-12)
        return Calibration(K=K, source="geocalib", gravity=grav,
                           focal_uncertainty=unc)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] GeoCalib failed on {getattr(image_path, 'name', image_path)}: {e}")
        return None


# --------------------------------------------------------------------------- #
# 3. EXIF
# --------------------------------------------------------------------------- #
def K_from_exif(image_path: Path, w: int, h: int) -> np.ndarray | None:
    """Derive focal length in pixels from EXIF metadata, if present.

    Preferred tag is `FocalLengthIn35mmFilm`: 35 mm format is 36 mm wide, and
    that 36 mm maps to the image's LONGER side, so
        f_px = (f_35mm / 36) * max(w, h)
    Falls back to `FocalLength` + `FocalPlaneXResolution`, which together give
    the physical sensor width.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        with Image.open(image_path) as im:
            raw = im.getexif()
            if not raw:
                return None
            exif = {TAGS.get(k, k): v for k, v in raw.items()}
            # merge the ExifOffset sub-IFD, where focal-plane tags usually live
            try:
                for k, v in raw.get_ifd(0x8769).items():
                    exif.setdefault(TAGS.get(k, k), v)
            except Exception:  # noqa: BLE001
                pass

        f35 = exif.get("FocalLengthIn35mmFilm")
        if f35:
            f35 = float(f35)
            if f35 > 0:
                return K_from_focal(f35 / 36.0 * max(w, h), w, h)

        f_mm = exif.get("FocalLength")
        fp_res = exif.get("FocalPlaneXResolution")
        fp_unit = exif.get("FocalPlaneResolutionUnit", 2)  # 2 = inch, 3 = cm
        if f_mm and fp_res:
            f_mm = float(f_mm)
            fp_res = float(fp_res)
            per_mm = fp_res / (25.4 if int(fp_unit) == 2 else 10.0)
            sensor_w_mm = w / per_mm
            if sensor_w_mm > 0:
                return K_from_focal(f_mm / sensor_w_mm * w, w, h)
    except Exception:  # noqa: BLE001
        return None
    return None


# --------------------------------------------------------------------------- #
# The chain
# --------------------------------------------------------------------------- #
def calibrate(image_path: Path | None, base: str | None, w: int, h: int,
              orig_w: int | None = None, orig_h: int | None = None,
              device=None, allow_geocalib: bool = True) -> Calibration:
    """Best available intrinsics for one image, scaled to (w, h).

    `orig_w/orig_h` describe the resolution the source calibration refers to
    (the image on disk); K is rescaled to the working resolution (w, h).
    """
    orig_w = orig_w or w
    orig_h = orig_h or h

    def _scaled(K):
        return scale_K(K, w / orig_w, h / orig_h) if (orig_w, orig_h) != (w, h) else K

    for source in config.INTRINSICS_PRIORITY:
        if source == "kitti_calib" and base:
            K = K_from_kitti(base)
            if K is not None:
                return Calibration(K=_scaled(K), source="kitti_calib")

        elif source == "geocalib" and allow_geocalib and image_path is not None:
            cal = K_from_geocalib(Path(image_path), device)
            if cal is not None:
                # GeoCalib runs on the file, so its K refers to the on-disk size.
                cal.K = _scaled(cal.K)
                return cal

        elif source == "exif" and image_path is not None:
            K = K_from_exif(Path(image_path), orig_w, orig_h)
            if K is not None:
                return Calibration(K=_scaled(K), source="exif")

        elif source == "fov_prior":
            return Calibration(K=K_from_fov(w, h, config.GEOM_FALLBACK_HFOV_DEG),
                               source="fov_prior")

    return Calibration(K=K_from_fov(w, h, config.GEOM_FALLBACK_HFOV_DEG),
                       source="fov_prior")
