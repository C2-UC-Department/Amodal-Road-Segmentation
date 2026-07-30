"""Bird's-eye view: metric top-down rasters of the fitted ground plane.

No new geometry is estimated here. Step 7 already caches, per image, the
intrinsics K and the RANSAC ground plane n in the `n . X = 1` parameterisation
(see src/geometry.py). Those two alone define an exact homography between the
image and a metric top-down grid on that plane:

    a ground point (X, Z) has  Y = (1 - n_x X - n_z Z) / n_y

which is AFFINE in (X, Z), so the whole image <-> BEV relation is a single 3x3
matrix -- no per-pixel scatter, no hole filling, one cv2.warpPerspective.

This is the analytic inverse of `geometry.derive_ground_fields`, which computes
the same plane's forward ray intersection G = t * K^-1 [u,v,1]^T. Consequence
worth stating: the BEV is guaranteed consistent with the geometry OFRSNet was
actually trained on -- same plane, same intrinsics, same sign convention -- and
`tests/test_bev.py` asserts the two agree rather than trusting a fixture.

Conventions (identical to src/geometry.py): OpenCV camera frame, x right,
y DOWN, z forward; n in 1/metres; K in pixels at the resolution it was built
for. BEV row v=0 is the FAR edge, so a raster reads like a map with the camera
at the bottom.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BevGrid:
    """A metric top-down raster: ppm pixels per metre over [x, z] extents."""

    ppm: float
    x_min: float
    x_max: float
    z_min: float
    z_max: float

    @classmethod
    def from_config(cls, ppm: float | None = None,
                    x_range: tuple[float, float] | None = None,
                    z_range: tuple[float, float] | None = None) -> "BevGrid":
        xr = x_range if x_range is not None else config.BEV_RANGE_X_M
        zr = z_range if z_range is not None else config.BEV_RANGE_Z_M
        return cls(ppm=float(ppm if ppm is not None else config.BEV_PPM),
                   x_min=float(xr[0]), x_max=float(xr[1]),
                   z_min=float(zr[0]), z_max=float(zr[1]))

    @property
    def width(self) -> int:
        return max(1, int(round((self.x_max - self.x_min) * self.ppm)))

    @property
    def height(self) -> int:
        return max(1, int(round((self.z_max - self.z_min) * self.ppm)))

    @property
    def cell_area_m2(self) -> float:
        return 1.0 / (self.ppm ** 2)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def world_to_pixel(self, x_m: float, z_m: float) -> tuple[int, int]:
        """Ground (X, Z) metres -> (u, v) BEV pixel. Inverse of `_S`."""
        u = (x_m - self.x_min) * self.ppm - 0.5
        v = (self.z_max - z_m) * self.ppm - 0.5
        return int(round(u)), int(round(v))


# --------------------------------------------------------------------------- #
# Plane sanity & metric scale
# --------------------------------------------------------------------------- #
def implied_camera_height(n: np.ndarray) -> float:
    """Distance from the camera centre to the plane, in metres.

    With `n . X = 1`, the origin (camera centre) is at signed distance
    -1/||n|| from the plane, so the plane sits 1/||n|| away. For a ground
    plane that IS the camera height -- the quantity the README compares
    against KITTI's true ~1.65 m.
    """
    norm = float(np.linalg.norm(np.asarray(n, dtype=np.float64)))
    if norm < 1e-9:
        return float("inf")
    return 1.0 / norm


def plane_is_usable(n: np.ndarray | None) -> bool:
    """Reject plane fits that cannot describe a ground plane below the camera.

    `n_y > 0` is a real constraint, not just a divide-by-zero guard: the plane's
    nearest point to the camera is n/||n||^2, so for the plane to be BELOW the
    camera (y is DOWN) its normal must have a positive y component. A fit with
    n_y <= 0 describes a plane above or through the camera and would produce a
    mirrored, meaningless raster.
    """
    if n is None:
        return False
    n = np.asarray(n, dtype=np.float64)
    if n.shape != (3,) or not np.all(np.isfinite(n)):
        return False
    if np.linalg.norm(n) < 1e-9 or n[1] <= 1e-9:
        return False
    h = implied_camera_height(n)
    return config.BEV_MIN_CAMERA_HEIGHT_M <= h <= config.BEV_MAX_CAMERA_HEIGHT_M


def rescale_plane_to_height(n: np.ndarray,
                            camera_height_m: float) -> tuple[np.ndarray, float]:
    """Rescale the metric world so the camera sits at a known height.

    Monocular metric depth carries a scale error (the README measures ~2.71 m
    implied camera height against a true ~1.65 m on KITTI), which would inflate
    any AREA by that factor squared. Because the plane fit exposes the camera
    height directly as 1/||n||, a uniform rescale of n corrects the whole metric
    world -- and, unlike a post-hoc area fudge, it also makes BEV_RANGE_X_M /
    BEV_RANGE_Z_M mean actual metres so the raster covers the intended ground.

    Returns (n_scaled, implied_height_before). Scaling n by s maps the implied
    height to (1/||n||)/s, so s = implied / target.
    """
    n = np.asarray(n, dtype=np.float64)
    implied = implied_camera_height(n)
    if not np.isfinite(implied) or camera_height_m <= 0:
        return n, implied
    return n * (implied / float(camera_height_m)), implied


# --------------------------------------------------------------------------- #
# The homography
# --------------------------------------------------------------------------- #
def _S(grid: BevGrid) -> np.ndarray:
    """BEV pixel [u, v, 1] -> ground [X, Z, 1], metres.

    Pixel (u, v) samples the CENTRE of its cell, hence the +0.5 offsets --
    without them a mask's area would be biased by half a cell along each axis.
    """
    inv = 1.0 / grid.ppm
    return np.array([
        [inv,  0.0, grid.x_min + 0.5 * inv],
        [0.0, -inv, grid.z_max - 0.5 * inv],
        [0.0,  0.0, 1.0],
    ], dtype=np.float64)


def _M(n: np.ndarray) -> np.ndarray:
    """Ground [X, Z, 1] -> camera-frame [Xc, Yc, Zc], via Y on the plane."""
    n = np.asarray(n, dtype=np.float64)
    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
    return np.array([
        [1.0, 0.0, 0.0],
        [-nx / ny, -nz / ny, 1.0 / ny],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)


def homography_bev_to_image(K: np.ndarray, n: np.ndarray,
                            grid: BevGrid) -> np.ndarray:
    """3x3 mapping a BEV pixel to the image pixel that observes that ground cell.

    Feed this straight to cv2.warpPerspective with WARP_INVERSE_MAP (that flag
    means "the matrix I gave you maps dst to src", which is exactly this).

    Zc == Z > 0 for every cell because the grid starts at z_min > 0, so no cell
    can land behind the camera and no horizon test is needed -- cells simply fall
    outside the image, which `bev_validity` reports.
    """
    if not plane_is_usable(n):
        raise ValueError(f"plane normal {np.asarray(n).tolist()} is not a usable "
                         "ground plane (see bev.plane_is_usable)")
    return np.asarray(K, dtype=np.float64) @ _M(n) @ _S(grid)


# --------------------------------------------------------------------------- #
# Warping
# --------------------------------------------------------------------------- #
def warp_to_bev(src: np.ndarray, H: np.ndarray, grid: BevGrid,
                nearest: bool = False) -> np.ndarray:
    """Warp an image-space array into the BEV grid, preserving its dtype.

    cv2.warpPerspective has no int32/bool support, so labels and masks round-trip
    through a type it does handle; `nearest` is forced for anything discrete so
    label values are never blended into ids that do not exist.
    """
    src = np.asarray(src)
    kind = src.dtype.kind
    if src.dtype == bool:
        work, restore = src.astype(np.uint8), lambda a: a > 0
        nearest = True
    elif kind in "iu" and src.dtype != np.uint8:
        # int32 instance-id maps: float32 is exact for ids well under 2^24.
        work, restore = src.astype(np.float32), lambda a: a.astype(src.dtype)
        nearest = True
    else:
        work, restore = src, lambda a: a

    flags = (cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR) | cv2.WARP_INVERSE_MAP
    out = cv2.warpPerspective(work, np.asarray(H, dtype=np.float64),
                              (grid.width, grid.height), flags=flags,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return restore(out)


def bev_validity(H: np.ndarray, grid: BevGrid,
                 image_hw: tuple[int, int]) -> np.ndarray:
    """(Hb, Wb) bool: which BEV cells are actually observed by the image.

    Warping a solid image through the same H makes the border handling define
    validity, so this can never disagree with `warp_to_bev`.
    """
    solid = np.full(tuple(image_hw), 255, np.uint8)
    return warp_to_bev(solid, H, grid, nearest=True) > 127


# --------------------------------------------------------------------------- #
# How much ground one camera pixel is responsible for
# --------------------------------------------------------------------------- #
def ground_m2_per_pixel(H: np.ndarray, grid: BevGrid) -> np.ndarray:
    """(Hb, Wb) float64: ground area each BEV cell's SOURCE pixel covers.

    Being in frame is not the same as being measurable. Ground resolution decays
    with roughly the CUBE of range (a pixel's depth extent goes as Z^2/(fy*h) and
    its width as Z/fx), so for a typical dashcam one pixel covers ~0.001 m^2 at
    10 m but ~0.08 m^2 at 40 m. Near the horizon a couple of pixels of mask
    disagreement therefore become tens of square metres, and any area difference
    measured out there is boundary noise wearing a metric costume.

    For a homography the Jacobian determinant has a closed form,
    det J = det(H) / (h3 . q)^3 at q = [u, v, 1], so this costs one array
    expression and needs no numerical differencing.
    """
    H = np.asarray(H, dtype=np.float64)
    us, vs = np.meshgrid(np.arange(grid.width, dtype=np.float64),
                         np.arange(grid.height, dtype=np.float64))
    denom = H[2, 0] * us + H[2, 1] * vs + H[2, 2]
    # det J = image px^2 per BEV cell; invert to get ground area per image pixel.
    img_px_per_cell = np.abs(np.linalg.det(H)) / np.maximum(np.abs(denom) ** 3, 1e-30)
    return grid.cell_area_m2 / np.maximum(img_px_per_cell, 1e-30)


def measurable_mask(H: np.ndarray, grid: BevGrid, image_hw: tuple[int, int],
                    max_m2_per_px: float | None = None) -> tuple[np.ndarray, dict]:
    """BEV cells that are both observed AND resolved well enough to measure.

    Returns (mask, info) where info reports the furthest range that survived, so
    callers can tell the user what was actually measured rather than implying the
    whole grid was.
    """
    limit = (config.BEV_MAX_M2_PER_PIXEL if max_m2_per_px is None
             else float(max_m2_per_px))
    in_frame = bev_validity(H, grid, image_hw)
    resolved = ground_m2_per_pixel(H, grid) <= limit
    mask = in_frame & resolved

    rows = np.nonzero(mask.any(axis=1))[0]
    # Row 0 is the far edge, so the smallest surviving row index is the range cap.
    z_max_used = (grid.z_max - (rows.min() + 0.5) / grid.ppm) if len(rows) else 0.0
    return mask, {
        "max_m2_per_px": limit,
        "measurable_range_m": round(float(z_max_used), 2),
        "in_frame_pct": round(100.0 * float(in_frame.mean()), 2),
        "measurable_pct": round(100.0 * float(mask.mean()), 2),
    }


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def area_m2(mask: np.ndarray, grid: BevGrid) -> float:
    """Ground area of a BEV mask. Every cell is the same size -- that is the
    whole point of rectifying to the plane first."""
    return float(np.count_nonzero(mask)) * grid.cell_area_m2


# --------------------------------------------------------------------------- #
# Drawing (grid furniture only; mask compositing lives in src/disturbance.py)
# --------------------------------------------------------------------------- #
def draw_grid_overlay(canvas: np.ndarray, grid: BevGrid, step_m: float = 5.0,
                      color=(70, 70, 70), text_color=(150, 150, 150)) -> np.ndarray:
    """Metre gridlines, the X=0 centreline, and a camera marker."""
    out = canvas.copy()
    z = np.ceil(grid.z_min / step_m) * step_m
    while z <= grid.z_max:
        _, v = grid.world_to_pixel(0.0, z)
        if 0 <= v < grid.height:
            cv2.line(out, (0, v), (grid.width - 1, v), color, 1)
            cv2.putText(out, f"{z:.0f}m", (4, max(12, v - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1, cv2.LINE_AA)
        z += step_m

    x = np.ceil(grid.x_min / step_m) * step_m
    while x <= grid.x_max:
        u, _ = grid.world_to_pixel(x, grid.z_min)
        if 0 <= u < grid.width:
            cv2.line(out, (u, 0), (u, grid.height - 1), color, 1)
        x += step_m

    u0, _ = grid.world_to_pixel(0.0, grid.z_min)
    if 0 <= u0 < grid.width:
        cv2.line(out, (u0, 0), (u0, grid.height - 1), (110, 110, 110), 1)
        # The camera sits at Z=0, which is outside the grid (z_min > 0); mark the
        # near edge instead so the viewer knows which way is "towards the camera".
        cv2.drawMarker(out, (u0, grid.height - 6), (255, 255, 255),
                       cv2.MARKER_TRIANGLE_UP, 10, 2)
    return out
