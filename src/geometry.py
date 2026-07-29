"""Ground-manifold geometry, derived from GroundNet's depth stream.

GroundNet (Man, Weng, Li, Kitani -- "GroundNet: Monocular Ground Plane Normal
Estimation with Geometric Consistency", 2019) estimates a ground-plane normal
by unprojecting ground-segmented pixels to 3D with monocular depth and fitting
a plane (Eq. 1-5), optionally cross-checking it against an independently
predicted surface-normal map (Eq. 6-7, 11).

What we take from it, and what we change:

  * Ground segmentation stream -> we REUSE our existing Mask2Former OFRS
    semantic map (the `road` class) instead of training GroundNet's VGG head.
    No duplicate model, and it is the same segmentation the rest of the
    pipeline already trusts.
  * Depth stream -> Depth Anything V2 (metric, outdoor) in place of
    GroundNet's jointly-trained depth head, then the paper's own geometry:
    pinhole unprojection (Eq. 1-2), ground isolation (Eq. 3), plane fit
    (Eq. 4-5) with RANSAC (the inference-time equivalent of the paper's DSAC,
    whose differentiability only matters for *training* hypothesis selection).
  * Surface-normal stream -> OPTIONAL and off by default (needs `diffusers`).
    The paper's geometric-consistency loss (Eq. 11) is a *training* signal for
    their jointly-trained heads; with independently pretrained substitutes we
    can still compute the consistency ANGLE as a quality/trust signal, but it
    cannot back-propagate into anything here.
  * NEW, and the reason this module exists: instead of collapsing everything
    to one 3-vector normal (GroundNet's final output), we keep two DENSE
    per-pixel fields that a downstream network can do message passing over --
    see `derive_ground_fields`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import calibration  # noqa: E402

_MODELS: dict = {}


# --------------------------------------------------------------------------- #
# Camera intrinsics
# --------------------------------------------------------------------------- #
# Intrinsics estimation lives in src/calibration.py (KITTI calib -> GeoCalib ->
# EXIF -> assumed FOV). These are thin re-exports so callers and older code
# keep working.
intrinsics_from_fov = calibration.K_from_fov
scale_intrinsics = calibration.scale_K


def calibrate_sample(image_path, base: str | None, w: int, h: int,
                     orig_w: int | None = None, orig_h: int | None = None,
                     device=None) -> calibration.Calibration:
    """Best-available intrinsics (+ gravity when GeoCalib supplies it)."""
    return calibration.calibrate(image_path, base, w, h,
                                 orig_w=orig_w, orig_h=orig_h, device=device)


# --------------------------------------------------------------------------- #
# Depth stream
# --------------------------------------------------------------------------- #
def get_depth_model(device):
    if "depth" not in _MODELS:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        mid = config.DEPTH_MODEL_ID
        _MODELS["depth_proc"] = AutoImageProcessor.from_pretrained(mid)
        _MODELS["depth"] = AutoModelForDepthEstimation.from_pretrained(mid).to(device).eval()
    return _MODELS["depth_proc"], _MODELS["depth"]


def estimate_depth_metric(rgb: np.ndarray, device) -> np.ndarray:
    """Raw metric depth in metres, (H, W) float32.

    Uses `post_process_depth_estimation` directly rather than the HF `pipeline`
    wrapper, which quantises its output to an 8-bit visualisation image.
    """
    import torch
    proc, model = get_depth_model(device)
    h, w = rgb.shape[:2]
    with torch.no_grad():
        inp = proc(images=rgb, return_tensors="pt").to(device)
        out = model(**inp)
        depth = proc.post_process_depth_estimation(out, target_sizes=[(h, w)])[0]
        depth = depth["predicted_depth"]
    return depth.float().cpu().numpy()


# --------------------------------------------------------------------------- #
# Eq. 1-2: pinhole unprojection;  Eq. 3: ground isolation
# --------------------------------------------------------------------------- #
def unproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Lift EVERY pixel to 3D camera coordinates -> (H, W, 3) float32."""
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    # Clamp non-finite/extreme depth so downstream products cannot overflow.
    z = np.nan_to_num(depth.astype(np.float32), nan=0.0,
                      posinf=0.0, neginf=0.0)
    z = np.clip(z, 0.0, 1e4)
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def unproject_ground_points(depth, ground_mask, K,
                            max_depth=config.GEOM_MAX_DEPTH_M) -> np.ndarray:
    """Ground-masked 3D point cloud, (N, 3). Paper caps depth at 30 m so far,
    noisy pixels do not bias the plane fit (Sec. 5.4)."""
    ys, xs = np.where(ground_mask)
    if len(ys) == 0:
        return np.empty((0, 3), np.float32)
    z = depth[ys, xs].astype(np.float64)
    keep = np.isfinite(z) & (z > 0.1) & (z < max_depth)
    xs, ys, z = xs[keep], ys[keep], z[keep]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    pts = np.column_stack([X, Y, z])
    return pts[np.all(np.isfinite(pts), axis=1)].astype(np.float32)


# --------------------------------------------------------------------------- #
# Eq. 4-5: least-squares plane;  RANSAC variant (inference-time DSAC analogue)
# --------------------------------------------------------------------------- #
def fit_plane_ls(points: np.ndarray) -> np.ndarray | None:
    """Solve C n = 1 in the least-squares sense (Eq. 4-5).

    Returns n with the plane parameterised as n . X = 1, so ||n|| is 1/distance
    to the plane -- we deliberately do NOT normalise here, because the offset
    is what lets us measure height above the plane later.
    """
    if len(points) < 3:
        return None
    C = points.astype(np.float64)
    try:
        n = np.linalg.solve(C.T @ C, C.T @ np.ones(len(C)))
    except np.linalg.LinAlgError:
        return None
    return n if np.all(np.isfinite(n)) and np.linalg.norm(n) > 1e-9 else None


def fit_plane_ransac(points: np.ndarray,
                     dist_thresh=config.GEOM_RANSAC_THRESH_M,
                     iters=config.GEOM_RANSAC_ITERS, seed=0) -> np.ndarray | None:
    """RANSAC plane fit, then refit on inliers.

    The paper uses DSAC (differentiable RANSAC) so hypothesis selection can be
    back-propagated through during *training*. At inference DSAC's expected-loss
    selection reduces to ordinary RANSAC scoring + inlier refit, which is what
    we do -- we never train these weights, so differentiability buys nothing.
    """
    points = np.asarray(points, dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        return None
    rng = np.random.default_rng(seed)
    n_pts = len(points)
    best_inliers, best_count = None, -1
    for _ in range(iters):
        idx = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(normal)
        if nn < 1e-9 or not np.isfinite(nn):
            continue
        normal = normal / nn
        # einsum, not `@` -- see the note in derive_ground_fields about the
        # spurious BLAS FP flag on this platform.
        dist = np.abs(np.einsum("nc,c->n", points - p0, normal))
        inliers = np.isfinite(dist) & (dist < dist_thresh)
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers
    if best_inliers is None or best_inliers.sum() < 3:
        return fit_plane_ls(points)
    return fit_plane_ls(points[best_inliers])


def canonical_unit_normal(n: np.ndarray | None) -> np.ndarray | None:
    """Unit-length plane normal, sign-fixed to point 'up' -- for REPORTING only.

    In OpenCV camera convention (x right, y DOWN, z forward) up is -y.

    Do NOT apply this to the `n . X = 1` plane parameterisation used elsewhere
    in this module: there the sign is fixed by the data (it encodes the plane's
    offset as well as its direction), so negating n describes a DIFFERENT plane
    -- the mirror image through the camera centre -- rather than the same plane
    with a flipped normal. This function is only for producing a comparable
    normal direction (e.g. against GroundNet's reported n, or for a horizon
    line), never for feeding `derive_ground_fields`.
    """
    if n is None:
        return None
    u = np.asarray(n, dtype=np.float64)
    u = u / (np.linalg.norm(u) + 1e-12)
    return -u if u[1] > 0 else u


def fit_plane_with_up_prior(points: np.ndarray, up: np.ndarray) -> np.ndarray | None:
    """Fit only the plane OFFSET, with the normal direction fixed by `up`.

    A 1-DOF fit instead of 3-DOF. When the normal direction comes from an
    independent, reliable source (GeoCalib's gravity), this is far more robust
    than a free fit on noisy monocular depth -- especially when only a small
    strip of road is visible, where a free RANSAC fit can latch onto a wrong
    surface entirely.

    Plane: u . X = c, solved robustly with the median; returned in the project's
    `n . X = 1` form as n = u / c (which also fixes the sign automatically).
    """
    points = np.asarray(points, dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        return None
    u = np.asarray(up, dtype=np.float64)
    u = u / (np.linalg.norm(u) + 1e-12)
    c = float(np.median(np.einsum("nc,c->n", points, u)))
    if abs(c) < 1e-6:
        return None
    return u / c


def estimate_ground_plane(depth, ground_mask, K, seed=0, gravity=None):
    """Depth-stream ground plane, optionally cross-checked against gravity.

    Returns (n, info) where `n` uses the `n . X = 1` parameterisation (Eq. 4-5)
    with its sign determined by the fit -- see `canonical_unit_normal` for why
    we must not "canonicalise" it here -- and `info` records which estimate was
    used and how well the two streams agreed.

    With `gravity` supplied (from GeoCalib), this implements GroundNet's
    two-stream idea with a genuinely independent second stream: the free depth
    fit and the gravity direction are compared (Eq. 11), and if they disagree
    by more than config.GEOM_GRAVITY_MAX_ANGLE_DEG the depth fit is rejected in
    favour of the gravity-constrained fit.
    """
    info = {"n_points": 0, "mode": "none", "consistency_deg": float("nan")}
    pts = unproject_ground_points(depth, ground_mask, K)
    info["n_points"] = len(pts)
    if len(pts) < 3:
        return None, info

    n_free = fit_plane_ransac(pts, seed=seed)

    if gravity is None:
        info["mode"] = "depth_ransac"
        return n_free, info

    n_prior = fit_plane_with_up_prior(pts, gravity)
    if n_free is None:
        info["mode"] = "gravity_prior"
        return n_prior, info

    ang = consistency_angle_deg(canonical_unit_normal(n_free),
                               canonical_unit_normal(gravity))
    info["consistency_deg"] = ang
    if np.isfinite(ang) and ang > config.GEOM_GRAVITY_MAX_ANGLE_DEG and n_prior is not None:
        info["mode"] = "gravity_prior"      # depth fit disagrees -> distrust it
        return n_prior, info
    info["mode"] = "depth_ransac"           # streams agree -> keep the free fit
    return n_free, info


# --------------------------------------------------------------------------- #
# Eq. 11: geometric consistency (optional surface-normal stream)
# --------------------------------------------------------------------------- #
def consistency_angle_deg(n_a, n_b) -> float:
    if n_a is None or n_b is None:
        return float("nan")
    a = n_a / (np.linalg.norm(n_a) + 1e-12)
    b = n_b / (np.linalg.norm(n_b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


# --------------------------------------------------------------------------- #
# THE DENSE FIELDS -- what the multimodal context module actually consumes
# --------------------------------------------------------------------------- #
def derive_ground_fields(depth: np.ndarray, n: np.ndarray | None, K: np.ndarray,
                         max_ground_dist=config.GEOM_MAX_GROUND_DIST_M):
    """Turn (depth, plane, intrinsics) into the two dense fields below.

    Returns (G, h, gvalid):

      G       (H, W, 3) float32 -- GROUND FOOTPRINT. For each pixel, where its
              viewing ray meets the ground plane. Depends only on K and the
              plane, NOT on depth, so it is defined even where depth is wrong
              or the pixel is occluded. This is the key quantity for amodal
              completion: for a pixel on a car roof, G is exactly the patch of
              road that car is hiding, so "which pixels describe nearby road?"
              stays answerable through the occluder.
      h       (H, W) float32 -- signed height above the plane of the pixel's
              OWN 3D point (from depth). |h| ~ 0 means the pixel really lies on
              the ground manifold; large |h| means it is something standing on
              (or floating above) the road. This is the manifold-membership
              signal.
      gvalid  (H, W) bool -- ray actually meets the plane in front of the
              camera. False above the horizon, where no footprint exists.

    With the plane parameterised n . X = 1:
        ray(u,v) = K^-1 [u, v, 1]^T ,  t = 1 / (n . ray) ,  G = t * ray
        h(p)     = n . Q(p) - 1  , scaled to metres by 1/||n||
    """
    hgt, wid = depth.shape
    if n is None or not np.all(np.isfinite(n)) or np.linalg.norm(n) < 1e-9:
        return (np.zeros((hgt, wid, 3), np.float32),
                np.zeros((hgt, wid), np.float32),
                np.zeros((hgt, wid), bool))

    n = np.asarray(n, dtype=np.float64)
    n_norm = np.linalg.norm(n)

    # --- G: ray-plane intersection, per pixel (depth-independent) ---
    # NOTE: `einsum` rather than `@` throughout this function. They are
    # mathematically identical here, but on this platform BLAS raises a
    # spurious "divide by zero encountered in matmul" FP flag for `@` on
    # (H, W, 3) arrays; einsum computes the same values without it (verified:
    # np.allclose(matmul, einsum) is True).
    us, vs = np.meshgrid(np.arange(wid, dtype=np.float64),
                         np.arange(hgt, dtype=np.float64))
    pix = np.stack([us, vs, np.ones_like(us)], axis=-1)      # (H, W, 3)
    rays = np.einsum("hwc,dc->hwd", pix, np.linalg.inv(K))    # (H, W, 3)
    denom = np.einsum("hwc,c->hw", rays, n)                   # (H, W)
    gvalid = denom > 1e-6                                     # in front of camera
    t = np.where(gvalid, 1.0 / np.where(gvalid, denom, 1.0), 0.0)
    G = rays * t[..., None]

    # Clamp far-field: near the horizon t -> inf, which would dominate any
    # pairwise distance term. Rescale over-long rays back onto the cap.
    dist = np.linalg.norm(G, axis=-1)
    too_far = dist > max_ground_dist
    scale = np.where(too_far & (dist > 1e-9), max_ground_dist / np.maximum(dist, 1e-9), 1.0)
    G = G * scale[..., None]
    G = np.where(gvalid[..., None], G, 0.0)

    # --- h: height of each pixel's own 3D point above the plane ---
    Q = unproject(depth, K).astype(np.float64)                # (H, W, 3)
    h = (np.einsum("hwc,c->hw", Q, n) - 1.0) / n_norm         # metres
    h = np.where(np.isfinite(h), h, 0.0)

    return G.astype(np.float32), h.astype(np.float32), gvalid


# --------------------------------------------------------------------------- #
# Cache IO
# --------------------------------------------------------------------------- #
def geometry_path(base: str) -> Path:
    return config.GEOMETRY_DIR / f"{base}.npz"


def save_geometry(base: str, depth, n, K, calibrated: bool,
                  n_ground_pts: int, consistency_deg: float = float("nan"),
                  k_source: str = "unknown", gravity=None,
                  plane_mode: str = "unknown") -> None:
    config.GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        geometry_path(base),
        depth=depth.astype(np.float16),          # smooth field; fp16 is plenty
        n=(np.zeros(3) if n is None else np.asarray(n)).astype(np.float32),
        K=np.asarray(K, np.float32),
        valid=np.array(n is not None),
        calibrated=np.array(bool(calibrated)),
        n_ground_pts=np.array(int(n_ground_pts)),
        consistency_deg=np.array(float(consistency_deg), np.float32),
        # Provenance: which intrinsics source was used, whether gravity was
        # available, and which plane estimate won. Recorded so a cache built
        # with a weak source can be spotted and re-exported later.
        k_source=np.array(str(k_source)),
        gravity=(np.zeros(3) if gravity is None else np.asarray(gravity)).astype(np.float32),
        has_gravity=np.array(gravity is not None),
        plane_mode=np.array(str(plane_mode)),
    )


def load_geometry(base: str):
    """Return dict with depth/n/K/valid + provenance, or None if not cached."""
    p = geometry_path(base)
    if not p.exists():
        return None
    z = np.load(p)

    def _opt(key, default):
        return z[key] if key in z.files else default

    return dict(
        depth=z["depth"].astype(np.float32),
        n=z["n"].astype(np.float64),
        K=z["K"].astype(np.float64),
        valid=bool(z["valid"]),
        calibrated=bool(z["calibrated"]),
        n_ground_pts=int(z["n_ground_pts"]),
        consistency_deg=float(z["consistency_deg"]),
        k_source=str(_opt("k_source", "unknown")),
        gravity=(np.asarray(_opt("gravity", np.zeros(3)), np.float64)
                 if bool(_opt("has_gravity", False)) else None),
        plane_mode=str(_opt("plane_mode", "unknown")),
    )


def load_ground_fields(base: str, out_hw: tuple[int, int] | None = None):
    """Load the cache and derive (G, h, gvalid), optionally resized to out_hw.

    Resizing rescales K accordingly so the geometry stays metrically correct.
    """
    geo = load_geometry(base)
    if geo is None:
        return None
    depth, K = geo["depth"], geo["K"]
    if out_hw is not None and depth.shape != tuple(out_hw):
        oh, ow = depth.shape
        th, tw = out_hw
        depth = cv2.resize(depth, (tw, th), interpolation=cv2.INTER_AREA)
        K = scale_intrinsics(K, tw / ow, th / oh)
    n = geo["n"] if geo["valid"] else None
    G, h, gvalid = derive_ground_fields(depth, n, K)
    return dict(G=G, h=h, gvalid=gvalid, valid=geo["valid"])
