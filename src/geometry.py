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
    """Best-available intrinsics K for this image (see src/calibration.py)."""
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


def estimate_ground_plane(depth, ground_mask, K, seed=0):
    """Full depth stream: returns (n, n_points_used). n is None on failure.

    `n` uses the `n . X = 1` parameterisation (Eq. 4-5) with its sign determined
    by the fit -- see `canonical_unit_normal` for why we must not "canonicalise"
    it here.

    This is GroundNet's depth stream unchanged: unproject the road pixels with
    K, then RANSAC a plane. The ONLY thing calibration affects is K (see
    src/calibration.py) -- a better K gives a less sheared point cloud and
    therefore a better plane, but the estimator itself is untouched.
    """
    pts = unproject_ground_points(depth, ground_mask, K)
    if len(pts) < 3:
        return None, 0
    return fit_plane_ransac(pts, seed=seed), len(pts)


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
                  n_ground_pts: int, k_source: str = "unknown") -> None:
    config.GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        geometry_path(base),
        depth=depth.astype(np.float16),          # smooth field; fp16 is plenty
        n=(np.zeros(3) if n is None else np.asarray(n)).astype(np.float32),
        K=np.asarray(K, np.float32),
        valid=np.array(n is not None),
        calibrated=np.array(bool(calibrated)),
        n_ground_pts=np.array(int(n_ground_pts)),
        # Which intrinsics source produced K, so a cache built with the weak
        # heuristic can be spotted and re-exported.
        k_source=np.array(str(k_source)),
    )


def load_geometry(base: str):
    """Return dict with depth/n/K/valid + k_source, or None if not cached."""
    p = geometry_path(base)
    if not p.exists():
        return None
    z = np.load(p)
    return dict(
        depth=z["depth"].astype(np.float32),
        n=z["n"].astype(np.float64),
        K=z["K"].astype(np.float64),
        valid=bool(z["valid"]),
        calibrated=bool(z["calibrated"]),
        n_ground_pts=int(z["n_ground_pts"]),
        # Older caches predate this field.
        k_source=str(z["k_source"]) if "k_source" in z.files else "unknown",
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


def resolve_plane(base: str, sem: np.ndarray, rgb: np.ndarray | None = None,
                  image_path: Path | None = None, device=None):
    """The raw primitives (depth, n, K) for one image, at `sem`'s resolution.

    Prefers the Step-7 cache; falls back to computing depth on the fly for images
    that were never precomputed (a brand-new photo, an uploaded frame). Returns
    None when no plane is available -- consumers then fall back to semantic-only
    behaviour (OFRSNet) or report the BEV as unavailable (src/disturbance.py).

    This exists because two different consumers need the same three quantities
    but derive different things from them: `derive_ground_fields` for the network,
    and the ground-plane homography for the BEV (see src/bev.py). Keeping the
    cache/compute decision in one place means they can never disagree about which
    plane an image has.

    `sem` is the OFRS semantic map, used both to size the output and (on the
    compute path) to isolate road pixels for the plane fit.
    """
    hgt, wid = sem.shape
    geo = load_geometry(base)
    if geo is not None:
        depth, K = geo["depth"], geo["K"]
        if depth.shape != (hgt, wid):
            oh, ow = depth.shape
            depth = cv2.resize(depth, (wid, hgt), interpolation=cv2.INTER_AREA)
            K = scale_intrinsics(K, wid / ow, hgt / oh)
        # NOTE: no GEOM_MIN_GROUND_PX check here. Step 7 already applied it when
        # building the cache and recorded the outcome in `valid`, so re-testing
        # against a *different* semantic map would be able to reject a plane the
        # cache considers good.
        return dict(depth=depth, n=(geo["n"] if geo["valid"] else None), K=K,
                    valid=geo["valid"], calibrated=geo["calibrated"],
                    k_source=geo["k_source"], cached=True)

    if rgb is None:
        return None
    cal = calibrate_sample(image_path, base, wid, hgt,
                           orig_w=rgb.shape[1], orig_h=rgb.shape[0], device=device)
    rgb_small = (rgb if rgb.shape[:2] == (hgt, wid) else
                 cv2.resize(rgb, (wid, hgt), interpolation=cv2.INTER_AREA))
    depth = estimate_depth_metric(rgb_small, device)
    ground = sem == config.OFRS_CLASSES.index("road")
    if int(ground.sum()) < config.GEOM_MIN_GROUND_PX:
        return None
    n_plane, _ = estimate_ground_plane(depth, ground, cal.K)
    if n_plane is None:
        return None
    return dict(depth=depth, n=n_plane, K=cal.K, valid=True,
                calibrated=cal.calibrated, k_source=cal.source, cached=False)
