"""
Zero-shot, pretrained-component reimplementation of the GEOMETRIC COMPUTATION
GRAPH of GroundNet (Man, Weng, Li, Kitani -- ACM MM 2019).

GroundNet trains a single VGG-16 encoder end-to-end with three heads (depth,
surface normal, ground segmentation) on KITTI/ApolloScape ground-truth, and
fuses two independently-computed ground-plane normals with a geometric
consistency loss. No public pretrained checkpoint exists for GroundNet itself
(verified: not in the paper, not on GitHub), and training one requires
per-pixel depth/normal/segmentation GT we do not have -- so a literal
zero-shot load is impossible.

Instead, this module reimplements the paper's INFERENCE-TIME geometry
(Eq. 1-18 of the paper) exactly, substituting one independently-pretrained
off-the-shelf model per stream in place of GroundNet's jointly-trained heads:

  stream              | GroundNet (paper)         | substitute here
  --------------------|----------------------------|---------------------------------
  ground segmentation | VGG head, KITTI/Apollo GT  | Mask2Former (Mapillary Vistas 'road')
  depth               | VGG head, KITTI/Apollo GT  | Depth Anything V2 (metric,outdoor)
  surface normal      | VGG head, KITTI/Apollo GT  | Marigold Normals (LCM, diffusion)

All three are genuinely independent pretrained networks (different papers,
different training data), so the paper's core idea -- cross-checking a
depth-derived plane normal against an independently-predicted normal map via
a geometric consistency angle -- remains meaningful here, not tautological.
"""
import sys, time
import numpy as np
import cv2
import torch

sys.path.insert(0, "scratchpad")
import vp_core as vc
from seg_bev import get_segmenter, segment, MVD

_MODELS = {}


def get_depth_model():
    if "depth" not in _MODELS:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        mid = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
        _MODELS["depth_proc"] = AutoImageProcessor.from_pretrained(mid)
        _MODELS["depth"] = AutoModelForDepthEstimation.from_pretrained(mid).eval()
    return _MODELS["depth_proc"], _MODELS["depth"]


def estimate_depth_metric(rgb):
    """Raw metric depth in meters (H,W) float32 -- bypasses the HF `pipeline`
    convenience wrapper, which quantizes output to an 8-bit visualization."""
    proc, model = get_depth_model()
    h, w = rgb.shape[:2]
    with torch.no_grad():
        inp = proc(images=rgb, return_tensors="pt")
        out = model(**inp)
        depth = proc.post_process_depth_estimation(
            out, target_sizes=[(h, w)])[0]["predicted_depth"]
    return depth.numpy().astype(np.float32)


def get_normal_model():
    if "normal" not in _MODELS:
        from diffusers import MarigoldNormalsPipeline
        _MODELS["normal"] = MarigoldNormalsPipeline.from_pretrained(
            "prs-eth/marigold-normals-lcm-v0-1", torch_dtype=torch.float32)
    return _MODELS["normal"]


def _marigold_to_opencv(normal_map):
    """Marigold predicts normals in graphics/OpenGL camera space (X-right,
    Y-up, Z-toward-viewer). Our pipeline (and the paper's Eq. 1-2 pinhole
    model) use OpenCV/image camera space (X-right, Y-down, Z-forward into the
    scene). The two are related by a 180 deg rotation about X: flip Y and Z.
    Empirically validated (see notebook Sec. 2) against an independently
    computed ground pitch from the LSD/RANSAC pipeline of the first notebook."""
    out = normal_map.copy()
    out[..., 1] *= -1
    out[..., 2] *= -1
    return out


def estimate_normal_map(rgb, infer_w=384, steps=4):
    """Dense unit surface-normal map (H,W,3) in OpenCV camera convention, at
    full input resolution (resized back up after inference)."""
    pipe = get_normal_model()
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (infer_w, int(infer_w * h / w)))
    out = pipe(small, num_inference_steps=steps, output_type="np",
              generator=torch.Generator().manual_seed(0))
    n_small = _marigold_to_opencv(out.prediction[0].astype(np.float32))
    n = cv2.resize(n_small, (w, h), interpolation=cv2.INTER_LINEAR)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / (norm + 1e-8)


# --------------------------------------------------------------------------
# Eq. 1-2 (pinhole unprojection), Eq. 3 (ground isolation)
# --------------------------------------------------------------------------
def unproject_ground_points(depth, mask, K, max_depth=30.0):
    """Lift masked ground pixels to a 3D point cloud in camera coordinates.
    The paper caps valid depth at 30 m to keep far, noisy pixels out of the
    plane fit (Sec. 5.4, 'valid depth threshold... 30m')."""
    ys, xs = np.where(mask)
    z = depth[ys, xs]
    keep = (z > 0.1) & (z < max_depth)
    xs, ys, z = xs[keep], ys[keep], z[keep]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    return np.column_stack([X, Y, z])


# --------------------------------------------------------------------------
# Eq. 4-5: least-squares plane module  (n = (C^T C)^-1 C^T 1 / ||...||)
# --------------------------------------------------------------------------
def fit_plane_ls(points):
    if len(points) < 3:
        return None
    C = points
    ones = np.ones(len(C))
    try:
        n = np.linalg.solve(C.T @ C, C.T @ ones)
    except np.linalg.LinAlgError:
        return None
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-9 else None


# --------------------------------------------------------------------------
# RANSAC plane module (inference-time substitute for DSAC -- DSAC's role is
# to make the *training* hypothesis-selection differentiable; at inference
# it reduces to RANSAC hypothesis scoring + inlier refit, which is what we
# implement directly here).
# --------------------------------------------------------------------------
def fit_plane_ransac(points, dist_thresh=0.05, iters=500, seed=0):
    if len(points) < 3:
        return None
    rng = np.random.default_rng(seed)
    n = len(points)
    best_inliers, best_count = None, -1
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]
        v1, v2 = p1 - p0, p2 - p0
        normal = np.cross(v1, v2)
        nn = np.linalg.norm(normal)
        if nn < 1e-9:
            continue
        normal /= nn
        d = normal @ p0
        dist = np.abs(points @ normal - d)
        inliers = dist < dist_thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers
    if best_inliers is None or best_inliers.sum() < 3:
        return fit_plane_ls(points)
    return fit_plane_ls(points[best_inliers])


# --------------------------------------------------------------------------
# Eq. 6-7: normal-stream ground plane normal (average over ground mask)
# --------------------------------------------------------------------------
def average_normal_over_mask(normal_map, mask):
    vecs = normal_map[mask]
    if len(vecs) == 0:
        return None
    avg = vecs.mean(axis=0)
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 1e-9 else None


def canonicalize_up(n):
    """Plane normals are only defined up to sign; fix the convention to
    'pointing up' (negative y in our OpenCV camera frame), matching vp_core."""
    if n is None:
        return None
    return -n if n[1] > 0 else n


# --------------------------------------------------------------------------
# Eq. 11: geometric consistency between the two streams
# --------------------------------------------------------------------------
def consistency_angle_deg(n_d, n_n):
    if n_d is None or n_n is None:
        return np.nan
    cos = np.clip(np.dot(n_d, n_n) / (np.linalg.norm(n_d) * np.linalg.norm(n_n)), -1, 1)
    return float(np.degrees(np.arccos(cos)))


# --------------------------------------------------------------------------
# Eq. 13-18: ground plane normal -> horizon line (theta, rho)
# --------------------------------------------------------------------------
def normal_to_horizon(n, K):
    """A = K^-T n; horizon line a*x + b*y + c = 0 in pixel coords (Eq. 14-16);
    (theta, rho) horizon-line parameterisation (Eq. 17-18)."""
    if n is None:
        return None, np.nan, np.nan
    A = np.linalg.inv(K).T @ n
    a, b, c = A
    if abs(c) < 1e-12:
        return None, np.nan, np.nan
    ac, bc = a / c, b / c
    theta = np.degrees(np.arctan2(-ac, bc))
    Wp, Hp = K[0, 2], K[1, 2]
    rho = -abs(ac * Wp + bc * Hp + 1) / np.sqrt(ac**2 + bc**2 + 1e-12)
    line = np.array([ac, bc, 1.0])
    return line, theta, rho


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------
def run_groundnet(bgr, proc_width=768, cam_height=1.5, max_depth=30.0,
                  normal_infer_w=384, normal_steps=4, seed=0):
    t0 = time.time()
    H0, W0 = bgr.shape[:2]
    scale = proc_width / W0
    proc = cv2.resize(bgr, (proc_width, int(round(H0 * scale))),
                      interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
    h, w = proc.shape[:2]
    K, _, _ = vc.build_intrinsics(w, h)

    # -- ground segmentation stream (substitute) --
    t1 = time.time()
    seg = segment(rgb)
    ground = (seg == MVD["road"])
    ground[int(0.88 * h):, :] = False        # exclude hood/dashboard strip
    t_seg = time.time() - t1

    # -- depth stream --
    t1 = time.time()
    depth = estimate_depth_metric(rgb)
    pts = unproject_ground_points(depth, ground, K, max_depth=max_depth)
    n_d_ls = canonicalize_up(fit_plane_ls(pts))
    n_d = canonicalize_up(fit_plane_ransac(pts, dist_thresh=0.05, iters=500, seed=seed))
    t_depth = time.time() - t1

    # -- surface normal stream --
    t1 = time.time()
    normal_map = estimate_normal_map(rgb, infer_w=normal_infer_w, steps=normal_steps)
    n_n = canonicalize_up(average_normal_over_mask(normal_map, ground))
    t_normal = time.time() - t1

    # -- geometric consistency + fused output (paper: final = average of n_d, n_n) --
    consistency = consistency_angle_deg(n_d, n_n)
    if n_d is not None and n_n is not None:
        n_final = n_d + n_n
        n_final = n_final / (np.linalg.norm(n_final) + 1e-9)
    else:
        n_final = n_d if n_d is not None else n_n

    horizon, theta, rho = normal_to_horizon(n_final, K) if n_final is not None else (None, np.nan, np.nan)
    n_points = int(ground.sum())

    return dict(
        proc=proc, rgb=rgb, seg=seg, ground_mask=ground, K=K,
        depth=depth, points=pts, normal_map=normal_map,
        n_d_ls=n_d_ls, n_d=n_d, n_n=n_n, n_final=n_final,
        consistency_deg=consistency, horizon=horizon, theta_deg=theta, rho=rho,
        n_ground_px=n_points, n_ground_pts_used=len(pts),
        timing=dict(seg=round(t_seg, 2), depth=round(t_depth, 2),
                    normal=round(t_normal, 2), total=round(time.time() - t0, 2)),
    )


if __name__ == "__main__":
    import glob, os
    files = ["IMG_0876", "IMG_0865", "IMG_0880"]
    for name in files:
        bgr = cv2.imread(f"illegal parking/{name}.jpg")
        r = run_groundnet(bgr)
        print(f"{name}: ground_px={r['n_ground_px']} pts_used={r['n_ground_pts_used']} "
              f"n_d={np.round(r['n_d'],3) if r['n_d'] is not None else None} "
              f"n_n={np.round(r['n_n'],3) if r['n_n'] is not None else None} "
              f"consistency={r['consistency_deg']:.1f}deg "
              f"theta={r['theta_deg']:.1f} rho={r['rho']:.1f} "
              f"t={r['timing']}")
