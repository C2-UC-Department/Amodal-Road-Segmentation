"""Regression tests for `geometry.resolve_plane` (src/geometry.py).

`resolve_plane` factored the cache-or-compute decision for the raw (depth, n, K)
primitives out of `predict.compute_geometry`, so that the network path
(`derive_ground_fields`) and the BEV path (`bev.homography_bev_to_image`) can
never disagree about which plane an image has.

That refactor must be behaviour-preserving, and it cannot be checked by comparing
predicted masks: the on-the-fly depth path is NOT deterministic on this platform
(two runs of identical code produce different masks). So instead these tests pin
the depth model to a fixed synthetic output and compare the refactored
`compute_geometry` against the pre-refactor implementation, reproduced verbatim
below, on both branches.

Fully synthetic -- no dataset, no checkpoints, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import geometry as geo  # noqa: E402
from src import predict  # noqa: E402

H, W = 240, 640
ROAD_IDX = config.OFRS_CLASSES.index("road")


def _sem() -> np.ndarray:
    """OFRS semantic map: road across the bottom half, a vehicle blob on it."""
    sem = np.full((H, W), config.OFRS_CLASSES.index("building"), np.int64)
    sem[H // 2:] = ROAD_IDX
    sem[H // 2 + 20:H // 2 + 70, 260:380] = config.OFRS_CLASSES.index("vehicle")
    return sem


def _rgb() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (H, W, 3), dtype=np.uint8)


def _synthetic_depth(rgb: np.ndarray, device) -> np.ndarray:
    """Deterministic stand-in for Depth-Anything: a plausible ground ramp.

    Depth falls off towards the horizon so a RANSAC plane fit on the road pixels
    actually succeeds, which is what makes the comparison meaningful.
    """
    h, w = rgb.shape[:2]
    v = np.arange(h, dtype=np.float32)[:, None]
    d = 400.0 / np.maximum(v - h * 0.45, 1.0)
    return np.clip(np.broadcast_to(d, (h, w)).copy(), 0.5, 80.0)


def _compute_geometry_pre_refactor(rgb, sem, base, device, image_path=None):
    """Verbatim src/predict.py:compute_geometry as of commit 14ffb6d."""
    fields = geo.load_ground_fields(base, out_hw=sem.shape)
    if fields is None:
        h_img, w_img = sem.shape
        cal = geo.calibrate_sample(image_path, base, w_img, h_img,
                                   orig_w=rgb.shape[1], orig_h=rgb.shape[0],
                                   device=device)
        rgb_small = (rgb if rgb.shape[:2] == sem.shape else
                     cv2.resize(rgb, (w_img, h_img), interpolation=cv2.INTER_AREA))
        depth = geo.estimate_depth_metric(rgb_small, device)
        ground = sem == ROAD_IDX
        if int(ground.sum()) < config.GEOM_MIN_GROUND_PX:
            return None
        n_plane, _ = geo.estimate_ground_plane(depth, ground, cal.K)
        if n_plane is None:
            return None
        G, hh, gvalid = geo.derive_ground_fields(depth, n_plane, cal.K)
        fields = dict(G=G, h=hh, gvalid=gvalid, valid=True)
    if not fields["valid"]:
        return None
    return fields


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    """Isolated geometry cache + a deterministic depth model."""
    monkeypatch.setattr(config, "GEOMETRY_DIR", tmp_path / "geometry")
    monkeypatch.setattr(geo, "estimate_depth_metric", _synthetic_depth)
    return tmp_path


def _assert_same_fields(a, b):
    assert (a is None) == (b is None), f"one returned None: {a is None} vs {b is None}"
    if a is None:
        return
    assert a.keys() == b.keys()
    assert a["valid"] == b["valid"]
    for k in ("G", "h", "gvalid"):
        assert a[k].shape == b[k].shape, k
        assert a[k].dtype == b[k].dtype, k
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


# --------------------------------------------------------------------------- #
# Branch 1: no cache -> compute on the fly
# --------------------------------------------------------------------------- #
def test_compute_geometry_matches_pre_refactor_uncached(pinned):
    sem, rgb = _sem(), _rgb()
    old = _compute_geometry_pre_refactor(rgb, sem, "synth", None)
    new = predict.compute_geometry(rgb, sem, "synth", None)
    assert new is not None, "the synthetic scene should yield a plane"
    _assert_same_fields(old, new)


def test_uncached_returns_none_when_too_little_road(pinned):
    """The GEOM_MIN_GROUND_PX guard must survive the refactor."""
    sem = np.full((H, W), config.OFRS_CLASSES.index("building"), np.int64)
    sem[:2, :10] = ROAD_IDX                    # far below GEOM_MIN_GROUND_PX
    rgb = _rgb()
    assert _compute_geometry_pre_refactor(rgb, sem, "synth", None) is None
    assert predict.compute_geometry(rgb, sem, "synth", None) is None


# --------------------------------------------------------------------------- #
# Branch 2: Step-7 cache present
# --------------------------------------------------------------------------- #
def _seed_cache(base: str, sem, K_scale: float = 1.0, valid: bool = True):
    """Write a Step-7-shaped cache entry, optionally at a different resolution
    than `sem` so the resize/rescale-K path is exercised."""
    h, w = int(sem.shape[0] * K_scale), int(sem.shape[1] * K_scale)
    depth = _synthetic_depth(np.zeros((h, w, 3), np.uint8), None)
    K = geo.intrinsics_from_fov(w, h, config.GEOM_FALLBACK_HFOV_DEG)
    n = None
    if valid:
        ground = cv2.resize(sem.astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_NEAREST) == ROAD_IDX
        n, _ = geo.estimate_ground_plane(depth, ground, K)
        assert n is not None
    geo.save_geometry(base, depth, n, K, True, 1234, k_source="kitti_calib")


def test_compute_geometry_matches_pre_refactor_cached(pinned):
    sem, rgb = _sem(), _rgb()
    _seed_cache("synth", sem)
    old = _compute_geometry_pre_refactor(rgb, sem, "synth", None)
    new = predict.compute_geometry(rgb, sem, "synth", None)
    assert new is not None
    _assert_same_fields(old, new)


def test_compute_geometry_matches_pre_refactor_cached_needs_resize(pinned):
    """Cache built at a different resolution: depth is INTER_AREA'd and K
    rescaled. This is the branch where a sign or transpose slip would show up."""
    sem, rgb = _sem(), _rgb()
    _seed_cache("synth", sem, K_scale=0.5)
    old = _compute_geometry_pre_refactor(rgb, sem, "synth", None)
    new = predict.compute_geometry(rgb, sem, "synth", None)
    assert new is not None
    _assert_same_fields(old, new)


def test_invalid_cache_entry_returns_none_both_ways(pinned):
    """Step 7 caches explicitly-invalid entries when the plane fit fails; both
    implementations must decline rather than consume a zero normal."""
    sem, rgb = _sem(), _rgb()
    _seed_cache("synth", sem, valid=False)
    assert _compute_geometry_pre_refactor(rgb, sem, "synth", None) is None
    assert predict.compute_geometry(rgb, sem, "synth", None) is None


# --------------------------------------------------------------------------- #
# resolve_plane's own contract
# --------------------------------------------------------------------------- #
def test_resolve_plane_reports_provenance(pinned):
    sem, rgb = _sem(), _rgb()

    got = geo.resolve_plane("synth", sem, rgb=rgb, device=None)
    assert got["cached"] is False and got["k_source"] == "fov_prior"
    assert got["calibrated"] is False        # fov_prior is a blind assumption

    _seed_cache("synth", sem)
    got = geo.resolve_plane("synth", sem, rgb=rgb, device=None)
    assert got["cached"] is True and got["k_source"] == "kitti_calib"
    assert got["calibrated"] is True
    assert got["depth"].shape == sem.shape
    assert got["n"].shape == (3,)


def test_resolve_plane_returns_none_without_rgb_or_cache(pinned):
    """The BEV path may be handed an image it cannot compute depth for; it must
    get None rather than an exception."""
    assert geo.resolve_plane("synth", _sem(), rgb=None, device=None) is None


def test_resolve_plane_output_is_a_usable_ground_plane(pinned):
    """The whole point of the shared helper: what it returns must be directly
    consumable by both derive_ground_fields and the BEV homography."""
    from src import bev

    sem, rgb = _sem(), _rgb()
    got = geo.resolve_plane("synth", sem, rgb=rgb, device=None)
    assert bev.plane_is_usable(got["n"]), got["n"]
    H_bev = bev.homography_bev_to_image(got["K"], got["n"], bev.BevGrid.from_config())
    assert np.all(np.isfinite(H_bev))
