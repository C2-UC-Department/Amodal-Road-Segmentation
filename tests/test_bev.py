"""Tests for the BEV ground-plane homography (src/bev.py).

Pure numpy -- no models, no dataset, no checkpoints. Run from the project root
so the `import config` bootstrap every module uses resolves:

    pytest tests/test_bev.py -v

The headline test does not trust a hand-written fixture: because
`bev.homography_bev_to_image` is the analytic inverse of
`geometry.derive_ground_fields`, the existing (already-shipped, already-trained-
against) forward mapping IS the ground truth, and we assert the two agree.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import bev  # noqa: E402
from src import geometry as geo  # noqa: E402

IMG_H, IMG_W = 720, 1280
CAM_H = 1.65


def _K() -> np.ndarray:
    return np.array([[800.0, 0.0, IMG_W / 2.0],
                     [0.0, 800.0, IMG_H / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _plane(height_m: float = CAM_H, tilted: bool = True) -> np.ndarray:
    """`n . X = 1` for a plane `height_m` below the camera.

    y is DOWN, so the unit normal (camera -> plane) has a positive y component.
    """
    u = np.array([0.06, 0.99, -0.12] if tilted else [0.0, 1.0, 0.0])
    u = u / np.linalg.norm(u)
    return u / height_m


# --------------------------------------------------------------------------- #
# The homography agrees with the existing forward mapping
# --------------------------------------------------------------------------- #
def test_homography_inverts_derive_ground_fields():
    """H must map each pixel's ground footprint back onto that pixel.

    `derive_ground_fields` gives G = where pixel (u,v)'s ray meets the plane.
    Feeding (G_x, G_z) through the BEV grid and then H must land back on (u,v).
    """
    K, n = _K(), _plane()
    grid = bev.BevGrid.from_config()

    # G depends only on (K, n), not on depth -- depth only drives the h field.
    depth = np.ones((IMG_H, IMG_W), np.float32)
    G, _, gvalid = geo.derive_ground_fields(depth, n, K)

    S_inv = np.linalg.inv(bev._S(grid))
    H = bev.homography_bev_to_image(K, n, grid)

    # derive_ground_fields RESCALES rays longer than GEOM_MAX_GROUND_DIST_M back
    # onto the cap (geometry.py:287-290); the homography does no such clamping,
    # so only compare where G is un-clamped.
    dist = np.linalg.norm(G, axis=-1)
    ok = gvalid & (dist < config.GEOM_MAX_GROUND_DIST_M - 1e-3) & (dist > 1e-6)
    assert ok.sum() > 50_000, "not enough un-clamped pixels to test"

    vs, us = np.nonzero(ok)
    sel = np.linspace(0, len(vs) - 1, 4000).astype(int)   # subsample; it's dense
    vs, us = vs[sel], us[sel]
    world = np.stack([G[vs, us, 0].astype(np.float64),
                      G[vs, us, 2].astype(np.float64),
                      np.ones(len(vs))])                  # (3, N)

    # errstate: this platform's BLAS raises spurious divide-by-zero/overflow FP
    # flags for `@`, exactly as documented at geometry.py:270-275. The values are
    # correct; only the flags are bogus.
    with np.errstate(all="ignore"):
        img_h = H @ (S_inv @ world)
    u_back = img_h[0] / img_h[2]
    v_back = img_h[1] / img_h[2]

    # G is cached and returned as float32, which at 60 m bounds the achievable
    # agreement at ~1e-4 px. A thousandth of a pixel is still a far stronger
    # statement than any fixture-based check would be.
    assert np.abs(u_back - us).max() < 1e-3
    assert np.abs(v_back - vs).max() < 1e-3


# --------------------------------------------------------------------------- #
# End-to-end: a ground rectangle of known area survives the round trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tilted", [False, True])
def test_ground_rectangle_area_roundtrip(tilted):
    """Project a known 4 m x 10 m ground patch into the image, warp it back,
    and recover 40 m^2. Exercises K, M, S, the warp and area_m2 together."""
    K, n = _K(), _plane(tilted=tilted)
    grid = bev.BevGrid.from_config()

    x0, x1, z0, z1 = -2.0, 2.0, 10.0, 20.0
    true_area = (x1 - x0) * (z1 - z0)

    # Ground -> camera -> pixels. A planar homography maps lines to lines, so
    # the four projected corners describe the region exactly (no curvature).
    M = bev._M(n)
    corners = []
    for x, z in [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]:
        p = K @ M @ np.array([x, z, 1.0])
        corners.append([p[0] / p[2], p[1] / p[2]])
    corners = np.array(corners)
    assert (corners[:, 0] >= 0).all() and (corners[:, 0] < IMG_W).all()
    assert (corners[:, 1] >= 0).all() and (corners[:, 1] < IMG_H).all()

    mask = np.zeros((IMG_H, IMG_W), np.uint8)
    cv2.fillPoly(mask, [np.round(corners).astype(np.int32)], 1)

    H = bev.homography_bev_to_image(K, n, grid)
    got = bev.area_m2(bev.warp_to_bev(mask.astype(bool), H, grid), grid)

    # Tolerance is boundary rasterisation only: ~perimeter/cell_size cells out of
    # 16000, i.e. a few percent.
    assert got == pytest.approx(true_area, rel=0.05), f"{got:.2f} vs {true_area:.2f}"


def test_bev_validity_marks_offscreen_cells_invalid():
    K, n = _K(), _plane()
    grid = bev.BevGrid.from_config()
    H = bev.homography_bev_to_image(K, n, grid)
    valid = bev.bev_validity(H, grid, (IMG_H, IMG_W))

    assert valid.shape == grid.shape
    assert valid.any() and not valid.all(), "a 20m-wide grid should be partly off-frame"

    u, v = grid.world_to_pixel(0.0, 12.0)
    assert valid[v, u], "straight ahead at mid range must be observed"

    # It is the NEAR band that falls outside the frame, not the far one: ground at
    # Z projects to v = cy + fy*h/Z, so small Z lands BELOW the image bottom. With
    # h=1.65 m and this FOV nothing closer than ~3.7 m is visible -- a real
    # property of the camera, and the reason bev_valid must gate every area count.
    assert valid[0, 0] and valid[0, -1], "far corners are within the frame"
    assert not valid[-1, 0] and not valid[-1, -1], "the near band is not observed"
    u, v = grid.world_to_pixel(0.0, 2.0)
    assert not valid[v, u]


# --------------------------------------------------------------------------- #
# Grid bookkeeping
# --------------------------------------------------------------------------- #
def test_grid_shape_and_cell_area():
    grid = bev.BevGrid(ppm=20.0, x_min=-10.0, x_max=10.0, z_min=0.5, z_max=40.5)
    assert (grid.width, grid.height) == (400, 800)
    assert grid.cell_area_m2 == pytest.approx(0.0025)
    # A full-grid mask must measure the grid's true extent.
    assert bev.area_m2(np.ones(grid.shape, bool), grid) == pytest.approx(20.0 * 40.0)


def test_world_to_pixel_inverts_S():
    grid = bev.BevGrid.from_config()
    S = bev._S(grid)
    for x, z in [(0.0, 10.0), (-7.5, 3.25), (4.0, 39.0)]:
        u, v = grid.world_to_pixel(x, z)
        back = S @ np.array([u, v, 1.0])
        assert back[0] == pytest.approx(x, abs=1.0 / grid.ppm)
        assert back[1] == pytest.approx(z, abs=1.0 / grid.ppm)


# --------------------------------------------------------------------------- #
# Metric scale
# --------------------------------------------------------------------------- #
def test_rescale_plane_to_height():
    n = _plane(height_m=6.92)          # the median implied height on our footage
    assert bev.implied_camera_height(n) == pytest.approx(6.92)

    n2, implied = bev.rescale_plane_to_height(n, 1.65)
    assert implied == pytest.approx(6.92)
    assert bev.implied_camera_height(n2) == pytest.approx(1.65)
    # Direction is unchanged -- only the metric scale moves.
    assert np.allclose(n2 / np.linalg.norm(n2), n / np.linalg.norm(n))


def test_rescale_changes_area_by_the_square_of_the_factor():
    """The whole reason m^2 is reported twice: a linear scale error on the world
    is a QUADRATIC error on area.

    The grid here is deliberately far larger than BEV_RANGE_* so neither raster
    clips -- with the default 40 m grid the uncorrected plane pushes the same
    image blob out past 130 m and both areas saturate, which would make the test
    pass for the wrong reason.
    """
    K = _K()
    grid = bev.BevGrid(ppm=10.0, x_min=-60.0, x_max=60.0, z_min=0.5, z_max=200.0)
    n_raw = _plane(height_m=6.92, tilted=False)
    n_cal, implied = bev.rescale_plane_to_height(n_raw, 1.65)

    # Low in the frame, so the footprint stays inside the grid under both planes.
    mask = np.zeros((IMG_H, IMG_W), bool)
    mask[600:700, 500:800] = True

    a_raw = bev.area_m2(bev.warp_to_bev(mask, bev.homography_bev_to_image(K, n_raw, grid), grid), grid)
    a_cal = bev.area_m2(bev.warp_to_bev(mask, bev.homography_bev_to_image(K, n_cal, grid), grid), grid)

    assert a_raw == pytest.approx(a_cal * (implied / 1.65) ** 2, rel=0.02)
    assert a_raw > 10 * a_cal, "a 4.2x height error must be a ~17x area error"


def test_uncorrected_plane_on_a_rescaled_grid_is_the_same_raster():
    """Underpins how src/disturbance.py reports uncalibrated areas.

    Reporting an uncalibrated area must measure the SAME PHYSICAL REGION as the
    calibrated one, or the two figures printed side by side are not comparable.
    Scaling the extents up by k and ppm down by k gives a pixel-identical raster
    whose cells are k^2 larger, so the uncalibrated figure is reachable two
    independent ways -- warp with the raw plane on the widened grid, or warp with
    the corrected plane and multiply by k^2 -- and the two must agree.
    """
    K = _K()
    target = CAM_H
    n_raw = _plane(height_m=6.92, tilted=True)
    n_cal, implied = bev.rescale_plane_to_height(n_raw, target)
    k = implied / target

    base = bev.BevGrid.from_config()
    wide = bev.BevGrid(ppm=base.ppm / k, x_min=base.x_min * k, x_max=base.x_max * k,
                       z_min=base.z_min * k, z_max=base.z_max * k)
    assert (wide.width, wide.height) == (base.width, base.height)

    mask = np.zeros((IMG_H, IMG_W), bool)
    mask[500:700, 400:900] = True

    cal = bev.warp_to_bev(mask, bev.homography_bev_to_image(K, n_cal, base), base)
    raw = bev.warp_to_bev(mask, bev.homography_bev_to_image(K, n_raw, wide), wide)

    # Agreement is to within floating-point rounding in warpPerspective's sampling,
    # which can flip a single boundary cell -- not bit-exact, but far tighter than
    # any tolerance the reported areas care about.
    assert cal.sum() > 1000, "the fixture should produce a substantial region"
    assert np.mean(cal == raw) > 0.9999
    assert bev.area_m2(raw, wide) == pytest.approx(bev.area_m2(cal, base) * k * k,
                                                  rel=1e-4)

    # And the reliability gate must select the same cells once its cap is scaled.
    m_cal, _ = bev.measurable_mask(bev.homography_bev_to_image(K, n_cal, base),
                                   base, (IMG_H, IMG_W), max_m2_per_px=0.02)
    m_raw, _ = bev.measurable_mask(bev.homography_bev_to_image(K, n_raw, wide),
                                   wide, (IMG_H, IMG_W), max_m2_per_px=0.02 * k * k)
    assert np.mean(m_cal == m_raw) > 0.9999


# --------------------------------------------------------------------------- #
# Degenerate plane rejection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,why", [
    (None, "no plane fitted"),
    (np.zeros(3), "zero normal"),
    (np.array([0.0, 0.0, 0.6]), "n_y == 0 (plane parallel to the view axis)"),
    (np.array([0.0, -0.6, 0.0]), "n_y < 0 (plane ABOVE the camera)"),
    (np.array([0.0, 5.0, 0.0]), "implied height 0.2m -- below the floor"),
    (np.array([0.0, 0.02, 0.0]), "implied height 50m -- absurd"),
    (np.array([np.nan, 1.0, 0.0]), "non-finite"),
])
def test_plane_is_usable_rejects(n, why):
    assert not bev.plane_is_usable(n), why


def test_plane_is_usable_accepts_a_real_plane():
    assert bev.plane_is_usable(_plane())


def test_homography_raises_on_unusable_plane():
    with pytest.raises(ValueError, match="usable ground plane"):
        bev.homography_bev_to_image(_K(), np.array([0.0, -1.0, 0.0]),
                                    bev.BevGrid.from_config())


# --------------------------------------------------------------------------- #
# Discrete data must not be blended
# --------------------------------------------------------------------------- #
def test_warp_preserves_instance_ids_exactly():
    """Instance ids are labels, not intensities: interpolating id 2 and id 8
    into id 5 would silently invent a vehicle."""
    K, n = _K(), _plane()
    grid = bev.BevGrid.from_config()
    H = bev.homography_bev_to_image(K, n, grid)

    ids = np.zeros((IMG_H, IMG_W), np.int32)
    ids[400:500, 300:600] = 2
    ids[400:500, 600:900] = 8

    out = bev.warp_to_bev(ids, H, grid)
    assert out.dtype == np.int32
    assert set(np.unique(out)).issubset({0, 2, 8})
    assert (out == 2).any() and (out == 8).any()


def test_warp_preserves_bool_dtype():
    K, n = _K(), _plane()
    grid = bev.BevGrid.from_config()
    H = bev.homography_bev_to_image(K, n, grid)
    out = bev.warp_to_bev(np.ones((IMG_H, IMG_W), bool), H, grid)
    assert out.dtype == bool and out.any()
