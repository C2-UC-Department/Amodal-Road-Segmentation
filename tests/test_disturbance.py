"""Tests for attribution and instance splitting (src/disturbance.py, src/instances.py).

Pure numpy -- the parts that decide the reported numbers are deliberately
separable from the models that produce the masks, so they can be pinned here
without a checkpoint or a GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import bev  # noqa: E402
from src import disturbance as dist  # noqa: E402
from src import instances as inst  # noqa: E402


class _V:
    """Minimal stand-in for inst.VehicleInstance (attribute only reads inst_id)."""

    def __init__(self, inst_id):
        self.inst_id = inst_id


# --------------------------------------------------------------------------- #
# Step 6: attribution must partition the occluded area exactly
# --------------------------------------------------------------------------- #
def test_attribution_partitions_the_total_exactly():
    grid = bev.BevGrid(ppm=1.0, x_min=0.0, x_max=8.0, z_min=0.0, z_max=8.0)
    occluded = np.zeros((8, 8), bool)
    occluded[1:4, 1:5] = True          # 12 cells
    occluded[5:7, 2:4] = True          # 4 cells -> 16 total

    inst_bev = np.zeros((8, 8), np.int32)
    inst_bev[1:4, 1:3] = 1             # claims 6 of the first block
    inst_bev[1:4, 3:5] = 2             # claims the other 6
    # the 4 cells at rows 5:7 stay id 0 -> unattributed

    per, un = dist.attribute(occluded, inst_bev, [_V(1), _V(2)], grid, scale_factor=1.0)

    assert per[1]["cells"] == 6 and per[2]["cells"] == 6
    assert un["cells"] == 4
    assert sum(m["cells"] for m in per.values()) + un["cells"] == int(occluded.sum())
    # ppm=1 -> 1 cell == 1 m^2
    assert per[1]["area_m2"] == pytest.approx(6.0)
    assert un["area_m2"] == pytest.approx(4.0)


def test_attribution_ignores_instance_pixels_outside_the_occluded_region():
    """An instance covering lots of BEV must only be credited with road it hides."""
    grid = bev.BevGrid(ppm=1.0, x_min=0.0, x_max=8.0, z_min=0.0, z_max=8.0)
    occluded = np.zeros((8, 8), bool)
    occluded[0:2, 0:2] = True                       # 4 cells
    inst_bev = np.ones((8, 8), np.int32)            # instance 1 covers everything

    per, un = dist.attribute(occluded, inst_bev, [_V(1)], grid, scale_factor=1.0)
    assert per[1]["cells"] == 4 and un["cells"] == 0


def test_attribution_reports_both_scales():
    grid = bev.BevGrid(ppm=1.0, x_min=0.0, x_max=4.0, z_min=0.0, z_max=4.0)
    occluded = np.zeros((4, 4), bool)
    occluded[0, 0:2] = True
    inst_bev = np.zeros((4, 4), np.int32)
    inst_bev[0, 0:2] = 1

    per, _ = dist.attribute(occluded, inst_bev, [_V(1)], grid, scale_factor=9.0)
    assert per[1]["area_m2"] == pytest.approx(2.0)
    assert per[1]["area_m2_raw"] == pytest.approx(18.0)


# --------------------------------------------------------------------------- #
# Step 4: ids must live on the support compose_amodal_mask patches
# --------------------------------------------------------------------------- #
def _scene():
    """Road across the bottom, two separate vehicle blobs standing on it."""
    sem = np.full((100, 200), config.OFRS_CLASSES.index("building"), np.int64)
    sem[60:, :] = inst.ROAD_IDX
    sem[40:65, 30:70] = inst.VEHICLE_IDX      # left vehicle
    sem[40:65, 120:160] = inst.VEHICLE_IDX    # right vehicle
    visible = sem == inst.ROAD_IDX
    return sem, visible


def test_ids_are_confined_to_the_occluder_support():
    sem, visible = _scene()
    ids, vehicles = inst.occluder_instance_ids(sem, visible, detections=None)
    support = inst.occluder_support(sem, visible)

    assert not (ids > 0)[~support].any(), "ids must never appear outside the support"
    assert len(vehicles) == 2, "two separate blobs -> two instances"
    assert {v.source for v in vehicles} == {"blob"}
    assert all(v.selectable for v in vehicles)


def test_ids_are_renumbered_by_descending_area():
    sem, visible = _scene()
    sem[40:65, 120:180] = inst.VEHICLE_IDX     # make the right blob bigger
    ids, vehicles = inst.occluder_instance_ids(sem, visible, detections=None)

    areas = [v.pixel_area for v in vehicles]
    assert areas == sorted(areas, reverse=True), "#1 should be the largest occluder"
    assert [v.inst_id for v in vehicles] == [1, 2]
    for v in vehicles:
        assert int((ids == v.inst_id).sum()) == v.pixel_area


def test_detections_split_a_merged_blob():
    """The reason the instance model is worth having: one touching blob, two cars."""
    sem = np.full((100, 200), config.OFRS_CLASSES.index("building"), np.int64)
    sem[60:, :] = inst.ROAD_IDX
    sem[40:65, 40:140] = inst.VEHICLE_IDX      # ONE connected blob
    visible = sem == inst.ROAD_IDX

    ids_blob, v_blob = inst.occluder_instance_ids(sem, visible, detections=None)
    assert len(v_blob) == 1, "connected components cannot separate touching vehicles"

    left = np.zeros_like(visible); left[40:65, 40:90] = True
    right = np.zeros_like(visible); right[40:65, 90:140] = True
    dets = [inst.Detection("car", 0.9, left, True),
            inst.Detection("truck", 0.8, right, True)]
    ids, vehicles = inst.occluder_instance_ids(sem, visible, dets)

    assert len(vehicles) == 2
    assert {v.label for v in vehicles} == {"car", "truck"}
    assert {v.source for v in vehicles} == {"instance"}
    # Still confined to the support, and still a partition of it.
    support = inst.occluder_support(sem, visible)
    assert not (ids > 0)[~support].any()


def test_higher_score_wins_an_overlap():
    sem, visible = _scene()
    a = np.zeros_like(visible); a[40:65, 30:70] = True
    b = np.zeros_like(visible); b[40:65, 40:70] = True     # overlaps a
    dets = [inst.Detection("car", 0.60, a, True),
            inst.Detection("bus", 0.95, b, True)]
    ids, vehicles = inst.occluder_instance_ids(sem, visible, dets)

    by_label = {v.label: v for v in vehicles}
    overlap_ids = np.unique(ids[b & inst.occluder_support(sem, visible)])
    overlap_ids = overlap_ids[overlap_ids > 0]
    assert overlap_ids.tolist() == [by_label["bus"].inst_id]


def test_detections_outside_the_support_are_dropped():
    """A car detected up on a bridge, nowhere near the road, is not an occluder."""
    sem, visible = _scene()
    far = np.zeros_like(visible); far[0:20, 0:20] = True   # in 'building' territory
    dets = [inst.Detection("car", 0.99, far, True)]
    ids, vehicles = inst.occluder_instance_ids(sem, visible, dets)
    assert all(v.source == "blob" for v in vehicles), "the off-support car must not appear"


def test_person_blobs_are_not_selectable():
    sem, visible = _scene()
    sem[40:65, 30:70] = inst.PERSON_IDX
    _, vehicles = inst.occluder_instance_ids(sem, visible, detections=None)
    labels = {v.label: v.selectable for v in vehicles}
    assert labels["person"] is False
    assert labels["vehicle"] is True


def test_slivers_below_the_threshold_are_left_unattributed():
    """Better an honest unattributed remainder than a fake one-pixel 'vehicle'."""
    sem, visible = _scene()
    support = inst.occluder_support(sem, visible)
    big = np.zeros_like(visible); big[40:65, 30:70] = True
    dets = [inst.Detection("car", 0.9, big, True)]
    ids, vehicles = inst.occluder_instance_ids(sem, visible, dets)

    leftover = support & (ids == 0)
    assert leftover.sum() >= config.INSTANCE_MIN_BLOB_PX or not leftover.any()
    assert all(v.pixel_area >= config.INSTANCE_MIN_BLOB_PX for v in vehicles
               if v.source == "blob")


def test_no_occluders_yields_no_instances():
    sem = np.full((100, 200), inst.ROAD_IDX, np.int64)
    ids, vehicles = inst.occluder_instance_ids(sem, sem == inst.ROAD_IDX, None)
    assert vehicles == [] and not ids.any()


def test_vehicle_class_ids_exclude_people():
    id2label = {0: "person", 1: "rider", 2: "car", 3: "truck", 4: "bus",
                5: "train", 6: "motorcycle", 7: "bicycle"}
    got = inst.build_vehicle_class_ids(id2label)
    assert got == {2, 3, 4, 5, 6, 7}
    assert 0 not in got and 1 not in got


# --------------------------------------------------------------------------- #
# The measurement-reliability gate
# --------------------------------------------------------------------------- #
def test_measurable_mask_excludes_the_unresolved_far_field():
    """Ground resolution decays as ~Z^3, so a strict cap must pull the usable
    range in, and a loose one must push it back out."""
    K = np.array([[691.0, 0.0, 360.0], [0.0, 691.0, 640.0], [0.0, 0.0, 1.0]])
    n = np.array([0.0, 1.0 / 1.65, 0.0])
    grid = bev.BevGrid.from_config()
    H = bev.homography_bev_to_image(K, n, grid)

    strict, s_info = bev.measurable_mask(H, grid, (1280, 720), max_m2_per_px=0.005)
    loose, l_info = bev.measurable_mask(H, grid, (1280, 720), max_m2_per_px=0.08)

    assert s_info["measurable_range_m"] < l_info["measurable_range_m"]
    assert strict.sum() < loose.sum()
    assert np.all(loose[strict]), "a looser cap must be a superset"
    # The gate only ever removes cells that were in frame.
    in_frame = bev.bev_validity(H, grid, (1280, 720))
    assert np.all(in_frame[loose])


# --------------------------------------------------------------------------- #
# Automatic camera-height estimation from vehicle roof heights
# --------------------------------------------------------------------------- #
H, W = 200, 300


def _car_mask(v0, v1, u0=50, u1=150):
    m = np.zeros((H, W), bool)
    m[v0:v1, u0:u1] = True
    return m


def _uniform_h(value, mask=None):
    """h field that is `value` everywhere, or just within `mask` (rest 0)."""
    h = np.zeros((H, W), np.float32)
    h[mask if mask is not None else slice(None)] = value
    return h


def test_roofline_height_reads_the_top_slice_only():
    """A car mask with a roofline value up top and a different value below --
    the reading must come from the top slice, not be blended with the body."""
    mask = _car_mask(20, 120)          # 100 rows tall
    h = np.zeros((H, W), np.float32)
    h[20:30, 50:150] = -1.5            # roofline: 8-10 rows, per SCALE_EST_TOP_FRAC
    h[30:120, 50:150] = -0.3           # rest of the body, much lower
    height, ok = inst._roofline_height_m(h, mask)
    assert ok
    assert height == pytest.approx(1.5, abs=0.05)


def test_roofline_rejects_mask_touching_image_top():
    mask = _car_mask(0, 80)            # v0 == 0 -> roofline possibly cut off
    h = _uniform_h(-1.5, mask)
    height, ok = inst._roofline_height_m(h, mask)
    assert not ok and height is None


def test_roofline_rejects_degenerate_mask():
    assert inst._roofline_height_m(np.zeros((H, W), np.float32),
                                   np.zeros((H, W), bool)) == (None, False)


def _det(label="car", score=0.95, v0=20, v1=120, u0=50, u1=150, is_vehicle=True):
    return inst.Detection(label=label, score=score, mask=_car_mask(v0, v1, u0, u1),
                          is_vehicle=is_vehicle)


def _set_roof(det, h, roof_value_m):
    """Paint det's mask with a uniform NEGATIVE h (= height above plane)."""
    h[det.mask] = -roof_value_m
    return h


def test_estimate_returns_none_with_no_qualifying_detection():
    h = np.zeros((H, W), np.float32)
    assert inst.estimate_camera_height_from_vehicles(h, [], implied_biased_height=3.0) is None
    # A detection that fails every filter should also yield None, not crash.
    bad = _det(label="truck")           # not in VEHICLE_ROOF_HEIGHT_PRIOR_M
    assert inst.estimate_camera_height_from_vehicles(h, [bad], 3.0) is None


def test_estimate_single_vehicle_computes_expected_height():
    """implied_biased * (prior / roof_biased) is the whole mechanism -- pin it."""
    d = _det()
    biased_roof = 2.0
    prior = config.VEHICLE_ROOF_HEIGHT_PRIOR_M["car"]
    h = _set_roof(d, np.zeros((H, W), np.float32), biased_roof)

    est = inst.estimate_camera_height_from_vehicles(h, [d], implied_biased_height=3.0)
    assert est is not None
    assert est.n_samples == 1
    assert est.k_median == pytest.approx(prior / biased_roof)
    assert est.camera_height_m == pytest.approx(3.0 * prior / biased_roof)
    assert est.k_spread == 0.0          # nothing to disagree with at n=1
    assert est.per_vehicle[0]["label"] == "car"


def test_estimate_combines_multiple_vehicles_by_median():
    d1 = _det(v0=20, v1=120, u0=20, u1=100)
    d2 = _det(v0=20, v1=120, u0=150, u1=230)
    h = np.zeros((H, W), np.float32)
    _set_roof(d1, h, 1.0)   # k = prior/1.0
    _set_roof(d2, h, 3.0)   # k = prior/3.0

    prior = config.VEHICLE_ROOF_HEIGHT_PRIOR_M["car"]
    est = inst.estimate_camera_height_from_vehicles(h, [d1, d2], implied_biased_height=2.0)
    assert est.n_samples == 2
    expected_k_med = float(np.median([prior / 1.0, prior / 3.0]))
    assert est.k_median == pytest.approx(expected_k_med)
    assert est.camera_height_m == pytest.approx(2.0 * expected_k_med)
    assert est.k_spread > 0, "two disagreeing vehicles must show a nonzero spread"


@pytest.mark.parametrize("mutate,why", [
    (lambda d: setattr(d, "score", config.SCALE_EST_MIN_SCORE - 0.01), "low score"),
    (lambda d: setattr(d, "is_vehicle", False), "not a vehicle"),
    (lambda d: setattr(d, "label", "bus"), "no height prior for this class"),
])
def test_estimate_filters_reject_bad_detections(mutate, why):
    d = _det()
    h = _set_roof(d, np.zeros((H, W), np.float32), 2.0)
    mutate(d)
    assert inst.estimate_camera_height_from_vehicles(h, [d], 3.0) is None, why


def test_estimate_rejects_small_and_short_masks():
    tiny = _det(v0=20, v1=25, u0=50, u1=55)     # well under SCALE_EST_MIN_PIXELS
    h = _set_roof(tiny, np.zeros((H, W), np.float32), 2.0)
    assert inst.estimate_camera_height_from_vehicles(h, [tiny], 3.0) is None

    flat = _det(v0=20, v1=20 + config.SCALE_EST_MIN_MASK_ROWS - 1, u0=20, u1=280)
    h2 = _set_roof(flat, np.zeros((H, W), np.float32), 2.0)
    assert inst.estimate_camera_height_from_vehicles(h2, [flat], 3.0) is None


def test_estimate_rejects_truncated_roofline():
    d = _det(v0=0, v1=120)      # touches the image's top row
    h = _set_roof(d, np.zeros((H, W), np.float32), 2.0)
    assert inst.estimate_camera_height_from_vehicles(h, [d], 3.0) is None


def test_estimate_mixes_qualifying_and_disqualifying_detections():
    """One good car and one bad one -> the good one alone must still produce a
    result; the bad one must not corrupt or block it."""
    good = _det(v0=20, v1=120, u0=20, u1=100)
    bad = _det(v0=20, v1=120, u0=150, u1=230, score=0.1)   # fails score filter
    h = np.zeros((H, W), np.float32)
    _set_roof(good, h, 2.0)
    _set_roof(bad, h, 99.0)     # if this leaked in, k/height would be wildly different

    prior = config.VEHICLE_ROOF_HEIGHT_PRIOR_M["car"]
    est = inst.estimate_camera_height_from_vehicles(h, [good, bad], implied_biased_height=3.0)
    assert est.n_samples == 1
    assert est.camera_height_m == pytest.approx(3.0 * prior / 2.0)


def test_ground_resolution_grows_with_range():
    K = np.array([[691.0, 0.0, 360.0], [0.0, 691.0, 640.0], [0.0, 0.0, 1.0]])
    n = np.array([0.0, 1.0 / 1.65, 0.0])
    grid = bev.BevGrid.from_config()
    H = bev.homography_bev_to_image(K, n, grid)
    m2px = bev.ground_m2_per_pixel(H, grid)

    at = []
    for z in (5.0, 10.0, 20.0, 35.0):
        u, v = grid.world_to_pixel(0.0, z)
        at.append(m2px[v, u])
    assert at == sorted(at), "further away must mean coarser"
    assert at[-1] > 100 * at[0], "the decay is steep -- roughly cubic in range"


# --------------------------------------------------------------------------- #
# bottom_contour_points: the ground-contact contour a vehicle mask projects from
# --------------------------------------------------------------------------- #
def test_bottom_contour_points_basic_correctness():
    mask = np.zeros((50, 50), bool)
    mask[10:30, 5:20] = True                 # rectangle; bottom row is 29
    pts, coverage = inst.bottom_contour_points(mask)

    assert coverage == pytest.approx(1.0)
    assert len(pts) == 15                    # columns 5..19
    assert np.all(pts[:, 1] == 29)           # every point on the true bottom row
    assert np.array_equal(pts[:, 0], np.sort(pts[:, 0])), "left-to-right by column"


def test_bottom_contour_points_ignores_holes_above_the_true_bottom():
    """A hole/gap above the lowest pixel must not confuse the bottom-row pick."""
    mask = np.zeros((50, 50), bool)
    mask[10:15, 5:20] = True                 # roof fragment, disconnected
    mask[25:30, 5:20] = True                 # body reaching the true bottom (row 29)
    pts, _ = inst.bottom_contour_points(mask)
    assert np.all(pts[:, 1] == 29)


def test_bottom_contour_points_excludes_frame_truncated_columns():
    h_img = 50
    mask = np.zeros((h_img, 50), bool)
    mask[10:h_img, 5:15] = True              # touches the last row: truncated
    mask[10:30, 20:30] = True                # a normal, non-truncated column range

    pts, coverage = inst.bottom_contour_points(mask)
    assert 0.0 < coverage < 1.0
    assert set(pts[:, 0].tolist()) == set(range(20, 30)), \
        "only the non-truncated columns should survive"
    assert np.all(pts[:, 1] == 29)


def test_bottom_contour_points_fully_truncated_mask():
    h_img = 50
    mask = np.zeros((h_img, 50), bool)
    mask[10:h_img, 5:20] = True              # every column touches the last row
    pts, coverage = inst.bottom_contour_points(mask)
    assert coverage == pytest.approx(0.0)
    assert len(pts) == 0


def test_bottom_contour_points_empty_mask():
    pts, coverage = inst.bottom_contour_points(np.zeros((50, 50), bool))
    assert len(pts) == 0 and coverage == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# _ground_contact_seed_ids: contours -> a labeled BEV seed image
# --------------------------------------------------------------------------- #
class _Veh:
    """Minimal stand-in for inst.VehicleInstance (._ground_contact_seed_ids only
    reads .inst_id and .pixel_area)."""

    def __init__(self, inst_id, pixel_area):
        self.inst_id = inst_id
        self.pixel_area = pixel_area


def _synthetic_H(grid):
    K = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    u = np.array([0.0, 1.0, 0.0]) / 1.65
    return bev.homography_bev_to_image(K, u, grid)


def test_ground_contact_seed_ids_stamps_each_vehicle_and_flags_low_coverage():
    grid = bev.BevGrid.from_config()
    H = _synthetic_H(grid)

    inst_ids = np.zeros((720, 1280), np.int32)
    inst_ids[600:650, 300:400] = 1           # a normal, fully-visible base
    inst_ids[700:720, 800:900] = 2           # base cropped at the image bottom

    vehicles = [_Veh(1, pixel_area=5000), _Veh(2, pixel_area=2000)]
    seed_ids, low_coverage = dist._ground_contact_seed_ids(inst_ids, vehicles, H, grid)

    assert (seed_ids == 1).any(), "vehicle 1 must get a seed band"
    assert low_coverage == [2], "vehicle 2's cropped base must be flagged"


def test_ground_contact_seed_ids_larger_vehicle_wins_direct_overlap():
    """Ascending pixel_area order -> the larger vehicle paints last and wins,
    mirroring occluder_instance_ids' overlap rule."""
    grid = bev.BevGrid.from_config()
    H = _synthetic_H(grid)

    inst_ids = np.zeros((720, 1280), np.int32)
    inst_ids[600:650, 300:340] = 1           # small vehicle
    inst_ids[600:650, 320:400] = 2           # large vehicle, overlapping columns 320-340

    small_first = [_Veh(1, pixel_area=100), _Veh(2, pixel_area=100000)]
    seed_ids, _ = dist._ground_contact_seed_ids(inst_ids, small_first, H, grid)

    # Where both vehicles' bands land on the exact same BEV cells, vehicle 2
    # (processed last, being larger) must be the one that ends up stamped.
    assert (seed_ids == 2).sum() > (seed_ids == 1).sum()


def test_ground_contact_seed_ids_empty_vehicles():
    grid = bev.BevGrid.from_config()
    H = _synthetic_H(grid)
    seed_ids, low_coverage = dist._ground_contact_seed_ids(
        np.zeros((720, 1280), np.int32), [], H, grid)
    assert not seed_ids.any() and low_coverage == []


# --------------------------------------------------------------------------- #
# width_disturbance: functional ROAD-WIDTH blockage, distinct from area
# --------------------------------------------------------------------------- #
def _width_grid():
    return bev.BevGrid(ppm=20.0, x_min=-10.0, x_max=10.0, z_min=0.5, z_max=40.0)


def test_width_disturbance_basic_correctness():
    grid = _width_grid()
    amodal_bev = np.zeros(grid.shape, bool)
    occluded_bev = np.zeros(grid.shape, bool)
    inst_bev = np.zeros(grid.shape, np.int32)

    row = 300
    amodal_bev[row, 100:200] = True             # 100 cells = 5.0 m road width
    occluded_bev[row, 100:130] = True           # 30 cells = 1.5 m blocked
    inst_bev[row, 100:130] = 1

    per_vehicle, total = dist.width_disturbance(amodal_bev, occluded_bev, inst_bev,
                                                [_V(1)], grid)

    expected_pct = 100.0 * (30 / 100)
    assert per_vehicle[1]["width_max_pct"] == pytest.approx(expected_pct)
    assert per_vehicle[1]["width_mean_pct"] == pytest.approx(expected_pct)
    assert per_vehicle[1]["width_max_m"] == pytest.approx(30 / grid.ppm)
    assert per_vehicle[1]["width_road_m_at_max"] == pytest.approx(100 / grid.ppm)
    assert total["width_max_pct"] == pytest.approx(expected_pct)

    expected_z = grid.z_max - (row + 0.5) / grid.ppm
    assert per_vehicle[1]["width_max_at_z_m"] == pytest.approx(expected_z, abs=0.01)


def test_width_disturbance_max_picks_the_worst_row_mean_averages():
    grid = _width_grid()
    amodal_bev = np.zeros(grid.shape, bool)
    occluded_bev = np.zeros(grid.shape, bool)
    inst_bev = np.zeros(grid.shape, np.int32)

    amodal_bev[300, 100:200] = True    # 5.0 m road
    occluded_bev[300, 100:110] = True  # 10% blocked
    inst_bev[300, 100:110] = 1

    amodal_bev[350, 100:200] = True    # same 5.0 m road, worse blockage
    occluded_bev[350, 100:180] = True  # 80% blocked
    inst_bev[350, 100:180] = 1

    per_vehicle, total = dist.width_disturbance(amodal_bev, occluded_bev, inst_bev,
                                                [_V(1)], grid)
    assert per_vehicle[1]["width_max_pct"] == pytest.approx(80.0)
    assert per_vehicle[1]["width_mean_pct"] == pytest.approx((10.0 + 80.0) / 2)
    assert total["width_max_pct"] == pytest.approx(80.0)


def test_width_disturbance_excludes_narrow_rows():
    """A road span under WIDTH_MIN_ROAD_SPAN_M is measurement-boundary noise,
    not a real usable width, and must not contribute at all."""
    grid = _width_grid()
    amodal_bev = np.zeros(grid.shape, bool)
    occluded_bev = np.zeros(grid.shape, bool)
    inst_bev = np.zeros(grid.shape, np.int32)

    narrow_cells = int(config.WIDTH_MIN_ROAD_SPAN_M * grid.ppm) - 1
    amodal_bev[300, 100:100 + narrow_cells] = True      # too narrow to count
    occluded_bev[300, 100:100 + narrow_cells] = True    # "fully" blocked, but excluded
    inst_bev[300, 100:100 + narrow_cells] = 1

    per_vehicle, total = dist.width_disturbance(amodal_bev, occluded_bev, inst_bev,
                                                [_V(1)], grid)
    assert per_vehicle[1]["width_max_pct"] == 0.0
    assert per_vehicle[1]["width_max_at_z_m"] is None
    assert total["width_max_pct"] == 0.0


def test_width_disturbance_vehicle_with_no_occlusion_is_zero_not_nan():
    grid = _width_grid()
    amodal_bev = np.zeros(grid.shape, bool)
    amodal_bev[300, 100:200] = True
    occluded_bev = np.zeros(grid.shape, bool)          # nothing occluded anywhere
    inst_bev = np.zeros(grid.shape, np.int32)

    per_vehicle, total = dist.width_disturbance(amodal_bev, occluded_bev, inst_bev,
                                                [_V(1)], grid)
    assert per_vehicle[1] == {"width_max_pct": 0.0, "width_mean_pct": 0.0,
                              "width_max_m": 0.0, "width_road_m_at_max": 0.0,
                              "width_max_at_z_m": None}
    assert total["width_max_pct"] == 0.0


def test_width_disturbance_never_exceeds_100_percent():
    """occluded_bev is a subset of amodal_bev by construction elsewhere in the
    pipeline (disturbance.py: occluded = amodal & ~visible), so blocked can never
    exceed road width when that invariant holds -- confirm the metric reflects it
    rather than silently clipping a bug."""
    grid = _width_grid()
    amodal_bev = np.zeros(grid.shape, bool)
    occluded_bev = np.zeros(grid.shape, bool)
    inst_bev = np.zeros(grid.shape, np.int32)

    amodal_bev[300, 100:200] = True
    occluded_bev[300, 100:200] = True     # the entire road width, fully blocked
    inst_bev[300, 100:200] = 1

    per_vehicle, total = dist.width_disturbance(amodal_bev, occluded_bev, inst_bev,
                                                [_V(1)], grid)
    assert per_vehicle[1]["width_max_pct"] == pytest.approx(100.0)
    assert total["width_max_pct"] == pytest.approx(100.0)


def test_width_disturbance_multiple_vehicles_share_a_row_independently():
    grid = _width_grid()
    amodal_bev = np.zeros(grid.shape, bool)
    occluded_bev = np.zeros(grid.shape, bool)
    inst_bev = np.zeros(grid.shape, np.int32)

    amodal_bev[300, 0:200] = True                     # 10.0 m road
    occluded_bev[300, 0:20] = True; inst_bev[300, 0:20] = 1     # vehicle 1: 1.0 m
    occluded_bev[300, 100:160] = True; inst_bev[300, 100:160] = 2  # vehicle 2: 3.0 m

    per_vehicle, total = dist.width_disturbance(amodal_bev, occluded_bev, inst_bev,
                                                [_V(1), _V(2)], grid)
    assert per_vehicle[1]["width_max_pct"] == pytest.approx(100.0 * 20 / 200)
    assert per_vehicle[2]["width_max_pct"] == pytest.approx(100.0 * 60 / 200)
    # Total combines BOTH vehicles' contributions in that row -- must not equal
    # either vehicle's own figure alone.
    assert total["width_max_pct"] == pytest.approx(100.0 * 80 / 200)
