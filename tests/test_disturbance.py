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
