"""Step 9: parking disturbance -- the road area a parked vehicle takes out of use.

Turns the amodal mask into a NUMBER. Two top-down rasters of the same scene are
compared:

  amodal BEV   the road as it actually is        (Mask2Former road + OFRSNet patch)
  visible BEV  the road as the camera sees it    (Mask2Former road alone)

Their difference is road that exists but is not usable, i.e. the functional space
the occluders consume. Rectifying to the ground plane FIRST is what makes this
measurable: in the image a far-away square metre is a handful of pixels and a
near one is thousands, whereas every BEV cell is the same size, so counting cells
is measuring area (src/bev.py).

Per-vehicle attribution is then geometric, not a full-mask warp. The ground-plane
homography is only valid for pixels that truly lie on the plane; a vehicle's
roof/hood/windshield do not, and warping them smears its claimed BEV region away
from the camera past its true position. Only a vehicle's BOTTOM CONTOUR is near
the plane, so attribution seeds come from that contour alone, projected to BEV
and rasterised into a small footprint band (src/bev.py), then propagated to the
rest of its occluded shadow by nearest seed.

Metric scale (camera height) is estimated AUTOMATICALLY by default, from
detected cars' roof heights above the fitted plane (src/instances.py
`estimate_camera_height_from_vehicles`) -- no manual input needed. It falls
back to a fixed assumption only when no car in the scene qualifies. Pass
--camera-height to override with a known value.

Usage:
    python -m src.disturbance --input data/RealWorld/IMG_0864.jpg
    python -m src.disturbance --input img.jpg --vehicle 2 --json
    python -m src.disturbance --input data/footage_frames --limit 10
    python -m src.disturbance --input img.jpg --no-instance-model    # blobs only
    python -m src.disturbance --input img.jpg --camera-height 1.4    # known height

For interactive vehicle selection see `streamlit run app.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import bev  # noqa: E402
from src import common  # noqa: E402
from src import geometry as geo  # noqa: E402
from src import instances as inst  # noqa: E402
from src import predict  # noqa: E402
from src import s3_detect_foreground as s3  # noqa: E402
from src import s5_export_semantics as s5  # noqa: E402

# Reuse predict.py's colour language so a BEV reads like its overlays.
COL_VISIBLE = predict.COL_ROAD      # green  -- road the camera can see
COL_OCCLUDED = predict.COL_PATCH    # cyan   -- road recovered behind occluders
COL_SELECTED = (255, 0, 200)        # magenta -- the selected vehicle's share
COL_OBJECT = predict.COL_OBJECT     # red    -- occluder footprints
COL_FOOTPRINT = (255, 255, 255)     # white  -- each vehicle's own ground-contact band


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class DisturbanceResult:
    image_path: Path
    rgb: np.ndarray
    sem: np.ndarray
    visible: np.ndarray               # image-space modal road
    amodal: np.ndarray                # image-space amodal road
    inst_ids: np.ndarray              # image-space instance ids
    vehicles: list                    # list[inst.VehicleInstance]

    grid: bev.BevGrid | None = None
    H: np.ndarray | None = None
    visible_bev: np.ndarray | None = None
    amodal_bev: np.ndarray | None = None
    occluded_bev: np.ndarray | None = None
    inst_bev: np.ndarray | None = None
    bev_valid: np.ndarray | None = None
    footprint_seeds: np.ndarray | None = None  # each vehicle's own ground-contact
                                                # band, pre-propagation -- for display

    total: dict = field(default_factory=dict)
    per_vehicle: dict = field(default_factory=dict)
    unattributed: dict = field(default_factory=dict)
    scale: dict = field(default_factory=dict)
    resolution: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when a BEV was actually produced (a usable plane was available)."""
        return self.occluded_bev is not None

    def selectable(self) -> list:
        return [v for v in self.vehicles if v.selectable]

    def summary(self) -> dict:
        """JSON-serialisable report."""
        return {
            "image": str(self.image_path),
            "ok": self.ok,
            "measures": "occlusion shadow (amodal road minus visible road), "
                        "attributed by which occluder hides each BEV cell",
            "scale": self.scale,
            "resolution": self.resolution,
            "total": self.total,
            "unattributed": self.unattributed,
            "vehicles": [
                {"id": v.inst_id, "label": v.label,
                 "score": (None if np.isnan(v.score) else round(v.score, 4)),
                 "source": v.source, "selectable": v.selectable,
                 "pixel_area": v.pixel_area, "centroid": list(v.centroid),
                 **self.per_vehicle.get(v.inst_id, {})}
                for v in self.vehicles
            ],
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def load_models(ckpt: Path | str | None = None, device=None,
                use_instance_model: bool = True) -> dict:
    """Every model this needs, in one call (so the app caches one object)."""
    device = device or common.pick_device()
    print(f"[device] {device}")
    processor, m2f, _ = s3.load_model(device)
    id2label = {int(k): v for k, v in m2f.config.id2label.items()}
    return {
        "device": device,
        "m2f": (processor, m2f),
        "lut": s5.build_mapillary_to_ofrs(id2label),
        "ofrsnet": predict.load_ofrsnet(
            Path(ckpt or (config.CKPT_DIR / "ofrsnet_best.pt")), device),
        "instances": inst.load_instance_model(device) if use_instance_model else None,
    }


# --------------------------------------------------------------------------- #
# Steps 3 & 6: compare, then attribute
# --------------------------------------------------------------------------- #
def _ground_contact_seed_ids(inst_ids: np.ndarray, vehicles: list, H: np.ndarray,
                             grid: bev.BevGrid) -> tuple[np.ndarray, list[int]]:
    """Build the attribution seed image: each vehicle's ground-contact band,
    projected to BEV and stamped with its id (see the "Ground-contact footprint
    attribution" section of src/bev.py for why only the contour, not the whole
    mask, gets projected).

    Vehicles are processed smallest-first, so that on any direct band overlap
    the larger/more prominent vehicle paints last and wins -- mirroring the
    ascending-score overlap rule already used in `instances.occluder_instance_ids`.

    Returns (seed_ids, low_coverage_ids) -- the latter lists the ids of vehicles
    whose ground contact was mostly cropped by the frame (low `coverage_frac`),
    for a caller warning.
    """
    seed_ids = np.zeros(grid.shape, np.int32)
    low_coverage: list[int] = []
    for v in sorted(vehicles, key=lambda v: v.pixel_area):
        pts_uv, coverage = inst.bottom_contour_points(inst_ids == v.inst_id)
        if coverage < config.FOOTPRINT_MIN_COVERAGE:
            low_coverage.append(v.inst_id)
        if len(pts_uv) == 0:
            continue
        pts_bev = bev.project_points_to_bev(pts_uv, H, grid)
        band = bev.rasterize_footprint_band(pts_bev, grid, config.FOOTPRINT_BAND_WIDTH_M)
        seed_ids[band] = v.inst_id
    return seed_ids, low_coverage


def attribute(occluded_bev: np.ndarray, inst_bev: np.ndarray, vehicles: list,
              grid: bev.BevGrid, scale_factor: float) -> tuple[dict, dict]:
    """Split the occluded BEV area between the instances that hide it.

    `scale_factor` converts calibrated area to the raw (uncalibrated) figure;
    both are reported because the calibrated one depends on a camera-height prior
    while the raw one inherits monocular depth's scale error.
    """
    def measure(mask):
        a = bev.area_m2(mask, grid)
        return {"area_m2": round(a, 3),
                "area_m2_raw": round(a * scale_factor, 3),
                "cells": int(np.count_nonzero(mask))}

    per_vehicle = {v.inst_id: measure(occluded_bev & (inst_bev == v.inst_id))
                   for v in vehicles}
    unattributed = measure(occluded_bev & (inst_bev == 0))
    return per_vehicle, unattributed


def width_disturbance(amodal_bev: np.ndarray, occluded_bev: np.ndarray,
                      inst_bev: np.ndarray, vehicles: list,
                      grid: bev.BevGrid) -> tuple[dict, dict]:
    """At each along-road position (a BEV row), what fraction of the road's
    CROSS-SECTION is blocked -- distinct from area on purpose: a small occluded
    patch on a wide road and a full-width blockage on a narrow one can have the
    same area but very different real-world consequences (only the second one
    actually stops other traffic passing).

    Road width per row is the OUTER SPAN of `amodal_bev` (tolerant of a small
    internal gap); blocked width is a plain COUNT of `occluded_bev` (a visible
    gap between two occluders in the same row is real, passable road and must
    not be swept into "blocked"). See `bev.row_span_m` / `bev.row_count_m`.

    Rows narrower than config.WIDTH_MIN_ROAD_SPAN_M are excluded -- the BEV
    wedge narrows near the camera, so a very thin span there is measurement
    boundary noise, not a real usable road width.
    """
    road_span = bev.row_span_m(amodal_bev, grid)
    usable = road_span >= config.WIDTH_MIN_ROAD_SPAN_M
    z_of_row = grid.z_max - (np.arange(grid.height) + 0.5) / grid.ppm

    def measure(mask):
        blocked = bev.row_count_m(mask, grid)
        rows = np.nonzero(usable & (blocked > 0))[0]
        if len(rows) == 0:
            return {"width_max_pct": 0.0, "width_mean_pct": 0.0,
                    "width_max_m": 0.0, "width_road_m_at_max": 0.0,
                    "width_max_at_z_m": None}
        frac = blocked[rows] / road_span[rows]
        row = rows[np.argmax(frac)]
        return {"width_max_pct": round(100.0 * float(frac.max()), 2),
                "width_mean_pct": round(100.0 * float(frac.mean()), 2),
                "width_max_m": round(float(blocked[row]), 3),
                "width_road_m_at_max": round(float(road_span[row]), 3),
                "width_max_at_z_m": round(float(z_of_row[row]), 2)}

    per_vehicle = {v.inst_id: measure(occluded_bev & (inst_bev == v.inst_id))
                  for v in vehicles}
    total = measure(occluded_bev)
    return per_vehicle, total


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #
def analyze(image_path: Path, models: dict, *, camera_height_m: float | None = None,
            threshold: float = 0.5, min_score: float | None = None,
            grid: bev.BevGrid | None = None, use_height_prior: bool = True,
            auto_camera_height: bool = True,
            max_m2_per_px: float | None = None) -> DisturbanceResult:
    """Full workflow for one image. The CLI and the Streamlit app both call only
    this, so they can never drift apart."""
    device = models["device"]
    processor, m2f = models["m2f"]
    grid = grid or bev.BevGrid.from_config()
    warnings: list[str] = []

    rgb = common.read_rgb(image_path)
    image = Image.fromarray(rgb)

    # --- semantics -> the two road masks (steps 1 and 2) ---
    seg = s3.segment(processor, m2f, device, image)
    sem = models["lut"][np.clip(seg, 0, len(models["lut"]) - 1)]
    visible = sem == predict.ROAD_IDX
    geo_fields = (predict.compute_geometry(rgb, sem, image_path.stem, device,
                                           image_path=image_path)
                  if models["ofrsnet"].use_geometry else None)
    ofrs_road = predict.predict_amodal(models["ofrsnet"], sem, device,
                                       threshold=threshold, geo_fields=geo_fields)
    amodal, _patch = predict.compose_amodal_mask(sem, ofrs_road)

    # --- step 4: instances, on the support compose_amodal_mask patches ---
    detections = inst.detect(models.get("instances"), device, image,
                             min_score=min_score)
    if models.get("instances") is None:
        warnings.append("Instance model unavailable: occluders split by connected "
                        "components, so touching vehicles may be merged.")
    inst_ids, vehicles = inst.occluder_instance_ids(sem, visible, detections)

    result = DisturbanceResult(image_path=image_path, rgb=rgb, sem=sem,
                               visible=visible, amodal=amodal, inst_ids=inst_ids,
                               vehicles=vehicles, warnings=warnings)

    # --- the ground plane (shared with the network path via resolve_plane) ---
    plane = geo.resolve_plane(image_path.stem, sem, rgb=rgb,
                              image_path=image_path, device=device)
    if plane is None or not plane["valid"]:
        warnings.append("No ground plane could be fitted for this image "
                        "(too little visible road, or the RANSAC fit failed), "
                        "so no BEV or area can be produced.")
        return result
    if not plane["calibrated"]:
        warnings.append(
            f"Camera intrinsics came from '{plane['k_source']}', which is a blind "
            "assumption -- on our own footage it underestimates focal length by "
            "35-45%. Every area below inherits that error. Install GeoCalib or use "
            "images with EXIF for a real estimate.")

    n_raw, K = plane["n"], plane["K"]
    implied = bev.implied_camera_height(n_raw)

    # --- metric scale (see bev.rescale_plane_to_height) ---
    if use_height_prior:
        est = None
        if camera_height_m is None and auto_camera_height:
            # A detected car's ROOF sits a roughly known height above the ground
            # plane regardless of the car's yaw relative to the camera (unlike
            # its apparent WIDTH, which is view-angle-dependent), so it is a
            # per-scene scale anchor that needs no manual input. See
            # config.VEHICLE_ROOF_HEIGHT_PRIOR_M for validation numbers -- this
            # is a real improvement over one fixed constant for unknown cameras,
            # not a precise measurement, hence the sample count/spread reported
            # alongside it rather than a single silent number.
            _, h_field, _ = geo.derive_ground_fields(plane["depth"], n_raw, K)
            est = inst.estimate_camera_height_from_vehicles(h_field, detections, implied)

        if camera_height_m is not None:
            target, mode = camera_height_m, "manual"
        elif est is not None:
            target, mode = est.camera_height_m, "auto_vehicle_height"
        else:
            target, mode = config.BEV_CAMERA_HEIGHT_M, "fixed_fallback"
            if auto_camera_height:
                warnings.append(
                    "No car in this scene met the quality bar for automatic "
                    f"camera-height estimation (need score>={config.SCALE_EST_MIN_SCORE}, "
                    f">={config.SCALE_EST_MIN_PIXELS}px, an unclipped roofline); "
                    f"falling back to the fixed {config.BEV_CAMERA_HEIGHT_M:.2f} m "
                    "assumption. Pass --camera-height if you know this camera's "
                    "real height.")

        n_use, _ = bev.rescale_plane_to_height(n_raw, target)
        # Raw = what the uncorrected plane would have said. A linear world-scale
        # error is a quadratic area error, hence the square.
        scale_factor = (implied / target) ** 2
        result.scale = {"mode": mode,
                        "camera_height_m": round(target, 3),
                        "implied_camera_height_m": round(implied, 3),
                        "area_correction": round(1.0 / scale_factor, 5)}
        if est is not None:
            result.scale.update(n_samples=est.n_samples, k_spread=round(est.k_spread, 3),
                                per_vehicle=est.per_vehicle)
            if est.n_samples == 1:
                warnings.append(
                    "Automatic camera-height estimate is based on a single "
                    "qualifying car -- treat it as a rough estimate, not a "
                    "precise one. More vehicles in frame (or --camera-height) "
                    "would tighten it.")
            elif est.k_spread > 0.3:
                warnings.append(
                    f"The {est.n_samples} cars used for automatic camera-height "
                    f"estimation disagreed substantially (spread {est.k_spread:.0%} "
                    "of the median) -- the estimate may be unreliable for this scene.")
        if implied / target > 2.0:
            warnings.append(
                f"Monocular depth implies a camera height of {implied:.2f} m against "
                f"the {'estimated' if mode == 'auto_vehicle_height' else 'assumed'} "
                f"{target:.2f} m -- a {implied / target:.1f}x scale error, so "
                f"uncorrected areas would be {scale_factor:.0f}x too large. "
                "'area_m2' is corrected; 'area_m2_raw' is not.")
    else:
        n_use, scale_factor = n_raw, 1.0
        # The grid is relabelled into the depth model's own (larger) world so that
        # the uncalibrated figure measures the SAME PHYSICAL REGION as the
        # calibrated one. Reusing the grid as-is would still produce numbers, but
        # from a different swath of road (a 40 m grid reaches ~35 m of raw-world
        # range against ~25 m of corrected range on our footage), so 'area_m2' and
        # 'area_m2_raw' would not be comparable -- which is the whole point of
        # printing them side by side.
        #
        # Scaling the extents up by k and ppm down by k yields a PIXEL-IDENTICAL
        # raster (width = (x_max*k - x_min*k) * ppm/k is unchanged, and the k-fold
        # larger ground point projects through the k-fold smaller normal to the
        # same pixel) whose cells are k^2 larger in area -- exactly the raw figure.
        k = implied / config.BEV_CAMERA_HEIGHT_M
        grid = bev.BevGrid(ppm=grid.ppm / k, x_min=grid.x_min * k,
                           x_max=grid.x_max * k, z_min=grid.z_min * k,
                           z_max=grid.z_max * k)
        max_m2_per_px = ((config.BEV_MAX_M2_PER_PIXEL if max_m2_per_px is None
                          else max_m2_per_px) * k * k)
        result.scale = {"mode": "uncalibrated",
                        "implied_camera_height_m": round(implied, 3),
                        "grid_scaled_by": round(k, 4)}
        warnings.append(
            "No height prior applied: areas are in the depth model's own scale and "
            "are NOT trustworthy in absolute terms. The BEV grid was widened "
            f"{k:.2f}x to match that scale, so its axis labels are in the depth "
            "model's metres, not real ones.")

    if not bev.plane_is_usable(n_use):
        warnings.append(
            f"The fitted plane is not a usable ground plane (implied camera height "
            f"{bev.implied_camera_height(n_use):.2f} m, n_y={n_use[1]:+.4f}); "
            "refusing to rasterise a BEV from it.")
        return result

    # --- steps 1-3: rasterise and compare ---
    H = bev.homography_bev_to_image(K, n_use, grid)
    img_hw = sem.shape
    # Every area is confined to cells that are both in frame and resolved well
    # enough to measure -- without the resolution cap, near-horizon boundary
    # disagreement dominates the result (see bev.ground_m2_per_pixel).
    bev_valid, res_info = bev.measurable_mask(H, grid, img_hw, max_m2_per_px)
    result.resolution = res_info
    visible_bev = bev.warp_to_bev(visible, H, grid) & bev_valid
    amodal_bev = bev.warp_to_bev(amodal, H, grid) & bev_valid
    seed_ids, low_coverage_ids = _ground_contact_seed_ids(inst_ids, vehicles, H, grid)
    inst_bev = bev.nearest_label_bev(seed_ids, grid, config.FOOTPRINT_MAX_ATTRIBUTION_DIST_M)
    occluded_bev = amodal_bev & ~visible_bev & bev_valid
    if res_info["measurable_range_m"] < grid.z_max - 1.0:
        warnings.append(
            f"Areas are measured only out to {res_info['measurable_range_m']:.1f} m "
            f"(of the {grid.z_max:.0f} m grid): beyond that one camera pixel covers "
            f"more than {res_info['max_m2_per_px']} m^2 of road and a difference of a "
            "few pixels would swamp the result. Raise --max-m2-per-px to extend the "
            "range at the cost of precision.")
    if low_coverage_ids:
        warnings.append(
            f"Vehicle(s) {', '.join(f'#{i}' for i in low_coverage_ids)} have their "
            "ground contact mostly cropped by the frame edge, so their footprint "
            "(and any area attributed to them) may be incomplete.")

    # --- step 6: attribute ---
    per_vehicle, unattributed = attribute(occluded_bev, inst_bev, vehicles, grid,
                                          scale_factor)

    # Area alone conflates a small patch on a wide road with a full-width
    # blockage on a narrow one -- this is the same per_vehicle/total split as
    # attribute(), but measuring what fraction of the road's WIDTH is blocked
    # at each vehicle's own worst along-road position, not its area.
    width_per_vehicle, width_total = width_disturbance(amodal_bev, occluded_bev,
                                                       inst_bev, vehicles, grid)
    for vid, w in width_per_vehicle.items():
        per_vehicle[vid].update(w)

    amodal_area = bev.area_m2(amodal_bev, grid)
    occluded_area = bev.area_m2(occluded_bev, grid)
    result.grid, result.H = grid, H          # the grid actually used, post-rescale
    result.visible_bev, result.amodal_bev = visible_bev, amodal_bev
    result.occluded_bev, result.inst_bev, result.bev_valid = occluded_bev, inst_bev, bev_valid
    result.footprint_seeds = seed_ids
    result.per_vehicle, result.unattributed = per_vehicle, unattributed
    result.total = {
        "amodal_road_m2": round(amodal_area, 3),
        "visible_road_m2": round(bev.area_m2(visible_bev, grid), 3),
        "occluded_road_m2": round(occluded_area, 3),
        "occluded_road_m2_raw": round(occluded_area * scale_factor, 3),
        # Scale-immune, and therefore the figure to trust.
        "occluded_pct": (round(100.0 * occluded_area / amodal_area, 2)
                         if amodal_area > 0 else 0.0),
        **width_total,
    }

    # An occluder hides the road BEHIND it all the way to the horizon, so when the
    # shadow reaches the far measurement boundary the reported area is whatever
    # fitted inside the range -- a lower bound, and range-dependent. Saying so is
    # the difference between a measurement and a number.
    cols = np.nonzero(bev_valid.any(axis=0))[0]
    if len(cols):
        far_row = np.argmax(bev_valid[:, cols], axis=0)
        if bool(occluded_bev[far_row, cols].any()):
            result.total["shadow_truncated"] = True
            warnings.append(
                "The occluded region reaches the far measurement boundary: an "
                "occluder shadows the road behind it indefinitely, so these areas "
                "are LOWER BOUNDS truncated at "
                f"{res_info['measurable_range_m']:.1f} m, not the full shadow.")

    if unattributed["cells"] and occluded_area > 0:
        frac = 100.0 * unattributed["area_m2"] / occluded_area
        if frac > 20.0:
            warnings.append(
                f"{frac:.0f}% of the occluded area is not attributed to any "
                "instance -- the instance masks and the Mapillary occluder support "
                "disagree substantially on this image.")
    return result


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_bev(result: DisturbanceResult, selected: int | None = None,
               scale: int = 1) -> np.ndarray:
    """The BEV panel: visible road, occluded road, and one vehicle's share.

    Layers composite with `common.overlay_mask`, the same primitive every preview
    in the repo uses, so the colours mean what they mean elsewhere.
    """
    grid = result.grid
    canvas = np.zeros((grid.height, grid.width, 3), np.uint8)
    canvas = common.overlay_mask(canvas, result.bev_valid, (26, 26, 30), 1.0)
    canvas = common.overlay_mask(canvas, result.visible_bev, COL_VISIBLE, 0.75)
    canvas = common.overlay_mask(canvas, result.occluded_bev, COL_OCCLUDED, 0.85)
    # Every vehicle's own ground-contact band -- the direct visual evidence that
    # attribution is now anchored to each vehicle's true base rather than a
    # smeared full-body warp. Drawn BEFORE the selection highlight so the
    # selected vehicle's own band still reads as selected, not white.
    if result.footprint_seeds is not None:
        canvas = common.overlay_mask(canvas, result.footprint_seeds > 0,
                                     COL_FOOTPRINT, 0.9)
    if selected is not None:
        own = result.occluded_bev & (result.inst_bev == selected)
        canvas = common.overlay_mask(canvas, own, COL_SELECTED, 0.95)
    canvas = bev.draw_grid_overlay(canvas, grid)
    if scale != 1:
        canvas = cv2.resize(canvas, (grid.width * scale, grid.height * scale),
                            interpolation=cv2.INTER_NEAREST)
    return canvas


def render_candidates(result: DisturbanceResult,
                      selected: int | None = None) -> np.ndarray:
    """Image-space overlay with every occluder numbered, for picking one."""
    out = common.overlay_mask(result.rgb, result.amodal, COL_VISIBLE, 0.35)
    out = common.overlay_mask(out, result.inst_ids > 0, COL_OBJECT, 0.30)
    if selected is not None:
        out = common.overlay_mask(out, result.inst_ids == selected, COL_SELECTED, 0.50)

    for v in result.vehicles:
        u0, v0, u1, v1 = v.bbox
        hot = (v.inst_id == selected)
        colour = COL_SELECTED if hot else (COL_OBJECT if v.selectable else (150, 150, 150))
        cv2.rectangle(out, (u0, v0), (u1, v1), colour, 2 if hot else 1)
        tag = f"#{v.inst_id} {v.label}"
        cu, cv_ = v.centroid
        cv2.putText(out, tag, (max(2, cu - 30), max(14, cv_)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, tag, (max(2, cu - 30), max(14, cv_)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_report(result: DisturbanceResult, selected: int | None) -> None:
    print(f"\n=== {result.image_path.name} ===")
    for w in result.warnings:
        print(f"  [warn] {w}")
    if not result.ok:
        print("  no BEV produced.")
        return

    t, s = result.total, result.scale
    height_note = f"camera height {s['camera_height_m']:.2f} m"
    if s["mode"] == "auto_vehicle_height":
        height_note += (f"  (auto, {s['n_samples']} car(s), "
                        f"spread {s['k_spread']:.0%})")
    elif s["mode"] == "fixed_fallback":
        height_note += "  (no qualifying car -- fixed assumption)"
    elif s["mode"] == "manual":
        height_note += "  (manual)"
    print(f"  scale            {s['mode']:<20} {height_note}")
    print(f"  implied (biased) {s['implied_camera_height_m']:9.2f} m")
    print(f"  amodal road      {t['amodal_road_m2']:9.2f} m^2")
    print(f"  visible road     {t['visible_road_m2']:9.2f} m^2")
    print(f"  OCCLUDED road    {t['occluded_road_m2']:9.2f} m^2   "
          f"({t['occluded_pct']:.2f}% of amodal road)")
    print(f"    uncalibrated   {t['occluded_road_m2_raw']:9.2f} m^2   "
          "<- monocular depth scale, do not trust absolutely")
    if t["width_max_at_z_m"] is not None:
        print(f"  max width blocked{t['width_max_pct']:8.2f} %   "
              f"({t['width_max_m']:.2f} / {t['width_road_m_at_max']:.2f} m "
              f"at {t['width_max_at_z_m']:.1f} m out)")
    r = result.resolution
    print(f"  measured to      {r['measurable_range_m']:.1f} m  "
          f"({r['measurable_pct']:.1f}% of grid; {r['in_frame_pct']:.1f}% in frame)")
    print(f"  {'id':>3}  {'label':<12} {'score':>6} {'src':<9} "
          f"{'area m^2':>9} {'raw m^2':>9} {'max w%':>7}  sel")
    # Ids stay area-of-pixels ordered so `--vehicle N` is stable, but the table
    # reads by how much road each one actually costs. Occluders hiding no
    # measurable road are counted, not listed -- a scene can have dozens.
    ranked = sorted(result.vehicles,
                    key=lambda v: -result.per_vehicle.get(v.inst_id, {}).get("area_m2", 0.0))
    n_zero = 0
    for v in ranked:
        m = result.per_vehicle.get(v.inst_id, {})
        if m.get("area_m2", 0.0) <= 0.0 and v.inst_id != selected:
            n_zero += 1
            continue
        star = " <-- selected" if v.inst_id == selected else ""
        score = "  n/a" if np.isnan(v.score) else f"{v.score:.3f}"
        print(f"  {v.inst_id:>3}  {v.label:<12} {score:>6} {v.source:<9} "
              f"{m.get('area_m2', 0):9.2f} {m.get('area_m2_raw', 0):9.2f} "
              f"{m.get('width_max_pct', 0):7.1f}  "
              f"{'y' if v.selectable else '-'}{star}")
    if n_zero:
        print(f"  {'':>3}  ... and {n_zero} more occluder(s) hiding no measurable road")
    u = result.unattributed
    print(f"  {'-':>3}  {'unattributed':<12} {'':>6} {'':<9} "
          f"{u['area_m2']:9.2f} {u['area_m2_raw']:9.2f}")

    # The invariant worth shouting about if it ever breaks.
    total_cells = sum(m["cells"] for m in result.per_vehicle.values()) + u["cells"]
    assert total_cells == int(result.occluded_bev.sum()), \
        f"attribution lost cells: {total_cells} != {int(result.occluded_bev.sum())}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="image file or directory")
    ap.add_argument("--out", default=str(config.DATA_DIR / "disturbance"))
    ap.add_argument("--ckpt", default=str(config.CKPT_DIR / "ofrsnet_best.pt"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--vehicle", type=int, default=None,
                    help="instance id to treat as the parked vehicle "
                         "(default: report all)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="OFRSNet road-probability threshold")
    ap.add_argument("--min-score", type=float, default=None,
                    help=f"instance score threshold (default {config.INSTANCE_MIN_SCORE})")
    ap.add_argument("--camera-height", type=float, default=None,
                    help="true camera height in metres for the metric scale "
                         "correction; overrides automatic estimation. Without "
                         "this, height is estimated automatically from detected "
                         f"cars' roof heights, falling back to "
                         f"{config.BEV_CAMERA_HEIGHT_M} m only when no car qualifies")
    ap.add_argument("--no-auto-height", action="store_true",
                    help=f"skip automatic camera-height estimation; use the fixed "
                         f"{config.BEV_CAMERA_HEIGHT_M} m assumption directly "
                         "(ignored if --camera-height is set)")
    ap.add_argument("--no-height-prior", action="store_true",
                    help="report only uncorrected areas in the depth model's scale")
    ap.add_argument("--no-instance-model", action="store_true",
                    help="skip the instance checkpoint; split occluders by "
                         "connected components instead")
    ap.add_argument("--ppm", type=float, default=None,
                    help=f"BEV pixels per metre (default {config.BEV_PPM})")
    ap.add_argument("--max-m2-per-px", type=float, default=None,
                    help="exclude BEV cells whose source pixel covers more ground "
                         f"than this (default {config.BEV_MAX_M2_PER_PIXEL}); raising "
                         "it extends range but admits far-field boundary noise")
    ap.add_argument("--json", action="store_true", help="also write a .json report")
    args = ap.parse_args()

    images = predict.list_images(Path(args.input))
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found under {args.input}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = load_models(args.ckpt, use_instance_model=not args.no_instance_model)
    grid = bev.BevGrid.from_config(ppm=args.ppm)

    for path in tqdm(images, desc="disturbance"):
        result = analyze(path, models, camera_height_m=args.camera_height,
                         threshold=args.threshold, min_score=args.min_score,
                         grid=grid, use_height_prior=not args.no_height_prior,
                         auto_camera_height=not args.no_auto_height,
                         max_m2_per_px=args.max_m2_per_px)
        _print_report(result, args.vehicle)

        Image.fromarray(render_candidates(result, args.vehicle)).save(
            out_dir / f"{path.stem}_candidates.png")
        if result.ok:
            Image.fromarray(render_bev(result, args.vehicle)).save(
                out_dir / f"{path.stem}_bev.png")
        if args.json:
            summary = result.summary()
            summary["selected_vehicle"] = args.vehicle
            (out_dir / f"{path.stem}.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[ok] wrote reports to {out_dir}")


if __name__ == "__main__":
    main()
