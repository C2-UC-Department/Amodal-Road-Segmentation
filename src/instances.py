"""Per-vehicle instances, defined on the occluder support the pipeline patches.

Everything upstream of here is SEMANTIC: `sem == vehicle` says which pixels are
vehicle, not which vehicle. Measuring the road one *particular* parked car hides
needs instances, so this module adds a Mask2Former instance head (same
`transformers` family as config.MODEL_ID -- no new dependency).

The load-bearing design decision is what the ids are defined ON.
`predict.compose_amodal_mask` only ever patches inside
`common.occluder_blob_mask(...)` -- whole connected blobs of the Mapillary
person/vehicle classes that touch the near-road band. The occluded region we
measure is therefore *defined* by that support. If ids came straight from the
instance model instead, every pixel where Cityscapes and Mapillary disagree about
a vehicle's extent would fall outside the support and its occluded area would
vanish into an unexplained remainder.

So: take the support as given, and use the instance model only to SPLIT it.
Consequences worth having --

  * Attribution is complete by construction, and the leftover figure becomes a
    real diagnostic (instance/semantic disagreement) instead of a mystery.
  * The instance checkpoint is OPTIONAL. Without it, the support is split by
    connected components -- exactly the abstraction `occluder_blob_mask` already
    uses -- so the whole pipeline still runs with nothing new downloaded. It
    merges touching vehicles, which is precisely what the instance model fixes.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import common  # noqa: E402

ROAD_IDX = config.OFRS_CLASSES.index("road")
PERSON_IDX = config.OFRS_CLASSES.index("person")
VEHICLE_IDX = config.OFRS_CLASSES.index("vehicle")

_CACHE: dict = {}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_vehicle_class_ids(id2label: dict[int, str]) -> set[int]:
    """Class ids whose label matches a vehicle keyword.

    Substring matching against id2label, exactly like
    `s3_detect_foreground.build_foreground_class_ids`, so the checkpoint's label
    spelling does not matter. Cityscapes-instance ships 8 'thing' classes
    (person, rider, car, truck, bus, train, motorcycle, bicycle); person and
    rider deliberately do not match -- they are occluders but not parked
    vehicles.
    """
    keywords = [k.lower() for k in config.VEHICLE_INSTANCE_KEYWORDS]
    return {int(cid) for cid, label in id2label.items()
            if any(k in label.lower() for k in keywords)}


def load_instance_model(device):
    """(processor, model, vehicle_ids), or None if the checkpoint is unavailable.

    Returning None rather than raising is deliberate: `occluder_instance_ids`
    falls back to connected components, so a missing/undownloadable checkpoint
    degrades the result instead of breaking the run (the same posture
    `calibration.geocalib_available` takes for its optional dependency).
    """
    key = f"instance::{config.INSTANCE_MODEL_ID}"
    if key in _CACHE:
        return _CACHE[key]
    try:
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        print(f"[model] loading {config.INSTANCE_MODEL_ID} on {device} ...")
        processor = AutoImageProcessor.from_pretrained(config.INSTANCE_MODEL_ID)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(config.INSTANCE_MODEL_ID)
        model.to(device).eval()
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        vehicle_ids = build_vehicle_class_ids(id2label)
        print(f"[model] {len(vehicle_ids)} selectable vehicle classes: "
              + ", ".join(sorted(id2label[c] for c in vehicle_ids)))
        _CACHE[key] = (processor, model, vehicle_ids)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] instance model unavailable ({type(exc).__name__}: {exc}).\n"
              "       Falling back to connected components of the occluder mask; "
              "touching vehicles will merge.")
        _CACHE[key] = None
    return _CACHE[key]


@dataclass
class Detection:
    """One raw instance from the model, before the support restriction."""

    label: str
    score: float
    mask: np.ndarray          # bool (H, W)
    is_vehicle: bool


@torch.no_grad()
def detect(models, device, image: Image.Image,
           min_score: float | None = None) -> list[Detection]:
    """Run the instance head over one image."""
    if models is None:
        return []
    processor, model, vehicle_ids = models
    thresh = config.INSTANCE_MIN_SCORE if min_score is None else min_score

    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    res = processor.post_process_instance_segmentation(
        outputs, target_sizes=[image.size[::-1]], threshold=thresh,
    )[0]

    seg = res["segmentation"]
    seg = seg.cpu().numpy() if hasattr(seg, "cpu") else np.asarray(seg)
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    out: list[Detection] = []
    for info in res.get("segments_info", []):
        score = float(info.get("score", 1.0))
        if score < thresh:
            continue
        mask = seg == int(info["id"])
        if not mask.any():
            continue
        label_id = int(info["label_id"])
        out.append(Detection(label=id2label.get(label_id, str(label_id)),
                             score=score, mask=mask,
                             is_vehicle=label_id in vehicle_ids))
    return out


# --------------------------------------------------------------------------- #
# Automatic camera-height estimation from detected vehicles
# --------------------------------------------------------------------------- #
@dataclass
class ScaleEstimate:
    """A per-scene metric-scale correction derived from detected vehicles.

    `camera_height_m` is directly usable as the `camera_height_m` target for
    `bev.rescale_plane_to_height` -- it is already in real metres, not a raw
    correction factor.
    """

    camera_height_m: float
    n_samples: int
    k_median: float                    # correction factor actually applied
    k_spread: float                    # (p75-p25)/median of the k's that combined; 0 if n=1
    per_vehicle: list[dict]            # diagnostics: label, roof_height_biased_m, k


def _roofline_height_m(h_field: np.ndarray, mask: np.ndarray) -> tuple[float | None, bool]:
    """Median height-above-plane (positive, metres, in the depth model's OWN
    biased scale) over the top slice of a vehicle mask -- its roofline.

    Broad percentiles over the WHOLE vehicle body are unreliable: a close/cropped
    vehicle's mask may not reach the ground at all, mixing hood- and
    roof-height pixels indiscriminately. The roofline specifically is what a
    fixed real-world height prior (a car's actual roof height) can be compared
    against regardless of viewing angle or how much of the body is visible.

    Returns (height_m, ok). ok is False when the mask touches the image's top
    row -- the roofline itself may be cut off by the frame, which would silently
    bias the reading low.
    """
    vs, us = np.nonzero(mask)
    if len(vs) == 0:
        return None, False
    v0, v1 = int(vs.min()), int(vs.max())
    if v0 <= 1:
        return None, False           # roofline possibly truncated by the frame edge
    span = max(1, v1 - v0)
    cutoff = v0 + max(config.SCALE_EST_MIN_TOP_ROWS,
                      int(round(span * config.SCALE_EST_TOP_FRAC)))
    top = mask & (np.arange(mask.shape[0])[:, None] <= cutoff)
    if int(top.sum()) < 3:
        return None, False
    # h is negative above the plane (see geometry.derive_ground_fields); height
    # above ground is therefore -h.
    height = float(-np.median(h_field[top]))
    return height, True


def bottom_contour_points(mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Per image column, the bottom-most (max-row) True pixel of `mask`.

    Only this contour is actually near the ground plane -- everything above it
    (bonnet, roof) is not, and would smear if projected through a ground-plane
    homography (see the "Ground-contact footprint attribution" section of
    src/bev.py). Points come out already ordered left-to-right by column, ready
    for `bev.rasterize_footprint_band`. Public (not `_`-prefixed) because
    src/disturbance.py calls this directly, on masks it reads back off `inst_ids`.

    Columns whose bottom pixel sits on the image's last row (or the one above
    it) are excluded: the vehicle's true ground contact may extend below the
    visible frame, so that "contour" point is not trustworthy -- mirroring
    `_roofline_height_m`'s `v0 <= 1` guard at the opposite edge. `coverage_frac`
    is the fraction of spanned columns that were NOT excluded this way; low
    coverage means the footprint is likely incomplete.

    Returns ((N, 2) uv points, coverage_frac). N may be 0 for a degenerate mask.
    """
    if not mask.any():
        return np.zeros((0, 2)), 0.0
    h_img, _ = mask.shape
    row_idx = np.where(mask, np.arange(h_img)[:, None], -1)
    bottom_row = row_idx.max(axis=0)              # (W,); -1 = no pixel in that column
    cols = np.nonzero(bottom_row >= 0)[0]
    if len(cols) == 0:
        return np.zeros((0, 2)), 0.0

    truncated = bottom_row[cols] >= h_img - 2
    coverage_frac = float(1.0 - truncated.mean())
    usable = cols[~truncated]
    if len(usable) == 0:
        return np.zeros((0, 2)), coverage_frac
    points = np.stack([usable, bottom_row[usable]], axis=1).astype(np.float64)
    return points, coverage_frac


def estimate_camera_height_from_vehicles(h_field: np.ndarray,
                                         detections: list[Detection],
                                         implied_biased_height: float,
                                         ) -> ScaleEstimate | None:
    """Estimate the camera's TRUE height from cars' roof heights above the plane.

    For each qualifying car, `k = real_roof_height_prior / roof_height_biased`
    is the same correction factor that would fix the plane's implied camera
    height too (both are derived from the same depth, so they share the same
    scale bias) -- so `implied_biased_height * median(k)` is the estimate.

    Returns None when no detection meets the quality bar (config.SCALE_EST_*);
    callers should fall back to a fixed assumption in that case. A result with
    `n_samples == 1` is a real estimate, not a placeholder, but has no
    cross-vehicle agreement to report -- callers should treat it as
    lower-confidence than n_samples >= 2.
    """
    per_vehicle: list[dict] = []
    ks: list[float] = []
    for d in detections:
        if not d.is_vehicle or d.label not in config.VEHICLE_ROOF_HEIGHT_PRIOR_M:
            continue
        if d.score < config.SCALE_EST_MIN_SCORE:
            continue
        if int(d.mask.sum()) < config.SCALE_EST_MIN_PIXELS:
            continue
        vs, _ = np.nonzero(d.mask)
        if len(vs) == 0 or (vs.max() - vs.min()) < config.SCALE_EST_MIN_MASK_ROWS:
            continue
        roof_biased, ok = _roofline_height_m(h_field, d.mask)
        if not ok or roof_biased is None or roof_biased < config.SCALE_EST_MIN_ROOF_HEIGHT_M:
            continue
        k = config.VEHICLE_ROOF_HEIGHT_PRIOR_M[d.label] / roof_biased
        ks.append(k)
        per_vehicle.append({"label": d.label, "score": round(d.score, 3),
                            "pixel_area": int(d.mask.sum()),
                            "roof_height_biased_m": round(roof_biased, 3),
                            "k": round(k, 4)})
    if not ks:
        return None

    k_med = float(np.median(ks))
    spread = ((np.percentile(ks, 75) - np.percentile(ks, 25)) / k_med
             if len(ks) > 1 and k_med > 0 else 0.0)
    return ScaleEstimate(camera_height_m=implied_biased_height * k_med,
                         n_samples=len(ks), k_median=k_med,
                         k_spread=float(spread), per_vehicle=per_vehicle)


# --------------------------------------------------------------------------- #
# Instances on the occluder support
# --------------------------------------------------------------------------- #
@dataclass
class VehicleInstance:
    """One occluder in the support, with the id it carries in the id map."""

    inst_id: int                       # 1-based; matches `inst_ids == inst_id`
    label: str
    score: float
    pixel_area: int
    centroid: tuple[int, int]          # (u, v)
    bbox: tuple[int, int, int, int]    # (u0, v0, u1, v1), exclusive upper
    selectable: bool                   # a candidate "parked vehicle"
    source: str                        # "instance" | "blob"


def occluder_support(sem: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """The exact region `predict.compose_amodal_mask` is allowed to patch.

    Kept as its own function so there is one definition of "the occluders that
    matter", shared by the id map and by any caller that needs to reason about
    what the amodal patch could possibly cover.
    """
    occluders = (sem == VEHICLE_IDX) | (sem == PERSON_IDX)
    return common.occluder_blob_mask(occluders, visible, config.ROAD_NEIGHBOURHOOD_PX)


def _describe(mask: np.ndarray) -> tuple[int, tuple[int, int], tuple[int, int, int, int]]:
    vs, us = np.nonzero(mask)
    return (int(mask.sum()),
            (int(us.mean()), int(vs.mean())),
            (int(us.min()), int(vs.min()), int(us.max()) + 1, int(vs.max()) + 1))


def _dominant_label(sem: np.ndarray, mask: np.ndarray) -> tuple[str, bool]:
    """Name an un-matched blob from the semantics that produced it."""
    n_veh = int((sem[mask] == VEHICLE_IDX).sum())
    n_per = int((sem[mask] == PERSON_IDX).sum())
    return ("vehicle", True) if n_veh >= n_per else ("person", False)


def occluder_instance_ids(sem: np.ndarray, visible: np.ndarray,
                          detections: list[Detection] | None = None,
                          ) -> tuple[np.ndarray, list[VehicleInstance]]:
    """Split the occluder support into instances.

    Returns (inst_ids (H, W) int32 with 0 = not an occluder, instances). Ids are
    renumbered 1..N by DESCENDING pixel area, so `#1` is the most prominent
    occluder and the numbering a user sees is stable and meaningful.

    Support pixels no detection claims are kept as their own `source="blob"`
    instances when they are big enough to matter, and left at 0 otherwise -- in
    which case their occluded area is reported as unattributed rather than being
    quietly folded into a neighbour.
    """
    support = occluder_support(sem, visible)
    ids = np.zeros(support.shape, np.int32)
    meta: dict[int, tuple[str, float, bool, str]] = {}
    next_id = 1

    # Ascending score, so the most confident detection paints last and wins any
    # overlap. Ids are handed out in the same order; the authoritative mask for
    # each instance is read back off `ids` afterwards, so a detection buried by a
    # better one simply ends up empty and is dropped.
    for det in sorted(detections or [], key=lambda d: d.score):
        m = det.mask & support
        if not m.any():
            continue
        ids[m] = next_id
        meta[next_id] = (det.label, det.score, det.is_vehicle, "instance")
        next_id += 1

    leftover = support & (ids == 0)
    if leftover.any():
        n_lab, labels = cv2.connectedComponents(leftover.astype(np.uint8), connectivity=8)
        for lab in range(1, n_lab):
            comp = labels == lab
            if int(comp.sum()) < config.INSTANCE_MIN_BLOB_PX:
                continue          # sliver: leave at 0 -> counted as unattributed
            label, selectable = _dominant_label(sem, comp)
            ids[comp] = next_id
            meta[next_id] = (label, float("nan"), selectable, "blob")
            next_id += 1

    # Renumber by descending area so the user-visible numbering is meaningful.
    present = [(i, int((ids == i).sum())) for i in sorted(meta)]
    present = [(i, a) for i, a in present if a > 0]
    present.sort(key=lambda t: -t[1])

    lut = np.zeros(next_id, np.int32)
    for new, (old, _) in enumerate(present, start=1):
        lut[old] = new
    ids = lut[ids]

    instances: list[VehicleInstance] = []
    for new, (old, _) in enumerate(present, start=1):
        label, score, selectable, source = meta[old]
        area, centroid, bbox = _describe(ids == new)
        instances.append(VehicleInstance(inst_id=new, label=label, score=score,
                                         pixel_area=area, centroid=centroid,
                                         bbox=bbox, selectable=selectable,
                                         source=source))
    return ids, instances
