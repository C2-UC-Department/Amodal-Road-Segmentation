"""Central configuration for the amodal road-segmentation dataset pipeline.

All paths are relative to the project root so the project stays portable.
Import `config` anywhere and read the constants below.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                    # KITTI download lands here
KITTI_ROOT = RAW_DIR / "data_road"           # data_road/training, data_road/testing

# Processed / intermediate artefacts (one PNG per KITTI image, shared base name)
PROC_DIR = DATA_DIR / "processed"
VISIBLE_DIR = PROC_DIR / "visible_road"      # binary visible-road mask (KITTI GT decoded)
FOREGROUND_DIR = PROC_DIR / "foreground"     # binary foreground-object mask (Mask2Former)
OCCLUSION_DIR = PROC_DIR / "occlusion"       # occlusion-hint mask (foreground near road)
PREVIEW_DIR = PROC_DIR / "preview"           # human-readable overlays (optional QA)
SEMANTIC_DIR = PROC_DIR / "semantic_ofrs"    # 11-class OFRS semantic label map (OFRSNet INPUT)

# Final, human-corrected amodal ground truth (annotator output)
AMODAL_DIR = DATA_DIR / "amodal_road"

# Real-world footage (your own videos) and extracted image snapshots
FOOTAGE_DIR = DATA_DIR / "footage"
FRAMES_DIR = DATA_DIR / "footage_frames"

# OFRSNet training artefacts
CKPT_DIR = ROOT / "checkpoints"

# KITTI only ships ground truth for the *training* split, so amodal labels can
# only be produced there. Set to "testing" if you just want foreground masks.
SPLIT = "training"

# --------------------------------------------------------------------------- #
# KITTI Road dataset
# --------------------------------------------------------------------------- #
KITTI_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip"
KITTI_ZIP = RAW_DIR / "data_road.zip"

# --------------------------------------------------------------------------- #
# Mask2Former (foreground detection)
# --------------------------------------------------------------------------- #
# Semantic Mask2Former trained on Mapillary Vistas (65 classes). Semantic is
# enough here: we only need "which pixels are foreground objects", not instances.
MODEL_ID = "facebook/mask2former-swin-large-mapillary-vistas-semantic"

# Mapillary Vistas class *names* we treat as occluding foreground. We match by
# substring against the model's id2label (case-insensitive) so we are robust to
# the exact label spelling ("Car", "car", "Other Vehicle", ...).
FOREGROUND_KEYWORDS = [
    "person",
    "rider",          # Bicyclist / Motorcyclist / Other Rider
    "bicyclist",
    "motorcyclist",
    "car",
    "truck",
    "bus",
    "caravan",
    "trailer",
    "motorcycle",
    "bicycle",
    "on rails",       # trams / trains
    "boat",
    "wheeled slow",
    "other vehicle",
    "van",
]

# Labels to reject even if they match a keyword above. "Car Mount" contains
# "car" but is the camera rig / ego-vehicle mount, not a road user.
FOREGROUND_EXCLUDE_KEYWORDS = [
    "car mount",
    "ego",
]

# Grow foreground masks slightly so anti-aliased object boundaries are fully
# covered before we subtract them from the road.
FOREGROUND_DILATE_PX = 5

# For the occlusion hint: how far below/around the road we consider a foreground
# object "capable of occluding the road" (dilation of the road region in px).
ROAD_NEIGHBOURHOOD_PX = 25

# Inference device preference order. "mps" = Apple Silicon GPU.
DEVICE_PREFERENCE = ["mps", "cuda", "cpu"]

# --------------------------------------------------------------------------- #
# OFRSNet (Sensors 2019, 19(21), 4711) — input semantic scheme & training
# --------------------------------------------------------------------------- #
# OFRSNet's INPUT is a multi-class semantic map (one-hot), NOT the RGB image.
# The paper unifies everything to these 11 classes. Index == channel.
OFRS_CLASSES = [
    "road",         # 0
    "sidewalk",     # 1
    "building",     # 2
    "wall",         # 3
    "fence",        # 4
    "pole",         # 5
    "traffic_sign", # 6
    "vegetation",   # 7
    "person",       # 8
    "vehicle",      # 9
    "unlabeled",    # 10  (void / everything else)
]
OFRS_NUM_CLASSES = len(OFRS_CLASSES)
OFRS_UNLABELED_ID = OFRS_CLASSES.index("unlabeled")

# Map Mapillary-Vistas label *names* -> OFRS class, by substring keyword, in
# PRIORITY ORDER (first group that matches wins). Anything unmatched -> unlabeled.
# `exclude` rejects a label from a group even if a keyword matched.
OFRS_CLASS_KEYWORDS = [
    # (ofrs_class,       [keywords],                                              [exclude])
    ("unlabeled",     ["car mount", "ego", "void", "unlabeled"],                 []),
    ("traffic_sign",  ["traffic sign", "signage", "back", "front"],              []),
    ("person",        ["person", "rider", "bicyclist", "motorcyclist"],          []),
    ("vehicle",       ["car", "truck", "bus", "caravan", "trailer", "motorcycle",
                       "bicycle", "on rails", "boat", "wheeled slow",
                       "other vehicle", "van", "vehicle"],                        ["mount"]),
    ("road",          ["road", "lane marking", "service lane", "crosswalk"],      []),
    ("sidewalk",      ["sidewalk", "curb", "pedestrian area"],                    []),
    ("building",      ["building"],                                               []),
    ("wall",          ["wall"],                                                   []),
    ("fence",         ["fence", "guard rail", "barrier"],                         []),
    ("pole",          ["pole", "street light", "traffic light", "utility"],       []),
    ("vegetation",    ["vegetation", "terrain"],                                  []),
]

# OFRSNet is fully convolutional (global context uses adaptive pooling; every
# up-sampling stage explicitly targets its skip connection's own shape), so it
# accepts ANY input resolution/aspect ratio -- no fixed canvas is required.
# We only:
#   1. cap the longer side to OFRS_MAX_SIDE (uniform scale, aspect PRESERVED --
#      unlike the old fixed-(384,1248) resize, this never stretches/squashes),
#      purely to bound compute on very large photos;
#   2. pad up to a multiple of OFRS_NET_STRIDE so the /8 down/up-sampling
#      inside the network lines up exactly, then crop the padding back off.
# This replaces the previous behaviour of forcing every image into a fixed
# 384x1248 landscape canvas, which silently stretched/squashed any image
# whose native aspect ratio differed (e.g. portrait phone photos).
OFRS_MAX_SIDE = 1024
OFRS_NET_STRIDE = 8  # matches the 3 stride-2 DownBlocks in ofrs/model.py

# Spatially-weighted cross-entropy (paper: heavier at road edges & far from
# image centre). base weight + edge bonus within `edge_px`, scaled by radial dist.
OFRS_LOSS_EDGE_PX = 10
OFRS_LOSS_EDGE_BONUS = 2.0      # extra weight added on road-boundary pixels
OFRS_LOSS_CENTER_GAMMA = 1.0    # radial term: w *= (1 + gamma * dist_from_centre[0..1])

# Label smoothing on the one-hot semantic INPUT (paper: alpha in [0.1, 0.2]).
OFRS_LABEL_SMOOTH = 0.1

# Train/val split of annotated samples (deterministic, seeded).
OFRS_VAL_FRACTION = 0.2
OFRS_SPLIT_SEED = 1234

# Training hyper-parameters
OFRS_EPOCHS = 80
OFRS_BATCH_SIZE = 4
OFRS_LR = 1e-3

# --------------------------------------------------------------------------- #
# Geometry stream (GroundNet-derived) -- the "ground manifold" signal
# --------------------------------------------------------------------------- #
# Reformulation: road completion as message passing on a ground manifold.
# Mask2Former says WHAT each visible pixel is; this stream says WHERE the road
# surface is in 3D, so the context module can propagate semantic evidence
# preferentially between pixels that share a ground surface.
#
# We implement GroundNet's DEPTH stream (Man et al., ICCV-W 2019, Eq. 1-5):
# monocular depth -> unproject ground pixels -> RANSAC plane fit -> normal n.
# GroundNet's second (surface-normal) stream needs Marigold/`diffusers`, which
# is not installed; it is optional here (see src/geometry.py). Per the paper's
# own ablation (Table 4) the depth+RANSAC stream is the stronger of the two
# individually (2.92 deg vs 6.73 deg on KITTI 5/0.05), so depth-only is a
# defensible default -- but it IS a reduction from the full paper method.
GEOMETRY_DIR = PROC_DIR / "geometry"

# Monocular metric depth model (outdoor). Small = fast enough to precompute.
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"

# Plane fitting (paper Sec. 5.4: "valid depth threshold ... 30m").
GEOM_MAX_DEPTH_M = 30.0
GEOM_RANSAC_THRESH_M = 0.05
GEOM_RANSAC_ITERS = 500
GEOM_MIN_GROUND_PX = 500      # below this, the plane fit is not trustworthy

# Camera intrinsics. Every 3D quantity is back-projected through K, so a wrong
# focal length shears the whole ground manifold. Sources are tried in order and
# the one actually used is recorded in the geometry cache (see src/calibration.py):
#   kitti_calib -- exact, per-image (KITTI calib/*.txt, P2)
#   geocalib    -- GeoCalib (ECCV 2024): learned focal length from one image.
#                  Optional dependency; skipped automatically if not installed.
#   exif        -- FocalLengthIn35mmFilm etc. Free, but ffmpeg strips it when
#                  extracting video frames, so footage frames rarely have it.
#   fov_prior   -- blind assumption below. Measurably poor: on our own footage
#                  it underestimates the focal length by 35-45%.
INTRINSICS_PRIORITY = ["kitti_calib", "geocalib", "exif", "fov_prior"]
GEOCALIB_WEIGHTS = "pinhole"     # or "distorted" for lens distortion
GEOM_FALLBACK_HFOV_DEG = 65.0

# Clamp on the ground-footprint distance (metres). Rays near the horizon
# intersect the plane arbitrarily far away; without a cap the pairwise
# distance term blows up and swamps the softmax.
GEOM_MAX_GROUND_DIST_M = 60.0

# --------------------------------------------------------------------------- #
# Multimodal (geometry-guided) context module
# --------------------------------------------------------------------------- #
# Tier 1: ground-weighted global pooling. Pixels are weighted by
#   exp(-|h| / tau_h), h = signed height above the fitted ground plane, so the
#   pooled "global context" summarises the road surface instead of averaging
#   in cars, buildings and sky.
# Tier 2: geometry-gated non-local attention. Query i attends to key j with
#   logit  <q_i,k_j>/sqrt(d)  -  alpha * ( ||G_i - G_j||^2 / tau_d + |h_j| / tau_h )
#   where G is each pixel's GROUND FOOTPRINT (where its viewing ray meets the
#   plane). Using G rather than the pixel's own 3D point is what makes this
#   work for amodal completion: a pixel on a car roof is far off-manifold, but
#   its footprint is exactly the road location it occludes, so it still gathers
#   evidence from the right neighbourhood.
OFRS_USE_GEOMETRY = True       # master switch; False = original semantic-only module
OFRS_ATTN_DIM = 64             # query/key dim for the non-local block
OFRS_ATTN_KV_STRIDE = 4        # pool keys/values by this factor (cost: N x N/stride^2)
OFRS_TAU_H_INIT = 0.5          # metres; manifold-membership softness
OFRS_TAU_D_INIT = 25.0         # metres^2; ground-plane neighbourhood (~5 m)

# --------------------------------------------------------------------------- #
# Bird's-eye view & parking disturbance (src/bev.py, src/disturbance.py)
# --------------------------------------------------------------------------- #
# The BEV needs no new estimation: the same (K, n) already cached by Step 7
# defines an exact ground-plane homography, and that homography is the analytic
# inverse of geometry.derive_ground_fields' ray-plane intersection. So the BEV
# raster is guaranteed consistent with the geometry OFRSNet was trained on.
#
# Grid convention: X is lateral (camera at X=0, +X right), Z is forward. BEV row
# v=0 is the FAR edge (z_max), so the raster reads like a map with the camera at
# the bottom.
BEV_PPM = 20.0                    # BEV pixels per metre -> 5 cm cells
BEV_RANGE_X_M = (-10.0, 10.0)     # lateral extent
BEV_RANGE_Z_M = (0.5, 40.0)       # forward extent (z_min > 0: the camera is not on the plane)

# Metric scale prior. The fitted plane gives camera height directly as 1/||n||,
# but monocular metric depth overestimates it (~2.71 m implied vs ~1.65 m true on
# KITTI -- see the README caveats), which would inflate any area by that factor
# SQUARED. Rescaling n to a known camera height corrects the whole metric world
# and makes BEV_RANGE_* mean actual metres. Areas are always reported both ways.
#
# This is the FALLBACK when automatic estimation (below) finds no usable vehicle
# in the scene -- it is no longer the primary source of the height assumption.
BEV_CAMERA_HEIGHT_M = 1.65
# Reject plainly broken plane fits rather than rasterising garbage.
BEV_MIN_CAMERA_HEIGHT_M = 0.3
BEV_MAX_CAMERA_HEIGHT_M = 10.0

# --------------------------------------------------------------------------- #
# Automatic camera-height estimation from detected vehicles (src/instances.py)
# --------------------------------------------------------------------------- #
# The single fixed BEV_CAMERA_HEIGHT_M assumption above is badly wrong for
# unknown cameras -- on our own footage the RANSAC-implied height ranges 2.5-7.7m
# (median 6.9m), nothing like 1.65m, so forcing every image through that one
# constant just replaces one error with a different, silent one.
#
# Instead: a detected car's ROOF sits a roughly known height above the ground
# plane, regardless of the vehicle's yaw relative to the camera (unlike its
# apparent WIDTH, which is view-angle-dependent and unusable for this). The
# geometry module already computes height-above-plane per pixel (`h` in
# derive_ground_fields) at the SAME biased scale as the RANSAC-implied camera
# height, so comparing a car's measured roof height to this prior gives a
# per-scene correction factor with no new estimation machinery.
#
# Validated against KITTI (true camera height ~1.65m, independently documented,
# not derived from this pipeline): with these thresholds, single-frame estimates
# were typically within +-12% of ground truth (occasionally worse; not every
# frame has a qualifying car), pooling across several cars/frames converged much
# tighter (median correction factor 0.97, i.e. within 3% of the target on
# average). This is a real improvement over the fixed constant for unknown
# cameras, NOT a precise measurement -- report the sample count/spread alongside
# any estimate rather than presenting it as exact.
VEHICLE_ROOF_HEIGHT_PRIOR_M = {
    "car": 1.5,      # sedan/hatchback roof height above ground; Cityscapes' "car"
                      # also catches some taller SUVs, which is the main source of
                      # residual per-vehicle variance. Deliberately not extended to
                      # truck/bus/motorcycle yet -- their real height variance is
                      # much wider, and a bad prior there would hurt more than help.
}
SCALE_EST_MIN_SCORE = 0.85        # instance-segmentation confidence floor
SCALE_EST_MIN_PIXELS = 3000       # rejects small/distant/partially-occluded cars
SCALE_EST_MIN_MASK_ROWS = 20      # vertical extent floor -- a sliver has no reliable roofline
SCALE_EST_TOP_FRAC = 0.08         # fraction of the mask's own height treated as "roofline"
SCALE_EST_MIN_TOP_ROWS = 6        # ...but never fewer than this many rows (small cars)
SCALE_EST_MIN_ROOF_HEIGHT_M = 0.3  # sanity floor against near-zero/degenerate measurements

# --------------------------------------------------------------------------- #
# Ground-contact footprint attribution (src/bev.py, src/disturbance.py)
# --------------------------------------------------------------------------- #
# Warping a vehicle's ENTIRE mask through the ground-plane homography is only
# valid for pixels that truly lie on the plane; roofline/hood/windshield pixels
# do not (see BEV_CAMERA_HEIGHT_M above -- the same "elevated pixels don't
# project where you'd expect" issue, here affecting WHICH vehicle a shadow cell
# is attributed to rather than the metric scale). Only a vehicle's BOTTOM
# CONTOUR is actually near the ground, so attribution seeds are built from that
# contour alone: projected to BEV, then rasterised into a small band -- not the
# vehicle's full projected silhouette.
FOOTPRINT_BAND_WIDTH_M = 0.4        # real-world width of the rasterised contact
                                     # band -- a tire's contact zone plus a small
                                     # margin for contour/depth noise; deliberately
                                     # far short of the vehicle's own body length
FOOTPRINT_MAX_ATTRIBUTION_DIST_M = 35.0  # safety net, not the primary mechanism --
                                     # that's the band's shape. Generous enough that
                                     # a legitimate single-vehicle shadow within the
                                     # measurable range (25-35m on our own footage)
                                     # is never truncated by it; exists only to stop
                                     # a degenerate/erroneous band from claiming
                                     # implausibly distant cells.
FOOTPRINT_MIN_COVERAGE = 0.5        # below this fraction of non-truncated contour
                                     # columns, a vehicle's footprint is likely
                                     # incomplete (its base is mostly cropped by
                                     # the frame) -- surfaced as a warning.

# --------------------------------------------------------------------------- #
# Road-width disturbance (src/bev.py, src/disturbance.py)
# --------------------------------------------------------------------------- #
# Area alone conflates two very different situations: a small occluded patch on
# a wide road, and a full-width blockage on a narrow one -- only the second one
# actually stops other traffic passing. A BEV row is a fixed-distance
# cross-section of the road, so reducing amodal_bev/occluded_bev row-by-row
# gives "what fraction of the road's WIDTH is blocked here" for free, no new
# segmentation or geometry needed. Validated on real KITTI and footage scenes
# (see the "Road-Width Disturbance Metric" design note): per-vehicle figures of
# 9-38%, a worst-case combined total of 69%, nothing pathological or over 100%.
#
# Rows narrower than this are excluded from width stats entirely -- the BEV
# wedge narrows near the camera (see bev.bev_validity), so a very thin span near
# the edge of measurement is boundary noise, not a real usable road width.
WIDTH_MIN_ROAD_SPAN_M = 1.0

# Measurement reliability cap. Ground resolution decays with roughly the CUBE of
# range, so past some distance one camera pixel is responsible for a large patch
# of road and a two-pixel mask disagreement becomes tens of square metres. Cells
# whose source pixel covers more ground than this are in frame but NOT measurable,
# and are excluded from every area. 0.02 m^2 ~ a 14 cm square, which on our own
# footage (fx~690, 720x1280) keeps everything inside ~25 m.
BEV_MAX_M2_PER_PIXEL = 0.02

# Per-vehicle attribution needs INSTANCES, not just the vehicle class. Same
# Mask2Former family as MODEL_ID, so no new dependency -- but this checkpoint is
# optional: without it, src/instances.py falls back to connected components of
# the occluder mask (the abstraction common.occluder_blob_mask already uses).
INSTANCE_MODEL_ID = "facebook/mask2former-swin-large-cityscapes-instance"
INSTANCE_MIN_SCORE = 0.5
# Occluder-support pixels no detection claimed are kept as their own blob
# instance above this size, and left unattributed below it. Slivers along an
# instance boundary are disagreement noise, not vehicles.
INSTANCE_MIN_BLOB_PX = 200

# Cityscapes "thing" classes we offer as selectable parked vehicles. Matched by
# substring against id2label, exactly like FOREGROUND_KEYWORDS above.
VEHICLE_INSTANCE_KEYWORDS = [
    "car",
    "truck",
    "bus",
    "caravan",
    "trailer",
    "motorcycle",
    "bicycle",
    "train",          # Cityscapes' on-rails class
]
