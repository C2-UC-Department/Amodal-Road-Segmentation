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

# Camera intrinsics. KITTI ships real calibration (calib/*.txt, P2 = the
# rectified colour camera that produces image_2) and we parse it. Arbitrary
# phone photos have NO intrinsics, so we fall back to an assumed horizontal
# FOV -- a standard approximation, but a genuine source of error the paper
# never faced (it evaluated only on calibrated KITTI/ApolloScape).
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
