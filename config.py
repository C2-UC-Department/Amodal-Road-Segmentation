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

# OFRSNet input/target spatial size (paper uses 384x1248). H, W.
OFRS_INPUT_SIZE = (384, 1248)

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
