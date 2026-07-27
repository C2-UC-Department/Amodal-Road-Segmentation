"""OFRS PyTorch Dataset.

Yields, per annotated image (KITTI or footage):
  * input   : one-hot OFRS semantic map, shape (C=11, H, W), float32 in [0,1]
              (with optional label smoothing, matching the paper)
  * target  : occlusion-free road labels, shape (H, W), int64 in {0,1}
              (0 = non-road, 1 = road)
  * weight  : spatial CE weight map, shape (H, W), float32
              (heavier on road edges and away from the image centre)

A sample is included only if BOTH the semantic map (Step 5) and the
hand-annotated amodal mask (Step 4) exist.

Unlike the old fixed-384x1248 resize (which stretched/squashed any image
whose native aspect ratio differed, e.g. portrait phone photos), each sample
here keeps its own native resolution -- only downscaled (uniformly, aspect
preserved) if it exceeds config.OFRS_MAX_SIDE. Since samples can therefore
have different shapes, batching requires the custom `ofrs_collate` below,
which pads every sample in a batch up to the batch's own max (H, W) (rounded
up to config.OFRS_NET_STRIDE) using the 'unlabeled' class for the semantic
input and weight=0 for the loss weight -- so collate-padding never affects
gradients or evaluation metrics (see train.py, which filters by `weight > 0`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config  # noqa: E402
from src import common  # noqa: E402


def annotated_bases() -> list[str]:
    """Bases with both a semantic map and a hand-corrected amodal mask.

    Pooled across ALL sources (KITTI + footage) -- both write into the same
    flat SEMANTIC_DIR/AMODAL_DIR namespace with source-distinct base names
    (KITTI: um_/umm_/uu_..., footage: IMG_.../ScreenRecording_...), so a plain
    filename intersection is a correct union of everything annotated so far.
    """
    sem_bases = {p.stem for p in config.SEMANTIC_DIR.glob("*.png")}
    amodal_bases = {p.stem for p in config.AMODAL_DIR.glob("*.png")}
    return sorted(sem_bases & amodal_bases)


def split_bases(bases: list[str]) -> tuple[list[str], list[str]]:
    """Deterministic train/val split (seeded)."""
    rng = np.random.default_rng(config.OFRS_SPLIT_SEED)
    order = np.array(bases)
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * config.OFRS_VAL_FRACTION)))
    val = list(order[:n_val])
    train = list(order[n_val:])
    return train, val


def build_weight_map(road: np.ndarray) -> np.ndarray:
    """Spatially-dependent CE weight (paper Eq. for CE-SW).

    road : (H, W) bool  — the amodal road target.
    """
    h, w = road.shape
    weight = np.ones((h, w), np.float32)

    # (a) Emphasise the road boundary: morphological gradient, then dilate to
    #     an `edge_px`-wide band.
    r = road.astype(np.uint8)
    grad = cv2.morphologyEx(r, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    k = 2 * config.OFRS_LOSS_EDGE_PX + 1
    band = cv2.dilate(grad, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    weight += config.OFRS_LOSS_EDGE_BONUS * band.astype(np.float32)

    # (b) Radial term: increase weight with distance from the image centre.
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dist = np.sqrt(((ys - cy) / cy) ** 2 + ((xs - cx) / cx) ** 2) / np.sqrt(2.0)
    weight *= (1.0 + config.OFRS_LOSS_CENTER_GAMMA * dist)
    return weight


class OFRSDataset(Dataset):
    def __init__(self, bases: list[str], split: str = config.SPLIT,
                 train: bool = True):
        self.bases = bases
        self.split = split
        self.train = train
        self.C = config.OFRS_NUM_CLASSES

    def __len__(self) -> int:
        return len(self.bases)

    def __getitem__(self, i: int):
        base = self.bases[i]
        sem = np.asarray(cv2.imread(str(config.SEMANTIC_DIR / f"{base}.png"),
                                    cv2.IMREAD_GRAYSCALE))
        amodal = common.read_mask(config.AMODAL_DIR / f"{base}.png").astype(np.uint8)

        # Cap to OFRS_MAX_SIDE with a UNIFORM scale (aspect preserved) -- only
        # activates for oversized images; most samples pass through untouched
        # at their own native resolution. No fixed-shape resize happens here.
        sem, scale = common.fit_within_max_side(sem, config.OFRS_MAX_SIDE)
        if scale < 1.0:
            h, w = sem.shape
            amodal = cv2.resize(amodal, (w, h), interpolation=cv2.INTER_NEAREST)

        sem = np.clip(sem, 0, self.C - 1).astype(np.int64)
        amodal = amodal.astype(bool)

        # One-hot the semantic input -> (C, H, W), at this sample's own shape.
        onehot = np.eye(self.C, dtype=np.float32)[sem].transpose(2, 0, 1)

        # Label smoothing on the input (paper: alpha in [0.1, 0.2]).
        if self.train and config.OFRS_LABEL_SMOOTH > 0:
            a = config.OFRS_LABEL_SMOOTH
            onehot = onehot * (1.0 - a) + a / self.C

        weight = build_weight_map(amodal)

        return (
            torch.from_numpy(onehot),                       # (C, h, w) float32
            torch.from_numpy(amodal.astype(np.int64)),      # (h, w) int64
            torch.from_numpy(weight),                       # (h, w) float32
            base,
        )


def ofrs_collate(batch):
    """Pad every sample in a batch to the batch's own max (H, W), rounded up
    to config.OFRS_NET_STRIDE -- NOT to any fixed global shape. Samples keep
    their individual native aspect ratio right up until this point; only
    batching forces a shared canvas, and only as large as that particular
    batch actually needs.

    Padding uses the 'unlabeled' class for the semantic input and weight=0
    for the loss weight, so collate-padding never contributes to gradients.
    Returns an extra `valid` mask (weight > 0) for masking evaluation metrics.
    """
    onehots, targets, weights, bases = zip(*batch)
    n, c = len(batch), onehots[0].shape[0]
    s = config.OFRS_NET_STRIDE

    max_h = max(t.shape[0] for t in targets)
    max_w = max(t.shape[1] for t in targets)
    max_h = -(-max_h // s) * s
    max_w = -(-max_w // s) * s

    batch_onehot = torch.zeros(n, c, max_h, max_w, dtype=torch.float32)
    batch_onehot[:, config.OFRS_UNLABELED_ID, :, :] = 1.0
    batch_target = torch.zeros(n, max_h, max_w, dtype=torch.int64)
    batch_weight = torch.zeros(n, max_h, max_w, dtype=torch.float32)

    for i in range(n):
        h, w = targets[i].shape
        batch_onehot[i, :, :h, :w] = onehots[i]
        batch_target[i, :h, :w] = targets[i]
        batch_weight[i, :h, :w] = weights[i]

    valid = batch_weight > 0
    return batch_onehot, batch_target, batch_weight, valid, list(bases)
