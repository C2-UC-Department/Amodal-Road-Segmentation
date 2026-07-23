"""OFRS PyTorch Dataset.

Yields, per annotated KITTI image:
  * input   : one-hot OFRS semantic map, shape (C=11, H, W), float32 in [0,1]
              (with optional label smoothing, matching the paper)
  * target  : occlusion-free road labels, shape (H, W), int64 in {0,1}
              (0 = non-road, 1 = road)
  * weight  : spatial CE weight map, shape (H, W), float32
              (heavier on road edges and away from the image centre)

A sample is included only if BOTH the semantic map (Step 5) and the
hand-annotated amodal mask (Step 4) exist. Everything is resized to
config.OFRS_INPUT_SIZE (semantic/target with NEAREST to preserve labels).
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


def annotated_bases(split: str = config.SPLIT) -> list[str]:
    """Bases that have both a semantic map and an amodal (target) mask."""
    out = []
    for s in common.iter_samples(split):
        if (config.SEMANTIC_DIR / f"{s.base}.png").exists() and \
           (config.AMODAL_DIR / f"{s.base}.png").exists():
            out.append(s.base)
    return out


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
        self.size = config.OFRS_INPUT_SIZE  # (H, W)
        self.C = config.OFRS_NUM_CLASSES

    def __len__(self) -> int:
        return len(self.bases)

    def _resize_labels(self, arr: np.ndarray) -> np.ndarray:
        h, w = self.size
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)

    def __getitem__(self, i: int):
        base = self.bases[i]
        sem = np.asarray(cv2.imread(str(config.SEMANTIC_DIR / f"{base}.png"),
                                    cv2.IMREAD_GRAYSCALE))
        amodal = common.read_mask(config.AMODAL_DIR / f"{base}.png")

        sem = self._resize_labels(sem).astype(np.int64)
        amodal = self._resize_labels(amodal.astype(np.uint8)).astype(bool)

        # One-hot the semantic input -> (C, H, W).
        sem = np.clip(sem, 0, self.C - 1)
        onehot = np.eye(self.C, dtype=np.float32)[sem].transpose(2, 0, 1)

        # Label smoothing on the input (paper: alpha in [0.1, 0.2]).
        if self.train and config.OFRS_LABEL_SMOOTH > 0:
            a = config.OFRS_LABEL_SMOOTH
            onehot = onehot * (1.0 - a) + a / self.C

        weight = build_weight_map(amodal)

        return (
            torch.from_numpy(onehot),                       # (C, H, W) float32
            torch.from_numpy(amodal.astype(np.int64)),      # (H, W) int64
            torch.from_numpy(weight),                       # (H, W) float32
            base,
        )
