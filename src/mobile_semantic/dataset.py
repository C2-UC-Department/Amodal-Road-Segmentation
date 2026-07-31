"""PyTorch Dataset for the mobile semantic student (src/mobile_semantic/model.py).

Yields, per sample with a teacher label (any of kitti/footage/realworld --
see src/s5_export_semantics.py, which must be run first for each source):
  * rgb        : (3, H, W) float32 in [0, 1]
  * label      : (H, W) int64 in 0..OFRS_NUM_CLASSES-1 -- the TEACHER's own
                 OFRS-11 map (data/processed/semantic_ofrs/<base>.png), the
                 distillation target.
  * road_gt    : (H, W) float32 in {0, 1} -- REAL hand-labelled road mask,
                 decoded from KITTI's own gt_image_2 (src/common.py's
                 decode_kitti_road_gt). Only KITTI ships this; everywhere
                 else it's all -1 (see `has_road_gt`), meaning "no real
                 supervision available, don't use this sample for the
                 auxiliary road loss."
  * has_road_gt: scalar bool

Mirrors src/ofrs/dataset.py's shape (uniform-scale resize capped at a max
side, custom collate padding samples in a batch to a shared canvas) rather
than reinventing that pattern.
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


class LabeledSample:
    __slots__ = ("base", "image_path", "gt_path")

    def __init__(self, base: str, image_path: Path, gt_path: Path | None):
        self.base = base
        self.image_path = image_path
        self.gt_path = gt_path


def all_labeled_samples(sources: tuple[str, ...] = ("kitti", "footage", "realworld")
                        ) -> list[LabeledSample]:
    """Every sample across `sources` that already has a teacher OFRS-11 label
    (i.e. src/s5_export_semantics.py has been run for it). Pooling across
    sources mirrors src/ofrs/dataset.py::annotated_bases -- each source's
    base names are namespace-distinct (KITTI: um_/umm_/uu_..., footage/
    realworld: original filenames), so a flat pool can't collide.
    """
    sem_bases = {p.stem for p in config.SEMANTIC_DIR.glob("*.png")}
    samples: list[LabeledSample] = []
    for source in sources:
        for s in common.iter_source_samples(source):
            if s.base in sem_bases:
                samples.append(LabeledSample(s.base, s.image_path, s.gt_path))
    return samples


def split_samples(samples: list[LabeledSample]) -> tuple[list[LabeledSample], list[LabeledSample]]:
    """Deterministic train/val split (seeded), same recipe as src/ofrs/dataset.py."""
    rng = np.random.default_rng(config.MOBILE_SEM_SPLIT_SEED)
    order = np.array(samples, dtype=object)
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * config.MOBILE_SEM_VAL_FRACTION)))
    val = list(order[:n_val])
    train = list(order[n_val:])
    return train, val


class MobileSemanticDataset(Dataset):
    def __init__(self, samples: list[LabeledSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        rgb = common.read_rgb(s.image_path)
        label = np.asarray(cv2.imread(str(config.SEMANTIC_DIR / f"{s.base}.png"),
                                      cv2.IMREAD_GRAYSCALE))

        # Uniform-scale cap (aspect preserved), same recipe as
        # src/ofrs/dataset.py -- the label is the authority on shape (it's
        # what fit_within_max_side was written for); rgb just follows it.
        label, scale = common.fit_within_max_side(label, config.MOBILE_SEM_MAX_SIDE)
        h, w = label.shape
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

        road_gt = np.full((h, w), -1.0, np.float32)
        has_road_gt = False
        if s.gt_path is not None:
            real_road = common.decode_kitti_road_gt(s.gt_path)
            if real_road.shape != (h, w):
                real_road = cv2.resize(real_road.astype(np.uint8), (w, h),
                                       interpolation=cv2.INTER_NEAREST) > 0
            road_gt = real_road.astype(np.float32)
            has_road_gt = True

        rgb_t = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1) / 255.0)
        label_t = torch.from_numpy(np.clip(label, 0, config.OFRS_NUM_CLASSES - 1).astype(np.int64))
        road_gt_t = torch.from_numpy(road_gt)
        return rgb_t, label_t, road_gt_t, torch.tensor(has_road_gt), s.base


def mobile_semantic_collate(batch):
    """Pad every sample in a batch to the batch's own max (H, W), rounded up
    to config.MOBILE_SEM_NET_STRIDE -- same rationale as src/ofrs/dataset.py's
    ofrs_collate: samples keep native aspect ratio right up to batching, only
    padded as much as that batch actually needs. Padded pixels get label
    OFRS_UNLABELED_ID and are excluded from the loss via `valid`.
    """
    rgbs, labels, road_gts, has_road_gts, bases = zip(*batch)
    n = len(batch)
    s = config.MOBILE_SEM_NET_STRIDE

    max_h = max(t.shape[0] for t in labels)
    max_w = max(t.shape[1] for t in labels)
    max_h = -(-max_h // s) * s
    max_w = -(-max_w // s) * s

    batch_rgb = torch.zeros(n, 3, max_h, max_w, dtype=torch.float32)
    batch_label = torch.full((n, max_h, max_w), config.OFRS_UNLABELED_ID, dtype=torch.int64)
    batch_road_gt = torch.full((n, max_h, max_w), -1.0, dtype=torch.float32)
    batch_valid = torch.zeros(n, max_h, max_w, dtype=torch.bool)

    for i in range(n):
        h, w = labels[i].shape
        batch_rgb[i, :, :h, :w] = rgbs[i]
        batch_label[i, :h, :w] = labels[i]
        batch_road_gt[i, :h, :w] = road_gts[i]
        batch_valid[i, :h, :w] = True

    return (batch_rgb, batch_label, batch_road_gt,
            torch.stack(has_road_gts), batch_valid, list(bases))
