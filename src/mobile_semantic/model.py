"""The mobile semantic student -- iOS migration Phase 2 (see the migration
plan's Section 1: "Mobile segmentation model strategy").

Distills the Mask2Former/Mapillary-Vistas teacher (826MB, Swin-Large,
research-scale -- see src/s3_detect_foreground.py, src/s5_export_semantics.py)
into a small mobile-native backbone that emits the SAME 11-class OFRS output
directly, at a fraction of the size. Nothing downstream cares how that array
was produced (predict.py/disturbance.py only ever consume a dense (H,W)
argmax over config.OFRS_CLASSES) -- this model is a drop-in replacement for
the teacher's output, not a new pipeline stage.

Architecture: torchvision's LR-ASPP head over a MobileNetV3-Large backbone --
exactly the "lightweight backbone + Lite R-ASPP decoder" combination the
migration plan recommends, already implemented and Core-ML-exportable via
coremltools/Ultralytics-style tooling, so no custom architecture or new
dependency is needed. ~3.2M params (OFRSNet, for comparison, is ~2.1M).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config  # noqa: E402

# ImageNet normalization -- required by the pretrained MobileNetV3 backbone
# (weights_backbone="DEFAULT" below). Baked into the model's forward so
# callers deal in plain 0..1 RGB tensors everywhere else in this package,
# the same convention src/common.py's read_rgb uses.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class MobileSemanticNet(nn.Module):
    """RGB (N, 3, H, W) in [0, 1] -> (N, OFRS_NUM_CLASSES, H, W) logits."""

    def __init__(self, num_classes: int = config.OFRS_NUM_CLASSES,
                pretrained_backbone: bool = True):
        super().__init__()
        self.net = lraspp_mobilenet_v3_large(
            weights=None, num_classes=num_classes,
            weights_backbone="DEFAULT" if pretrained_backbone else None,
        )
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = (rgb - self.mean) / self.std
        return self.net(x)["out"]
