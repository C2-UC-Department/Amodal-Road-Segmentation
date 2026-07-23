"""Spatially-weighted cross-entropy (CE-SW) — the OFRSNet training loss.

Paper: Loss(y, p) = sum_ij -w(i,j) [ y_ij log p_ij + (1-y_ij) log(1-p_ij) ],
with w(i,j) higher on road edges and increasing with distance from the image
centre. The weight map w is precomputed per sample in the dataset
(`build_weight_map`); this module just applies it to per-pixel CE.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatiallyWeightedCELoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                weight: torch.Tensor) -> torch.Tensor:
        # logits (N,2,H,W); target (N,H,W) long; weight (N,H,W) float
        ce = F.cross_entropy(logits, target, reduction="none")  # (N,H,W)
        wce = ce * weight
        return wce.sum() / (weight.sum() + 1e-6)
