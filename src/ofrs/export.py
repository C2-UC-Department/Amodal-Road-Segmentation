"""Export-ready OFRSNet for Core ML conversion (iOS migration, Phase 1).

Deliberately a SEPARATE module from `model.py`, not a modification of it:
`model.py` is exercised by training (`train.py`), the dataset loader
(`dataset.py`), and inference (`src/predict.py`, `src/disturbance.py`), all
covered by the existing test suite -- changing its op graph to suit Core ML
would risk that working pipeline for the sake of a target it doesn't need to
satisfy. `OFRSNetExport` is built FROM a trained `OFRSNet` (weights
transplanted via `from_trained`, never retrained) and differs only in HOW
equivalent results are computed, addressing three specific Core ML conversion
risks identified for `MultimodalContextModule` and `UpBlock`:

  1. `torch.einsum` -> `torch.bmm`+`transpose`. coremltools' einsum support is
     pattern-limited; verified numerically identical for all three sites
     (see the einsum-to-bmm spike this module's history is built on).
  2. `geo=None` branch removed -- `MultimodalContextModuleExport.forward`
     always requires a populated geo dict. Proven exactly (not just
     approximately) equivalent to the `geo=None` path when the dict is
     all-invalid (G=0, h=0, gvalid=all-False, valid=all-False) --
     see tests/test_ofrsnet_coreml_readiness.py. Core ML traces one
     control-flow path per graph; this removes the need for two.
  3. `F.interpolate(size=skip.shape[-2:])` -> `F.interpolate(scale_factor=2)`.
     Bit-identical given `OFRS_NET_STRIDE=8`-padded input, since DownBlock
     halves exactly at every stage (verified: no rounding is possible when
     H,W are multiples of 8). Removes a runtime-shape dependency the Core ML
     trace would otherwise bake in as a fixed literal anyway -- `scale_factor`
     makes that explicit instead of incidental. The final safety-resize in
     `OFRSNet.forward` (a Python `if` on `.shape`, which a single Core ML
     trace cannot represent both branches of) becomes provably unreachable
     once this holds, so it is simply not included here.

`ceil_mode=True` pooling (config.OFRS_ATTN_KV_STRIDE) is INTENTIONALLY left
unchanged from `model.py` pending the Core ML conversion spike -- an earlier
attempt to replace it with `adaptive_avg_pool2d`/`adaptive_max_pool2d` was
verified numerically to NOT reproduce ceil_mode's values except when the
pooled dimension is evenly divisible by the stride (spike: divisible cases
matched exactly, non-divisible cases differed by up to ~1.7 in raw logit
units -- not a rounding artifact). If the Core ML conversion spike finds
ceil_mode itself is the problem (not just a caution from the migration plan),
the fix is to constrain exported input resolutions to multiples of
`OFRS_NET_STRIDE * OFRS_ATTN_KV_STRIDE` (32, not just 8) so the divisible case
always holds, THEN swap to adaptive pooling -- not a blind swap.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.ofrs.model import (  # noqa: E402
    DownBlock, MultimodalContextModule, OFRSNet, UpBlock, conv_bn_relu,
)


class MultimodalContextModuleExport(MultimodalContextModule):
    """Same parameters as `MultimodalContextModule` (subclasses it -- no new
    layers, so `load_state_dict` transplant is a direct key-for-key match).
    Only `forward` differs, per this module's docstring.
    """

    def forward(self, x, geo):                       # geo is now required, not Optional
        n, c, hh, ww = x.shape
        g_map = self._resize_geo(geo["G"], (hh, ww))              # (N,3,h,w)
        h_map = self._resize_geo(geo["h"], (hh, ww))               # (N,1,h,w)
        gvalid = self._resize_geo(geo["gvalid"].float(), (hh, ww)) > 0.5
        sample_valid = geo["valid"].to(torch.bool)

        tau_h = self.log_tau_h.exp().clamp(min=1e-3)
        tau_d = self.log_tau_d.exp().clamp(min=1e-3)

        # ---------------- Tier 1: ground-weighted pooling ----------------
        w = torch.exp(-h_map.abs() / tau_h) * gvalid.float()
        uniform = torch.ones_like(w)
        use = sample_valid.view(n, 1, 1, 1)
        w = torch.where(use, w, uniform)
        wsum = w.sum(dim=(2, 3), keepdim=True)
        degenerate = wsum < 1e-6
        w = torch.where(degenerate, uniform, w)
        wsum = torch.where(degenerate, uniform.sum(dim=(2, 3), keepdim=True), wsum)
        pooled = (x * w).sum(dim=(2, 3), keepdim=True) / wsum

        g = self.fc(pooled)
        out = x * self.gate(g) + g
        out = self.fuse(out)

        # ------------- Tier 2: geometry-gated non-local attention -------------
        s = self.kv_stride
        q = self.q(out).flatten(2)                                   # (N, d, HW)
        x_kv = F.avg_pool2d(out, s, ceil_mode=True) if s > 1 else out
        k = self.k(x_kv).flatten(2)                                  # (N, d, M)
        v = self.v(x_kv).flatten(2)                                  # (N, C, M)

        # bmm replaces einsum("ndq,ndk->nqk", q, k) -- verified identical.
        logits = torch.bmm(q.transpose(1, 2), k) / math.sqrt(self.attn_dim)

        gq = g_map.flatten(2)                                        # (N,3,HW)
        gk = (F.avg_pool2d(g_map, s, ceil_mode=True) if s > 1 else g_map).flatten(2)
        hk = (F.avg_pool2d(h_map, s, ceil_mode=True) if s > 1 else h_map).flatten(2)
        vk = (F.max_pool2d(gvalid.float(), s, ceil_mode=True)
              if s > 1 else gvalid.float()).flatten(2)                # (N,1,M)

        # bmm replaces the second einsum("ndq,ndk->nqk", gq, gk).
        d2 = (gq.pow(2).sum(1).unsqueeze(2)
              + gk.pow(2).sum(1).unsqueeze(1)
              - 2.0 * torch.bmm(gq.transpose(1, 2), gk)).clamp(min=0)

        geo_bias = -(d2 / tau_d + hk.abs().squeeze(1).unsqueeze(1) / tau_h)
        geo_bias = self.alpha * geo_bias

        key_ok = vk.squeeze(1).unsqueeze(1) > 0.5                    # (N,1,M)
        geo_bias = geo_bias.masked_fill(~key_ok, -1e4)

        has_key = key_ok.any(dim=2, keepdim=True)                    # (N,1,1)
        use_geo = sample_valid.view(n, 1, 1) & has_key
        logits = logits + torch.where(use_geo, geo_bias, torch.zeros_like(geo_bias))

        attn = logits.softmax(dim=-1)
        # bmm replaces einsum("nqk,nck->ncq", attn, v).
        msg = torch.bmm(v, attn.transpose(1, 2)).reshape(n, c, hh, ww)
        return out + self.gamma * self.proj(msg)


class OFRSNetExport(nn.Module):
    """Core ML export target. Build via `from_trained`, never `__init__` +
    fresh training -- this class exists purely to compute the SAME function
    as a trained `OFRSNet(use_geometry=True)` through an export-friendlier op
    graph, not to be trained itself.
    """

    def __init__(self, in_channels=11, num_classes=2, base=32):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.stem = nn.Sequential(conv_bn_relu(in_channels, c1),
                                  conv_bn_relu(c1, c1))
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)
        self.context = MultimodalContextModuleExport(c4)
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, num_classes, 1)

    @staticmethod
    def _up(block: UpBlock, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """UpBlock.forward, but scale_factor=2 instead of size=skip.shape --
        see this module's docstring for why that's bit-identical here."""
        x = block.reduce(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return block.fuse(torch.cat([x, skip], dim=1))

    def forward(self, x: torch.Tensor, geo: dict) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        g = self.context(s3, geo)
        d3 = self._up(self.up3, g, s2)
        d2 = self._up(self.up2, d3, s1)
        d1 = self._up(self.up1, d2, s0)
        return self.head(d1)          # no final safety resize -- see docstring

    @classmethod
    def from_trained(cls, model: OFRSNet) -> "OFRSNetExport":
        """Transplant a trained OFRSNet's weights, unchanged, into the
        export-friendly graph. Requires use_geometry=True (the shipped
        checkpoint) since `GlobalContextModule` has no export variant here --
        it has no einsum/geo=None/dynamic-shape issues to begin with, so it
        Core ML-converts directly from `model.py` with no rewrite needed.
        """
        if not model.use_geometry:
            raise ValueError(
                "OFRSNetExport only covers the MultimodalContextModule "
                "(use_geometry=True) path; a use_geometry=False checkpoint "
                "converts directly from model.py without this wrapper.")
        stem_conv = model.stem[0][0]
        export = cls(in_channels=stem_conv.in_channels,
                    num_classes=model.head.out_channels,
                    base=stem_conv.out_channels)
        export.load_state_dict(model.state_dict())   # exact key/shape match
        export.eval()
        return export
