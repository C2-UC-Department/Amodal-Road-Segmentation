"""OFRSNet — a lightweight encoder-decoder FCN with a global-context module.

Faithful to the paper's description ("a lightweight and efficient encoder-decoder
fully convolutional architecture ... down-sampling and up-sampling blocks combined
with global contextual operations ... a global context module is used to build up
the down-sampling and joint context up-sampling block"). Exact channel widths are
not published, so we use a compact, trainable configuration.

Input : (N, C=11, H, W) one-hot / probability semantic map,
        plus OPTIONAL dense ground-manifold geometry (see MultimodalContextModule).
Output: (N, 2, H, W) logits for {non-road, road}.

Reformulation of the context stage as message passing on a ground manifold
-------------------------------------------------------------------------
The original module (kept below as `GlobalContextModule`) pools ONE global
vector and broadcasts it identically to every pixel -- its attention weights
do not depend on the query pixel, so no two pixels ever exchange information
specifically. `MultimodalContextModule` replaces that with geometry-guided
message passing: semantic evidence is propagated preferentially between pixels
that the depth/plane geometry says lie on the same ground surface.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config  # noqa: E402


def conv_bn_relu(cin, cout, k=3, s=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride=s, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class GlobalContextModule(nn.Module):
    """Injects a global feature (SE-style channel gate + broadcast global cue)."""

    def __init__(self, ch, reduction=4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // reduction, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1),
        )
        self.gate = nn.Sigmoid()
        self.fuse = conv_bn_relu(ch, ch, k=1)

    def forward(self, x):
        g = self.fc(self.gap(x))          # (N, ch, 1, 1) global descriptor
        x = x * self.gate(g)              # channel re-weighting
        x = x + g                         # broadcast-add global cue
        return self.fuse(x)


class MultimodalContextModule(nn.Module):
    """Geometry-guided message passing over the ground manifold.

    Consumes, alongside the semantic feature map, three dense fields produced
    by src/geometry.py from GroundNet's depth stream:

        G       (N, 3, H, W)  ground footprint -- where each pixel's viewing ray
                              meets the fitted ground plane. Depth-INDEPENDENT,
                              so an occluder pixel's footprint is exactly the
                              road patch it hides.
        h       (N, 1, H, W)  signed height above the plane of the pixel's own
                              3D point. |h| ~ 0 <=> the pixel is ON the manifold.
        gvalid  (N, 1, H, W)  ray meets the plane in front of the camera
                              (False above the horizon).

    Tier 1 -- ground-weighted global pooling
        Weight each pixel by exp(-|h| / tau_h) before pooling, so the global
        summary describes the ROAD SURFACE rather than averaging in cars,
        buildings and sky.

    Tier 2 -- geometry-gated non-local attention
        logit(i, j) = <q_i, k_j> / sqrt(d)
                      - alpha * ( ||G_i - G_j||^2 / tau_d + |h_j| / tau_h )

        Two deliberate asymmetries:
          * The manifold gate |h| is applied to KEYS only. A query on a car
            body is off-manifold, but that is exactly the pixel we most need to
            fill in, so it must stay free to gather evidence; what we want to
            constrain is the SOURCE of that evidence (trust real road pixels).
          * Distance uses the ground footprint G, not the pixel's own 3D point.
            A car roof is metres off the manifold, yet its footprint sits on
            the road it occludes -- so it still gathers from the correct
            neighbourhood. Using the raw 3D point instead would push occluders
            towards other elevated things (other roofs), which is the opposite
            of what amodal completion needs.

        Keys/values are pooled by `kv_stride`, making the cost
        O(HW x HW/stride^2) instead of O((HW)^2).

    Falls back to plain (semantic-only) behaviour per-sample when geometry is
    missing or the plane fit failed, so a partially-precomputed dataset still
    trains.
    """

    def __init__(self, ch, reduction=4, attn_dim=None, kv_stride=None,
                 tau_h_init=None, tau_d_init=None):
        super().__init__()
        attn_dim = attn_dim or config.OFRS_ATTN_DIM
        self.kv_stride = kv_stride or config.OFRS_ATTN_KV_STRIDE
        tau_h_init = tau_h_init or config.OFRS_TAU_H_INIT
        tau_d_init = tau_d_init or config.OFRS_TAU_D_INIT

        # --- Tier 1: pooled global cue (same head as GlobalContextModule) ---
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // reduction, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1),
        )
        self.gate = nn.Sigmoid()
        self.fuse = conv_bn_relu(ch, ch, k=1)

        # --- Tier 2: non-local attention ---
        self.q = nn.Conv2d(ch, attn_dim, 1)
        self.k = nn.Conv2d(ch, attn_dim, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.attn_dim = attn_dim

        # Learnable, positive via exp(); the net tunes its own geometric
        # receptive field rather than trusting our hand-picked constants.
        self.log_tau_h = nn.Parameter(torch.tensor(math.log(tau_h_init)))
        self.log_tau_d = nn.Parameter(torch.tensor(math.log(tau_d_init)))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        # Zero-init residual scale for TIER 2 ONLY: the non-local block starts
        # as an exact no-op and learns to switch message passing on, which
        # keeps early training stable (a randomly-initialised global attention
        # can otherwise smear features across the whole image before anything
        # useful is learned). Tier 1 is intentionally NOT gated this way -- it
        # is only a reweighting of a pooled average, so it is safe from step
        # one and needs no warm-up.
        self.gamma = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resize_geo(t, size):
        return F.interpolate(t, size=size, mode="nearest")

    def forward(self, x, geo=None):
        n, c, hh, ww = x.shape

        if geo is None:
            h_map = None
            g_map = None
            gvalid = None
            sample_valid = torch.zeros(n, dtype=torch.bool, device=x.device)
        else:
            g_map = self._resize_geo(geo["G"], (hh, ww))            # (N,3,h,w)
            h_map = self._resize_geo(geo["h"], (hh, ww))            # (N,1,h,w)
            gvalid = self._resize_geo(geo["gvalid"].float(), (hh, ww)) > 0.5
            sample_valid = geo["valid"].to(torch.bool).to(x.device)  # (N,)

        tau_h = self.log_tau_h.exp().clamp(min=1e-3)
        tau_d = self.log_tau_d.exp().clamp(min=1e-3)

        # ---------------- Tier 1: ground-weighted pooling ----------------
        if h_map is None:
            pooled = x.mean(dim=(2, 3), keepdim=True)
        else:
            w = torch.exp(-h_map.abs() / tau_h) * gvalid.float()     # (N,1,h,w)
            # Where the plane fit failed, fall back to uniform weights.
            uniform = torch.ones_like(w)
            use = sample_valid.view(n, 1, 1, 1)
            w = torch.where(use, w, uniform)
            wsum = w.sum(dim=(2, 3), keepdim=True)
            # A degenerate weight map (e.g. nothing on the manifold) would give
            # 0/0; fall back to plain mean for those samples.
            degenerate = wsum < 1e-6
            w = torch.where(degenerate, uniform, w)
            wsum = torch.where(degenerate, uniform.sum(dim=(2, 3), keepdim=True), wsum)
            pooled = (x * w).sum(dim=(2, 3), keepdim=True) / wsum

        g = self.fc(pooled)                    # (N, C, 1, 1)
        out = x * self.gate(g) + g
        out = self.fuse(out)

        # ------------- Tier 2: geometry-gated non-local attention -------------
        s = self.kv_stride
        q = self.q(out).flatten(2)                                   # (N, d, HW)
        x_kv = F.avg_pool2d(out, s, ceil_mode=True) if s > 1 else out
        k = self.k(x_kv).flatten(2)                                  # (N, d, M)
        v = self.v(x_kv).flatten(2)                                  # (N, C, M)

        logits = torch.einsum("ndq,ndk->nqk", q, k) / math.sqrt(self.attn_dim)

        if h_map is not None:
            gq = g_map.flatten(2)                                    # (N,3,HW)
            gk = (F.avg_pool2d(g_map, s, ceil_mode=True) if s > 1 else g_map).flatten(2)
            hk = (F.avg_pool2d(h_map, s, ceil_mode=True) if s > 1 else h_map).flatten(2)
            vk = (F.max_pool2d(gvalid.float(), s, ceil_mode=True)
                  if s > 1 else gvalid.float()).flatten(2)           # (N,1,M)

            # ||G_i - G_j||^2 via the expansion, to avoid an (N,3,HW,M) tensor.
            d2 = (gq.pow(2).sum(1).unsqueeze(2)
                  + gk.pow(2).sum(1).unsqueeze(1)
                  - 2.0 * torch.einsum("ndq,ndk->nqk", gq, gk)).clamp(min=0)

            geo_bias = -(d2 / tau_d + hk.abs().squeeze(1).unsqueeze(1) / tau_h)
            geo_bias = self.alpha * geo_bias

            # Exclude keys with no ground footprint (above the horizon).
            key_ok = vk.squeeze(1).unsqueeze(1) > 0.5                # (N,1,M)
            geo_bias = geo_bias.masked_fill(~key_ok, -1e4)

            # Only use geometry where the plane fit succeeded AND at least one
            # key is usable; otherwise leave pure content attention (bias 0).
            has_key = key_ok.any(dim=2, keepdim=True)                # (N,1,1)
            use_geo = sample_valid.view(n, 1, 1) & has_key
            logits = logits + torch.where(use_geo, geo_bias,
                                          torch.zeros_like(geo_bias))

        attn = logits.softmax(dim=-1)
        msg = torch.einsum("nqk,nck->ncq", attn, v).reshape(n, c, hh, ww)
        return out + self.gamma * self.proj(msg)


class DownBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            conv_bn_relu(cin, cout, s=2),  # halve spatial size
            conv_bn_relu(cout, cout),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Joint context up-sampling: upsample, concat skip, fuse."""

    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.reduce = conv_bn_relu(cin, cout, k=1)
        self.fuse = nn.Sequential(
            conv_bn_relu(cout + cskip, cout),
            conv_bn_relu(cout, cout),
        )

    def forward(self, x, skip):
        x = self.reduce(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class OFRSNet(nn.Module):
    """
    use_geometry=True swaps the semantic-only global context stage for the
    geometry-guided message-passing module. The geometry itself is optional at
    call time: `forward(x)` with no geo still works (falls back to plain
    pooling + content-only attention), so the same checkpoint can run on
    images whose geometry has not been precomputed.
    """

    def __init__(self, in_channels=11, num_classes=2, base=32,
                 use_geometry=None):
        super().__init__()
        if use_geometry is None:
            use_geometry = config.OFRS_USE_GEOMETRY
        self.use_geometry = use_geometry
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        self.stem = nn.Sequential(conv_bn_relu(in_channels, c1),
                                  conv_bn_relu(c1, c1))
        self.down1 = DownBlock(c1, c2)      # /2
        self.down2 = DownBlock(c2, c3)      # /4
        self.down3 = DownBlock(c3, c4)      # /8

        self.context = (MultimodalContextModule(c4) if use_geometry
                        else GlobalContextModule(c4))

        self.up3 = UpBlock(c4, c3, c3)      # -> /4
        self.up2 = UpBlock(c3, c2, c2)      # -> /2
        self.up1 = UpBlock(c2, c1, c1)      # -> /1
        self.head = nn.Conv2d(c1, num_classes, 1)

    def forward(self, x, geo=None):
        s0 = self.stem(x)     # c1, /1
        s1 = self.down1(s0)   # c2, /2
        s2 = self.down2(s1)   # c3, /4
        s3 = self.down3(s2)   # c4, /8

        g = self.context(s3, geo) if self.use_geometry else self.context(s3)

        d3 = self.up3(g, s2)
        d2 = self.up2(d3, s1)
        d1 = self.up1(d2, s0)
        out = self.head(d1)
        # Safety: match input resolution exactly.
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear",
                                align_corners=False)
        return out
