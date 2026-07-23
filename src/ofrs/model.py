"""OFRSNet — a lightweight encoder-decoder FCN with a global-context module.

Faithful to the paper's description ("a lightweight and efficient encoder-decoder
fully convolutional architecture ... down-sampling and up-sampling blocks combined
with global contextual operations ... a global context module is used to build up
the down-sampling and joint context up-sampling block"). Exact channel widths are
not published, so we use a compact, trainable configuration.

Input : (N, C=11, H, W) one-hot / probability semantic map.
Output: (N, 2, H, W) logits for {non-road, road}.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, in_channels=11, num_classes=2, base=32):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        self.stem = nn.Sequential(conv_bn_relu(in_channels, c1),
                                  conv_bn_relu(c1, c1))
        self.down1 = DownBlock(c1, c2)      # /2
        self.down2 = DownBlock(c2, c3)      # /4
        self.down3 = DownBlock(c3, c4)      # /8

        self.context = GlobalContextModule(c4)

        self.up3 = UpBlock(c4, c3, c3)      # -> /4
        self.up2 = UpBlock(c3, c2, c2)      # -> /2
        self.up1 = UpBlock(c2, c1, c1)      # -> /1
        self.head = nn.Conv2d(c1, num_classes, 1)

    def forward(self, x):
        s0 = self.stem(x)     # c1, /1
        s1 = self.down1(s0)   # c2, /2
        s2 = self.down2(s1)   # c3, /4
        s3 = self.down3(s2)   # c4, /8

        g = self.context(s3)

        d3 = self.up3(g, s2)
        d2 = self.up2(d3, s1)
        d1 = self.up1(d2, s0)
        out = self.head(d1)
        # Safety: match input resolution exactly.
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear",
                                align_corners=False)
        return out
