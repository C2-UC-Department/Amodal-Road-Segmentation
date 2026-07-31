"""iOS/Core ML migration, Phase 0 spike: does OFRSNet(x, geo=None) produce the
same output as OFRSNet(x, geo=<populated-but-invalid dict>)?

Core ML traces a single control-flow path. `MultimodalContextModule.forward`
has a Python-level `if geo is None` branch (src/ofrs/model.py:153), so a naive
export would need two graphs -- unless the two paths are provably equivalent,
in which case the model can always be called with a synthesized
all-invalid geo dict (G=0, h=0, gvalid=all-False, valid=all-False) and only
one graph is ever needed.

Walking the module confirms this by construction: with gvalid all-False,
`key_ok` is all-False (src/ofrs/model.py:213), so `has_key` is False, so
`use_geo` is False regardless of `sample_valid`, so
`torch.where(use_geo, geo_bias, zeros)` (line 220) selects zeros
unconditionally -- whatever `geo_bias` computed to is discarded, not blended.
Tier 1 (line 174) takes the identical `torch.where(use=False, w, uniform)` path
to the same uniform-mean fallback the `geo is None` branch computes directly.
So the two call paths should be exactly, not just approximately, equal -- and
this test asserts exactly that, both for a freshly-initialised model and the
real shipped checkpoint (random-init batchnorm/softmax numerics can sometimes
mask branch-dependent drift that real activation statistics would reveal).

This is the load-bearing fact behind the iOS migration plan's Core ML strategy
(see the plan's OFRSNet section): if this test ever fails, the "always
synthesize a populated geo dict, ship one graph" plan is invalid and the
documented fallback (a second .mlpackage using use_geometry=False) is needed
instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.ofrs.model import OFRSNet  # noqa: E402

N, C, H, W = 2, config.OFRS_NUM_CLASSES, 64, 96   # H,W multiples of OFRS_NET_STRIDE=8


def _zeroed_geo(n, h, w):
    return {
        "G": torch.zeros(n, 3, h, w),
        "h": torch.zeros(n, 1, h, w),
        "gvalid": torch.zeros(n, 1, h, w, dtype=torch.bool),
        "valid": torch.zeros(n, dtype=torch.bool),
    }


def test_geo_none_matches_zeroed_geo_dict_random_init():
    torch.manual_seed(0)
    model = OFRSNet(in_channels=C, num_classes=2, use_geometry=True).eval()
    x = torch.softmax(torch.randn(N, C, H, W), dim=1)

    with torch.no_grad():
        out_none = model(x, geo=None)
        out_zeroed = model(x, geo=_zeroed_geo(N, H, W))

    assert torch.equal(out_none, out_zeroed)


def test_geo_none_matches_zeroed_geo_dict_trained_checkpoint():
    ckpt_path = config.CKPT_DIR / "ofrsnet_best.pt"
    if not ckpt_path.exists():
        pytest.skip(f"no checkpoint at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not ckpt.get("use_geometry", False):
        pytest.skip("shipped checkpoint does not use MultimodalContextModule")

    model = OFRSNet(in_channels=C, num_classes=2, use_geometry=True).eval()
    model.load_state_dict(ckpt["model"])

    torch.manual_seed(1)
    x = torch.softmax(torch.randn(N, C, H, W), dim=1)

    with torch.no_grad():
        out_none = model(x, geo=None)
        out_zeroed = model(x, geo=_zeroed_geo(N, H, W))

    assert torch.equal(out_none, out_zeroed)
