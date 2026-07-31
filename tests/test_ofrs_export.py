"""Parity tests for src/ofrs/export.py (iOS migration, Phase 1).

`OFRSNetExport` computes the same function as a trained `OFRSNet` through an
op graph rewritten for Core ML conversion (einsum->bmm, no geo=None branch,
scale_factor=2 upsampling -- see export.py's module docstring for why each
rewrite is expected to be exact, not approximate). These tests are the
empirical half of that claim: `test_ofrsnet_coreml_readiness.py` proves the
geo=None equivalence in the ORIGINAL model; this file proves the EXPORT
variant reproduces the original bit-for-bit, on both synthetic and real data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import geometry as geo  # noqa: E402
from src.ofrs.export import OFRSNetExport  # noqa: E402
from src.ofrs.model import OFRSNet  # noqa: E402


def _zeroed_geo(n, h, w):
    return {
        "G": torch.zeros(n, 3, h, w),
        "h": torch.zeros(n, 1, h, w),
        "gvalid": torch.zeros(n, 1, h, w, dtype=torch.bool),
        "valid": torch.zeros(n, dtype=torch.bool),
    }


def _random_geo(n, h, w, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "G": torch.randn(n, 3, h, w, generator=g),
        "h": torch.randn(n, 1, h, w, generator=g),
        "gvalid": torch.rand(n, 1, h, w, generator=g) > 0.3,
        "valid": torch.ones(n, dtype=torch.bool),
    }


@pytest.fixture(scope="module")
def trained_pair():
    """(original, export) built from the same real checkpoint, or skip."""
    ckpt_path = config.CKPT_DIR / "ofrsnet_best.pt"
    if not ckpt_path.exists():
        pytest.skip(f"no checkpoint at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not ckpt.get("use_geometry", False):
        pytest.skip("shipped checkpoint does not use MultimodalContextModule")
    model = OFRSNet(in_channels=config.OFRS_NUM_CLASSES, num_classes=2,
                    use_geometry=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, OFRSNetExport.from_trained(model)


def _semantic_onehot(h, w, seed=0):
    rng = np.random.default_rng(seed)
    sem = rng.integers(0, config.OFRS_NUM_CLASSES, (h, w))
    onehot = np.eye(config.OFRS_NUM_CLASSES, dtype=np.float32)[sem]
    return torch.from_numpy(onehot.transpose(2, 0, 1))[None]


class TestExportMatchesOriginal:
    def test_zeroed_geo(self, trained_pair):
        model, export = trained_pair
        x = _semantic_onehot(64, 96)
        g = _zeroed_geo(1, 64, 96)
        with torch.no_grad():
            assert torch.equal(model(x, g), export(x, g))

    def test_populated_geo(self, trained_pair):
        model, export = trained_pair
        x = _semantic_onehot(64, 96, seed=1)
        g = _random_geo(1, 64, 96, seed=1)
        with torch.no_grad():
            assert torch.equal(model(x, g), export(x, g))

    @pytest.mark.parametrize("h,w", [(32, 32), (128, 256), (8, 16), (64, 96)])
    def test_various_shapes(self, trained_pair, h, w):
        """Multiples of OFRS_NET_STRIDE=8, as the real pipeline guarantees via
        common.pad_to_multiple -- the scale_factor=2 rewrite's precondition."""
        model, export = trained_pair
        x = _semantic_onehot(h, w, seed=2)
        g = _random_geo(1, h, w, seed=2)
        with torch.no_grad():
            out_m, out_e = model(x, g), export(x, g)
        assert out_m.shape == out_e.shape == (1, 2, h, w)
        assert torch.equal(out_m, out_e)

    def test_real_cached_geometry(self, trained_pair):
        """Real geometry-cache data, not synthetic -- exercises the actual
        (G, h, gvalid) distributions the model sees in production."""
        model, export = trained_pair
        cases = sorted(config.GEOMETRY_DIR.glob("*.npz"))[:5]
        if not cases:
            pytest.skip("no cached geometry under data/processed/geometry")

        checked = 0
        for p in cases:
            g = geo.load_geometry(p.stem)
            if g is None or not g["valid"]:
                continue
            hh, ww = g["depth"].shape
            ph, pw = ((hh + 7) // 8) * 8, ((ww + 7) // 8) * 8
            fields = geo.load_ground_fields(p.stem, out_hw=(ph, pw))
            if fields is None or not fields["valid"]:
                continue
            x = _semantic_onehot(ph, pw, seed=hash(p.stem) % 1000)
            g_t = {
                "G": torch.from_numpy(fields["G"].transpose(2, 0, 1))[None],
                "h": torch.from_numpy(fields["h"])[None, None],
                "gvalid": torch.from_numpy(fields["gvalid"])[None, None],
                "valid": torch.tensor([True]),
            }
            with torch.no_grad():
                assert torch.equal(model(x, g_t), export(x, g_t)), p.stem
            checked += 1
        if checked == 0:
            pytest.skip("no valid cached geometry entries found")


def test_from_trained_rejects_semantic_only_checkpoint():
    model = OFRSNet(in_channels=config.OFRS_NUM_CLASSES, num_classes=2,
                    use_geometry=False)
    with pytest.raises(ValueError, match="use_geometry"):
        OFRSNetExport.from_trained(model)
