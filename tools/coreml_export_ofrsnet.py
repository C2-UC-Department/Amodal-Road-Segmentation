"""Convert OFRSNetExport to a Core ML .mlpackage and check numeric parity.

iOS migration, Phase 1 (see the migration plan's OFRSNet section). This is the
reusable version of the Phase-0 conversion spike: given a trained checkpoint,
produce a .mlpackage and report how closely it reproduces the PyTorch model's
output, on both the input resolution used for tracing and (optionally) a
handful of real cached-geometry samples.

Usage:
    python -m tools.coreml_export_ofrsnet
    python -m tools.coreml_export_ofrsnet --height 512 --width 768
    python -m tools.coreml_export_ofrsnet --out /tmp/OFRSNet.mlpackage --no-real-geo

Requires `coremltools` (optional, iOS-migration-only dependency -- see
requirements.txt). Not part of the training/inference pipeline; nothing here
is imported by predict.py or disturbance.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import geometry as geo  # noqa: E402
from src.ofrs.export import OFRSNetExport  # noqa: E402
from src.ofrs.model import OFRSNet  # noqa: E402


class TracingWrapper(nn.Module):
    """Flattens the geo dict into positional float32 tensors: torch.jit.trace
    and Core ML's TensorType inputs both want plain tensors, not a dict, and
    plain float32 (rather than bool) sidesteps any bool-input support
    variance across coremltools/deployment-target combinations -- the
    boolean casts happen just inside the traced graph instead.
    """

    def __init__(self, net: OFRSNetExport):
        super().__init__()
        self.net = net

    def forward(self, x, G, h, gvalid_f, valid_f):
        geo_dict = {"G": G, "h": h, "gvalid": gvalid_f > 0.5, "valid": valid_f > 0.5}
        return self.net(x, geo_dict)


def load_export_model(ckpt_path: Path) -> OFRSNetExport:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not ckpt.get("use_geometry", False):
        raise SystemExit(
            f"{ckpt_path} was trained with use_geometry=False -- it converts "
            "directly from src/ofrs/model.py (no einsum/geo=None/dynamic-shape "
            "issues to work around), OFRSNetExport is not needed for it.")
    model = OFRSNet(in_channels=config.OFRS_NUM_CLASSES, num_classes=2,
                    use_geometry=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return OFRSNetExport.from_trained(model)


def _plausible_example_inputs(h: int, w: int):
    """A scene with actual road/occluder structure, not random noise -- a
    degenerate all-background input makes any parity metric meaningless
    (both models correctly predict nothing everywhere)."""
    ROAD_IDX = config.OFRS_CLASSES.index("road")
    VEHICLE_IDX = config.OFRS_CLASSES.index("vehicle")
    sem = np.full((h, w), config.OFRS_CLASSES.index("building"), np.int64)
    sem[h // 3:, :] = ROAD_IDX
    sem[h // 2:h // 2 + max(1, h // 12), w // 3:w // 3 + max(1, w // 6)] = VEHICLE_IDX
    x = torch.from_numpy(np.eye(config.OFRS_NUM_CLASSES, dtype=np.float32)[sem]
                         .transpose(2, 0, 1))[None]

    G = torch.zeros(1, 3, h, w)
    G[0, 2] = torch.linspace(2, 30, h).view(h, 1).expand(h, w)
    G[0, 0] = torch.linspace(-5, 5, w).view(1, w).expand(h, w)
    hh = torch.randn(1, 1, h, w) * 0.1
    gvalid_f = torch.zeros(1, 1, h, w)
    gvalid_f[0, 0, h // 3:, :] = 1.0
    valid_f = torch.ones(1)
    return x, G, hh, gvalid_f, valid_f


def convert(export_model: OFRSNetExport, h: int, w: int, out_path: Path,
           deployment_target: str = "iOS16"):
    import coremltools as ct

    wrapper = TracingWrapper(export_model).eval()
    x, G, hh, gvalid_f, valid_f = _plausible_example_inputs(h, w)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (x, G, hh, gvalid_f, valid_f))
        py_out = wrapper(x, G, hh, gvalid_f, valid_f).numpy()

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x", shape=x.shape),
            ct.TensorType(name="G", shape=G.shape),
            ct.TensorType(name="h", shape=hh.shape),
            ct.TensorType(name="gvalid_f", shape=gvalid_f.shape),
            ct.TensorType(name="valid_f", shape=valid_f.shape),
        ],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
        minimum_deployment_target=getattr(ct.target, deployment_target),
    )
    mlmodel.save(str(out_path))

    pred = mlmodel.predict({
        "x": x.numpy().astype(np.float32), "G": G.numpy().astype(np.float32),
        "h": hh.numpy().astype(np.float32),
        "gvalid_f": gvalid_f.numpy().astype(np.float32),
        "valid_f": valid_f.numpy().astype(np.float32),
    })
    cm_out = pred["logits"]

    py_road = py_out[:, 1] > py_out[:, 0]
    cm_road = cm_out[:, 1] > cm_out[:, 0]
    inter, union = int((py_road & cm_road).sum()), int((py_road | cm_road).sum())
    return {
        "max_abs_diff": float(np.abs(py_out - cm_out).max()),
        "mean_abs_diff": float(np.abs(py_out - cm_out).mean()),
        "road_iou": (inter / union) if union else float("nan"),
        "pixel_agreement": float((py_road == cm_road).mean()),
    }, mlmodel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(config.CKPT_DIR / "ofrsnet_best.pt"))
    ap.add_argument("--height", type=int, default=256,
                    help="export resolution height, must be a multiple of "
                         f"OFRS_NET_STRIDE={config.OFRS_NET_STRIDE}")
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--out", default=str(config.ROOT / "OFRSNetExport.mlpackage"))
    ap.add_argument("--deployment-target", default="iOS16")
    ap.add_argument("--no-real-geo", action="store_true",
                    help="skip the additional check at real cached-geometry resolutions")
    args = ap.parse_args()

    if args.height % config.OFRS_NET_STRIDE or args.width % config.OFRS_NET_STRIDE:
        raise SystemExit(f"--height/--width must be multiples of "
                         f"{config.OFRS_NET_STRIDE}")

    print(f"[load] {args.ckpt}")
    export_model = load_export_model(Path(args.ckpt))

    print(f"[convert] tracing at {args.height}x{args.width}, "
          f"target {args.deployment_target} ...")
    metrics, _ = convert(export_model, args.height, args.width, Path(args.out),
                         args.deployment_target)
    print(f"[ok] saved {args.out}")
    print(f"     max_abs_diff={metrics['max_abs_diff']:.4e}  "
          f"mean_abs_diff={metrics['mean_abs_diff']:.4e}")
    print(f"     road_iou={metrics['road_iou']:.6f}  "
          f"pixel_agreement={metrics['pixel_agreement']:.6f}")

    if not args.no_real_geo:
        # NOTE: this re-converts+re-checks at each REAL image's padded
        # resolution, with freshly synthesized plausible geometry (not the
        # actual cached G/h/gvalid values) -- it's a shape-robustness check
        # (does conversion hold up across the aspect ratios/sizes real images
        # actually produce), not a check against real geometry VALUES. Exact
        # PyTorch-side parity against real cached geometry values is already
        # covered exhaustively by tests/test_ofrs_export.py::test_real_cached_geometry.
        print("\n[shape-robustness check, at real cached-geometry resolutions]")
        cases = sorted(config.GEOMETRY_DIR.glob("*.npz"))[:5]
        if not cases:
            print("  no cached geometry found under data/processed/geometry, skipping")
        for p in cases:
            g = geo.load_geometry(p.stem)
            if g is None or not g["valid"]:
                continue
            hh_, ww_ = g["depth"].shape
            ph = ((hh_ + config.OFRS_NET_STRIDE - 1) // config.OFRS_NET_STRIDE) * config.OFRS_NET_STRIDE
            pw = ((ww_ + config.OFRS_NET_STRIDE - 1) // config.OFRS_NET_STRIDE) * config.OFRS_NET_STRIDE
            m, _ = convert(export_model, ph, pw, Path("/tmp/_ofrs_geo_check.mlpackage"),
                          args.deployment_target)
            print(f"  {p.stem:<24} {ph}x{pw}  max_diff={m['max_abs_diff']:.4e}  "
                  f"road_iou={m['road_iou']:.6f}")


if __name__ == "__main__":
    main()
