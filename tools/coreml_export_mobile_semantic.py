"""Convert the trained mobile semantic student to Core ML and check parity.

iOS migration, Phase 2 (see the migration plan's Section 1: "Mobile
segmentation model strategy"). Unlike OFRSNet (src/ofrs/export.py, which
needed einsum/ceil_mode/geo-dict rewrites to trace cleanly) or Depth-Anything-V2
(needed a position-embedding patch for an unsupported bicubic-interpolation
op), MobileSemanticNet (src/mobile_semantic/model.py) is a standard
feedforward CNN -- LR-ASPP over MobileNetV3-Large, a well-precedented Core ML
conversion target -- so no special-casing is expected here; this tool exists
to VERIFY that, not assume it.

Usage:
    python -m tools.coreml_export_mobile_semantic
    python -m tools.coreml_export_mobile_semantic --height 512 --width 512

Requires `coremltools` (see requirements.txt). Standalone tool, not imported
by predict.py/disturbance.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.mobile_semantic.model import MobileSemanticNet  # noqa: E402

ROAD_IDX = config.OFRS_CLASSES.index("road")


def load_model(ckpt_path: Path) -> MobileSemanticNet:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = MobileSemanticNet(pretrained_backbone=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def convert(model: MobileSemanticNet, h: int, w: int, out_path: Path,
           deployment_target: str = "iOS16"):
    import coremltools as ct

    rgb = torch.from_numpy(np.random.default_rng(0).random((1, 3, h, w)).astype(np.float32))

    with torch.no_grad():
        traced = torch.jit.trace(model, (rgb,))
        py_out = model(rgb).numpy()

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="rgb", shape=rgb.shape)],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
        minimum_deployment_target=getattr(ct.target, deployment_target),
    )
    mlmodel.save(str(out_path))

    pred = mlmodel.predict({"rgb": rgb.numpy().astype(np.float32)})
    cm_out = pred["logits"]

    py_cls = py_out.argmax(1)
    cm_cls = cm_out.argmax(1)
    py_road, cm_road = py_cls == ROAD_IDX, cm_cls == ROAD_IDX
    inter, union = int((py_road & cm_road).sum()), int((py_road | cm_road).sum())
    return {
        "max_abs_diff": float(np.abs(py_out - cm_out).max()),
        "mean_abs_diff": float(np.abs(py_out - cm_out).mean()),
        "road_iou": (inter / union) if union else float("nan"),
        "all_class_pixel_agreement": float((py_cls == cm_cls).mean()),
    }, mlmodel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(config.CKPT_DIR / "mobile_semantic_best.pt"))
    ap.add_argument("--height", type=int, default=config.MOBILE_SEM_MAX_SIDE,
                    help=f"must be a multiple of MOBILE_SEM_NET_STRIDE={config.MOBILE_SEM_NET_STRIDE}")
    ap.add_argument("--width", type=int, default=config.MOBILE_SEM_MAX_SIDE)
    ap.add_argument("--out", default=str(config.ROOT / "MobileSemanticNet.mlpackage"))
    ap.add_argument("--deployment-target", default="iOS16")
    args = ap.parse_args()

    if args.height % config.MOBILE_SEM_NET_STRIDE or args.width % config.MOBILE_SEM_NET_STRIDE:
        raise SystemExit(f"--height/--width must be multiples of {config.MOBILE_SEM_NET_STRIDE}")

    print(f"[load] {args.ckpt}")
    model = load_model(Path(args.ckpt))

    print(f"[convert] tracing at {args.height}x{args.width}, target {args.deployment_target} ...")
    metrics, _ = convert(model, args.height, args.width, Path(args.out), args.deployment_target)
    print(f"[ok] saved {args.out}")
    print(f"     max_abs_diff={metrics['max_abs_diff']:.4e}  "
          f"mean_abs_diff={metrics['mean_abs_diff']:.4e}")
    print(f"     road_iou={metrics['road_iou']:.6f}  "
          f"all_class_pixel_agreement={metrics['all_class_pixel_agreement']:.6f}")


if __name__ == "__main__":
    main()
