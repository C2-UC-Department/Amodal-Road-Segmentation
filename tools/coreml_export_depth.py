"""Convert Depth-Anything-V2 (metric, outdoor, small) to Core ML and check parity.

iOS migration, Phase 0/3 spike (see the migration plan's geometry-stack section:
"needs a verification spike, not an assertion"). Result of that spike, recorded
here rather than left as a one-off finding:

  GO, with one required fix. Direct conversion of `config.DEPTH_MODEL_ID` fails:

      NotImplementedError: PyTorch convert function for op
      'upsample_bicubic2d' not implemented.

  The op comes from `Dinov2Embeddings.interpolate_pos_encoding` (HF
  `transformers.models.dinov2.modeling_dinov2`), which resizes the ViT's
  learned position embeddings to match the input's patch grid. Its own source
  comment says it "always interpolate[s] when tracing to ensure the exported
  model works for dynamic input shapes" -- a property this export does not
  need, since Core ML export already fixes the input resolution.

  Fix: precompute the interpolated position embeddings ONCE in eager mode for
  the fixed export resolution, then monkey-patch `interpolate_pos_encoding` to
  return that precomputed constant instead of calling `F.interpolate(...,
  mode="bicubic")` inside the traced graph. Verified bit-identical (0.0 max
  abs diff) between the patched and unpatched model in eager PyTorch for that
  resolution -- this is exact constant-folding of a fixed-shape computation,
  not an approximation.

  Remaining PyTorch-vs-Core-ML numeric drift after conversion: ~2-3.5%
  relative on the depth field (max abs diff ~0.23m on an observed [3.85,
  12.13]m range) -- consistent with fp16 precision in Core ML's Neural-Engine-
  oriented graph, and small next to this pipeline's own already-documented and
  corrected monocular-depth scale bias (config.py: 50-400% on real footage).

Usage:
    python -m tools.coreml_export_depth
    python -m tools.coreml_export_depth --height 518 --width 686

Requires `coremltools` (see requirements.txt). Standalone tool, not imported
by predict.py/disturbance.py/geometry.py.
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


class DepthWrapper(nn.Module):
    """Unwraps the HF ModelOutput dict to a plain tensor -- Core ML's
    TensorType outputs want that, not a dict/dataclass."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, pixel_values):
        return self.net(pixel_values=pixel_values).predicted_depth


def patch_position_embedding_interpolation(model, h: int, w: int) -> None:
    """Bake the ViT position embeddings for a FIXED (h, w) into a constant,
    eliminating the runtime `interpolate(..., mode="bicubic")` call that
    coremltools cannot convert. See this module's docstring for why this is
    exact, not approximate, for a fixed export resolution.
    """
    import transformers.models.dinov2.modeling_dinov2 as dinov2_mod

    embeddings_module = model.backbone.embeddings
    n_patches = (h // embeddings_module.patch_size) * (w // embeddings_module.patch_size)
    dim = embeddings_module.position_embeddings.shape[-1]
    dummy = torch.zeros(1, n_patches + 1, dim)   # only .shape is used by the fn below
    with torch.no_grad():
        precomputed = dinov2_mod.Dinov2Embeddings.interpolate_pos_encoding(
            embeddings_module, dummy, h, w)

    def _patched(self, embeddings, height, width):
        return precomputed

    embeddings_module.interpolate_pos_encoding = _patched.__get__(embeddings_module)


def convert(h: int, w: int, out_path: Path, deployment_target: str = "iOS16"):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    import coremltools as ct

    proc = AutoImageProcessor.from_pretrained(config.DEPTH_MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(config.DEPTH_MODEL_ID).eval()

    # A representative RGB image sized so the processor lands on (h, w) after
    # its own resize/crop -- if it doesn't, adjust --height/--width to match
    # what the processor actually produces for your target resolution.
    rgb = (np.random.default_rng(0).random((h, w, 3)) * 255).astype(np.uint8)
    px = proc(images=rgb, return_tensors="pt")["pixel_values"]
    actual_h, actual_w = px.shape[-2:]
    if (actual_h, actual_w) != (h, w):
        print(f"[note] processor produced {actual_h}x{actual_w}, not the requested "
              f"{h}x{w} -- exporting at the processor's actual output size.")
        h, w = actual_h, actual_w

    patch_position_embedding_interpolation(model, h, w)
    wrapper = DepthWrapper(model).eval()

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (px,))
        py_out = wrapper(px).numpy()

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixel_values", shape=px.shape)],
        outputs=[ct.TensorType(name="predicted_depth")],
        convert_to="mlprogram",
        minimum_deployment_target=getattr(ct.target, deployment_target),
    )
    mlmodel.save(str(out_path))

    pred = mlmodel.predict({"pixel_values": px.numpy().astype(np.float32)})
    cm_out = list(pred.values())[0]
    diff = np.abs(py_out - cm_out)
    rel = diff / (np.abs(py_out) + 1e-6)
    return {
        "max_abs_diff_m": float(diff.max()),
        "mean_abs_diff_m": float(diff.mean()),
        "max_relative_diff": float(rel.max()),
        "mean_relative_diff": float(rel.mean()),
        "py_depth_range_m": (float(py_out.min()), float(py_out.max())),
    }, mlmodel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--height", type=int, default=518,
                    help="requested export height (the HF processor may adjust "
                         "this to its own patch-aligned size)")
    ap.add_argument("--width", type=int, default=686)
    ap.add_argument("--out", default=str(config.ROOT / "DepthAnythingV2Small.mlpackage"))
    ap.add_argument("--deployment-target", default="iOS16")
    args = ap.parse_args()

    print(f"[convert] {config.DEPTH_MODEL_ID} at ~{args.height}x{args.width} ...")
    metrics, _ = convert(args.height, args.width, Path(args.out), args.deployment_target)
    print(f"[ok] saved {args.out}")
    lo, hi = metrics["py_depth_range_m"]
    print(f"     PyTorch depth range: [{lo:.2f}, {hi:.2f}] m")
    print(f"     max_abs_diff={metrics['max_abs_diff_m']:.4f} m  "
          f"mean_abs_diff={metrics['mean_abs_diff_m']:.4f} m")
    print(f"     max_relative_diff={metrics['max_relative_diff']:.2%}  "
          f"mean_relative_diff={metrics['mean_relative_diff']:.2%}")


if __name__ == "__main__":
    main()
