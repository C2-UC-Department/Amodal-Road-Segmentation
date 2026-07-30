"""Inference + visualization: image directory -> OFRSNet amodal road.

Pipeline per image:
    RGB  --Mask2Former(Mapillary)-->  11-class OFRS semantic map
         --one-hot, resize 384x1248-->  OFRSNet  -->  road-probability map
         --bilinear resize back-->  threshold  -->  OFRSNet road prediction

The FINAL amodal mask is NOT OFRSNet's raw output. OFRSNet is only trusted to
patch the footprint of nearby occluders (person/vehicle classes touching the
visible road); the rest of the mask is Mask2Former's own visible-road
segmentation, untouched. See `compose_amodal_mask()`.

    final = mask2former_visible_road  OR  (ofrsnet_road AND occluder_footprint)

Writes, for each input image, a stacked visualization (original / semantic /
final amodal road, with OFRSNet's patch contribution highlighted) to the
output dir, plus the raw binary mask.

Usage:
    python -m src.predict --input path/to/images --out data/predictions
    python -m src.predict --input data/raw/data_road/testing/image_2 --limit 20
    python -m src.predict --input my_dashcam/ --ckpt checkpoints/ofrsnet_best.pt --no-stack
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import common  # noqa: E402
from src import s3_detect_foreground as s3  # noqa: E402
from src import s5_export_semantics as s5  # noqa: E402
from src import geometry as geo  # noqa: E402
from src.ofrs.model import OFRSNet  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
COL_ROAD = (40, 220, 90)     # final amodal road (Mask2Former visible + OFRSNet patch)
COL_PATCH = (0, 210, 255)    # the part of the mask actually contributed by OFRSNet
COL_OBJECT = (255, 60, 60)   # person/vehicle from semantics (red)

ROAD_IDX = config.OFRS_CLASSES.index("road")
PERSON_IDX = config.OFRS_CLASSES.index("person")
VEHICLE_IDX = config.OFRS_CLASSES.index("vehicle")


def list_images(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def load_ofrsnet(ckpt_path: Path, device):
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        # Rebuild the architecture the checkpoint was trained with, so a
        # geometry-trained model is never silently loaded as semantic-only.
        use_geo = bool(ckpt.get("use_geometry", False))
        model = OFRSNet(in_channels=config.OFRS_NUM_CLASSES, num_classes=2,
                        use_geometry=use_geo).to(device)
        model.load_state_dict(ckpt["model"])
        print(f"[ckpt] loaded {ckpt_path}  (epoch {ckpt.get('epoch','?')}, "
              f"val_road_IoU {ckpt.get('val_iou','?')}, geometry={use_geo})")
    else:
        model = OFRSNet(in_channels=config.OFRS_NUM_CLASSES, num_classes=2).to(device)
        print(f"[warn] no checkpoint at {ckpt_path}; using RANDOM weights "
              "(train OFRSNet first for meaningful output).")
    model.eval()
    return model


def compute_geometry(rgb: np.ndarray, sem: np.ndarray, base: str, device,
                     image_path: Path | None = None):
    """Ground-manifold fields for one image, as model-ready tensors.

    The cache-or-compute decision for the raw (depth, n, K) primitives lives in
    `geo.resolve_plane`, shared with the BEV path in src/disturbance.py so the
    two can never disagree about which plane an image has. Returns None if
    geometry is unavailable or the plane fit fails -- the model then falls back
    to semantic-only behaviour for that image.
    """
    plane = geo.resolve_plane(base, sem, rgb=rgb, image_path=image_path,
                              device=device)
    if plane is None or not plane["valid"]:
        return None
    G, hh, gvalid = geo.derive_ground_fields(plane["depth"], plane["n"], plane["K"])
    # Return raw numpy at NATIVE resolution; predict_amodal resizes and pads it
    # through exactly the same path as the semantic map so the two stay aligned.
    return dict(G=G, h=hh, gvalid=gvalid, valid=True)


def _geo_to_tensors(fields, small_hw, padded_hw, device):
    """Resize native-resolution geometry to the capped size, pad it to the
    network canvas, and convert to tensors.

    Must mirror `fit_within_max_side` + `pad_to_multiple` exactly, or the
    geometry would be spatially offset from the semantic features it gates.
    Padding sets gvalid=0, so padded pixels can never act as attention keys.
    """
    if fields is None:
        return None
    sh, sw = small_hw
    ph, pw = padded_hw
    G = cv2.resize(fields["G"], (sw, sh), interpolation=cv2.INTER_LINEAR)
    h_f = cv2.resize(fields["h"], (sw, sh), interpolation=cv2.INTER_LINEAR)
    gv = cv2.resize(fields["gvalid"].astype(np.uint8), (sw, sh),
                    interpolation=cv2.INTER_NEAREST)

    Gp = np.zeros((ph, pw, 3), np.float32); Gp[:sh, :sw] = G
    hp = np.zeros((ph, pw), np.float32);    hp[:sh, :sw] = h_f
    vp = np.zeros((ph, pw), np.float32);    vp[:sh, :sw] = gv

    return {
        "G": torch.from_numpy(Gp.transpose(2, 0, 1))[None].to(device),
        "h": torch.from_numpy(hp)[None, None].to(device),
        "gvalid": (torch.from_numpy(vp)[None, None] > 0.5).to(device),
        "valid": torch.tensor([True], device=device),
    }


@torch.no_grad()
def predict_amodal(model, sem_labels: np.ndarray, device,
                   threshold: float = 0.5, geo_fields=None) -> np.ndarray:
    """sem_labels (H,W) int 0..10, at NATIVE resolution -> predicted amodal
    road mask, same (H,W), bool.

    OFRSNet is fully convolutional (global context uses adaptive pooling;
    every up-sampling stage targets its skip connection's own shape), so no
    resize to a fixed canvas is needed or performed here -- that used to force
    every image into a 384x1248 landscape shape, stretching/squashing any
    image whose native aspect ratio differed (e.g. portrait phone photos).

    Instead: (1) if the image exceeds config.OFRS_MAX_SIDE, uniformly
    downscale (both axes by the same factor -- aspect ratio is exactly
    preserved, unlike the old fixed-shape resize) purely to bound compute;
    (2) pad up to a multiple of config.OFRS_NET_STRIDE so the network's /8
    down/up-sampling lines up; (3) run the model; (4) crop the padding back
    off and undo step (1)'s scale (bilinear, still aspect-preserving) to land
    back at the exact native resolution.
    """
    h0, w0 = sem_labels.shape
    sem_small, scale = common.fit_within_max_side(sem_labels, config.OFRS_MAX_SIDE)
    sem_padded = common.pad_to_multiple(sem_small, config.OFRS_NET_STRIDE,
                                        config.OFRS_UNLABELED_ID)
    sem_padded = np.clip(sem_padded, 0, config.OFRS_NUM_CLASSES - 1).astype(np.int64)

    onehot = np.eye(config.OFRS_NUM_CLASSES, dtype=np.float32)[sem_padded].transpose(2, 0, 1)
    x = torch.from_numpy(onehot).unsqueeze(0).to(device)

    h, w = sem_small.shape
    geo_t = _geo_to_tensors(geo_fields, (h, w), sem_padded.shape, device)
    logits = model(x, geo_t)                             # (1, 2, H', W') padded
    prob_road = F.softmax(logits, dim=1)[:, 1:2, :h, :w]  # crop padding off -> (1,1,h,w)

    if scale < 1.0:
        # Undo step (1): uniform (aspect-preserving) upsample back to native.
        prob_road = F.interpolate(prob_road, size=(h0, w0), mode="bilinear",
                                  align_corners=False)
    return (prob_road[0, 0] > threshold).cpu().numpy().astype(bool)


def compose_amodal_mask(sem_labels: np.ndarray,
                        ofrsnet_road: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge Mask2Former's visible road with ONLY the occluded-region patch
    predicted by OFRSNet -- OFRSNet's output never replaces or redraws the
    already-visible road, it only fills the footprint of nearby occluders.

    This keeps the boundary of the open road exactly as clean as Mask2Former's
    own segmentation; any residual jaggedness from OFRSNet is confined to the
    small patched regions instead of the whole mask.

    Returns (final_mask, patch_mask) -- `patch_mask` is the subset of
    `final_mask` that came from OFRSNet, useful for QA visualization.
    """
    visible = sem_labels == ROAD_IDX
    occluders = (sem_labels == PERSON_IDX) | (sem_labels == VEHICLE_IDX)
    # Whole-blob eligibility (common.occluder_blob_mask): if ANY part of a
    # person/vehicle blob touches the near-road band, its ENTIRE footprint is
    # eligible for patching -- not just the pixels within
    # config.ROAD_NEIGHBOURHOOD_PX of the visible road. A per-pixel version of
    # this used to only admit a vehicle's tire/undercarriage band (the part
    # literally close to the road) and silently clip off the rest of the
    # body, no matter how well OFRSNet reconstructed it.
    occlusion_region = common.occluder_blob_mask(occluders, visible,
                                                 config.ROAD_NEIGHBOURHOOD_PX)

    patch = ofrsnet_road & occlusion_region
    return (visible | patch), patch


def make_stack(rgb, sem_labels, amodal, patch) -> np.ndarray:
    """Vertical stack: original / semantic / final amodal overlay."""
    sem_color = s5.OFRS_PALETTE[sem_labels]
    sem_blend = (0.45 * rgb + 0.55 * sem_color).astype(np.uint8)

    obj = np.isin(sem_labels, [PERSON_IDX, VEHICLE_IDX])
    over = common.overlay_mask(rgb, amodal, COL_ROAD, 0.5)
    over = common.overlay_mask(over, obj, COL_OBJECT, 0.35)
    over = common.overlay_mask(over, patch, COL_PATCH, 0.6)  # highlight OFRSNet's contribution

    def label(img, text):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(img, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img

    return np.vstack([label(rgb, "input"),
                      label(sem_blend, "Mask2Former semantics (OFRS 11-class)"),
                      label(over, "Mask2Former road (green) + OFRSNet patch (cyan)")])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="image file or directory")
    ap.add_argument("--out", default=str(config.DATA_DIR / "predictions"))
    ap.add_argument("--ckpt", default=str(config.CKPT_DIR / "ofrsnet_best.pt"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="road-probability threshold after bilinear upsampling")
    ap.add_argument("--no-stack", action="store_true",
                    help="save only the amodal overlay, not the 3-panel stack")
    args = ap.parse_args()

    images = list_images(Path(args.input))
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found under {args.input}")

    out_dir = Path(args.out)
    (out_dir / "mask").mkdir(parents=True, exist_ok=True)

    device = common.pick_device()
    print(f"[device] {device}")

    # Upstream Mask2Former + Mapillary->OFRS LUT (shared with Step 5).
    processor, model_m2f, _ = s3.load_model(device)
    id2label = {int(k): v for k, v in model_m2f.config.id2label.items()}
    lut = s5.build_mapillary_to_ofrs(id2label)

    ofrsnet = load_ofrsnet(Path(args.ckpt), device)

    for path in tqdm(images, desc="predict"):
        rgb = common.read_rgb(path)
        image = Image.fromarray(rgb)
        seg = s3.segment(processor, model_m2f, device, image)      # mapillary ids
        sem = lut[np.clip(seg, 0, len(lut) - 1)]                    # OFRS 0..10
        geo_fields = (compute_geometry(rgb, sem, path.stem, device, image_path=path)
                      if ofrsnet.use_geometry else None)
        ofrs_road = predict_amodal(ofrsnet, sem, device, threshold=args.threshold,
                                   geo_fields=geo_fields)
        amodal, patch = compose_amodal_mask(sem, ofrs_road)

        common.write_mask(out_dir / "mask" / f"{path.stem}.png", amodal)
        if args.no_stack:
            viz = common.overlay_mask(rgb, amodal, COL_ROAD, 0.5)
            viz = common.overlay_mask(viz, patch, COL_PATCH, 0.6)
        else:
            viz = make_stack(rgb, sem, amodal, patch)
        Image.fromarray(viz).save(out_dir / f"{path.stem}_viz.png")

    print(f"[ok] wrote predictions to {out_dir}")


if __name__ == "__main__":
    main()
