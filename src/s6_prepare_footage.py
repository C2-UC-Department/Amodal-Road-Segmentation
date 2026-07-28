"""Step 6: Prepare real-world footage frames (data/footage_frames) for annotation.

KITTI ships ground truth to seed the annotator from; your own footage has
none. Instead, this script seeds each footage frame's amodal mask from the
MODEL'S OWN current prediction -- Mask2Former's visible road merged with
OFRSNet's occluded-region patch (the same `compose_amodal_mask` logic used by
`src.predict`). You then open the annotator and correct the model's mistakes
directly (e.g. under motorcycles) instead of drawing every frame from scratch.

Safety: this script NEVER overwrites an existing data/amodal_road/<base>.png.
If you've already hand-corrected a frame, it is left untouched.

Usage:
    python -m src.s6_prepare_footage
    python -m src.s6_prepare_footage --limit 20
    python -m src.s6_prepare_footage --ckpt checkpoints/ofrsnet_best.pt

Then annotate:
    python -m src.annotator --source footage
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import common  # noqa: E402
from src import s3_detect_foreground as s3  # noqa: E402
from src import s5_export_semantics as s5  # noqa: E402
from src import predict  # noqa: E402

PERSON_IDX = config.OFRS_CLASSES.index("person")
VEHICLE_IDX = config.OFRS_CLASSES.index("vehicle")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ckpt", default=str(config.CKPT_DIR / "ofrsnet_best.pt"))
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    config.SEMANTIC_DIR.mkdir(parents=True, exist_ok=True)
    config.FOREGROUND_DIR.mkdir(parents=True, exist_ok=True)
    config.AMODAL_DIR.mkdir(parents=True, exist_ok=True)

    device = common.pick_device()
    print(f"[device] {device}")

    processor, model_m2f, _ = s3.load_model(device)
    id2label = {int(k): v for k, v in model_m2f.config.id2label.items()}
    lut = s5.build_mapillary_to_ofrs(id2label)
    ofrsnet = predict.load_ofrsnet(Path(args.ckpt), device)

    samples = list(common.iter_footage_samples())
    if args.limit:
        samples = samples[: args.limit]

    n_seeded = n_skipped_existing = 0
    for s in tqdm(samples, desc="prepare footage"):
        sem_path = config.SEMANTIC_DIR / f"{s.base}.png"
        fg_path = config.FOREGROUND_DIR / f"{s.base}.png"
        amodal_path = config.AMODAL_DIR / f"{s.base}.png"

        rgb = common.read_rgb(s.image_path)

        if sem_path.exists():
            sem = np.asarray(Image.open(sem_path).convert("L"))
        else:
            image = Image.fromarray(rgb)
            seg = s3.segment(processor, model_m2f, device, image)
            sem = lut[np.clip(seg, 0, len(lut) - 1)].astype(np.uint8)
            Image.fromarray(sem, mode="L").save(sem_path)

        if not fg_path.exists():
            fg = np.isin(sem, [PERSON_IDX, VEHICLE_IDX])
            common.write_mask(fg_path, fg)

        if amodal_path.exists():
            n_skipped_existing += 1
            continue  # never clobber a hand-corrected (or previously seeded) mask

        geo_fields = (predict.compute_geometry(rgb, sem, s.base, device)
                      if ofrsnet.use_geometry else None)
        ofrs_road = predict.predict_amodal(ofrsnet, sem, device,
                                           threshold=args.threshold,
                                           geo_fields=geo_fields)
        final, _patch = predict.compose_amodal_mask(sem, ofrs_road)
        common.write_mask(amodal_path, final)
        n_seeded += 1

    print(f"[ok] seeded {n_seeded} new amodal masks "
          f"({n_skipped_existing} already existed and were left untouched)")
    print("Next: python -m src.annotator --source footage")


if __name__ == "__main__":
    main()
