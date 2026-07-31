"""Build a YOLO-seg training set by running the existing Mask2Former/
Cityscapes-instance teacher (src/instances.py) over kitti+footage+realworld
and converting its vehicle masks to YOLO polygon labels
(src/mobile_instance/labels.py).

Single class ("vehicle", id 0): downstream (src/instances.py's own
VehicleInstance/occluder_support machinery) only ever distinguishes
vehicle-vs-not, never car-vs-truck-vs-bus, so training a finer-grained
detector would add label noise without anything consuming the extra
resolution.

Output layout (standard Ultralytics YOLO-seg):
    data/processed/yolo_instance/
        images/train/*.jpg  images/val/*.jpg   (copied from the source images)
        labels/train/*.txt  labels/val/*.txt   (YOLO-seg polygon lines)
        data.yaml

Usage:
    python -m tools.export_yolo_instance_dataset
    python -m tools.export_yolo_instance_dataset --sources kitti footage
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import common  # noqa: E402
from src import instances  # noqa: E402
from src.mobile_instance.labels import mask_to_yolo_polygons  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=["kitti", "footage", "realworld"])
    ap.add_argument("--min-score", type=float, default=config.INSTANCE_MIN_SCORE)
    args = ap.parse_args()

    device = common.pick_device()
    print(f"[device] {device}")
    models = instances.load_instance_model(device)
    if models is None:
        raise SystemExit(f"Instance model {config.INSTANCE_MODEL_ID} unavailable -- "
                         "cannot build the distillation dataset without the teacher.")

    samples = []
    for source in args.sources:
        samples.extend(common.iter_source_samples(source))
    print(f"[data] {len(samples)} images across {args.sources}")

    # NOTE: shuffle INDICES, not the Sample objects themselves -- Sample is a
    # NamedTuple, and np.array(samples, dtype=object) silently flattens each
    # one into its 3 fields instead of keeping it as one opaque object
    # (unlike src/ofrs/dataset.py::split_bases, which shuffles plain strings
    # and has no such issue).
    rng = np.random.default_rng(config.MOBILE_INST_SPLIT_SEED)
    order = rng.permutation(len(samples))
    n_val = max(1, int(round(len(order) * config.MOBILE_INST_VAL_FRACTION)))
    split_of = {samples[idx].base: ("val" if pos < n_val else "train")
               for pos, idx in enumerate(order)}

    for split in ("train", "val"):
        (config.YOLO_INSTANCE_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (config.YOLO_INSTANCE_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_with_vehicle = 0
    n_instances = 0
    for s in tqdm(samples, desc="YOLO-seg instance export"):
        split = split_of[s.base]
        image = Image.open(s.image_path).convert("RGB")
        dets = instances.detect(models, device, image, min_score=args.min_score)
        vehicle_dets = [d for d in dets if d.is_vehicle]

        lines = []
        for d in vehicle_dets:
            lines.extend(mask_to_yolo_polygons(d.mask, class_id=0))
        if lines:
            n_with_vehicle += 1
            n_instances += len(lines)

        ext = s.image_path.suffix
        shutil.copy(s.image_path, config.YOLO_INSTANCE_DIR / "images" / split / f"{s.base}{ext}")
        (config.YOLO_INSTANCE_DIR / "labels" / split / f"{s.base}.txt").write_text("\n".join(lines))

    yaml_text = (
        f"path: {config.YOLO_INSTANCE_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: vehicle\n"
    )
    (config.YOLO_INSTANCE_DIR / "data.yaml").write_text(yaml_text)

    print(f"[ok] {len(samples)} images | {n_with_vehicle} with >=1 vehicle | "
          f"{n_instances} vehicle instances total")
    print(f"     wrote {config.YOLO_INSTANCE_DIR}")


if __name__ == "__main__":
    main()
