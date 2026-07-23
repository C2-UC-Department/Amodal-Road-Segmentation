"""Run the automated pipeline (Steps 1-3) end to end.

    python run_pipeline.py                 # download -> visible masks -> foreground -> occlusion
    python run_pipeline.py --skip-download # if KITTI is already extracted
    python run_pipeline.py --limit 10      # only process 10 images (quick trial)

Step 4 (interactive annotation) is launched separately:
    python -m src.annotator
"""
from __future__ import annotations

import argparse
import runpy
import sys


def _run(module: str, argv: list[str]) -> None:
    print(f"\n=== {module} {' '.join(argv)} ===")
    sys.argv = [module] + argv
    runpy.run_module(module, run_name="__main__")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="limit foreground detection to N images")
    ap.add_argument("--preview", action="store_true", help="write QA overlays in steps 2 & 3")
    args = ap.parse_args()

    if not args.skip_download:
        _run("src.s1_download_kitti", [])
    _run("src.s2_prepare_visible_masks", [])

    fg_argv = []
    if args.limit:
        fg_argv += ["--limit", str(args.limit)]
    if args.preview:
        fg_argv += ["--preview"]
    _run("src.s3_detect_foreground", fg_argv)

    occ_argv = ["--preview"] if args.preview else []
    _run("src.s4_build_incomplete", occ_argv)

    # Step 5: OFRSNet input — 11-class semantic maps.
    sem_argv = list(fg_argv)  # same --limit/--preview flags apply
    _run("src.s5_export_semantics", sem_argv)

    print("\n[done] Automated steps complete. Now annotate:")
    print("       python -m src.annotator")
    print("Then train OFRSNet:")
    print("       python -m src.ofrs.train")


if __name__ == "__main__":
    main()
