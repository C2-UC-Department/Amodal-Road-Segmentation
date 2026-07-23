"""Step 1: Download and unpack the original KITTI Road dataset.

Usage:
    python -m src.s1_download_kitti          # download + extract
    python -m src.s1_download_kitti --force  # re-download even if present

If the automatic download fails (KITTI occasionally rate-limits / requires a
browser), download `data_road.zip` manually from
    https://www.cvlibs.net/datasets/kitti/eval_road.php
and drop it at data/raw/data_road.zip, then re-run this script to extract.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def download(url: str, dest: Path, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[skip] {dest} already exists ({dest.stat().st_size / 1e6:.0f} MB)")
        return
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def extract(zip_path: Path, out_dir: Path) -> None:
    print(f"[extract] {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    # KITTI zip extracts to <out_dir>/data_road/{training,testing}
    train = config.KITTI_ROOT / "training" / "image_2"
    if not train.exists():
        raise RuntimeError(
            f"Extraction finished but {train} is missing. Check the zip contents."
        )
    print(f"[ok] KITTI ready at {config.KITTI_ROOT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download the zip")
    args = ap.parse_args()

    try:
        download(config.KITTI_URL, config.KITTI_ZIP, force=args.force)
    except Exception as e:  # noqa: BLE001
        print(f"[error] automatic download failed: {e}")
        if not config.KITTI_ZIP.exists():
            print(
                "Please download data_road.zip manually (see module docstring) "
                f"to {config.KITTI_ZIP} and re-run."
            )
            sys.exit(1)
        print("[info] using previously downloaded zip.")

    extract(config.KITTI_ZIP, config.RAW_DIR)


if __name__ == "__main__":
    main()
