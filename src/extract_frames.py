"""Extract 2D image snapshots from your own footage (data/footage) for
manual segmentation correction.

Handles iPhone/QuickTime video (.mov/.mp4/.m4v, incl. HEVC) via ffmpeg, which
auto-applies rotation metadata, and still images (.heic/.jpg/.png) via sips /
copy. Frames are sampled at a target rate so you get a manageable set to label.

Output (default data/footage_frames/):
    <video-stem>_f00001.jpg, <video-stem>_f00002.jpg, ...
    <image-stem>.jpg                       (for still images)

Usage:
    python -m src.extract_frames                       # 1 fps, native size
    python -m src.extract_frames --fps 0.25            # 1 frame / 4 s
    python -m src.extract_frames --max-size 1280       # cap longest side (downscale only)
    python -m src.extract_frames --max-per-video 30    # cap frames per clip
    python -m src.extract_frames --input some/dir --out some/frames --overwrite

Requires ffmpeg/ffprobe on PATH (brew install ffmpeg). No GPU/venv needed.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm"}
STILL_EXTS = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def require(tool: str) -> None:
    if shutil.which(tool) is None:
        raise SystemExit(f"'{tool}' not found on PATH. Install it (brew install ffmpeg).")


def safe_stem(path: Path) -> str:
    """Filesystem-friendly stem (spaces/odd chars -> underscore)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)


def probe_duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except Exception:  # noqa: BLE001
        return 0.0


def build_vf(fps: float, max_size: int | None) -> str:
    filters = [f"fps={fps}"]
    if max_size:
        # Downscale-only: fit within a max_size box, keep aspect, even dims.
        filters.append(
            f"scale='min({max_size},iw)':'min({max_size},ih)':"
            "force_original_aspect_ratio=decrease:force_divisible_by=2")
    return ",".join(filters)


def extract_video(path: Path, out_dir: Path, fps: float, max_size: int | None,
                  fmt: str, max_per_video: int, overwrite: bool) -> int:
    stem = safe_stem(path)
    existing = sorted(out_dir.glob(f"{stem}_f*.{fmt}"))
    if existing and not overwrite:
        return 0  # already done
    for p in existing:
        p.unlink()

    pattern = str(out_dir / f"{stem}_f%05d.{fmt}")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
           "-i", str(path), "-vf", build_vf(fps, max_size), "-vsync", "vfr"]
    if max_per_video > 0:
        cmd += ["-frames:v", str(max_per_video)]
    if fmt in ("jpg", "jpeg"):
        cmd += ["-q:v", "2"]           # high-quality JPEG
    cmd += [pattern]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[warn] ffmpeg failed on {path.name}: {res.stderr.strip()[:200]}")
        return 0
    return len(list(out_dir.glob(f"{stem}_f*.{fmt}")))


def extract_still(path: Path, out_dir: Path, fmt: str, max_size: int | None,
                  overwrite: bool) -> int:
    dst = out_dir / f"{safe_stem(path)}.{fmt}"
    if dst.exists() and not overwrite:
        return 0
    ext = path.suffix.lower()
    if ext in (".heic", ".heif"):
        # sips converts HEIC and can resample if a max size is given.
        cmd = ["sips", "-s", "format", "jpeg" if fmt in ("jpg", "jpeg") else fmt]
        if max_size:
            cmd += ["-Z", str(max_size)]
        cmd += [str(path), "--out", str(dst)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[warn] sips failed on {path.name}: {res.stderr.strip()[:200]}")
            return 0
    else:
        # ffmpeg handles ordinary raster stills (and optional downscale).
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
               "-i", str(path)]
        if max_size:
            cmd += ["-vf", f"scale='min({max_size},iw)':'min({max_size},ih)':"
                    "force_original_aspect_ratio=decrease:force_divisible_by=2"]
        if fmt in ("jpg", "jpeg"):
            cmd += ["-q:v", "2"]
        cmd += ["-y", str(dst)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[warn] ffmpeg failed on {path.name}: {res.stderr.strip()[:200]}")
            return 0
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(config.FOOTAGE_DIR))
    ap.add_argument("--out", default=str(config.FRAMES_DIR))
    ap.add_argument("--fps", type=float, default=1.0,
                    help="frames sampled per second of video (default 1.0)")
    ap.add_argument("--max-size", type=int, default=0,
                    help="cap longest side in px (downscale only; 0 = native)")
    ap.add_argument("--max-per-video", type=int, default=0,
                    help="cap frames per clip (0 = unlimited)")
    ap.add_argument("--format", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    require("ffmpeg")
    require("ffprobe")

    in_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_size = args.max_size or None

    files = sorted(p for p in in_dir.rglob("*")
                   if p.suffix.lower() in VIDEO_EXTS | STILL_EXTS)
    if not files:
        raise SystemExit(f"No videos/images found under {in_dir}")

    videos = [p for p in files if p.suffix.lower() in VIDEO_EXTS]
    stills = [p for p in files if p.suffix.lower() in STILL_EXTS]
    est = sum(probe_duration(v) for v in videos) * args.fps
    print(f"[plan] {len(videos)} videos (~{est:.0f} frames @ {args.fps} fps) + "
          f"{len(stills)} stills -> {out_dir}")

    total = 0
    for i, v in enumerate(videos, 1):
        n = extract_video(v, out_dir, args.fps, max_size, args.format,
                          args.max_per_video, args.overwrite)
        total += n
        print(f"  [{i}/{len(videos)}] {v.name}: {n} frames"
              + (" (skipped, exists)" if n == 0 and not args.overwrite else ""))
    for s in stills:
        total += extract_still(s, out_dir, args.format, max_size, args.overwrite)

    n_out = len(list(out_dir.glob(f"*.{args.format}")))
    print(f"[ok] {n_out} snapshots in {out_dir}")
    print("Next: run the model on them ->")
    print(f"      python -m src.predict --input {out_dir}")


if __name__ == "__main__":
    main()
