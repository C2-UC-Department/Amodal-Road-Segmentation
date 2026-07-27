"""Step 4: Interactive amodal road annotation tool (OpenCV HighGUI).

Loads, per image:
  * the original RGB image,
  * the visible-road mask     (blue overlay),
  * the foreground-object mask (red overlay),
  * the occlusion hint        (yellow overlay),
and lets you paint/erase the *amodal* road mask (green overlay) to reconstruct
road hidden behind foreground objects. The amodal mask is saved as a 0/255 PNG
to data/amodal_road/<base>.png, preserving the source's base name.

Two sources are supported (--source):
  kitti   (default) -- visible road comes from KITTI's ground truth; occlusion
          hint from data/processed/occlusion/; unannotated frames seed the
          amodal mask from the visible road alone.
  footage -- no ground truth exists. "Visible road" is instead Mask2Former's
          own road class (from data/processed/semantic_ofrs/, run
          `python -m src.s6_prepare_footage` first); the occlusion hint is
          computed on the fly. Unannotated frames are pre-seeded by
          s6_prepare_footage from the model's own composed prediction
          (Mask2Former road + OFRSNet patch), so you correct the model's
          mistakes directly instead of drawing from scratch.

This tool uses OpenCV's native window (the project's pyenv Python has no Tk).
It never imports torch -- footage seeding is a separate precompute step
(s6_prepare_footage.py), keeping this editor lightweight.

Run:
    python -m src.annotator
    python -m src.annotator --source footage
    python -m src.annotator --start 0

Controls
--------
  Left drag            paint amodal road (brush)
  Right drag           erase amodal road
  Middle drag          pan
  Mouse wheel          zoom towards cursor      (or  + / -  keys)
  h j k l              pan left/down/up/right
  [ / ]                brush smaller / larger   (or the 'brush' trackbar)
  , / .                overlay less/more opaque  (or the 'alpha' trackbar)
  1 2 3 4              toggle visible / foreground / occlusion / amodal overlays
  z / y                undo / redo
  f                    auto-fill: amodal = visible road + occlusion hint
  c                    clear amodal back to the visible road
  b                    fill brush area with FOREGROUND value (paint over object)
  r                    reset view (fit to window)
  s                    save now
  n / p                next / previous image (autosaves first)
  q / Esc              quit (autosaves)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import common  # noqa: E402

WINDOW = "amodal-road-annotator"
MAX_CANVAS_W, MAX_CANVAS_H = 1280, 760

# Overlay colours in RGB.
COL_VISIBLE = (60, 120, 255)
COL_FOREGROUND = (255, 60, 60)
COL_OCCLUSION = (255, 210, 0)
COL_AMODAL = (40, 220, 90)
UNDO_LIMIT = 40

ROAD_IDX = config.OFRS_CLASSES.index("road")


def _wheel_delta(flags: int) -> int:
    """Extract the scroll direction from an OpenCV mouse-event flags value.

    OpenCV 5 dropped cv2.getMouseWheelDelta; the signed delta lives in the
    high 16 bits of `flags` (same convention as the C++ implementation).
    """
    if hasattr(cv2, "getMouseWheelDelta"):
        return cv2.getMouseWheelDelta(flags)
    return int(np.int16((flags >> 16) & 0xFFFF))


class Annotator:
    def __init__(self, source: str, start: int):
        self.source = source
        self.samples = list(common.iter_source_samples(source))
        if not self.samples:
            raise SystemExit(f"No samples found for source '{source}'.")
        self.idx = max(0, min(start, len(self.samples) - 1))

        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.brush = 18
        self.alpha = 0.5
        self.show = {"visible": True, "foreground": True,
                     "occlusion": True, "amodal": True}

        self._drag = None            # 'paint' | 'erase' | 'pan' | None
        self._last_img_pt = None
        self._pan_anchor = None
        self._dirty = True

        # per-image data
        self.rgb = self.visible = self.foreground = self.occlusion = None
        self.amodal = None
        self.cw = MAX_CANVAS_W
        self.ch = MAX_CANVAS_H
        self.undo_stack: list[np.ndarray] = []
        self.redo_stack: list[np.ndarray] = []

        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        cv2.createTrackbar("alpha", WINDOW, int(self.alpha * 100), 90, self._tb_alpha)
        cv2.createTrackbar("brush", WINDOW, self.brush, 120, self._tb_brush)
        self.load_sample()

    # -------------------------------------------------------- trackbars
    def _tb_alpha(self, v):
        self.alpha = max(0.1, min(0.9, v / 100.0))
        self._dirty = True

    def _tb_brush(self, v):
        self.brush = max(1, v)

    # ------------------------------------------------------- data load
    def _opt_mask(self, folder: Path, base: str, shape) -> np.ndarray:
        p = folder / f"{base}.png"
        return common.read_mask(p) if p.exists() else np.zeros(shape, bool)

    def load_sample(self) -> None:
        s = self.samples[self.idx]
        self.rgb = common.read_rgb(s.image_path)
        h, w = self.rgb.shape[:2]
        self.foreground = self._opt_mask(config.FOREGROUND_DIR, s.base, (h, w))

        if self.source == "kitti":
            self.visible = self._opt_mask(config.VISIBLE_DIR, s.base, (h, w))
            self.occlusion = self._opt_mask(config.OCCLUSION_DIR, s.base, (h, w))
        else:
            # footage: no ground truth -- "visible road" is Mask2Former's own
            # road class (run s6_prepare_footage first); occlusion hint is
            # computed on the fly, same formula as Step 3 for KITTI.
            sem_path = config.SEMANTIC_DIR / f"{s.base}.png"
            if sem_path.exists():
                sem = cv2.imread(str(sem_path), cv2.IMREAD_GRAYSCALE)
                self.visible = sem == ROAD_IDX
            else:
                print(f"[warn] no semantic map for {s.base}; "
                      "run `python -m src.s6_prepare_footage` first")
                self.visible = np.zeros((h, w), bool)
            self.occlusion = common.occluder_blob_mask(self.foreground, self.visible,
                                                       config.ROAD_NEIGHBOURHOOD_PX)

        amodal_path = config.AMODAL_DIR / f"{s.base}.png"
        if amodal_path.exists():
            self.amodal = common.read_mask(amodal_path).astype(np.uint8)
        else:
            self.amodal = self.visible.astype(np.uint8).copy()

        # Canvas fits the image within the max window, keeping aspect ratio.
        scale = min(MAX_CANVAS_W / w, MAX_CANVAS_H / h, 4.0)
        self.cw = max(1, int(round(w * scale)))
        self.ch = max(1, int(round(h * scale)))

        self.undo_stack.clear()
        self.redo_stack.clear()
        self.reset_view()

    # ------------------------------------------------- view transforms
    def reset_view(self) -> None:
        h, w = self.rgb.shape[:2]
        self.zoom = min(self.cw / w, self.ch / h)
        self.pan_x = (w - self.cw / self.zoom) / 2.0
        self.pan_y = (h - self.ch / self.zoom) / 2.0
        self._dirty = True

    def canvas_to_image(self, cx, cy):
        return int(self.pan_x + cx / self.zoom), int(self.pan_y + cy / self.zoom)

    # --------------------------------------------------------- render
    def render(self) -> None:
        h, w = self.rgb.shape[:2]
        buf = np.full((self.ch, self.cw, 3), 32, np.uint8)  # dark grey background

        x0 = int(np.floor(max(0, self.pan_x)))
        y0 = int(np.floor(max(0, self.pan_y)))
        x1 = int(np.ceil(min(w, self.pan_x + self.cw / self.zoom)))
        y1 = int(np.ceil(min(h, self.pan_y + self.ch / self.zoom)))
        if x1 > x0 and y1 > y0:
            crop = self.rgb[y0:y1, x0:x1].astype(np.float32).copy()

            def blend(mask, color):
                m = mask[y0:y1, x0:x1].astype(bool)
                if m.any():
                    crop[m] = (1 - self.alpha) * crop[m] + self.alpha * np.asarray(color, np.float32)

            if self.show["visible"]:
                blend(self.visible, COL_VISIBLE)
            if self.show["foreground"]:
                blend(self.foreground, COL_FOREGROUND)
            if self.show["occlusion"]:
                blend(self.occlusion, COL_OCCLUSION)
            if self.show["amodal"]:
                blend(self.amodal.astype(bool), COL_AMODAL)
            crop = np.clip(crop, 0, 255).astype(np.uint8)

            rw = max(1, int(round((x1 - x0) * self.zoom)))
            rh = max(1, int(round((y1 - y0) * self.zoom)))
            interp = cv2.INTER_NEAREST if self.zoom >= 1 else cv2.INTER_AREA
            resized = cv2.resize(crop, (rw, rh), interpolation=interp)

            px = int(round((x0 - self.pan_x) * self.zoom))
            py = int(round((y0 - self.pan_y) * self.zoom))
            # Clip the paste into the buffer.
            bx0, by0 = max(0, px), max(0, py)
            sx0, sy0 = bx0 - px, by0 - py
            bx1 = min(self.cw, px + rw)
            by1 = min(self.ch, py + rh)
            if bx1 > bx0 and by1 > by0:
                buf[by0:by1, bx0:bx1] = resized[sy0:sy0 + (by1 - by0),
                                                sx0:sx0 + (bx1 - bx0)]

        self._draw_hud(buf)
        cv2.imshow(WINDOW, cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))
        self._dirty = False

    def _draw_hud(self, buf) -> None:
        s = self.samples[self.idx]
        layers = "".join(k[0].upper() if v else k[0].lower()
                         for k, v in self.show.items())
        txt = (f"[{self.idx + 1}/{len(self.samples)}] {s.base}  "
               f"zoom {self.zoom:.2f}  brush {self.brush}  a {self.alpha:.1f}  "
               f"VFOA:{layers}  px {int(self.amodal.sum())}")
        cv2.rectangle(buf, (0, 0), (self.cw, 22), (0, 0, 0), -1)
        cv2.putText(buf, txt, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # -------------------------------------------------------- painting
    def _push_undo(self):
        self.undo_stack.append(self.amodal.copy())
        if len(self.undo_stack) > UNDO_LIMIT:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _paint(self, cx, cy, value):
        ix, iy = self.canvas_to_image(cx, cy)
        r = max(1, int(self.brush))
        if self._last_img_pt is not None:
            cv2.line(self.amodal, self._last_img_pt, (ix, iy),
                     int(value), thickness=r * 2, lineType=cv2.LINE_8)
        cv2.circle(self.amodal, (ix, iy), r, int(value), -1)
        self._last_img_pt = (ix, iy)
        self._dirty = True

    # ----------------------------------------------------- mouse events
    def _on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_MOUSEWHEEL:
            self._zoom_at(x, y, 1 if _wheel_delta(flags) > 0 else -1)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self._push_undo()
            self._drag = "paint"
            self._last_img_pt = None
            self._paint(x, y, 1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._push_undo()
            self._drag = "erase"
            self._last_img_pt = None
            self._paint(x, y, 0)
        elif event == cv2.EVENT_MBUTTONDOWN:
            self._drag = "pan"
            self._pan_anchor = (x, y, self.pan_x, self.pan_y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._drag == "paint":
                self._paint(x, y, 1)
            elif self._drag == "erase":
                self._paint(x, y, 0)
            elif self._drag == "pan":
                ax, ay, px, py = self._pan_anchor
                self.pan_x = px - (x - ax) / self.zoom
                self.pan_y = py - (y - ay) / self.zoom
                self._dirty = True
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP,
                       cv2.EVENT_MBUTTONUP):
            self._drag = None
            self._last_img_pt = None

    def _zoom_at(self, cx, cy, direction):
        factor = 1.15 if direction > 0 else 1 / 1.15
        ix, iy = self.canvas_to_image(cx, cy)
        self.zoom = float(np.clip(self.zoom * factor, 0.05, 40.0))
        self.pan_x = ix - cx / self.zoom
        self.pan_y = iy - cy / self.zoom
        self._dirty = True

    def _pan_by(self, dx, dy):
        step = 60 / self.zoom
        self.pan_x += dx * step
        self.pan_y += dy * step
        self._dirty = True

    # ---------------------------------------------------------- edit ops
    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.amodal.copy())
            self.amodal = self.undo_stack.pop()
            self._dirty = True

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.amodal.copy())
            self.amodal = self.redo_stack.pop()
            self._dirty = True

    def auto_fill(self):
        self._push_undo()
        self.amodal = (self.amodal.astype(bool) | self.visible |
                       self.occlusion).astype(np.uint8)
        self._dirty = True

    def clear_to_visible(self):
        self._push_undo()
        self.amodal = self.visible.astype(np.uint8).copy()
        self._dirty = True

    # -------------------------------------------------------------- save
    def save(self):
        s = self.samples[self.idx]
        common.write_mask(config.AMODAL_DIR / f"{s.base}.png", self.amodal)

    # --------------------------------------------------------------- nav
    def next_image(self):
        self.save()
        if self.idx < len(self.samples) - 1:
            self.idx += 1
            self.load_sample()

    def prev_image(self):
        self.save()
        if self.idx > 0:
            self.idx -= 1
            self.load_sample()

    # -------------------------------------------------------------- loop
    def run(self):
        while True:
            if self._dirty:
                self.render()
            key = cv2.waitKey(15) & 0xFF
            if key == 255:
                # window closed via the red button?
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
            ch = chr(key) if 32 <= key < 127 else ""
            if ch in ("q",) or key == 27:  # q / Esc
                break
            elif ch == "n":
                self.next_image()
            elif ch == "p":
                self.prev_image()
            elif ch == "s":
                self.save()
            elif ch == "z":
                self.undo()
            elif ch == "y":
                self.redo()
            elif ch == "f":
                self.auto_fill()
            elif ch == "c":
                self.clear_to_visible()
            elif ch == "r":
                self.reset_view()
            elif ch == "[":
                self.brush = max(1, self.brush - 3)
                cv2.setTrackbarPos("brush", WINDOW, self.brush)
            elif ch == "]":
                self.brush = min(120, self.brush + 3)
                cv2.setTrackbarPos("brush", WINDOW, self.brush)
            elif ch in ("+", "="):
                self._zoom_at(self.cw // 2, self.ch // 2, 1)
            elif ch == "-":
                self._zoom_at(self.cw // 2, self.ch // 2, -1)
            elif ch == ",":
                self._tb_alpha(int(self.alpha * 100) - 8)
                cv2.setTrackbarPos("alpha", WINDOW, int(self.alpha * 100))
            elif ch == ".":
                self._tb_alpha(int(self.alpha * 100) + 8)
                cv2.setTrackbarPos("alpha", WINDOW, int(self.alpha * 100))
            elif ch in ("1", "2", "3", "4"):
                layer = ["visible", "foreground", "occlusion", "amodal"][int(ch) - 1]
                self.show[layer] = not self.show[layer]
                self._dirty = True
            elif ch == "h":
                self._pan_by(-1, 0)
            elif ch == "l":
                self._pan_by(1, 0)
            elif ch == "k":
                self._pan_by(0, -1)
            elif ch == "j":
                self._pan_by(0, 1)

        self.save()
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["kitti", "footage"], default="kitti")
    ap.add_argument("--start", type=int, default=0, help="index to start at")
    args = ap.parse_args()

    config.AMODAL_DIR.mkdir(parents=True, exist_ok=True)
    Annotator(source=args.source, start=args.start).run()


if __name__ == "__main__":
    main()
