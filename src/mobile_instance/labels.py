"""Binary instance mask -> YOLO-seg polygon label lines.

YOLO's segmentation label format (one .txt per image, one line per instance):
    <class_id> x1 y1 x2 y2 ... xn yn
with every coordinate normalized to [0, 1] by the image's own (w, h).

A mask can split into multiple disjoint contours (e.g. a parked car occluded
by a lamppost, or the teacher's mask having a small disconnected sliver) --
YOLO-seg wants one polygon per line, so each contour becomes its OWN label
line, same class id. This slightly over-counts instances in that specific
case, but there is only one class here ("vehicle"), so it never causes a
cross-class labeling error, only an occasional harmless split of one real
vehicle into two training instances.
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_CONTOUR_AREA_PX = 40   # drop slivers too small to be a meaningful instance


def mask_to_yolo_polygons(mask: np.ndarray, class_id: int = 0) -> list[str]:
    """`mask`: bool/uint8 (H, W). Returns formatted YOLO-seg label lines."""
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA_PX:
            continue
        # Simplify to keep label files small; epsilon scaled to the contour's
        # own perimeter so both tiny and large vehicles simplify proportionally.
        eps = 0.002 * cv2.arcLength(contour, True)
        poly = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
        if len(poly) < 3:
            continue
        coords = []
        for x, y in poly:
            coords.append(f"{x / w:.6f}")
            coords.append(f"{y / h:.6f}")
        lines.append(f"{class_id} " + " ".join(coords))
    return lines
