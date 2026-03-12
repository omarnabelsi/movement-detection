"""CS2-specific post-processing heuristics for detection filtering.

Applied **after** YOLO inference to reduce false positives by:
- Size filtering (too small / too large for a player)
- Aspect-ratio gating (player-shaped bounding boxes)
- Optional ROI masking (ignore HUD / minimap regions)
- UI element exclusion (top/bottom strips where HUD lives)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


BBox = Tuple[int, int, int, int]  # (x, y, w, h)


class CS2Heuristics:
    """Post-processing filters tuned for CS2 gameplay frames.

    Args:
        min_box_h:   Minimum bounding box height (px).
        max_box_h:   Maximum bounding box height (px).
        min_box_area: Minimum bounding box area in pixels (w*h).
        min_aspect:  Minimum width/height ratio for a player.
        max_aspect:  Maximum width/height ratio.
        hud_top:     Fraction of frame height at top to ignore (minimap, HP bar).
        hud_bottom:  Fraction of frame height at bottom to ignore (weapon bar).
        min_conf:    Additional confidence gate (applied *after* YOLO NMS).
        roi:         Optional (x, y, w, h) region of interest — only keep detections
                     whose center falls inside this rectangle.
    """

    def __init__(
        self,
        min_box_h: int = 20,
        max_box_h: int = 600,
        min_box_area: int = 0,
        min_aspect: float = 0.15,
        max_aspect: float = 0.80,
        hud_top: float = 0.0,
        hud_bottom: float = 0.0,
        min_conf: float = 0.0,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        self.min_box_h = min_box_h
        self.max_box_h = max_box_h
        self.min_box_area = max(0, int(min_box_area))
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.hud_top = hud_top
        self.hud_bottom = hud_bottom
        self.min_conf = min_conf
        self.roi = roi

    def filter(
        self,
        detections: List[Dict],
        frame_shape: Tuple[int, int] | None = None,
    ) -> List[Dict]:
        """Apply all heuristic filters to YOLO detection dicts.

        Args:
            detections:  List from ``YOLODetector.detect()``.
            frame_shape: (height, width) of the frame for HUD masking.

        Returns:
            Filtered list of detection dicts.
        """
        kept: List[Dict] = []

        fh, fw = frame_shape if frame_shape else (0, 0)

        for det in detections:
            x, y, w, h = det["bbox"]
            conf = det.get("conf", 1.0)

            # Confidence gate
            if conf < self.min_conf:
                continue

            # Size filter
            if h < self.min_box_h or h > self.max_box_h:
                continue
            if (w * h) < self.min_box_area:
                continue

            # Aspect ratio
            if w > 0 and h > 0:
                ar = w / h
                if ar < self.min_aspect or ar > self.max_aspect:
                    continue

            # HUD masking (ignore detections in top/bottom HUD strips)
            if fh > 0:
                cy = y + h / 2
                if self.hud_top > 0 and cy < fh * self.hud_top:
                    continue
                if self.hud_bottom > 0 and cy > fh * (1.0 - self.hud_bottom):
                    continue

            # Region-of-interest filter
            if self.roi is not None:
                rx, ry, rw, rh = self.roi
                cx = x + w / 2
                cy = y + h / 2
                if not (rx <= cx <= rx + rw and ry <= cy <= ry + rh):
                    continue

            kept.append(det)

        return kept

    def extract_bboxes(self, detections: List[Dict]) -> List[BBox]:
        """Extract plain (x, y, w, h) tuples from detection dicts."""
        return [d["bbox"] for d in detections]
