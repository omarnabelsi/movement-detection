"""CS2 Bot Detector — identifies enemy bots via motion + shape + color analysis.

Strategy
--------
1. Run the existing MOG2 motion detector to get candidate bounding boxes.
2. Apply a player-shape filter (aspect-ratio gate) to keep only regions that
   look like a standing/crouching CS2 player model.
3. Optionally run a colour pass that looks for the red/orange enemy indicator
   dots that CS2 draws above spotted enemies.
4. Merge both streams with greedy IoU-based NMS to eliminate duplicates.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from detection.motion_detector import MotionDetector

BBox = Tuple[int, int, int, int]   # (x, y, w, h)


class CS2BotDetector:
    """Detect CS2 enemy bots in a BGR frame.

    Args:
        motion_detector: A pre-configured :class:`MotionDetector` instance.
        use_color: Also search for red/orange enemy-indicator colours.
        min_h: Minimum bounding-box height (px) to consider a valid bot.
        max_h: Maximum bounding-box height (px).
    """

    # Enemy indicator: red in HSV (hue wraps around 180°)
    _HSV_RED_LO1 = np.array([0,   120, 120])
    _HSV_RED_HI1 = np.array([8,   255, 255])
    _HSV_RED_LO2 = np.array([172, 120, 120])
    _HSV_RED_HI2 = np.array([180, 255, 255])

    # CS2 enemy outline also uses orange
    _HSV_ORANGE_LO = np.array([8,  150, 150])
    _HSV_ORANGE_HI = np.array([25, 255, 255])

    # Acceptable width/height ratio for a CS2 player silhouette
    _MIN_ASPECT = 0.15   # very thin = far away
    _MAX_ASPECT = 0.75   # wide = close / crouching

    def __init__(
        self,
        motion_detector: MotionDetector,
        use_color: bool = True,
        min_h: int = 15,
        max_h: int = 500,
    ) -> None:
        self._motion = motion_detector
        self._use_color = use_color
        self._min_h = min_h
        self._max_h = max_h

    # ── Public API ────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[BBox]:
        """Return a list of (x, y, w, h) boxes for detected CS2 bots."""
        _, motion_bboxes = self._motion.detect(frame)
        player_bboxes = [b for b in motion_bboxes if self._is_player_shaped(b)]

        if not self._use_color:
            return player_bboxes

        color_bboxes = self._detect_by_color(frame)
        return self._nms(player_bboxes + color_bboxes)

    # ── Helpers ───────────────────────────────────────────────────────

    def _is_player_shaped(self, bbox: BBox) -> bool:
        x, y, w, h = bbox
        if h < self._min_h or h > self._max_h:
            return False
        if w == 0:
            return False
        return self._MIN_ASPECT <= (w / h) <= self._MAX_ASPECT

    def _detect_by_color(self, frame: np.ndarray) -> List[BBox]:
        """Locate red/orange enemy indicators and project a body region below them."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        red1   = cv2.inRange(hsv, self._HSV_RED_LO1,    self._HSV_RED_HI1)
        red2   = cv2.inRange(hsv, self._HSV_RED_LO2,    self._HSV_RED_HI2)
        orange = cv2.inRange(hsv, self._HSV_ORANGE_LO,  self._HSV_ORANGE_HI)
        mask   = cv2.bitwise_or(cv2.bitwise_or(red1, red2), orange)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask   = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        h_frame, w_frame = frame.shape[:2]
        bboxes: List[BBox] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 15:   # skip single-pixel noise
                continue
            cx, cy, cw, ch = cv2.boundingRect(cnt)

            # The indicator sits above the bot's head — estimate the full body
            # as ≈5× the indicator height, centred horizontally on the indicator
            body_h = max(int(ch * 5), 30)
            bx = max(0, cx - cw)
            by = cy
            bw = min(w_frame - bx, cw * 3)
            bh = min(h_frame - by, body_h)

            if bh < self._min_h:
                continue
            bboxes.append((bx, by, bw, bh))

        return bboxes

    def _nms(
        self,
        boxes: List[BBox],
        iou_threshold: float = 0.4,
    ) -> List[BBox]:
        """Greedy non-maximum suppression to remove redundant overlapping boxes."""
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        keep: List[BBox] = []
        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            boxes = [b for b in boxes if self._iou(best, b) < iou_threshold]
        return keep

    @staticmethod
    def _iou(a: BBox, b: BBox) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix  = max(ax, bx)
        iy  = max(ay, by)
        ix2 = min(ax + aw, bx + bw)
        iy2 = min(ay + ah, by + bh)
        if ix2 <= ix or iy2 <= iy:
            return 0.0
        inter = (ix2 - ix) * (iy2 - iy)
        union = aw * ah + bw * bh - inter
        return inter / max(union, 1e-6)
