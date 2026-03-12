"""SORT (Simple Online and Real-time Tracking) implementation.

Uses Kalman filters for motion prediction and the Hungarian algorithm
for detection-to-track assignment.  This gives much more stable IDs
than centroid-only matching for fast-moving CS2 bots.

Reference:
    Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


# ═══════════════════════════════════════════════════════════════════════
# Single-object Kalman track
# ═══════════════════════════════════════════════════════════════════════

class _KalmanTrack:
    """Internal Kalman-filter based track for one object."""

    _id_counter = 0

    def __init__(self, bbox: BBox) -> None:
        _KalmanTrack._id_counter += 1
        self.id = _KalmanTrack._id_counter
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

        # State: [cx, cy, s, r, dx, dy, ds]
        #   cx,cy = centre;  s = area;  r = aspect ratio (w/h)
        #   dx,dy,ds = velocities
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=float)

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = _bbox_to_z(bbox).reshape(4, 1)

    def predict(self) -> BBox:
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return _z_to_bbox(self.kf.x[:4].flatten())

    def update(self, bbox: BBox) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(_bbox_to_z(bbox))

    def get_state(self) -> BBox:
        return _z_to_bbox(self.kf.x[:4].flatten())


# ═══════════════════════════════════════════════════════════════════════
# Coordinate conversions
# ═══════════════════════════════════════════════════════════════════════

def _bbox_to_z(bbox: BBox) -> np.ndarray:
    """(x, y, w, h) → (cx, cy, area, aspect_ratio)."""
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    s = w * h
    r = w / max(h, 1e-6)
    return np.array([cx, cy, s, r])


def _z_to_bbox(z: np.ndarray) -> BBox:
    """(cx, cy, area, aspect_ratio) → (x, y, w, h)."""
    cx, cy, s, r = z
    s = max(s, 1e-6)
    r = max(r, 1e-6)
    w = np.sqrt(s * r)
    h = s / max(w, 1e-6)
    x = cx - w / 2.0
    y = cy - h / 2.0
    return (int(x), int(y), int(w), int(h))


# ═══════════════════════════════════════════════════════════════════════
# IoU computation
# ═══════════════════════════════════════════════════════════════════════

def _iou_batch(bb_a: np.ndarray, bb_b: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of (x, y, w, h) boxes."""
    # Convert to (x1, y1, x2, y2)
    a = np.column_stack([bb_a[:, 0], bb_a[:, 1],
                         bb_a[:, 0] + bb_a[:, 2], bb_a[:, 1] + bb_a[:, 3]])
    b = np.column_stack([bb_b[:, 0], bb_b[:, 1],
                         bb_b[:, 0] + bb_b[:, 2], bb_b[:, 1] + bb_b[:, 3]])

    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)

    area_a = a[:, 2] - a[:, 0]
    area_a *= a[:, 3] - a[:, 1]
    area_b = b[:, 2] - b[:, 0]
    area_b *= b[:, 3] - b[:, 1]

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


# ═══════════════════════════════════════════════════════════════════════
# SORT Tracker
# ═══════════════════════════════════════════════════════════════════════

class SORTTracker:
    """SORT multi-object tracker with Kalman prediction + Hungarian matching.

    Args:
        max_age:      Max frames a track survives without a match before deletion.
        min_hits:     Min detection hits before a track is considered confirmed.
        iou_threshold: Minimum IoU for a valid match.
    """

    def __init__(
        self,
        max_age: int = 15,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: List[_KalmanTrack] = []

    def update(self, bboxes: List[BBox]) -> Dict[int, BBox]:
        """Run one tracking step.

        Args:
            bboxes: Detected (x, y, w, h) bounding boxes for the current frame.

        Returns:
            Dict mapping track_id → (x, y, w, h) for active confirmed tracks.
        """
        # Predict existing tracks forward
        predicted = np.array([t.predict() for t in self._tracks]) if self._tracks else np.empty((0, 4))
        det_array = np.array(bboxes) if bboxes else np.empty((0, 4))

        # Match detections to tracks via IoU + Hungarian
        matched, unmatched_dets, unmatched_trks = self._associate(det_array, predicted)

        # Update matched tracks
        for d_idx, t_idx in matched:
            self._tracks[t_idx].update(tuple(det_array[d_idx].astype(int)))

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            self._tracks.append(_KalmanTrack(tuple(det_array[d_idx].astype(int))))

        # Remove dead tracks
        self._tracks = [
            t for t in self._tracks if t.time_since_update <= self.max_age
        ]

        # Build output — only confirmed tracks
        result: Dict[int, BBox] = {}
        for t in self._tracks:
            if t.time_since_update == 0 and t.hits >= self.min_hits:
                result[t.id] = t.get_state()
        return result

    def reset(self) -> None:
        """Clear all tracks."""
        self._tracks.clear()
        _KalmanTrack._id_counter = 0

    # ── Association ──────────────────────────────────────────────────

    def _associate(
        self,
        detections: np.ndarray,
        predictions: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Hungarian matching between detections and predicted track positions."""
        if len(predictions) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(predictions)))

        iou_matrix = _iou_batch(detections, predictions)
        cost = 1.0 - iou_matrix

        row_idx, col_idx = linear_sum_assignment(cost)

        matched: List[Tuple[int, int]] = []
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(range(len(predictions)))

        for r, c in zip(row_idx, col_idx):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched.append((r, c))
                unmatched_detections.discard(r)
                unmatched_tracks.discard(c)

        return matched, sorted(unmatched_detections), sorted(unmatched_tracks)
