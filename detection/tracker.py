"""
Centroid-based multi-object tracker.

Assigns a persistent integer ID to each detected bounding box and maintains
that ID across frames by matching centroids with Euclidean distance.

When a tracked object disappears for more than *max_disappeared* consecutive
frames it is deregistered and the ID is not reused.
"""
import numpy as np
from typing import Dict, List, Tuple


class CentroidTracker:
    """Tracks objects across frames using centroid matching.

    Args:
        max_disappeared: Number of consecutive frames an object may vanish
                         before it is removed from tracking.
    """

    def __init__(self, max_disappeared: int = 30):
        self._next_id: int = 1
        self._centroids: Dict[int, Tuple[float, float]] = {}
        self._bboxes: Dict[int, Tuple[int, int, int, int]] = {}
        self._disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared

    # ── Public API ───────────────────────────────────────────────────

    def update(
        self, bboxes: List[Tuple[int, int, int, int]]
    ) -> Dict[int, Tuple[int, int, int, int]]:
        """Match new detections against existing objects.

        Args:
            bboxes: List of (x, y, w, h) bounding boxes for the current frame.

        Returns:
            Dict mapping object_id → (x, y, w, h) for every active object.
        """
        # ── No detections this frame ─────────────────────────────────
        if not bboxes:
            for oid in list(self._disappeared):
                self._disappeared[oid] += 1
                if self._disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return dict(self._bboxes)

        # ── Compute input centroids ──────────────────────────────────
        input_centroids = np.array(
            [(x + w / 2.0, y + h / 2.0) for x, y, w, h in bboxes]
        )

        # ── First frame — register everything ────────────────────────
        if not self._centroids:
            for i, bbox in enumerate(bboxes):
                self._register(input_centroids[i], bbox)
            return dict(self._bboxes)

        # ── Match existing objects → new detections by distance ──────
        obj_ids = list(self._centroids.keys())
        obj_cents = np.array([self._centroids[oid] for oid in obj_ids])

        # Pairwise Euclidean distance matrix (N_existing × N_new)
        diff = obj_cents[:, np.newaxis, :] - input_centroids[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))

        # Greedy assignment: shortest distance first
        rows = dists.min(axis=1).argsort()
        cols = dists.argmin(axis=1)[rows]

        used_rows: set = set()
        used_cols: set = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            oid = obj_ids[row]
            self._centroids[oid] = tuple(input_centroids[col])
            self._bboxes[oid] = bboxes[col]
            self._disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects (increment disappeared count)
        for row in range(len(obj_ids)):
            if row not in used_rows:
                oid = obj_ids[row]
                self._disappeared[oid] += 1
                if self._disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)

        # Handle unmatched new detections (register as new objects)
        for col in range(len(bboxes)):
            if col not in used_cols:
                self._register(input_centroids[col], bboxes[col])

        return dict(self._bboxes)

    def reset(self):
        """Clear all tracked objects and reset the ID counter."""
        self._centroids.clear()
        self._bboxes.clear()
        self._disappeared.clear()
        self._next_id = 1

    # ── Internal helpers ─────────────────────────────────────────────

    def _register(self, centroid, bbox):
        oid = self._next_id
        self._next_id += 1
        self._centroids[oid] = tuple(centroid)
        self._bboxes[oid] = bbox
        self._disappeared[oid] = 0

    def _deregister(self, oid):
        del self._centroids[oid]
        del self._bboxes[oid]
        del self._disappeared[oid]
