"""CS2 Bot Detection Pipeline — end-to-end orchestrator.

Ties together YOLO detection, CS2 heuristic filters, SORT tracking,
active learning, and annotation rendering into a single callable
that processes one frame at a time.

This is the main "brain" that the UI worker thread calls.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from detector.yolo_detector import YOLODetector
from detector.cs2_heuristics import CS2Heuristics
from tracker.sort_tracker import SORTTracker
from active_learning.loop import ActiveLearningLoop


BBox = Tuple[int, int, int, int]


class CS2DetectorPipeline:
    """Full CS2 bot detection + tracking pipeline.

    Args:
        model_path:    Path to trained YOLO ``.pt`` weights.
        conf:          Detection confidence threshold.
        iou:           NMS IoU threshold.
        device:        ``""`` = auto-select GPU/CPU.
        tracker_max_age:    SORT tracker max frames without match.
        tracker_min_hits:   SORT minimum hits to confirm a track.
        tracker_iou:        SORT IoU matching threshold.
        heuristics:         Optional pre-configured :class:`CS2Heuristics`.
        active_learning:    Optional :class:`ActiveLearningLoop`.
    """

    # ── Annotation colors (BGR) ──────────────────────────────────────
    COLOR_BOT = (0, 165, 255)       # orange
    COLOR_CONFIRMED = (0, 255, 0)   # green (high-confidence)
    COLOR_UNCERTAIN = (0, 255, 255) # yellow (low-confidence)

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.35,
        iou: float = 0.45,
        device: str = "",
        tracker_max_age: int = 15,
        tracker_min_hits: int = 3,
        tracker_iou: float = 0.3,
        heuristics: Optional[CS2Heuristics] = None,
        active_learning: Optional[ActiveLearningLoop] = None,
    ) -> None:
        self._detector = YOLODetector(
            model_path=model_path,
            conf_threshold=conf,
            iou_threshold=iou,
            device=device,
        )
        self._heuristics = heuristics or CS2Heuristics()
        self._tracker = SORTTracker(
            max_age=tracker_max_age,
            min_hits=tracker_min_hits,
            iou_threshold=tracker_iou,
        )
        self._active_learning = active_learning
        self._frame_count = 0

    # ── Main processing step ─────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Run full pipeline on one BGR frame.

        Returns:
            (annotated_frame, info_dict)
            info_dict keys:
                ``detections``    — raw YOLO detections (post-heuristics)
                ``tracked``       — dict of track_id → bbox
                ``tracked_count`` — number of active tracks
                ``fps``           — (caller should fill in)
                ``cs2_mode``      — always True
        """
        self._frame_count += 1

        # 1) YOLO detection
        raw_dets = self._detector.detect(frame)

        # 2) CS2 heuristic filtering
        filtered = self._heuristics.filter(raw_dets, frame_shape=frame.shape[:2])

        # 3) Extract bboxes for tracker
        bboxes = self._heuristics.extract_bboxes(filtered)

        # 4) SORT tracking
        tracked = self._tracker.update(bboxes)

        # 5) Active learning (save uncertain frames)
        if self._active_learning:
            self._active_learning.maybe_save(frame, filtered, self._frame_count)

        # 6) Draw annotations
        annotated = self._annotate(frame, filtered, tracked)

        info = {
            "raw_detections": len(raw_dets),
            "filtered_detections": len(filtered),
            "tracked": tracked,
            "tracked_count": len(tracked),
            "cs2_mode": True,
        }
        return annotated, info

    # ── Annotation rendering ─────────────────────────────────────────

    def _annotate(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        tracked: Dict[int, BBox],
    ) -> np.ndarray:
        """Draw bounding boxes and labels on the frame."""
        out = frame.copy()

        # Draw tracked objects (stable IDs)
        for tid, (x, y, w, h) in tracked.items():
            clr = self.COLOR_BOT
            cv2.rectangle(out, (x, y), (x + w, y + h), clr, 2)

            label = f"BOT {tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ly = max(y - th - 8, 0)
            cv2.rectangle(out, (x, ly), (x + tw + 6, ly + th + 6), clr, -1)
            cv2.putText(
                out, label, (x + 3, ly + th + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )

        # Status overlay
        n = len(tracked)
        status = f"CS2: {n} bot{'s' if n != 1 else ''}"
        (sw, sh), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (8, 8), (sw + 16, sh + 18), (0, 0, 0), -1)
        clr = self.COLOR_BOT if n else (140, 140, 140)
        cv2.putText(
            out, status, (12, sh + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 2, cv2.LINE_AA,
        )

        return out

    # ── Accessors ────────────────────────────────────────────────────

    def set_conf(self, value: float) -> None:
        self._detector.set_conf(value)

    def set_iou(self, value: float) -> None:
        self._detector.set_iou(value)

    def reset_tracker(self) -> None:
        self._tracker.reset()

    def set_min_box_area(self, area: int) -> None:
        """Update heuristic minimum bbox area at runtime."""
        self._heuristics.min_box_area = max(0, int(area))

    @property
    def detector(self) -> YOLODetector:
        return self._detector

    @property
    def tracker(self) -> SORTTracker:
        return self._tracker
