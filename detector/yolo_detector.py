"""YOLOv8 object detector wrapper for CS2 bot detection.

Provides a unified interface for inference with confidence filtering,
NMS, size filtering, and optional GPU acceleration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


BBox = Tuple[int, int, int, int]  # (x, y, w, h)


class YOLODetector:
    """YOLOv8-based detector for CS2 player/bot detection.

    Args:
        model_path: Path to a ``.pt`` weights file **or** a built-in name
            like ``"yolov8n.pt"`` to auto-download.
        conf_threshold: Minimum detection confidence to keep.
        iou_threshold:  IoU threshold for non-maximum suppression.
        device:         ``"cuda"``, ``"cpu"``, or ``""`` (auto-select).
        classes:        List of class indices to detect (``None`` = all).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "",
        classes: Optional[List[int]] = None,
    ) -> None:
        self._model_path = model_path
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._classes = classes

        # Auto-select GPU if available
        if device == "":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        self._model: Optional[YOLO] = None

    # ── Lazy loading ─────────────────────────────────────────────────

    def load(self) -> None:
        """Load the YOLO model into memory."""
        print(f"[yolo] Loading model: {self._model_path}  (device={self._device})")
        self._model = YOLO(self._model_path)
        # Warm-up with a dummy tensor on the target device
        dummy = torch.zeros(1, 3, 640, 640).to(self._device)
        self._model.predict(dummy, verbose=False)
        print("[yolo] Model loaded and warmed up")

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    # ── Inference ────────────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        conf: float | None = None,
        iou: float | None = None,
    ) -> List[Dict]:
        """Run detection on a single BGR frame.

        Returns:
            List of dicts, each with:
                ``bbox``  — (x, y, w, h) in pixels
                ``conf``  — confidence score
                ``cls``   — class index
                ``label`` — class name string
        """
        self._ensure_loaded()
        results = self._model.predict(
            frame,
            conf=conf or self._conf,
            iou=iou or self._iou,
            device=self._device,
            classes=self._classes,
            verbose=False,
        )

        detections: List[Dict] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                w, h = x2 - x1, y2 - y1
                detections.append({
                    "bbox": (int(x1), int(y1), int(w), int(h)),
                    "conf": float(boxes.conf[i].cpu()),
                    "cls": int(boxes.cls[i].cpu()),
                    "label": self._model.names[int(boxes.cls[i].cpu())],
                })
        return detections

    def detect_batch(
        self,
        frames: List[np.ndarray],
        conf: float | None = None,
        iou: float | None = None,
    ) -> List[List[Dict]]:
        """Run batch inference on multiple frames.

        Returns:
            List (one per frame) of detection-dict lists.
        """
        self._ensure_loaded()
        results = self._model.predict(
            frames,
            conf=conf or self._conf,
            iou=iou or self._iou,
            device=self._device,
            classes=self._classes,
            verbose=False,
        )

        batch_dets: List[List[Dict]] = []
        for r in results:
            dets: List[Dict] = []
            boxes = r.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    w, h = x2 - x1, y2 - y1
                    dets.append({
                        "bbox": (int(x1), int(y1), int(w), int(h)),
                        "conf": float(boxes.conf[i].cpu()),
                        "cls": int(boxes.cls[i].cpu()),
                        "label": self._model.names[int(boxes.cls[i].cpu())],
                    })
            batch_dets.append(dets)
        return batch_dets

    # ── Accessors ────────────────────────────────────────────────────

    @property
    def class_names(self) -> Dict[int, str]:
        self._ensure_loaded()
        return dict(self._model.names)

    @property
    def device(self) -> str:
        return self._device

    def set_conf(self, value: float) -> None:
        self._conf = max(0.01, min(1.0, value))

    def set_iou(self, value: float) -> None:
        self._iou = max(0.01, min(1.0, value))
