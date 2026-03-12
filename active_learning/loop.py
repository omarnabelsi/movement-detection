"""Active learning loop for CS2 bot detection.

Saves frames where the model is uncertain (low confidence or border-line
detections) so they can be manually reviewed, corrected, and added to
the training dataset for the next retraining cycle.

Usage (in pipeline):
    loop = ActiveLearningLoop("data/active_learning")
    loop.maybe_save(frame, detections)      # called each frame
    loop.export_for_labeling()              # batch export when ready

Usage (CLI):
    python -m active_learning.loop --dir data/active_learning --summary
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


class ActiveLearningLoop:
    """Collects uncertain frames for human review and retraining.

    Frames are saved when:
    - No detections at all (possible false negative scenario).
    - Maximum detection confidence is below ``uncertain_threshold``.
    - Multiple overlapping detections (ambiguous scene).

    Args:
        output_dir:           Where to save uncertain frames.
        uncertain_threshold:  Confidence below which a detection is flagged.
        save_no_detections:   Also save frames with zero detections.
        max_saved:            Cap on how many frames to store (0 = unlimited).
        cooldown_sec:         Minimum seconds between saves (avoid flooding).
    """

    def __init__(
        self,
        output_dir: str = "data/active_learning",
        uncertain_threshold: float = 0.45,
        save_no_detections: bool = True,
        max_saved: int = 2000,
        cooldown_sec: float = 1.0,
    ) -> None:
        self._dir = Path(output_dir)
        self._img_dir = self._dir / "images"
        self._meta_dir = self._dir / "metadata"
        self._img_dir.mkdir(parents=True, exist_ok=True)
        self._meta_dir.mkdir(parents=True, exist_ok=True)

        self.uncertain_threshold = uncertain_threshold
        self.save_no_detections = save_no_detections
        self.max_saved = max_saved
        self.cooldown_sec = cooldown_sec

        self._count = len(list(self._img_dir.glob("*.png")))
        self._last_save_time = 0.0

    # ── Per-frame hook ───────────────────────────────────────────────

    def maybe_save(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        frame_id: int = 0,
    ) -> bool:
        """Check if this frame should be saved for review.

        Args:
            frame:      BGR image.
            detections: List of detection dicts from ``YOLODetector.detect()``.
            frame_id:   Optional sequential frame number.

        Returns:
            True if the frame was saved.
        """
        if self.max_saved and self._count >= self.max_saved:
            return False

        now = time.time()
        if now - self._last_save_time < self.cooldown_sec:
            return False

        should_save = False
        reason = ""

        if not detections and self.save_no_detections:
            should_save = True
            reason = "no_detections"
        elif detections:
            max_conf = max(d.get("conf", 1.0) for d in detections)
            if max_conf < self.uncertain_threshold:
                should_save = True
                reason = f"low_conf_{max_conf:.3f}"

        if not should_save:
            return False

        # Save image
        ts = int(now * 1000)
        name = f"uncertain_{ts}_{frame_id:06d}"
        cv2.imwrite(str(self._img_dir / f"{name}.png"), frame)

        # Save metadata
        meta = {
            "timestamp": ts,
            "frame_id": frame_id,
            "reason": reason,
            "detections": [
                {
                    "bbox": list(d["bbox"]),
                    "conf": d.get("conf", 0),
                    "cls": d.get("cls", 0),
                    "label": d.get("label", ""),
                }
                for d in detections
            ],
        }
        with open(self._meta_dir / f"{name}.json", "w") as f:
            json.dump(meta, f, indent=2)

        self._count += 1
        self._last_save_time = now
        return True

    # ── Export for labeling ──────────────────────────────────────────

    def export_for_labeling(self, dest: str | None = None) -> str:
        """Copy saved images to a flat directory ready for labeling tools.

        Returns:
            Path to the export directory.
        """
        if dest is None:
            dest = str(self._dir / "to_label")
        out = Path(dest)
        out.mkdir(parents=True, exist_ok=True)

        count = 0
        for img in self._img_dir.glob("*.png"):
            shutil.copy2(img, out / img.name)
            count += 1

        print(f"[active_learning] Exported {count} frames → {out}")
        return str(out)

    # ── Stats ────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        """Return a summary of saved uncertain frames."""
        reasons = {}
        for meta_file in self._meta_dir.glob("*.json"):
            with open(meta_file) as f:
                m = json.load(f)
            r = m.get("reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1

        info = {"total_saved": self._count, "reasons": reasons}
        print(f"[active_learning] {self._count} uncertain frames saved")
        for r, c in sorted(reasons.items()):
            print(f"  {r}: {c}")
        return info

    @property
    def count(self) -> int:
        return self._count


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Active learning loop manager")
    parser.add_argument("--dir", default="data/active_learning")
    parser.add_argument("--summary", action="store_true", help="Print summary")
    parser.add_argument("--export", action="store_true", help="Export for labeling")
    args = parser.parse_args()

    loop = ActiveLearningLoop(output_dir=args.dir)
    if args.summary:
        loop.summary()
    if args.export:
        loop.export_for_labeling()


if __name__ == "__main__":
    main()
