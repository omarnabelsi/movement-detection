"""
Screen capture source using the MSS library.

Captures a full monitor at high speed and converts BGRA → BGR for OpenCV.
"""
import numpy as np
import mss
import cv2


class ScreenSource:
    """Captures screen frames using MSS.

    Args:
        monitor_index: 1-based monitor index (1 = primary display).
    """

    def __init__(self, monitor_index: int = 1):
        self._monitor_index = monitor_index
        self._sct = None
        self._monitor = None

    # ── Public interface ─────────────────────────────────────────────

    def open(self, monitor_index: int = None) -> bool:
        """Start screen capture on the given monitor.

        If *monitor_index* is provided it overrides the constructor value.
        Returns True on success.
        """
        if monitor_index is not None:
            self._monitor_index = monitor_index
        try:
            self._sct = mss.mss()
            monitors = self._sct.monitors
            # Index 0 is the virtual "all monitors" screen — skip it.
            if self._monitor_index < 1 or self._monitor_index >= len(monitors):
                self._monitor_index = 1
            self._monitor = monitors[self._monitor_index]
            return True
        except Exception:
            self._sct = None
            return False

    def get_frame(self):
        """Grab a single screenshot.

        Returns:
            (True, BGR_frame) on success, (False, None) on failure.
        """
        if self._sct is None or self._monitor is None:
            return False, None
        try:
            shot = self._sct.grab(self._monitor)
            frame = np.array(shot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return True, frame
        except Exception:
            return False, None

    def is_opened(self) -> bool:
        """Check whether screen capture is active."""
        return self._sct is not None and self._monitor is not None

    def release(self):
        """Release screen capture resources."""
        if self._sct is not None:
            self._sct.close()
            self._sct = None
            self._monitor = None

    # ── Static helpers ───────────────────────────────────────────────

    @staticmethod
    def get_available_monitors() -> list:
        """Return a list of physical monitors (excluding the virtual combined screen).

        Each entry is a dict with keys: index, left, top, width, height.
        """
        try:
            with mss.mss() as sct:
                return [
                    {
                        "index": i,
                        "left": m["left"],
                        "top": m["top"],
                        "width": m["width"],
                        "height": m["height"],
                    }
                    for i, m in enumerate(sct.monitors)
                    if i > 0
                ]
        except Exception:
            return []
