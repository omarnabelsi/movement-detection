"""Utility constants and helpers for the gesture control system."""

import time


# ---------------------------------------------------------------------------
# Gesture label mapping
# ---------------------------------------------------------------------------
GESTURE_LABELS = {
    0: "move",
    1: "click",
    2: "scroll",
    3: "drag",
    4: "volume_up",
    5: "volume_down",
}

NUM_GESTURES = len(GESTURE_LABELS)
NUM_LANDMARKS = 21
INPUT_DIM = NUM_LANDMARKS * 2  # x, y per landmark


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------
class FPSCounter:
    """Rolling FPS estimate over the last *avg_frames* frames."""

    def __init__(self, avg_frames: int = 30):
        self._avg_frames = avg_frames
        self._timestamps: list[float] = []

    def tick(self) -> float:
        """Call once per frame; returns current FPS."""
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) > self._avg_frames:
            self._timestamps = self._timestamps[-self._avg_frames:]
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


# ---------------------------------------------------------------------------
# Screen helpers
# ---------------------------------------------------------------------------
def get_screen_size() -> tuple[int, int]:
    """Return (width, height) of the primary monitor."""
    import pyautogui
    return pyautogui.size()
