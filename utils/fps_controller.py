"""
FPS controller and measurement utilities.

FPSController — limits processing speed to a target frame rate using
                time.sleep for smooth, predictable timing.
FPSCounter    — measures actual throughput over a sliding window.
"""
import time
from collections import deque


class FPSController:
    """Limits frame rate to a user-defined target using time.sleep.

    Call :meth:`tick` once per frame. It will block until the correct
    amount of time has elapsed to maintain the desired FPS.

    Args:
        target_fps: Maximum frames per second (default 30).
    """

    def __init__(self, target_fps: int = 30):
        self._target_fps = max(1, target_fps)
        self._interval = 1.0 / self._target_fps
        self._last_time = 0.0

    def tick(self):
        """Sleep if necessary to maintain the target frame interval."""
        now = time.perf_counter()
        elapsed = now - self._last_time
        remaining = self._interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_time = time.perf_counter()

    def set_fps(self, fps: int):
        """Change the target FPS at runtime."""
        self._target_fps = max(1, int(fps))
        self._interval = 1.0 / self._target_fps

    @property
    def target_fps(self) -> int:
        return self._target_fps


class FPSCounter:
    """Measures actual FPS over a sliding window of recent frames.

    Call :meth:`tick` after each frame is processed to record its timestamp.
    Read :attr:`fps` to get the current frames-per-second.

    Args:
        window: Number of recent timestamps to keep (larger → smoother).
    """

    def __init__(self, window: int = 30):
        self._stamps = deque(maxlen=window)
        self._fps = 0.0

    def tick(self):
        """Record a new frame timestamp and recompute FPS."""
        now = time.perf_counter()
        self._stamps.append(now)
        if len(self._stamps) >= 2:
            elapsed = self._stamps[-1] - self._stamps[0]
            if elapsed > 0:
                self._fps = (len(self._stamps) - 1) / elapsed

    @property
    def fps(self) -> float:
        return self._fps

    def reset(self):
        self._stamps.clear()
        self._fps = 0.0
