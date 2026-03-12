"""Prediction smoothing via sliding-window majority vote.

Smooths noisy per-frame gesture predictions over the last *window* frames
to produce stable, flicker-free gesture output.
"""

from collections import Counter, deque


class GestureSmoother:
    """Buffer the last *window* predictions and return the majority label."""

    def __init__(self, window: int = 5):
        self._history: deque[int] = deque(maxlen=window)

    def update(self, label: int) -> int:
        """Add a new prediction and return the smoothed (majority) label."""
        self._history.append(label)
        counts = Counter(self._history)
        return counts.most_common(1)[0][0]

    def reset(self):
        """Clear the history (e.g. when hand disappears)."""
        self._history.clear()
