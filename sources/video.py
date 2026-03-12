"""
Video file source using OpenCV VideoCapture.

Reads frames sequentially from an on-disk video file.
Returns (False, None) when the file ends so the caller can handle gracefully.
"""
import cv2


class VideoSource:
    """Reads frames from a video file.

    Args:
        path: Filesystem path to the video file.
    """

    def __init__(self, path: str = ""):
        self._path = path
        self._cap = None

    # ── Public interface ─────────────────────────────────────────────

    def open(self, path: str = None) -> bool:
        """Open a video file. Returns True on success."""
        if path is not None:
            self._path = path
        if not self._path:
            return False
        self._cap = cv2.VideoCapture(self._path)
        return self.is_opened()

    def get_frame(self):
        """Read the next frame from the video.

        Returns:
            (True, frame) on success, (False, None) when video ends or on error.
        """
        if not self.is_opened():
            return False, None
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        return True, frame

    def is_opened(self) -> bool:
        """Check whether the video file is open."""
        return self._cap is not None and self._cap.isOpened()

    def get_fps(self) -> float:
        """Return the video file's native FPS."""
        if self._cap is not None:
            return self._cap.get(cv2.CAP_PROP_FPS)
        return 0.0

    def release(self):
        """Release the video file handle."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
