"""
Webcam video source using OpenCV VideoCapture.

Provides a simple interface to capture frames from an attached camera.
"""
import cv2


class CameraSource:
    """Captures frames from a webcam.

    Args:
        index: Camera device index (0 = default webcam).
        width:  Requested capture width in pixels.
        height: Requested capture height in pixels.
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self._index = index
        self._width = width
        self._height = height
        self._cap = None

    # ── Public interface ─────────────────────────────────────────────

    def open(self) -> bool:
        """Open the camera device. Returns True on success."""
        self._cap = cv2.VideoCapture(self._index, cv2.CAP_DSHOW)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        return self.is_opened()

    def get_frame(self):
        """Read one frame from the webcam.

        Returns:
            (True, frame) on success, (False, None) on failure.
        """
        if not self.is_opened():
            return False, None
        return self._cap.read()

    def is_opened(self) -> bool:
        """Check whether the camera is currently open."""
        return self._cap is not None and self._cap.isOpened()

    def release(self):
        """Release the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
