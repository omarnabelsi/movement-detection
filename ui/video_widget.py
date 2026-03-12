"""
QLabel-based video display widget.

Converts OpenCV BGR frames to QPixmap and scales them to fit the widget
while keeping the original aspect ratio.
"""
import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


class VideoWidget(QLabel):
    """Displays OpenCV frames inside a PyQt5 layout."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 240)
        self.setStyleSheet(
            "background-color: #090c10; color: #484f58; font-size: 15px;"
        )
        self.setText("No Video Source")
        self._pixmap = None

    def update_frame(self, frame: np.ndarray):
        """Convert a BGR OpenCV frame and display it.

        Args:
            frame: BGR numpy array from OpenCV.
        """
        if frame is None:
            return

        # Clear placeholder text on first real frame
        if self.text():
            self.setText("")

        # BGR → RGB, ensure contiguous memory for QImage
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h, w, ch = rgb.shape

        # QImage wraps the buffer — .copy() to decouple from numpy lifetime
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg.copy())
        self._fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def clear_display(self):
        """Reset to placeholder state."""
        self.clear()
        self._pixmap = None
        self.setText("No Video Source")

    # ── Internal ─────────────────────────────────────────────────────

    def _fit(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)
