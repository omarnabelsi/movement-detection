"""
Motion detection using MOG2 adaptive background subtraction.

Pipeline (each step applied every frame):
    1. Convert BGR → grayscale
    2. Gaussian blur to suppress sensor noise
    3. MOG2 background subtraction
    4. Binary thresholding
    5. Erode — removes small specks of noise
    6. Dilate — closes gaps in detected regions
    7. Find external contours
    8. Filter by minimum contour area → bounding boxes

Only produces bounding boxes for *real* moving objects.
"""
import cv2
import numpy as np
from typing import List, Tuple


class MotionDetector:
    """Detects motion regions in video frames.

    Args:
        history:          Number of frames the background model remembers.
        var_threshold:    MOG2 variance threshold (lower → more sensitive).
        min_area:         Minimum contour area in pixels to count as motion.
        blur_size:        Gaussian blur kernel size (must be odd).
        erode_iterations: Erosion passes for noise removal.
        dilate_iterations: Dilation passes to fill gaps.
    """

    def __init__(
        self,
        history: int = 500,
        var_threshold: int = 25,
        min_area: int = 500,
        blur_size: int = 21,
        erode_iterations: int = 1,
        dilate_iterations: int = 3,
    ):
        # Background subtractor — detectShadows=False improves speed
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False,
        )
        self.min_area = min_area
        self._blur_ksize = (blur_size, blur_size)
        self._erode_iter = erode_iterations
        self._dilate_iter = dilate_iterations
        # Elliptical kernel works better than rectangular for organic shapes
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # ── Public API ───────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """Run the full motion detection pipeline on one BGR frame.

        Returns:
            (binary_mask, bboxes)
            bboxes is a list of (x, y, w, h) tuples for detected regions.
        """
        # Step 1 — Grayscale conversion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 2 — Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, self._blur_ksize, 0)

        # Step 3 — Background subtraction
        fg_mask = self._subtractor.apply(blurred)

        # Step 4 — Binary threshold (drop shadow/weak values)
        _, binary = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Step 5 — Erode to remove small noise
        binary = cv2.erode(binary, self._morph_kernel, iterations=self._erode_iter)

        # Step 6 — Dilate to close gaps
        binary = cv2.dilate(binary, self._morph_kernel, iterations=self._dilate_iter)

        # Step 7 — Find external contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 8 — Keep only contours above the minimum area
        bboxes: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            if cv2.contourArea(cnt) >= self.min_area:
                bboxes.append(cv2.boundingRect(cnt))

        return binary, bboxes

    def reset(self):
        """Re-create the background model so it re-learns from scratch."""
        h = self._subtractor.getHistory()
        v = self._subtractor.getVarThreshold()
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=h, varThreshold=v, detectShadows=False,
        )
