"""Hand tracking module using MediaPipe Hands.

Extracts all 21 hand landmarks per frame and provides drawing utilities.
"""

import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    """Detects hands and extracts normalised landmarks via MediaPipe."""

    def __init__(
        self,
        max_hands: int = 1,
        detection_conf: float = 0.7,
        tracking_conf: float = 0.6,
    ):
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray):
        """Run hand detection on a BGR frame.

        Returns MediaPipe hand results (may contain no hands).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        return self._hands.process(rgb)

    def get_landmarks(self, results) -> np.ndarray | None:
        """Extract normalised (x, y) for 21 landmarks → flat array of 42.

        Returns None when no hand is detected.
        """
        if not results or not results.multi_hand_landmarks:
            return None
        hand = results.multi_hand_landmarks[0]
        coords = []
        for lm in hand.landmark:
            coords.extend([lm.x, lm.y])
        return np.array(coords, dtype=np.float32)

    def get_pixel_landmarks(
        self, results, frame_w: int, frame_h: int
    ) -> np.ndarray | None:
        """Extract pixel-space (x, y) for 21 landmarks → shape (21, 2).

        Returns None when no hand is detected.
        """
        if not results or not results.multi_hand_landmarks:
            return None
        hand = results.multi_hand_landmarks[0]
        pts = []
        for lm in hand.landmark:
            pts.append([int(lm.x * frame_w), int(lm.y * frame_h)])
        return np.array(pts, dtype=np.int32)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw hand landmarks and connections on *frame* (in-place)."""
        if results and results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_styles.get_default_hand_landmarks_style(),
                    self._mp_styles.get_default_hand_connections_style(),
                )
        return frame

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Release MediaPipe resources."""
        self._hands.close()
