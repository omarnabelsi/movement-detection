"""Map recognised gestures to OS-level actions using PyAutoGUI.

Gesture → Action:
    move         → move cursor (index-finger tip drives position)
    click        → left click
    scroll       → scroll up
    drag         → toggle drag (mouseDown / mouseUp)
    volume_up    → volume-up media key
    volume_down  → volume-down media key
"""

import time

import pyautogui

from utils import get_screen_size

# PyAutoGUI safety: moving mouse to corner still triggers failsafe
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0  # we handle cooldown ourselves


class MouseController:
    """Translate gesture labels into mouse / keyboard actions."""

    def __init__(self, cooldown: float = 0.4, move_smooth: float = 0.35):
        """
        Args:
            cooldown:    minimum seconds between discrete actions (click, etc.)
            move_smooth: interpolation factor for cursor movement (0–1).
                         Lower = smoother but laggier.
        """
        self._screen_w, self._screen_h = get_screen_size()
        self._cooldown = cooldown
        self._smooth = move_smooth
        self._last_action: float = 0.0
        self._dragging = False
        self._prev_x: float | None = None
        self._prev_y: float | None = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def execute(self, gesture: str, pixel_landmarks=None,
                cam_w: int = 640, cam_h: int = 480):
        """Run the action that maps to *gesture*.

        Args:
            gesture:          one of the GESTURE_LABELS values.
            pixel_landmarks:  (21, 2) int array of hand landmarks in pixel
                              coords — only needed for 'move' / 'drag'.
            cam_w, cam_h:     webcam capture resolution (for coordinate mapping).
        """
        if gesture == "move":
            self._move_cursor(pixel_landmarks, cam_w, cam_h)
            return

        # Cooldown gate for discrete (non-continuous) actions
        now = time.time()
        if now - self._last_action < self._cooldown:
            return
        self._last_action = now

        if gesture == "click":
            self._end_drag()
            pyautogui.click()
        elif gesture == "scroll":
            pyautogui.scroll(3)
        elif gesture == "drag":
            self._toggle_drag()
        elif gesture == "volume_up":
            pyautogui.press("volumeup")
        elif gesture == "volume_down":
            pyautogui.press("volumedown")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _move_cursor(self, landmarks, cam_w: int, cam_h: int):
        """Move the cursor toward the index-finger-tip position."""
        if landmarks is None:
            return
        # Landmark 8 = index finger tip
        ix, iy = int(landmarks[8][0]), int(landmarks[8][1])

        # Remap webcam pixel coords → screen coords
        target_x = int(ix / cam_w * self._screen_w)
        target_y = int(iy / cam_h * self._screen_h)

        # Exponential smoothing for jitter reduction
        if self._prev_x is not None:
            target_x = int(self._prev_x + self._smooth * (target_x - self._prev_x))
            target_y = int(self._prev_y + self._smooth * (target_y - self._prev_y))
        self._prev_x, self._prev_y = target_x, target_y

        pyautogui.moveTo(target_x, target_y, _pause=False)

    def _toggle_drag(self):
        if self._dragging:
            pyautogui.mouseUp()
            self._dragging = False
        else:
            pyautogui.mouseDown()
            self._dragging = True

    def _end_drag(self):
        if self._dragging:
            pyautogui.mouseUp()
            self._dragging = False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def release(self):
        """Call on shutdown to ensure drag is released."""
        self._end_drag()
