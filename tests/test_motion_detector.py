"""Tests for detection.motion_detector."""
import unittest
import numpy as np
import cv2
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.motion_detector import MotionDetector


class TestMotionDetector(unittest.TestCase):

    def setUp(self):
        self.det = MotionDetector(min_area=100)

    def test_no_motion_on_static_frame(self):
        static = np.full((480, 640, 3), 128, dtype=np.uint8)
        for _ in range(15):
            self.det.detect(static)
        mask, bboxes = self.det.detect(static)
        self.assertEqual(len(bboxes), 0)

    def test_motion_detected_on_change(self):
        bg = np.full((480, 640, 3), 50, dtype=np.uint8)
        for _ in range(20):
            self.det.detect(bg)
        mv = bg.copy()
        cv2.rectangle(mv, (200, 200), (400, 400), (255, 255, 255), -1)
        mask, bboxes = self.det.detect(mv)
        self.assertGreater(len(bboxes), 0)

    def test_mask_is_binary(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mask, _ = self.det.detect(frame)
        self.assertTrue(set(np.unique(mask)).issubset({0, 255}))

    def test_min_area_filters_small(self):
        det = MotionDetector(min_area=50000)
        bg = np.full((480, 640, 3), 50, dtype=np.uint8)
        for _ in range(20):
            det.detect(bg)
        mv = bg.copy()
        cv2.rectangle(mv, (300, 300), (310, 310), (255, 255, 255), -1)
        _, bboxes = det.detect(mv)
        self.assertEqual(len(bboxes), 0)

    def test_reset_recreates_subtractor(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.det.detect(frame)
        self.det.reset()
        # After reset, subtractor should still work
        mask, _ = self.det.detect(frame)
        self.assertEqual(mask.shape, (480, 640))

    def test_small_noise_filtered(self):
        det = MotionDetector(min_area=500)
        bg = np.full((480, 640, 3), 100, dtype=np.uint8)
        for _ in range(20):
            det.detect(bg)
        noisy = bg.copy()
        for _ in range(5):
            x = np.random.randint(0, 640)
            y = np.random.randint(0, 480)
            noisy[y, x] = [255, 255, 255]
        _, bboxes = det.detect(noisy)
        self.assertEqual(len(bboxes), 0)


if __name__ == "__main__":
    unittest.main()
