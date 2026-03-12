"""Tests for detection.tracker (CentroidTracker)."""
import unittest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.tracker import CentroidTracker


class TestCentroidTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = CentroidTracker(max_disappeared=5)

    def test_register_new_objects(self):
        bboxes = [(100, 100, 50, 50), (300, 300, 60, 60)]
        tracked = self.tracker.update(bboxes)
        self.assertEqual(len(tracked), 2)

    def test_persistent_ids(self):
        tracked1 = self.tracker.update([(100, 100, 50, 50)])
        id1 = list(tracked1.keys())[0]
        # Same position next frame
        tracked2 = self.tracker.update([(102, 102, 50, 50)])
        id2 = list(tracked2.keys())[0]
        self.assertEqual(id1, id2, "Same object should keep the same ID")

    def test_object_disappears_after_max_frames(self):
        self.tracker.update([(100, 100, 50, 50)])
        for _ in range(6):
            self.tracker.update([])
        tracked = self.tracker.update([])
        self.assertEqual(len(tracked), 0)

    def test_new_object_gets_new_id(self):
        t1 = self.tracker.update([(100, 100, 50, 50)])
        t2 = self.tracker.update([(100, 100, 50, 50), (400, 400, 50, 50)])
        ids = list(t2.keys())
        self.assertEqual(len(ids), 2)
        self.assertNotEqual(ids[0], ids[1])

    def test_reset_clears_all(self):
        self.tracker.update([(100, 100, 50, 50)])
        self.tracker.reset()
        tracked = self.tracker.update([])
        self.assertEqual(len(tracked), 0)

    def test_empty_input_no_crash(self):
        tracked = self.tracker.update([])
        self.assertEqual(len(tracked), 0)


if __name__ == "__main__":
    unittest.main()
