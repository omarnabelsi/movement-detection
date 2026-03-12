"""Automated gesture dataset recorder.

Show your hand to the webcam and press the corresponding digit key to
record labelled samples.  Press 'q' or ESC to save and exit.

Gestures:
    0 = move        1 = click       2 = scroll
    3 = drag        4 = volume_up   5 = volume_down
"""

import csv
import os

import cv2

from hand_tracker import HandTracker
from utils import GESTURE_LABELS, FPSCounter

DATASET_DIR = "dataset"
DATASET_FILE = os.path.join(DATASET_DIR, "gestures.csv")


def _csv_header() -> list[str]:
    """Build the CSV header: x0, y0, x1, y1, … x20, y20, label."""
    header = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}"])
    header.append("label")
    return header


def record():
    """Open the webcam and interactively record gesture samples."""
    os.makedirs(DATASET_DIR, exist_ok=True)

    tracker = HandTracker()
    fps = FPSCounter()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        return

    samples: list[list[float]] = []
    counts = {i: 0 for i in GESTURE_LABELS}

    print("=== Gesture Dataset Recorder ===")
    print("Show your hand and press a key to record:")
    for k, v in GESTURE_LABELS.items():
        print(f"  [{k}] {v}")
    print("Press 'q' or ESC to save and exit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # mirror for natural feeling

        results = tracker.process(frame)
        landmarks = tracker.get_landmarks(results)
        frame = tracker.draw(frame, results)

        # ---- HUD ---------------------------------------------------------
        current_fps = fps.tick()
        cv2.putText(
            frame, f"FPS: {current_fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        y_off = 60
        for k, v in GESTURE_LABELS.items():
            cv2.putText(
                frame, f"[{k}] {v}: {counts[k]}", (10, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            )
            y_off += 25

        status = "Hand detected" if landmarks is not None else "No hand"
        colour = (0, 255, 0) if landmarks is not None else (0, 0, 255)
        cv2.putText(
            frame, status, (10, y_off + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2,
        )

        cv2.imshow("Gesture Recorder", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("q"):
            break

        # Record sample when a digit key 0–5 is pressed and hand is visible
        if landmarks is not None and ord("0") <= key <= ord("5"):
            label = key - ord("0")
            row = landmarks.tolist() + [float(label)]
            samples.append(row)
            counts[label] += 1
            print(f"  Recorded '{GESTURE_LABELS[label]}' — total {counts[label]}")

    # ---- Save & cleanup --------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()

    if not samples:
        print("No samples recorded.")
        return

    file_exists = os.path.isfile(DATASET_FILE) and os.path.getsize(DATASET_FILE) > 0
    with open(DATASET_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(_csv_header())
        writer.writerows(samples)

    total = sum(counts.values())
    print(f"\nSaved {total} samples → {DATASET_FILE}")
    for k, v in GESTURE_LABELS.items():
        print(f"  {v}: {counts[k]}")


if __name__ == "__main__":
    record()
