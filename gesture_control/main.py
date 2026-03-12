"""Hand Gesture Computer Control — main entry point.

Modes:
    python main.py                                          → real-time gesture control
    python main.py --record                                 → open the dataset recorder
    python main.py --from-images --images-root PATH         → extract landmarks from image dataset
    python main.py --from-hagrid-json --json-dir PATH       → extract landmarks from HaGRID JSONs
    python main.py --train                                  → train the gesture classifier

Press ESC to exit any mode cleanly.
"""

import argparse

import cv2

from hand_tracker import HandTracker
from gesture_classifier import GestureClassifier
from mouse_controller import MouseController
from smoothing import GestureSmoother
from utils import FPSCounter, GESTURE_LABELS


def run_inference():
    """Real-time gesture recognition and computer control loop."""
    tracker = HandTracker()
    classifier = GestureClassifier()
    controller = MouseController(cooldown=0.4)
    smoother = GestureSmoother(window=5)
    fps_counter = FPSCounter()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: cannot open webcam.")
        return

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam opened at {cam_w}×{cam_h}. Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Hand detection
        results = tracker.process(frame)
        landmarks = tracker.get_landmarks(results)
        pixel_lms = tracker.get_pixel_landmarks(results, w, h)
        frame = tracker.draw(frame, results)

        gesture_text = "No hand"
        conf_text = ""

        if landmarks is not None:
            # Classify and smooth
            label_id, _, confidence = classifier.predict(landmarks)
            smoothed_id = smoother.update(label_id)
            gesture_name = GESTURE_LABELS[smoothed_id]
            gesture_text = gesture_name
            conf_text = f"{confidence:.0%}"

            # Execute mapped action
            controller.execute(gesture_name, pixel_lms, cam_w, cam_h)
        else:
            smoother.reset()

        # ---- HUD ---------------------------------------------------------
        current_fps = fps_counter.tick()
        cv2.putText(
            frame, f"FPS: {current_fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        cv2.putText(
            frame, f"Gesture: {gesture_text}", (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2,
        )
        if conf_text:
            cv2.putText(
                frame, f"Conf: {conf_text}", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1,
            )

        cv2.imshow("Gesture Control", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    # ---- Cleanup ---------------------------------------------------------
    controller.release()
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print("Exited cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Hand Gesture Computer Control")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--record", action="store_true",
                       help="Open the dataset recorder")
    group.add_argument("--train", action="store_true",
                       help="Train the gesture classifier")
    group.add_argument("--from-images", action="store_true",
                       help="Extract landmarks from a HaGRID image dataset")
    group.add_argument("--from-hagrid-json", action="store_true",
                       help="Extract landmarks from HaGRID annotation JSON files (no images needed)")
    parser.add_argument("--images-root", default=None,
                        help="HaGRID root folder (required with --from-images)")
    parser.add_argument("--json-dir", default=None,
                        help="Folder with downloaded HaGRID JSON files (required with --from-hagrid-json)")
    parser.add_argument("--max-per-class", type=int, default=500,
                        help="Max images per gesture class when using --from-images (default 400)")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Training epochs (default 60)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default 0.001)")
    parser.add_argument("--batch", type=int, default=64,
                        help="Batch size (default 64)")
    args = parser.parse_args()

    if args.record:
        from dataset_recorder import record
        record()
    elif args.train:
        from train_model import train
        train(epochs=args.epochs, lr=args.lr, batch_size=args.batch)
    elif args.from_images:
        if not args.images_root:
            parser.error("--from-images requires --images-root PATH")
        from image_dataset_to_csv import extract_landmarks_from_images
        extract_landmarks_from_images(
            root=args.images_root,
            max_per_class=args.max_per_class,
        )
    elif args.from_hagrid_json:
        if not args.json_dir:
            parser.error("--from-hagrid-json requires --json-dir PATH")
        from hagrid_json_to_csv import extract_landmarks_from_json
        extract_landmarks_from_json(
            json_dir=args.json_dir,
            max_per_class=args.max_per_class,
        )
    else:
        run_inference()


if __name__ == "__main__":
    main()
